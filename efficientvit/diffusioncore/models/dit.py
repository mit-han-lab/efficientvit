# modified from https://github.com/facebookresearch/DiT
import math
from dataclasses import dataclass
from typing import Optional

import diffusers
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed

from efficientvit.diffusioncore.models.dit_sampler import create_diffusion
from efficientvit.models.utils.network import get_device

__all__ = ["DiTConfig", "DiT", "dc_ae_dit_xl_in_512px"]


@dataclass
class DiTConfig:
    name: str = "DiT"

    input_size: int = 32
    patch_size: int = 2
    in_channels: int = 4
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    post_norm: bool = False
    class_dropout_prob: float = 0.1
    num_classes: int = 1000
    learn_sigma: bool = True
    unconditional: bool = False
    use_checkpoint: bool = True

    pretrained_path: Optional[str] = None
    pretrained_source: str = "dc-ae"  # dit

    eval_scheduler: str = "GaussianDiffusion"
    num_inference_steps: int = 250
    train_scheduler: str = "GaussianDiffusion"


def modulate(x, shift, scale, base: float = 1):
    return x * (base + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, post_norm=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.post_norm = post_norm
        if not post_norm:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        else:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 4 * hidden_size, bias=True))

    def forward(self, x, c):
        if not self.post_norm:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            shift_msa, scale_msa, shift_mlp, scale_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
            x = x + modulate(self.norm1(self.attn(x)), shift_msa, scale_msa, base=0)
            x = x + modulate(self.norm2(self.mlp(x)), shift_mlp, scale_mlp, base=0)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg

        self.out_channels = cfg.in_channels * 2 if cfg.learn_sigma else cfg.in_channels

        self.x_embedder = PatchEmbed(cfg.input_size, cfg.patch_size, cfg.in_channels, cfg.hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(cfg.hidden_size)
        if not cfg.unconditional:
            self.y_embedder = LabelEmbedder(cfg.num_classes, cfg.hidden_size, cfg.class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, cfg.hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(cfg.hidden_size, cfg.num_heads, mlp_ratio=cfg.mlp_ratio, post_norm=cfg.post_norm)
                for _ in range(cfg.depth)
            ]
        )
        self.final_layer = FinalLayer(cfg.hidden_size, cfg.patch_size, self.out_channels)
        if cfg.pretrained_path is not None:
            self.load_model()
        else:
            self.initialize_weights()

        # scheduler
        if cfg.eval_scheduler == "GaussianDiffusion":
            self.eval_scheduler = create_diffusion(str(250))
        elif cfg.eval_scheduler == "UniPC":
            self.eval_scheduler = diffusers.UniPCMultistepScheduler(
                solver_order=3,
                # rescale_betas_zero_snr=False,
                prediction_type="epsilon",
            )
        else:
            raise NotImplementedError(f"eval_scheduler {cfg.eval_scheduler} is not supported")

        if cfg.train_scheduler == "GaussianDiffusion":
            self.train_scheduler = create_diffusion(timestep_respacing="")
        else:
            raise NotImplementedError(f"train_scheduler {cfg.train_scheduler} is not supported")

    def get_trainable_modules(self) -> nn.ModuleDict:
        return nn.ModuleDict({"dit": self})

    def load_model(self):
        checkpoint = torch.load(self.cfg.pretrained_path, map_location="cpu", weights_only=True)
        if self.cfg.pretrained_source == "dit":
            if "ema" in checkpoint:
                checkpoint = checkpoint["ema"]
            self.load_state_dict(checkpoint)
        elif self.cfg.pretrained_source == "dc-ae":
            checkpoint = list(checkpoint["ema"].values())[0]
            self.get_trainable_modules().load_state_dict(checkpoint)
        else:
            raise NotImplementedError(f"pretrained source {self.cfg.pretrained_source} is not supported")

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        if not self.cfg.unconditional:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def forward_without_cfg(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)  # (N, D)
        if self.cfg.unconditional:
            c = t
        else:
            y = self.y_embedder(y, self.training)  # (N, D)
            c = t + y  # (N, D)
        for block in self.blocks:
            if self.cfg.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c)  # (N, T, D)
            else:
                x = block(x, c)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward_without_cfg(combined, t, y)
        eps, rest = model_out[:, : self.cfg.in_channels], model_out[:, self.cfg.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def forward(self, x, y, generator: Optional[torch.Generator] = None):
        info = {}
        device = x.device
        if self.cfg.train_scheduler == "GaussianDiffusion":
            model_kwargs = dict(y=y)
            timesteps = torch.randint(0, self.train_scheduler.num_timesteps, (x.shape[0],), device=device)
            loss_dict = self.train_scheduler.training_losses(self.forward_without_cfg, x, timesteps, model_kwargs)
            loss = loss_dict["loss"].mean()
        else:
            raise NotImplementedError(f"train scheduler {self.cfg.train_scheduler} is not supported")
        info["loss_dict"] = {"loss": loss}
        return loss, info

    @torch.no_grad()
    def generate(
        self, inputs, null_inputs, scale: float = 1.5, generator: Optional[torch.Generator] = None, progress=False
    ):
        device = get_device(self)
        samples = torch.randn(
            (inputs.shape[0], self.cfg.in_channels, self.cfg.input_size, self.cfg.input_size),
            generator=generator,
            device=device,
        )

        if scale != 1.0:
            assert null_inputs is not None
            samples = torch.cat([samples, samples], dim=0)
            inputs = torch.cat([inputs, null_inputs], dim=0)
            if self.cfg.eval_scheduler == "GaussianDiffusion":
                model_kwargs = dict(y=inputs, cfg_scale=scale)
                samples = self.eval_scheduler.p_sample_loop(
                    self.forward_with_cfg,
                    samples.shape,
                    samples,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=progress,
                    device=device,
                )
            elif self.cfg.eval_scheduler == "UniPC":
                self.eval_scheduler.set_timesteps(num_inference_steps=self.cfg.num_inference_steps)
                for t in self.eval_scheduler.timesteps:
                    timesteps = torch.tensor([t] * samples.shape[0], device=device).int()
                    model_output = self.forward_with_cfg(samples, timesteps, inputs, scale)
                    if self.cfg.learn_sigma:
                        model_output = model_output[:, : self.cfg.in_channels]
                    samples = self.eval_scheduler.step(model_output, t, samples).prev_sample
            else:
                raise NotImplementedError(f"eval scheduler {self.cfg.eval_scheduler} is not supported")
            samples, _ = samples.chunk(2, dim=0)
        else:
            if self.cfg.eval_scheduler == "GaussianDiffusion":
                model_kwargs = dict(y=inputs)
                samples = self.eval_scheduler.p_sample_loop(
                    self.forward_without_cfg,
                    samples.shape,
                    samples,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=progress,
                    device=device,
                )
            elif self.cfg.eval_scheduler == "UniPC":
                self.eval_scheduler.set_timesteps(num_inference_steps=self.cfg.num_inference_steps)
                for t in self.eval_scheduler.timesteps:
                    timesteps = torch.tensor([t] * samples.shape[0], device=device).int()
                    model_output = self.forward_without_cfg(samples, timesteps, inputs)
                    if self.cfg.learn_sigma:
                        model_output = model_output[:, : self.cfg.in_channels]
                    samples = self.eval_scheduler.step(model_output, t, samples).prev_sample
            else:
                raise NotImplementedError(f"eval scheduler {self.cfg.eval_scheduler} is not supported")

        return samples


def dc_ae_dit_xl_in_512px(ae_name: str, scaling_factor: float, in_channels: int, pretrained_path: Optional[str]) -> str:
    return (
        f"autoencoder={ae_name} scaling_factor={scaling_factor} "
        f"model=dit dit.depth=28 dit.hidden_size=1152 dit.num_heads=16 dit.in_channels={in_channels} dit.patch_size=1 "
        f"dit.pretrained_path={'null' if pretrained_path is None else pretrained_path} "
        "fid.ref_path=assets/data/fid/imagenet_512_train.npz"
    )
