# modified from https://github.com/baofff/U-ViT
import math
from dataclasses import dataclass
from functools import partial
from typing import Optional

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from timm.models.vision_transformer import Mlp, trunc_normal_

from efficientvit.apps.utils.dist import is_master
from efficientvit.diffusioncore.models.uvit_sampler.dpm_solver_pp import DPM_Solver, NoiseScheduleVP
from efficientvit.models.utils.network import get_device, get_submodule_weights

from .sit_sampler import Sampler as SiTSampler
from .sit_sampler import create_transport as sit_create_transport

__all__ = ["UViTConfig", "UViT", "dc_ae_uvit_s_in_512px", "dc_ae_uvit_h_in_512px"]


if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
    ATTENTION_MODE = "flash"
else:
    ATTENTION_MODE = "math"


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=timesteps.device
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def patchify(imgs, patch_size):
    x = einops.rearrange(imgs, "B C (h p1) (w p2) -> B (h w) (p1 p2 C)", p1=patch_size, p2=patch_size)
    return x


def unpatchify(x, channels=3):
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
    x = einops.rearrange(x, "B (h w) (p1 p2 C) -> B C (h p1) (w p2)", h=h, p1=patch_size, p2=patch_size)
    return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, proj_bias=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5 if qk_scale is None else qk_scale

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == "flash":
            qkv = einops.rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, "B H L D -> B L (H D)")
        elif ATTENTION_MODE == "math":
            qkv = einops.rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        skip=False,
        use_checkpoint=False,
    ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


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


@dataclass
class UViTConfig:
    name: str = "UViT"

    input_size: int = 32
    patch_size: int = 2
    in_channels: int = 4
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    mlp_time_embed: bool = False
    qkv_bias: bool = False
    act_layer: str = "gelu"
    use_checkpoint: bool = True

    class_dropout_prob: float = 0.1
    num_classes: int = 1000

    pretrained_path: Optional[str] = None
    pretrained_source: str = "dc-ae"  # uvit

    eval_scheduler: str = "DPM_Solver"
    num_inference_steps: int = 30
    train_scheduler: str = "DPM_Solver"

    attn_mode: Optional[str] = None


def get_skip(alphas, betas):
    N = len(betas) - 1
    skip_alphas = np.ones([N + 1, N + 1], dtype=betas.dtype)
    for s in range(N + 1):
        skip_alphas[s, s + 1 :] = alphas[s + 1 :].cumprod()
    skip_betas = np.zeros([N + 1, N + 1], dtype=betas.dtype)
    for t in range(N + 1):
        prod = betas[1 : t + 1] * skip_alphas[1 : t + 1, t]
        skip_betas[:t, t] = (prod[::-1].cumsum())[::-1]
    return skip_alphas, skip_betas


def stp(s, ts: torch.Tensor):  # scalar tensor product
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts


class UViTSchedule(object):  # discrete time
    def __init__(self, _betas):
        r"""_betas[0...999] = betas[1...1000]
        for n>=1, betas[n] is the variance of q(xn|xn-1)
        for n=0,  betas[0]=0
        """

        self._betas = _betas
        self.betas = np.append(0.0, _betas)
        self.alphas = 1.0 - self.betas
        self.N = len(_betas)

        assert isinstance(self.betas, np.ndarray) and self.betas[0] == 0
        assert isinstance(self.alphas, np.ndarray) and self.alphas[0] == 1
        assert len(self.betas) == len(self.alphas)

        # skip_alphas[s, t] = alphas[s + 1: t + 1].prod()
        self.skip_alphas, self.skip_betas = get_skip(self.alphas, self.betas)
        self.cum_alphas = self.skip_alphas[0]  # cum_alphas = alphas.cumprod()
        self.cum_betas = self.skip_betas[0]
        self.snr = self.cum_alphas / self.cum_betas

    def tilde_beta(self, s, t):
        return self.skip_betas[s, t] * self.cum_betas[s] / self.cum_betas[t]

    def sample(self, x0):  # sample from q(xn|x0), where n is uniform
        n = np.random.choice(list(range(1, self.N + 1)), (len(x0),))
        eps = torch.randn_like(x0)
        xn = stp(self.cum_alphas[n] ** 0.5, x0) + stp(self.cum_betas[n] ** 0.5, eps)
        return torch.tensor(n, device=x0.device), eps, xn

    def __repr__(self):
        return f"Schedule({self.betas[:10]}..., {self.N})"


class UViT(nn.Module):
    def __init__(self, cfg: UViTConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.attn_mode is not None:
            global ATTENTION_MODE
            ATTENTION_MODE = cfg.attn_mode
        if is_master():
            print(f"attention mode is {ATTENTION_MODE}")

        self.patch_embed = PatchEmbed(patch_size=cfg.patch_size, in_chans=cfg.in_channels, embed_dim=cfg.hidden_size)
        num_patches = (cfg.input_size // cfg.patch_size) ** 2

        self.time_embed = (
            nn.Sequential(
                nn.Linear(cfg.hidden_size, 4 * cfg.hidden_size),
                nn.SiLU(),
                nn.Linear(4 * cfg.hidden_size, cfg.hidden_size),
            )
            if cfg.mlp_time_embed
            else nn.Identity()
        )

        if cfg.num_classes > 0:
            self.label_emb = LabelEmbedder(
                cfg.num_classes, cfg.hidden_size, cfg.class_dropout_prob
            )  # nn.Embedding(cfg.num_classes, cfg.hidden_size)
            self.extras = 2
        else:
            self.extras = 1

        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, cfg.hidden_size))

        if cfg.act_layer == "gelu":
            act_layer = nn.GELU
        elif cfg.act_layer == "silu":
            act_layer = partial(nn.SiLU, inplace=True)
        else:
            raise NotImplementedError(f"act_layer {act_layer} is not supported")

        self.in_blocks = nn.ModuleList(
            [
                Block(
                    dim=cfg.hidden_size,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    qkv_bias=cfg.qkv_bias,
                    qk_scale=None,
                    act_layer=act_layer,
                    norm_layer=nn.LayerNorm,
                    use_checkpoint=cfg.use_checkpoint,
                )
                for _ in range(cfg.depth // 2)
            ]
        )

        self.mid_block = Block(
            dim=cfg.hidden_size,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            qkv_bias=cfg.qkv_bias,
            qk_scale=None,
            act_layer=act_layer,
            norm_layer=nn.LayerNorm,
            use_checkpoint=cfg.use_checkpoint,
        )

        self.out_blocks = nn.ModuleList(
            [
                Block(
                    dim=cfg.hidden_size,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    qkv_bias=cfg.qkv_bias,
                    qk_scale=None,
                    act_layer=act_layer,
                    norm_layer=nn.LayerNorm,
                    skip=True,
                    use_checkpoint=cfg.use_checkpoint,
                )
                for _ in range(cfg.depth // 2)
            ]
        )

        self.norm = nn.LayerNorm(cfg.hidden_size)
        self.patch_dim = cfg.patch_size**2 * cfg.in_channels
        self.decoder_pred = nn.Linear(cfg.hidden_size, self.patch_dim, bias=True)
        self.final_layer = nn.Identity()

        if cfg.pretrained_path is not None:
            self.load_model()
        else:
            trunc_normal_(self.pos_embed, std=0.02)
            self.apply(self._init_weights)

        # scheduler
        if cfg.train_scheduler == "DPM_Solver":
            _betas = (torch.linspace(0.00085**0.5, 0.0120**0.5, 1000, dtype=torch.float64) ** 2).numpy()
            self.train_scheduler = UViTSchedule(_betas)
        elif cfg.train_scheduler == "SiTSampler":
            self.transport = sit_create_transport("Linear", "velocity", None, None, None)
        else:
            raise NotImplementedError(f"train_scheduler {cfg.train_scheduler} is not supported")

        if cfg.eval_scheduler == "DPM_Solver":
            device = torch.device("cuda")
            _betas = (torch.linspace(0.00085**0.5, 0.0120**0.5, 1000, dtype=torch.float64) ** 2).numpy()
            self.eval_scheduler = NoiseScheduleVP(
                schedule="discrete", betas=torch.tensor(_betas, device=device).float()
            )
        elif cfg.eval_scheduler in ["ODE_dopri5", "ODE_heun2"]:
            assert cfg.train_scheduler == "SiTSampler"
            sampler = SiTSampler(self.transport)
            if cfg.eval_scheduler == "ODE_dopri5":
                self.eval_scheduler = sampler.sample_ode(
                    sampling_method="dopri5", num_steps=cfg.num_inference_steps, atol=1e-6, rtol=0.001, reverse=False
                )
            elif cfg.eval_scheduler == "ODE_heun2":
                self.eval_scheduler = sampler.sample_ode(
                    sampling_method="heun2", num_steps=cfg.num_inference_steps, atol=1e-6, rtol=0.001, reverse=False
                )
            else:
                raise ValueError(f"eval scheduler {cfg.eval_scheduler} is not supported")
        else:
            raise NotImplementedError(f"eval_scheduler {cfg.eval_scheduler} is not supported")

    def get_trainable_modules(self) -> nn.ModuleDict:
        return nn.ModuleDict({"uvit": self})

    def load_model(self):
        checkpoint = torch.load(self.cfg.pretrained_path, map_location="cpu")
        if self.cfg.pretrained_source == "uvit":
            if "ema" in checkpoint:
                checkpoint = checkpoint["ema"]
            self.patch_embed.load_state_dict(get_submodule_weights(checkpoint, "patch_embed."))
            self.time_embed.load_state_dict(get_submodule_weights(checkpoint, "time_embed."))
            if self.cfg.num_classes > 0:
                self.label_emb.embedding_table.load_state_dict(get_submodule_weights(checkpoint, "label_emb."))
            self.pos_embed.data = checkpoint["pos_embed"]
            self.in_blocks.load_state_dict(get_submodule_weights(checkpoint, "in_blocks."))
            self.mid_block.load_state_dict(get_submodule_weights(checkpoint, "mid_block."))
            self.out_blocks.load_state_dict(get_submodule_weights(checkpoint, "out_blocks."))
            self.norm.load_state_dict(get_submodule_weights(checkpoint, "norm."))
            self.decoder_pred.load_state_dict(get_submodule_weights(checkpoint, "decoder_pred."))
            self.final_layer.load_state_dict(get_submodule_weights(checkpoint, "final_layer."))
        elif self.cfg.pretrained_source == "dc-ae":
            checkpoint = list(checkpoint["ema"].values())[0]
            self.get_trainable_modules().load_state_dict(checkpoint)
        else:
            raise NotImplementedError(f"pretrained source {self.cfg.pretrained_source} is not supported")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    def forward_without_cfg(self, x, timesteps, y=None):
        x = self.patch_embed(x)
        _, L, _ = x.shape

        time_token = self.time_embed(timestep_embedding(timesteps, self.cfg.hidden_size))
        time_token = time_token.unsqueeze(dim=1)
        x = torch.cat((time_token, x), dim=1)
        if y is not None and self.cfg.num_classes > 0:
            label_emb = self.label_emb(y, self.training)
            label_emb = label_emb.unsqueeze(dim=1)
            x = torch.cat((label_emb, x), dim=1)
        x = x + self.pos_embed

        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        x = self.norm(x)
        x = self.decoder_pred(x)
        assert x.size(1) == self.extras + L
        x = x[:, self.extras :, :]
        x = unpatchify(x, self.cfg.in_channels)
        x = self.final_layer(x)
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
        if self.cfg.train_scheduler == "DPM_Solver":
            n, eps, xn = self.train_scheduler.sample(x)  # n in {1, ..., 1000}
            eps_pred = self.forward_without_cfg(xn, n, y)
            loss = (eps - eps_pred).square().mean()
        elif self.cfg.train_scheduler == "SiTSampler":
            model_kwargs = dict(y=y)
            loss_dict = self.transport.training_losses(self.forward_without_cfg, x, model_kwargs)
            loss = loss_dict["loss"].mean()
        else:
            raise NotImplementedError(f"train scheduler {self.cfg.train_scheduler} is not supported")
        info["loss_dict"] = {"loss": loss}
        return loss, info

    @torch.no_grad()
    def generate(self, inputs, null_inputs, scale: float = 1.5, generator: Optional[torch.Generator] = None):
        device = get_device(self)
        samples = torch.randn(
            (inputs.shape[0], self.cfg.in_channels, self.cfg.input_size, self.cfg.input_size),
            generator=generator,
            device=device,
        )

        if self.cfg.eval_scheduler == "DPM_Solver":
            N = self.eval_scheduler.total_N

            def model_fn(x, t_continuous):
                t = t_continuous * N
                if scale == 1.0:
                    eps_pre = self.forward_without_cfg(x, t, inputs)
                else:
                    assert null_inputs is not None
                    _cond = self.forward_without_cfg(x, t, inputs)
                    _uncond = self.forward_without_cfg(x, t, null_inputs)
                    eps_pre = _cond + (scale - 1) * (_cond - _uncond)
                return eps_pre

            dpm_solver = DPM_Solver(model_fn, self.eval_scheduler, predict_x0=True, thresholding=False)
            samples = dpm_solver.sample(samples, steps=self.cfg.num_inference_steps, eps=1.0 / N, T=1.0)
        elif self.cfg.eval_scheduler in ["ODE_dopri5", "ODE_heun2"]:
            if scale != 1.0:
                assert null_inputs is not None
                samples = torch.cat([samples, samples], dim=0)
                inputs = torch.cat([inputs, null_inputs], dim=0)
                samples = self.eval_scheduler(samples, self.forward_with_cfg, y=inputs, cfg_scale=scale)[-1]
                samples, _ = samples.chunk(2, dim=0)
            else:
                samples = self.eval_scheduler(samples, self.forward_without_cfg, y=inputs)[-1]
        else:
            raise NotImplementedError(f"eval scheduler {self.cfg.eval_scheduler} is not supported")

        return samples


def dc_ae_uvit_s_in_512px(ae_name: str, scaling_factor: float, in_channels: int, pretrained_path: Optional[str]) -> str:
    return (
        f"autoencoder={ae_name} scaling_factor={scaling_factor} "
        f"model=uvit uvit.depth=12 uvit.hidden_size=512 uvit.num_heads=8 uvit.in_channels={in_channels} uvit.patch_size=1 "
        f"uvit.pretrained_path={'null' if pretrained_path is None else pretrained_path} "
        "fid.ref_path=assets/data/fid/imagenet_512_train.npz"
    )


def dc_ae_uvit_h_in_512px(ae_name: str, scaling_factor: float, in_channels: int, pretrained_path: Optional[str]) -> str:
    return (
        f"autoencoder={ae_name} scaling_factor={scaling_factor} "
        f"model=uvit uvit.depth=28 uvit.hidden_size=1152 uvit.num_heads=16 uvit.in_channels={in_channels} uvit.patch_size=1 "
        f"uvit.pretrained_path={'null' if pretrained_path is None else pretrained_path} "
        "fid.ref_path=assets/data/fid/imagenet_512_train.npz"
    )


def dc_ae_uvit_2b_in_512px(
    ae_name: str, scaling_factor: float, in_channels: int, pretrained_path: Optional[str]
) -> str:
    return (
        f"autoencoder={ae_name} scaling_factor={scaling_factor} "
        f"model=uvit uvit.depth=28 uvit.hidden_size=2048 uvit.num_heads=32 uvit.in_channels={in_channels} uvit.patch_size=1 "
        f"uvit.pretrained_path={'null' if pretrained_path is None else pretrained_path} "
        "fid.ref_path=assets/data/fid/imagenet_512_train.npz"
    )


def dc_ae_usit_h_in_512px(ae_name: str, scaling_factor: float, in_channels: int, pretrained_path: Optional[str]) -> str:
    return (
        f"autoencoder={ae_name} scaling_factor={scaling_factor} "
        f"model=uvit uvit.depth=28 uvit.hidden_size=1152 uvit.num_heads=16 uvit.in_channels={in_channels} uvit.patch_size=1 "
        "uvit.train_scheduler=SiTSampler uvit.eval_scheduler=ODE_dopri5 uvit.num_inference_steps=250 "
        f"uvit.pretrained_path={'null' if pretrained_path is None else pretrained_path} "
        "fid.ref_path=assets/data/fid/imagenet_512_train.npz"
    )


def dc_ae_usit_2b_in_512px(
    ae_name: str, scaling_factor: float, in_channels: int, pretrained_path: Optional[str]
) -> str:
    return (
        f"autoencoder={ae_name} scaling_factor={scaling_factor} "
        f"model=uvit uvit.depth=28 uvit.hidden_size=2048 uvit.num_heads=32 uvit.in_channels={in_channels} uvit.patch_size=1 "
        "uvit.train_scheduler=SiTSampler uvit.eval_scheduler=ODE_dopri5 uvit.num_inference_steps=250 "
        f"uvit.pretrained_path={'null' if pretrained_path is None else pretrained_path} "
        "fid.ref_path=assets/data/fid/imagenet_512_train.npz"
    )
