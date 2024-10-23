# Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models

[[paper](https://arxiv.org/abs/2410.10733)]

![demo](../../assets/dc_ae_demo.gif)
<p align="center">
<b> Figure 1: We address the reconstruction accuracy drop of high spatial-compression autoencoders.
</p>

![demo](../../assets/dc_ae_diffusion_demo.gif)
<p align="center">
<b> Figure 2: DC-AE speeds up latent diffusion models.
</p>

<p align="left">
<img src="../../assets/Sana-0.6B-laptop.png"  width="1200">
</p>

<p align="center">
<img src="../../assets/dc_ae_sana.jpg"  width="1200">
</p>

<p align="center">
<b> Figure 3: DC-AE enables efficient text-to-image generation on the laptop. For more details, please check our text-to-image diffusion model <a href="https://nvlabs.github.io/Sana/">SANA</a>.
</p>

## Abstract

We present Deep Compression Autoencoder (DC-AE), a new family of autoencoder models for accelerating high-resolution diffusion models. Existing autoencoder models have demonstrated impressive results at a moderate spatial compression ratio (e.g., 8x), but fail to maintain satisfactory reconstruction accuracy for high spatial compression ratios (e.g., 64x). We address this challenge by introducing two key techniques: (1) Residual Autoencoding, where we design our models to learn residuals based on the space-to-channel transformed features to alleviate the optimization difficulty of high spatial-compression autoencoders; (2) Decoupled High-Resolution Adaptation, an efficient decoupled three-phases training strategy for mitigating the generalization penalty of high spatial-compression autoencoders. With these designs, we improve the autoencoder's spatial compression ratio up to 128 while maintaining the reconstruction quality. Applying our DC-AE to latent diffusion models, we achieve significant speedup without accuracy drop. For example, on ImageNet 512x512, our DC-AE provides 19.1x inference speedup and 17.9x training speedup on H100 GPU for UViT-H while achieving a better FID, compared with the widely used SD-VAE-f8 autoencoder.

## Usage

### Deep Compression Autoencoder

```python
# build DC-AE models
# full DC-AE model list: https://huggingface.co/collections/mit-han-lab/dc-ae-670085b9400ad7197bb1009b
from efficientvit.ae_model_zoo import DCAE_HF

dc_ae = DCAE_HF.from_pretrained(f"mit-han-lab/dc-ae-f64c128-in-1.0")

# encode
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from efficientvit.apps.utils.image import DMCrop

device = torch.device("cuda")
dc_ae = dc_ae.to(device).eval()

transform = transforms.Compose([
    DMCrop(512), # resolution
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
image = Image.open("assets/fig/girl.png")
x = transform(image)[None].to(device)
latent = dc_ae.encode(x)
print(latent.shape)

# decode
y = dc_ae.decode(latent)
save_image(y * 0.5 + 0.5, "demo_dc_ae.png")
```

### Efficient Diffusion Models with DC-AE

```python
# build DC-AE-Diffusion models
# full DC-AE-Diffusion model list: https://huggingface.co/collections/mit-han-lab/dc-ae-diffusion-670dbb8d6b6914cf24c1a49d
from efficientvit.diffusion_model_zoo import DCAE_Diffusion_HF

dc_ae_diffusion = DCAE_Diffusion_HF.from_pretrained(f"mit-han-lab/dc-ae-f64c128-in-1.0-uvit-h-in-512px-train2000k")

# denoising on the latent space
import torch
import numpy as np
from torchvision.utils import save_image

torch.set_grad_enabled(False)
device = torch.device("cuda")
dc_ae_diffusion = dc_ae_diffusion.to(device).eval()

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
eval_generator = torch.Generator(device=device)
eval_generator.manual_seed(seed)

prompts = torch.tensor(
    [279, 333, 979, 936, 933, 145, 497, 1, 248, 360, 793, 12, 387, 437, 938, 978], dtype=torch.int, device=device
)
num_samples = prompts.shape[0]
prompts_null = 1000 * torch.ones((num_samples,), dtype=torch.int, device=device)
latent_samples = dc_ae_diffusion.diffusion_model.generate(prompts, prompts_null, 6.0, eval_generator)
latent_samples = latent_samples / dc_ae_diffusion.scaling_factor

# decode
image_samples = dc_ae_diffusion.autoencoder.decode(latent_samples)
save_image(image_samples * 0.5 + 0.5, "demo_dc_ae_diffusion.png", nrow=int(np.sqrt(num_samples)))
```

## Evaluate Deep Compression Autoencoder

- Download the ImageNet dataset to `~/dataset/imagenet`.

- Generate reference for FID computation:

```bash
torchrun --nnodes 1 --nproc_per_node=8 -m applications.dc_ae.generate_reference \
    dataset=imagenet imagenet.resolution=512 imagenet.image_mean=[0.,0.,0.] imagenet.image_std=[1.,1.,1.] split=test \
    fid.save_path=assets/data/fid/imagenet_512_val.npz
```

- Run evaluation:

```bash
# full DC-AE model list: https://huggingface.co/collections/mit-han-lab/dc-ae-670085b9400ad7197bb1009b
torchrun --nnodes 1 --nproc_per_node=8 -m applications.dc_ae.eval_dc_ae_model dataset=imagenet_512 model=dc-ae-f64c128-in-1.0 run_dir=tmp

# Expected results:
#   fid: 0.2167766520628902
#   psnr: 26.1489275
#   ssim: 0.710486114025116
#   lpips: 0.0802311897277832
```

## Demo DC-AE-Diffusion Models

``` bash
# full DC-AE-Diffusion model list: https://huggingface.co/collections/mit-han-lab/dc-ae-diffusion-670dbb8d6b6914cf24c1a49d

torchrun --nnodes 1 --nproc_per_node=1 -m applications.dc_ae.demo_dc_ae_diffusion_model model=dc-ae-f64c128-in-1.0-uvit-h-in-512px-train2000k run_dir=.demo/diffusion/dc-ae-f64c128-in-1.0-uvit-h-in-512px-train2000k
```

Expected results:
<p align="left">
<img src="../../assets/dc_ae_diffusion_example.png"  width="600">
</p>

## Evaluate DC-AE-Diffusion Models

- Generate reference for FID computation:

```bash
# generate reference for FID computation
torchrun --nnodes 1 --nproc_per_node=8 -m applications.dc_ae.generate_reference \
    dataset=imagenet imagenet.resolution=512 imagenet.image_mean=[0.,0.,0.] imagenet.image_std=[1.,1.,1.] split=train \
    fid.save_path=assets/data/fid/imagenet_512_train.npz
```

- Run evaluation without cfg

```bash
# full DC-AE-Diffusion model list: https://huggingface.co/collections/mit-han-lab/dc-ae-diffusion-670dbb8d6b6914cf24c1a49d

torchrun --nnodes 1 --nproc_per_node=8 -m applications.dc_ae.eval_dc_ae_diffusion_model dataset=imagenet_512 model=dc-ae-f64c128-in-1.0-uvit-h-in-512px cfg_scale=1.0 run_dir=tmp
# Expected results:
#   fid: 13.754458694549271
```

- Run evaluation with cfg

```bash
# full DC-AE-Diffusion model list: https://huggingface.co/collections/mit-han-lab/dc-ae-diffusion-670dbb8d6b6914cf24c1a49d
# cfg=1.3 for mit-han-lab/dc-ae-f32c32-in-1.0-dit-xl-in-512px
# and cfg=1.5 for all other models
torchrun --nnodes 1 --nproc_per_node=8 -m applications.dc_ae.eval_dc_ae_diffusion_model dataset=imagenet_512 model=dc-ae-f64c128-in-1.0-uvit-h-in-512px cfg_scale=1.5 run_dir=tmp
# Expected results:
#   fid: 2.963459255529642
```

## Train DC-AE-Diffusion Models

- Generate and save latent:

```bash
torchrun --nnodes 1 --nproc_per_node=8 -m applications.dc_ae.dc_ae_generate_latent resolution=512 \
    image_root_path=~/dataset/imagenet/train batch_size=64 \
    model_name=dc-ae-f64c128-in-1.0 scaling_factor=0.2889 \
    latent_root_path=assets/data/latent/dc_ae_f64c128_in_1.0/imagenet_512
```

- Run training

``` bash
# Example: DC-AE-f64 + UViT-H on ImageNet 512x512
torchrun --nnodes 1 --nproc_per_node=8 -m applications.dc_ae.train_dc_ae_diffusion_model resolution=512 \
    train_dataset=latent_imagenet latent_imagenet.batch_size=128 latent_imagenet.data_dir=assets/data/latent/dc_ae_f64c128_in_1.0/imagenet_512 \
    evaluate_dataset=sample_class sample_class.num_samples=50000 \
    autoencoder=dc-ae-f64c128-in-1.0 scaling_factor=0.2889 \
    model=uvit uvit.depth=28 uvit.hidden_size=1152 uvit.num_heads=16 uvit.in_channels=128 uvit.patch_size=1 \
    uvit.train_scheduler=DPM_Solver uvit.eval_scheduler=DPM_Solver \
    optimizer.name=adamw optimizer.lr=2e-4 optimizer.weight_decay=0.03 optimizer.betas=[0.99,0.99] lr_scheduler.name=constant_with_warmup lr_scheduler.warmup_steps=5000 amp=bf16 \
    max_steps=500000 ema_decay=0.9999 \
    fid.ref_path=assets/data/fid/imagenet_512_train.npz \
    run_dir=.exp/diffusion/imagenet_512/dc_ae_f64c128_in_1.0/uvit_h_1/bs_1024_lr_2e-4_bf16 log=False
```

## Reference

If DC-AE is useful or relevant to your research, please kindly recognize our contributions by citing our papers:

```bibtex
@article{chen2024deep,
  title={Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models},
  author={Chen, Junyu and Cai, Han and Chen, Junsong and Xie, Enze and Yang, Shang and Tang, Haotian and Li, Muyang and Lu, Yao and Han, Song},
  journal={arXiv preprint arXiv:2410.10733},
  year={2024}
}
```
