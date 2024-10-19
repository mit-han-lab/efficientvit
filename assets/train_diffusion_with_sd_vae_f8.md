# download sd-vae-ft-ema
``` bash
python -c "import diffusers; diffusers.models.AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema')"
```

# extract latent
``` bash
torchrun --nnodes 1 --nproc_per_node=8 -m applications.dc_ae.dc_ae_generate_latent resolution=512 \
    image_root_path=~/dataset/imagenet/train batch_size=64 \
    model_name=stabilityai/sd-vae-ft-ema \
    latent_root_path=assets/data/latent/sd_vae_ft_ema/imagenet_512
```

# train uvit_s
``` bash
torchrun --nnodes 1 --nproc_per_node=8 -m applications.dc_ae.train_dc_ae_diffusion_model resolution=512 \
    train_dataset=latent_imagenet latent_imagenet.batch_size=128 latent_imagenet.data_dir=assets/data/latent/sd_vae_ft_ema/imagenet_512 \
    evaluate_dataset=sample_class sample_class.num_samples=50000 \
    autoencoder=stabilityai/sd-vae-ft-ema \
    model=uvit uvit.depth=12 uvit.hidden_size=512 uvit.num_heads=8 uvit.in_channels=4 uvit.patch_size=2 \
    uvit.train_scheduler=DPM_Solver uvit.eval_scheduler=DPM_Solver \
    optimizer.name=adamw optimizer.lr=2e-4 optimizer.weight_decay=0.03 optimizer.betas=[0.99,0.99] lr_scheduler.name=constant_with_warmup lr_scheduler.warmup_steps=5000 amp=bf16 \
    max_steps=500000 ema_decay=0.9999 \
    fid.ref_path=assets/data/fid/imagenet_512_train.npz \
    run_dir=.exp/diffusion/imagenet_512/sd_vae_ft_ema/uvit_s_2/bs_1024_lr_2e-4_bf16 log=False
```