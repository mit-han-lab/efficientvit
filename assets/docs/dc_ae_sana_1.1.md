# DC-AE-SANA-1.1

We released DC-AE-f32c32-SANA-1.1: [efficientvit format](https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.1), [diffusers format](https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers). It has the same encoder as DC-AE-f32c32-SANA-1.0 while having a slightly better decoder. Without training, it can be applied to all diffusion models trained with DC-AE-f32c32-SANA-1.0, including SANA models.


| Autoencoder                                                                         | MJHQ 1024 rFID | MJHQ 1024 PSNR | MJHQ 1024 FID | MJHQ 1024 CLIP Score |
| :---------------------------------------------------------------------------------: | :------------: | :------------: | :-----------: | :------------------: |
| [dc-ae-f32c32-sana-1.0](https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.0)   | 0.339          | 29.31          | 6.132         | 28.85                |
| [dc-ae-f32c32-sana-1.1](https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.1)   | 0.314	       | 29.74          | 6.003         | 28.85                |
