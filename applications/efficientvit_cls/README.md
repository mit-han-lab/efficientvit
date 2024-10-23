# EfficientViT Classification

<p align="left">
<img src="../../assets/efficientvit_cls_results.png"  width="600">
</p>

## Datasets

<details><summary>ImageNet: https://www.image-net.org/</summary>

```python
Our code expects the ImageNet dataset directory to follow the following structure:

imagenet
├── train
├── val
```

</details>

## Pretrained EfficientViT Classification Models

Latency/Throughput is measured on NVIDIA Jetson Nano, NVIDIA Jetson AGX Orin, and NVIDIA A100 GPU with TensorRT, fp16. Data transfer time is included.

### ImageNet

All EfficientViT classification models are trained on ImageNet-1K with random initialization (300 epochs + 20 warmup epochs) using supervised learning. Please put the downloaded checkpoints under *${efficientvit_repo}/assets/checkpoints/efficientvit_cls/*

| Model         |  Resolution | ImageNet Top1 Acc | ImageNet Top5 Acc |  Params |  MACs |  A100 Throughput | Checkpoint |
|----------------------|:----------:|:----------:|:---------:|:------------:|:---------:|:------------:|:------------:|
| EfficientNetV2-S | 384x384 | [83.9](https://github.com/google/automl/tree/master/efficientnetv2#2-pretrained-efficientnetv2-checkpoints) | - | 22M | 8.4G | 2869 image/s | - |
| EfficientNetV2-M | 480x480 | [85.2](https://github.com/google/automl/tree/master/efficientnetv2#2-pretrained-efficientnetv2-checkpoints) | - | 54M | 25G | 1160 image/s | - |
| |
| EfficientViT-L1 | 224x224 |  84.484 | 96.862 | 53M | 5.3G | 6207 image/s | [link](https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_l1_r224.pt) |
| |
| EfficientViT-L2 | 224x224 |  85.050 | 97.090 | 64M | 6.9G | 4998 image/s | [link](https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_l2_r224.pt) |
| EfficientViT-L2 | 256x256 |  85.366 | 97.216 | 64M | 9.1G | 3969 image/s | [link](https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_l2_r256.pt) |
| EfficientViT-L2 | 288x288 |  85.630 | 97.364 | 64M | 11G  | 3102 image/s | [link](https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_l2_r288.pt) |
| EfficientViT-L2 | 320x320 |  85.734 | 97.438 | 64M | 14G  | 2525 image/s | [link](https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_l2_r320.pt) |
| EfficientViT-L2 | 384x384 |  85.978 | 97.518 | 64M | 20G  | 1784 image/s | [link](https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_l2_r384.pt) |
| |
| EfficientViT-L3 | 224x224 | 85.814 | 97.198 | 246M | 28G | 2081 image/s | [link](https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_l3_r224.pt) |
| EfficientViT-L3 | 256x256 | 85.938 | 97.318 | 246M | 36G | 1641 image/s | [link](https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_l3_r256.pt) |
| EfficientViT-L3 | 288x288 | 86.070 | 97.440 | 246M | 46G | 1276 image/s | [link](https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_l3_r288.pt) |
| EfficientViT-L3 | 320x320 | 86.230 | 97.474 | 246M | 56G | 1049 image/s | [link](https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_l3_r320.pt) |
| EfficientViT-L3 | 384x384 | 86.408 | 97.632 | 246M | 81G | 724 image/s | [link](https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_l3_r384.pt) |

<details>
  <summary>EfficientViT B series</summary>

  | Model         |  Resolution | ImageNet Top1 Acc | ImageNet Top5 Acc |  Params |  MACs |  Jetson Nano (bs1) | Jetson Orin (bs1) | Checkpoint |
  |----------------------|:----------:|:----------:|:---------:|:------------:|:---------:|:------------:|:------------:|:------------:|
  | EfficientViT-B1 | 224x224 | 79.390 | 94.346 | 9.1M | 0.52G | 24.8ms | 1.48ms | [link](https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_b1_r224.pt) |
  | EfficientViT-B1 | 256x256 | 79.918 | 94.704 | 9.1M | 0.68G | 28.5ms | 1.57ms | [link](https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_b1_r256.pt) |
  | EfficientViT-B1 | 288x288 | 80.410 | 94.984 | 9.1M | 0.86G | 34.5ms | 1.82ms | [link](https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_b1_r288.pt) |
  | |
  | EfficientViT-B2 | 224x224 | 82.100 | 95.782 | 24M  | 1.6G  | 50.6ms | 2.63ms | [link](https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_b2_r224.pt) |
  | EfficientViT-B2 | 256x256 | 82.698 | 96.096 | 24M  | 2.1G  | 58.5ms | 2.84ms | [link](https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_b2_r256.pt) |
  | EfficientViT-B2 | 288x288 | 83.086 | 96.302 | 24M  | 2.6G  | 69.9ms | 3.30ms | [link](https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_b2_r288.pt) |
  | |
  | EfficientViT-B3 | 224x224 | 83.468 | 96.356 | 49M  | 4.0G  | 101ms  | 4.36ms | [link](https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_b3_r224.pt) |
  | EfficientViT-B3 | 256x256 | 83.806 | 96.514 | 49M  | 5.2G  | 120ms  | 4.74ms | [link](https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_b3_r256.pt) |
  | EfficientViT-B3 | 288x288 | 84.150 | 96.732 | 49M  | 6.5G  | 141ms  | 5.63ms | [link](https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_b3_r288.pt) |

</details>

## Usage

```python
# classification
from efficientvit.cls_model_zoo import create_efficientvit_cls_model

model = create_efficientvit_cls_model(name="efficientvit-l3-r384", pretrained=True)
```

## Evaluation

Please run [eval_efficientvit_cls_model.py](eval_efficientvit_cls_model.py) to evaluate our models.

Examples: [classification](../../assets/eval_efficientvit_cls_model.sh)

## Export

### Onnx

To generate ONNX files, please refer to [onnx_export.py](../../assets/onnx_export.py).

Example:

```bash
python assets/onnx_export.py --export_path assets/export_models/efficientvit_cls_l3_r224.onnx --model efficientvit-l3 --resolution 224 224 --bs 1
```

### TFLite

To generate TFLite files, please refer to [tflite_export.py](../../assets/tflite_export.py).

Example:

```bash
python assets/tflite_export.py --export_path assets/export_models/efficientvit_cls_b3_r224.tflite --model efficientvit-b3 --resolution 224 224
```

## Training

Please refer to [train_efficientvit_cls_model.py](train_efficientvit_cls_model.py) for training models on imagenet.

### EfficientViT L Series

```bash
torchrun --nnodes 1 --nproc_per_node=8 \
python applications/efficientvit_cls/train_efficientvit_cls_model.py applications/efficientvit_cls/configs/imagenet/efficientvit_l1.yaml --amp bf16 \
    --data_provider.data_dir ~/dataset/imagenet \
    --path .exp/efficientvit_cls/imagenet/efficientvit_l1_r224/
```

```bash
torchrun --nnodes 1 --nproc_per_node=8 \
python applications/efficientvit_cls/train_efficientvit_cls_model.py applications/efficientvit_cls/configs/imagenet/efficientvit_l2.yaml --amp bf16 \
    --data_provider.data_dir ~/dataset/imagenet \
    --path .exp/efficientvit_cls/imagenet/efficientvit_l2_r224/
```

```bash
torchrun --nnodes 1 --nproc_per_node=8 \
python applications/efficientvit_cls/train_efficientvit_cls_model.py applications/efficientvit_cls/configs/imagenet/efficientvit_l3.yaml --amp bf16 \
    --data_provider.data_dir ~/dataset/imagenet \
    --path .exp/efficientvit_cls/imagenet/efficientvit_l3_r224/
```

### EfficientViT B Series

```bash
torchrun --nnodes 1 --nproc_per_node=8 \
python applications/efficientvit_cls/train_efficientvit_cls_model.py applications/efficientvit_cls/configs/imagenet/efficientvit_b1.yaml \
    --data_provider.data_dir ~/dataset/imagenet \
    --path .exp/efficientvit_cls/imagenet/efficientvit_b1_r224/
```

```bash
torchrun --nnodes 1 --nproc_per_node=8 \
python applications/efficientvit_cls/train_efficientvit_cls_model.py applications/efficientvit_cls/configs/imagenet/efficientvit_b1.yaml \
    --data_provider.image_size "[128,160,192,224,256,288]" \
    --data_provider.data_dir ~/dataset/imagenet \
    --run_config.eval_image_size "[288]" \
    --path .exp/efficientvit_cls/imagenet/efficientvit_b1_r288/
```

```bash
torchrun --nnodes 1 --nproc_per_node=8 \
python applications/efficientvit_cls/train_efficientvit_cls_model.py applications/efficientvit_cls/configs/imagenet/efficientvit_b2.yaml \
    --data_provider.data_dir ~/dataset/imagenet \
    --path .exp/efficientvit_cls/imagenet/efficientvit_b2_r224/
```

```bash
torchrun --nnodes 1 --nproc_per_node=8 \
python applications/efficientvit_cls/train_efficientvit_cls_model.py applications/efficientvit_cls/configs/imagenet/efficientvit_b2.yaml \
    --data_provider.image_size "[128,160,192,224,256,288]" \
    --data_provider.data_dir ~/dataset/imagenet \
    --run_config.eval_image_size "[288]" \
    --data_provider.data_aug "{n:1,m:5}" \
    --path .exp/efficientvit_cls/imagenet/efficientvit_b2_r288/
```

```bash
torchrun --nnodes 1 --nproc_per_node=8 \
python applications/efficientvit_cls/train_efficientvit_cls_model.py applications/efficientvit_cls/configs/imagenet/efficientvit_b3.yaml \
    --data_provider.data_dir ~/dataset/imagenet \
    --path .exp/efficientvit_cls/imagenet/efficientvit_b3_r224/
```

## Reference

If EfficientViT is useful or relevant to your research, please kindly recognize our contributions by citing our paper:

```bibtex
@inproceedings{cai2023efficientvit,
  title={Efficientvit: Lightweight multi-scale attention for high-resolution dense prediction},
  author={Cai, Han and Li, Junyan and Hu, Muyan and Gan, Chuang and Han, Song},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={17302--17313},
  year={2023}
}
```
