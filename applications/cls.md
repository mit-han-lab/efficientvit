# Image Classification

## Datasets

<details>
<summary> ImageNet: https://www.image-net.org/</summary>

```python
Our code expects the ImageNet dataset directory to follow the following structure:

imagenet
├── train
├── val
```

</details>

## Pretrained Models

Latency/Throughput is measured on NVIDIA Jetson Nano, NVIDIA Jetson AGX Orin, and NVIDIA A100 GPU with TensorRT, fp16. Data transfer time is included.

### ImageNet

All EfficientViT classification models are trained on ImageNet-1K with random initialization (300 epochs + 20 warmup epochs) using supervised learning.

<p align="left">
<img src="../assets/files/cls_results.png"  width="450">
</p>

| Model         |  Resolution | ImageNet Top1 Acc | ImageNet Top5 Acc |  Params |  MACs |  A100 Throughput | Checkpoint |
|----------------------|:----------:|:----------:|:---------:|:------------:|:---------:|:------------:|:------------:|
| EfficientNetV2-S | 384x384 | [83.9](https://github.com/google/automl/tree/master/efficientnetv2#2-pretrained-efficientnetv2-checkpoints) | - | 22M | 8.4G | 2869 image/s | - |
| EfficientNetV2-M | 480x480 | [85.2](https://github.com/google/automl/tree/master/efficientnetv2#2-pretrained-efficientnetv2-checkpoints) | - | 54M | 25G | 1160 image/s | - |
| |
| EfficientViT-L1 | 224x224 |  84.484 | 96.862 | 53M | 5.3G | 6207 image/s | [link](https://drive.google.com/file/d/1q5y0YbN08O4ToUBK8RfZSDKp-s1y5_44/view?usp=sharing) |
| |
| EfficientViT-L2 | 224x224 |  85.050 | 97.090 | 64M | 6.9G | 4998 image/s | [link](https://drive.google.com/file/d/1FEjImtyIQhG4VsHsstLgNM09Y9qJn9Sk/view?usp=sharing) |
| EfficientViT-L2 | 256x256 |  85.366 | 97.216 | 64M | 9.1G | 3969 image/s | [link](https://drive.google.com/file/d/1pvYtY0ckAAMTkRq6TbwpQ0U1p_urz2fE/view?usp=sharing) |
| EfficientViT-L2 | 288x288 |  85.630 | 97.364 | 64M | 11G  | 3102 image/s | [link](https://drive.google.com/file/d/1GDr0y45YPX8iWEWNq5fEmjo0UgyZLpUs/view?usp=sharing) |
| EfficientViT-L2 | 320x320 |  85.734 | 97.438 | 64M | 14G  | 2525 image/s | [link](https://drive.google.com/file/d/1GDr0y45YPX8iWEWNq5fEmjo0UgyZLpUs/view?usp=sharing) |
| EfficientViT-L2 | 384x384 |  85.978 | 97.518 | 64M | 20G  | 1784 image/s | [link](https://drive.google.com/file/d/1MpjduiCTbUVS1XJri4_eqCbARJyYo74b/view?usp=sharing) |
| |
| EfficientViT-L3 | 224x224 | 85.814 | 97.198 | 246M | 28G | 2081 image/s | [link](https://huggingface.co/han-cai/efficientvit-imagenet/blob/main/l3-r224.pt) |
| EfficientViT-L3 | 256x256 | 85.938 | 97.318 | 246M | 36G | 1641 image/s | [link](https://huggingface.co/han-cai/efficientvit-imagenet/blob/main/l3-r256.pt) |
| EfficientViT-L3 | 288x288 | 86.070 | 97.440 | 246M | 46G | 1276 image/s | [link](https://huggingface.co/han-cai/efficientvit-imagenet/blob/main/l3-r288.pt) |
| EfficientViT-L3 | 320x320 | 86.230 | 97.474 | 246M | 56G | 1049 image/s | [link](https://huggingface.co/han-cai/efficientvit-imagenet/blob/main/l3-r320.pt) |
| EfficientViT-L3 | 384x384 | 86.408 | 97.632 | 246M | 81G | 724 image/s | [link](https://huggingface.co/han-cai/efficientvit-imagenet/blob/main/l3-r384.pt) |

<details>
  <summary>EfficientViT B series</summary>

  | Model         |  Resolution | ImageNet Top1 Acc | ImageNet Top5 Acc |  Params |  MACs |  Jetson Nano (bs1) | Jetson Orin (bs1) | Checkpoint |
  |----------------------|:----------:|:----------:|:---------:|:------------:|:---------:|:------------:|:------------:|:------------:|
  | EfficientViT-B1 | 224x224 | 79.390 | 94.346 | 9.1M | 0.52G | 24.8ms | 1.48ms | [link](https://drive.google.com/file/d/1hKN_hvLG4nmRzbfzKY7GlqwpR5uKpOOk/view?usp=share_link) |
  | EfficientViT-B1 | 256x256 | 79.918 | 94.704 | 9.1M | 0.68G | 28.5ms | 1.57ms | [link](https://drive.google.com/file/d/1hXcG_jB0ODMOESsSkzVye-58B4F3Cahs/view?usp=share_link) |
  | EfficientViT-B1 | 288x288 | 80.410 | 94.984 | 9.1M | 0.86G | 34.5ms | 1.82ms | [link](https://drive.google.com/file/d/1sE_Suz9gOOUO7o5r9eeAT4nKK8Hrbhsu/view?usp=share_link) |
  | |
  | EfficientViT-B2 | 224x224 | 82.100 | 95.782 | 24M  | 1.6G  | 50.6ms | 2.63ms | [link](https://drive.google.com/file/d/1DiM-iqVGTrq4te8mefHl3e1c12u4qR7d/view?usp=share_link) |
  | EfficientViT-B2 | 256x256 | 82.698 | 96.096 | 24M  | 2.1G  | 58.5ms | 2.84ms | [link](https://drive.google.com/file/d/192OOk4ISitwlyW979M-FSJ_fYMMW9HQz/view?usp=share_link) |
  | EfficientViT-B2 | 288x288 | 83.086 | 96.302 | 24M  | 2.6G  | 69.9ms | 3.30ms | [link](https://drive.google.com/file/d/1aodcepOyne667hvBAGpf9nDwmd5g0NpU/view?usp=share_link) |
  | |
  | EfficientViT-B3 | 224x224 | 83.468 | 96.356 | 49M  | 4.0G  | 101ms  | 4.36ms | [link](https://drive.google.com/file/d/18RZDGLiY8KsyJ7LGic4mg1JHwd-a_ky6/view?usp=share_link) |
  | EfficientViT-B3 | 256x256 | 83.806 | 96.514 | 49M  | 5.2G  | 120ms  | 4.74ms | [link](https://drive.google.com/file/d/1y1rnir4I0XiId-oTCcHhs7jqnrHGFi-g/view?usp=share_link) |
  | EfficientViT-B3 | 288x288 | 84.150 | 96.732 | 49M  | 6.5G  | 141ms  | 5.63ms | [link](https://drive.google.com/file/d/1KfwbGtlyFgslNr4LIHERv6aCfkItEvRk/view?usp=share_link) |
</details>

## Usage

```python
# classification
from efficientvit.cls_model_zoo import create_cls_model

model = create_cls_model(
  name="l3", weight_url="assets/checkpoints/cls/l3-r384.pt"
)
```

## Evaluation

Please run `eval_cls_model.py` to evaluate our models.

Examples: [classification](../assets/files/eval_cls_model.sh)

## Export

### Onnx

To generate ONNX files, please refer to `onnx_export.py`.

### TFLite

To generate TFLite files, please refer to `tflite_export.py`. It requires the TinyNN package.

```bash
pip install git+https://github.com/alibaba/TinyNeuralNetwork.git
```

Example:

```bash
python tflite_export.py --export_path model.tflite --model b3 --resolution 224 224
```

## Training

Please refer to `train_cls_model.py` for training models on imagenet.

Single-Node Training Examples:

```bash
torchrun --nnodes 1 --nproc_per_node=8 \
train_cls_model.py configs/cls/imagenet/b1.yaml \
    --data_provider.image_size "[128,160,192,224,256,288]" \
    --run_config.eval_image_size "[288]" \
    --path .exp/cls/imagenet/b1_r288/

torchpack dist-run -np 8 \
python train_cls_model.py configs/cls/imagenet/b1.yaml \
    --data_provider.image_size "[128,160,192,224,256,288]" \
    --run_config.eval_image_size "[288]" \
    --path .exp/cls/imagenet/b1_r288/
```

### EfficientViT L Series

```bash
torchpack dist-run -np 16 -H $server1:8,$server2:8 \
python train_cls_model.py configs/cls/imagenet/l1.yaml --fp16 \
    --path .exp/cls/imagenet/l1_r224/
```

```bash
torchpack dist-run -np 16 -H $server1:8,$server2:8 \
python train_cls_model.py configs/cls/imagenet/l2.yaml --fp16 \
    --path .exp/cls/imagenet/l2_r224/
```

```bash
torchpack dist-run -np 16 -H $server1:8,$server2:8 \
python train_cls_model.py configs/cls/imagenet/l3.yaml --fp16 \
    --path .exp/cls/imagenet/l3_r224/
```

### EfficientViT B Series

```bash
torchpack dist-run -np 16 -H $server1:8,$server2:8 \
python train_cls_model.py configs/cls/imagenet/b1.yaml \
    --path .exp/cls/imagenet/b1_r224/
```

```bash
torchpack dist-run -np 16 -H $server1:8,$server2:8 \
python train_cls_model.py configs/cls/imagenet/b1.yaml \
    --data_provider.image_size "[128,160,192,224,256,288]" \
    --run_config.eval_image_size "[288]" \
    --path .exp/cls/imagenet/b1_r288/
```

```bash
torchpack dist-run -np 16 -H $server1:8,$server2:8 \
python train_cls_model.py configs/cls/imagenet/b2.yaml \
    --path .exp/cls/imagenet/b2_r224/
```

```bash
torchpack dist-run -np 16 -H $server1:8,$server2:8 \
python train_cls_model.py configs/cls/imagenet/b2.yaml \
    --data_provider.image_size "[128,160,192,224,256,288]" \
    --run_config.eval_image_size "[288]" \
    --data_provider.data_aug "{n:1,m:5}" \
    --path .exp/cls/imagenet/b2_r288/
```

```bash
torchpack dist-run -np 16 -H $server1:8,$server2:8 \
python train_cls_model.py configs/cls/imagenet/b3.yaml \
    --path .exp/cls/imagenet/b3_r224/
```

## Citation

If EfficientViT is useful or relevant to your research, please kindly recognize our contributions by citing our paper:

```
@article{cai2022efficientvit,
  title={Efficientvit: Enhanced linear attention for high-resolution low-computation visual recognition},
  author={Cai, Han and Gan, Chuang and Han, Song},
  journal={arXiv preprint arXiv:2205.14756},
  year={2022}
}
```
