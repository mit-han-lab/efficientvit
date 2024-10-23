# EfficientViT Segmentation

![demo](../../assets/cityscapes_l1.gif)

<p align="left">
<img src="../../assets/city_results.png"  width="600">
</p>

<p align="left">
<img src="../../assets/ade_results.png"  width="600">
</p>

## Datasets

<details>
  <summary>Cityscapes: https://www.cityscapes-dataset.com/</summary>

  ```python
  Our code expects the Cityscapes dataset directory to follow the following structure:

  cityscapes
  ├── gtFine
  |   ├── train
  |   ├── val
  ├── leftImg8bit
  |   ├── train
  |   ├── val
  ```

</details>

<details>
  <summary>ADE20K: https://groups.csail.mit.edu/vision/datasets/ADE20K/</summary>

  ```python
  Our code expects the ADE20K dataset directory to follow the following structure:

  ade20k
  ├── annotations
  |   ├── training
  |   ├── validation
  ├── images
  |   ├── training
  |   ├── validation
  ```
  
</details>

## Pretrained EfficientViT Segmentation Models

Latency/Throughput is measured on NVIDIA Jetson Nano, NVIDIA Jetson AGX Orin, and NVIDIA A100 GPU with TensorRT, fp16. Data transfer time is included. Please put the downloaded checkpoints under *${efficientvit_repo}/assets/checkpoints/efficientvit_seg/*

### Cityscapes

| Model         |  Resolution | Cityscapes mIoU | Params |  MACs |  Jetson Orin Latency (bs1) | A100 Throughput (bs1) | Checkpoint |
|----------------------|:----------:|:----------:|:---------:|:------------:|:---------:|:---------:|:------------:|
| EfficientViT-L1 | 1024x2048 | 82.716 | 40M | 282G | 45.9ms  | 122 image/s | [link](https://huggingface.co/han-cai/efficientvit-seg/resolve/main/efficientvit_seg_l1_cityscapes.pt) |
| EfficientViT-L2 | 1024x2048 | 83.228 | 53M | 396G | 60.0ms  | 102 image/s | [link](https://huggingface.co/han-cai/efficientvit-seg/resolve/main/efficientvit_seg_l2_cityscapes.pt) |

<details>
  <summary>EfficientViT B series</summary>

  | Model         |  Resolution | Cityscapes mIoU | Params |  MACs |  Jetson Nano (bs1) | Jetson Orin (bs1) | Checkpoint |
  |----------------------|:----------:|:----------:|:---------:|:------------:|:---------:|:---------:|:------------:|
  | EfficientViT-B0 | 1024x2048 | 75.653 | 0.7M | 4.4G | 275ms  | 9.9ms  | [link](https://huggingface.co/han-cai/efficientvit-seg/resolve/main/efficientvit_seg_b0_cityscapes.pt) |
  | EfficientViT-B1 | 1024x2048 | 80.547 | 4.8M | 25G  | 819ms  | 24.3ms | [link](https://huggingface.co/han-cai/efficientvit-seg/resolve/main/efficientvit_seg_b1_cityscapes.pt) |
  | EfficientViT-B2 | 1024x2048 | 82.073 | 15M  | 74G  | 1676ms | 46.5ms | [link](https://huggingface.co/han-cai/efficientvit-seg/resolve/main/efficientvit_seg_b2_cityscapes.pt) |
  | EfficientViT-B3 | 1024x2048 | 83.016 | 40M  | 179G | 3192ms | 81.8ms | [link](https://huggingface.co/han-cai/efficientvit-seg/resolve/main/efficientvit_seg_b3_cityscapes.pt) |

</details>

### ADE20K

| Model         |  Resolution | ADE20K mIoU | Params |  MACs |  Jetson Orin Latency (bs1) | A100 Throughput (bs16) | Checkpoint |
|----------------------|:----------:|:----------:|:---------:|:------------:|:---------:|:---------:|:------------:|
| EfficientViT-L1 | 512x512 | 49.191 | 40M | 36G | 7.2ms  | 947 image/s | [link](https://huggingface.co/han-cai/efficientvit-seg/resolve/main/efficientvit_seg_l1_ade20k.pt) |
| EfficientViT-L2 | 512x512 | 50.702 | 51M | 45G | 9.0ms | 758 image/s | [link](https://huggingface.co/han-cai/efficientvit-seg/resolve/main/efficientvit_seg_l2_ade20k.pt) |

<details>
  <summary>EfficientViT B series</summary>

  | Model         |  Resolution | ADE20K mIoU | Params |  MACs |  Jetson Nano (bs1) | Jetson Orin (bs1) | Checkpoint |
  |----------------------|:----------:|:----------:|:---------:|:------------:|:---------:|:---------:|:------------:|
  | EfficientViT-B1 | 512x512 | 42.840 | 4.8M | 3.1G | 110ms | 4.0ms  | [link](https://huggingface.co/han-cai/efficientvit-seg/resolve/main/efficientvit_seg_b1_ade20k.pt) |
  | EfficientViT-B2 | 512x512 | 45.941 | 15M  | 9.1G | 212ms | 7.3ms  | [link](https://huggingface.co/han-cai/efficientvit-seg/resolve/main/efficientvit_seg_b2_ade20k.pt) |
  | EfficientViT-B3 | 512x512 | 49.013 | 39M  | 22G  | 411ms | 12.5ms | [link](https://huggingface.co/han-cai/efficientvit-seg/resolve/main/efficientvit_seg_b3_ade20k.pt) |

</details>

## Usage

```python
# semantic segmentation
from efficientvit.seg_model_zoo import create_efficientvit_seg_model

model = create_efficientvit_seg_model(name="efficientvit-seg-l2-cityscapes", pretrained=True)

model = create_efficientvit_seg_model(name="efficientvit-seg-l2-ade20k", pretrained=True)
```

## Evaluation

Please run [eval_efficientvit_seg_model.py](eval_efficientvit_seg_model.py) to evaluate our models.

Examples: [segmentation](../../assets/eval_efficientvit_seg_model.sh)

## Visualization

Please run [demo_efficientvit_seg_model.py](demo_efficientvit_seg_model.py) to visualize the models.

Example:

```bash
python applications/efficientvit_seg/demo_efficientvit_seg_model.py --image_path assets/fig/indoor.jpg --dataset ade20k --crop_size 512 --model efficientvit-seg-l2-ade20k

python applications/efficientvit_seg/demo_efficientvit_seg_model.py --image_path assets/fig/city.png --dataset cityscapes --crop_size 1024 --model efficientvit-seg-l2-cityscapes
```

## Export

### Onnx

To generate ONNX files, please refer to [onnx_export.py](../../assets/onnx_export.py).

Example:

```bash
python assets/onnx_export.py --export_path assets/export_models/efficientvit_seg_l2_cityscapes_r1024x2048.onnx --task seg --model efficientvit-seg-l2-cityscapes --resolution 1024 2048 --bs 1
```

### TFLite

To generate TFLite files, please refer to [tflite_export.py](../../assets/tflite_export.py).

Example:

```bash
python assets/tflite_export.py --export_path assets/export_models/efficientvit_seg_l2_ade20k_r512x512.onnx --task seg --model efficientvit-seg-l2-ade20k --resolution 512 512
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
