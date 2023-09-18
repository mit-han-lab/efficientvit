# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction

## News

**If you are interested in getting updates, please join our mailing list [here](https://forms.gle/Z6DNkRidJ1ouxmUk9).**

- [2023/09/18] The first version of the **EfficientViT Segment Anything Model** has been released.
- [2023/09/12] EfficientViT is highlighted by [MIT home page](https://www.mit.edu/archive/spotlight/efficient-computer-vision/) and [MIT News](https://news.mit.edu/2023/ai-model-high-resolution-computer-vision-0912).
- [2023/07/18] EfficientViT is accepted by ICCV 2023.

## Demo

EfficientViT-L0 for Segment Anything (1009 image/s on A100 GPU)
![demo](assets/demo/sam_l0_box.jpg)
![demo](assets/demo/sam_l0_point.jpg)
![demo](assets/demo/sam_l0_all.jpg)

EfficientViT-L1 for Semantic Segmentation (45.9ms on Nvidia Jetson AGX Orin, 82.7 mIoU on Cityscapes)

![demo](assets/demo/cityscapes_l1.gif)

## About EfficientViT Models

EfficientViT is a new family of vision models for efficient high-resolution dense prediction. The core building block of EfficientViT is a new lightweight multi-scale linear attention module that achieves global receptive field and multi-scale learning with only hardware-efficient operations.

## Getting Started

### Installation

```bash
conda create -n efficientvit python=3.10
conda activate efficientvit
conda install -c conda-forge mpi4py openmpi
pip install -r requirements.txt
```

### Dataset

<details>
  <summary>ImageNet: https://www.image-net.org/</summary>

  ```python
  Our code expects the ImageNet dataset directory to follow the following structure:

  imagenet
  ├── train
  ├── val
  ```

</details>

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

## Pretrained Models

Latency/Throughput is measured on NVIDIA Jetson Nano, NVIDIA Jetson AGX Orin, and NVIDIA A100 GPU with TensorRT, fp16. Data transfer time is included.

### Segment Anything

In this version, the EfficientViT segment anything models are trained using the image embedding extracted by [SAM ViT-H](https://github.com/facebookresearch/segment-anything) as the target. The prompt encoder and mask decoder are the same as [SAM ViT-H](https://github.com/facebookresearch/segment-anything).

| Image Encoder  | COCO-val2017 mIoU (all) | COCO-val2017 mIoU (large) | COCO-val2017 mIoU (medium) | COCO-val2017 mIoU (small) | Params |  MACs | A100 Throughput | Checkpoint |
|----------------------|:----------:|:----------:|:---------:|:------------:|:---------:|:---------:|:------------:|:------------:|
| NanoSAM  | 70.6 | 79.6 | 73.8 | 62.4 | - | - | 744 image/s | - |
| MobileSAM | 72.8 | 80.4 | 75.9 | 65.8 | - | - | 297 image/s | - |
||
| EfficientViT-L0 | 74.5 | 81.4 | 77.2 | 68.2 | 31M | 35G | 1009 image/s | [link](https://www.dropbox.com/scl/fi/jvemt62abv6tfpn33luy2/l0.pt?rlkey=ijo7i64n6kpnhnrwd3uux8vhf&dl=0) |
| EfficientViT-L1 | 75.2 | 81.8 | 78.1 | 68.9 | 44M | 49G | 815 image/s | [link](https://www.dropbox.com/scl/fi/7j968vmeyx1oiojubkqbm/l1.pt?rlkey=ogpprp1et1zgwucsy8731vv0h&dl=0) |

### ImageNet

All EfficientViT classification models are trained on ImageNet-1K with random initialization (300 epochs + 20 warmup epochs) using supervised learning.

<p align="left">
<img src="assets/files/cls_results.png"  width="450">
</p>

| Model         |  Resolution | ImageNet Top1 Acc | ImageNet Top5 Acc |  Params |  MACs |  A100 Throughput | Checkpoint |
|----------------------|:----------:|:----------:|:---------:|:------------:|:---------:|:------------:|:------------:|
| EfficientNetV2-S | 384x384 | 83.9 | - | 22M | 8.8G | 2869 image/s | - |
| EfficientNetV2-M | 480x480 | 85.1 | - | 54M | 24G | 1160 image/s | - |
| |
| EfficientViT-L1 | 224x224 |  84.5 | 96.9 | 53M | 5.3G | 6207 image/s | [link](https://drive.google.com/file/d/1q5y0YbN08O4ToUBK8RfZSDKp-s1y5_44/view?usp=sharing) |
| |
| EfficientViT-L2 | 224x224 |  85.0 | 97.1 | 64M | 6.9G | 4998 image/s | [link](https://drive.google.com/file/d/1FEjImtyIQhG4VsHsstLgNM09Y9qJn9Sk/view?usp=sharing) |
| EfficientViT-L2 | 256x256 |  85.4 | 97.2 | 64M | 9.1G | 3969 image/s | [link](https://drive.google.com/file/d/1pvYtY0ckAAMTkRq6TbwpQ0U1p_urz2fE/view?usp=sharing) |
| EfficientViT-L2 | 288x288 |  85.6 | 97.4 | 64M | 11G  | 3102 image/s | [link](https://drive.google.com/file/d/1GDr0y45YPX8iWEWNq5fEmjo0UgyZLpUs/view?usp=sharing) |
| EfficientViT-L2 | 320x320 |  85.8 | 97.4 | 64M | 14G  | 2525 image/s | [link](https://drive.google.com/file/d/1GDr0y45YPX8iWEWNq5fEmjo0UgyZLpUs/view?usp=sharing) |
| EfficientViT-L2 | 352x352 |  85.9 | 97.5 | 64M | 17G  | 2099 image/s | [link](https://drive.google.com/file/d/1gEkrj2JScJEcUgxeBSVKpUYBbple99yI/view?usp=sharing) |
| EfficientViT-L2 | 384x384 |  86.0 | 97.5 | 64M | 20G  | 1784 image/s | [link](https://drive.google.com/file/d/1MpjduiCTbUVS1XJri4_eqCbARJyYo74b/view?usp=sharing) |

<details>
  <summary>EfficientViT B series</summary>

  | Model         |  Resolution | ImageNet Top1 Acc | ImageNet Top5 Acc |  Params |  MACs |  Jetson Nano (bs1) | Jetson Orin (bs1) | Checkpoint |
  |----------------------|:----------:|:----------:|:---------:|:------------:|:---------:|:------------:|:------------:|:------------:|
  | EfficientViT-B1 | 224x224 | 79.4 | 94.3 | 9.1M | 0.52G | 24.8ms | 1.48ms | [link](https://drive.google.com/file/d/1hKN_hvLG4nmRzbfzKY7GlqwpR5uKpOOk/view?usp=share_link) |
  | EfficientViT-B1 | 256x256 | 79.9 | 94.7 | 9.1M | 0.68G | 28.5ms | 1.57ms | [link](https://drive.google.com/file/d/1hXcG_jB0ODMOESsSkzVye-58B4F3Cahs/view?usp=share_link) |
  | EfficientViT-B1 | 288x288 | 80.4 | 95.0 | 9.1M | 0.86G | 34.5ms | 1.82ms | [link](https://drive.google.com/file/d/1sE_Suz9gOOUO7o5r9eeAT4nKK8Hrbhsu/view?usp=share_link) |
  | |
  | EfficientViT-B2 | 224x224 | 82.1 | 95.8 | 24M  | 1.6G  | 50.6ms | 2.63ms | [link](https://drive.google.com/file/d/1DiM-iqVGTrq4te8mefHl3e1c12u4qR7d/view?usp=share_link) |
  | EfficientViT-B2 | 256x256 | 82.7 | 96.1 | 24M  | 2.1G  | 58.5ms | 2.84ms | [link](https://drive.google.com/file/d/192OOk4ISitwlyW979M-FSJ_fYMMW9HQz/view?usp=share_link) |
  | EfficientViT-B2 | 288x288 | 83.1 | 96.3 | 24M  | 2.6G  | 69.9ms | 3.30ms | [link](https://drive.google.com/file/d/1aodcepOyne667hvBAGpf9nDwmd5g0NpU/view?usp=share_link) |
  | |
  | EfficientViT-B3 | 224x224 | 83.5 | 96.4 | 49M  | 4.0G  | 101ms  | 4.36ms | [link](https://drive.google.com/file/d/18RZDGLiY8KsyJ7LGic4mg1JHwd-a_ky6/view?usp=share_link) |
  | EfficientViT-B3 | 256x256 | 83.8 | 96.5 | 49M  | 5.2G  | 120ms  | 4.74ms | [link](https://drive.google.com/file/d/1y1rnir4I0XiId-oTCcHhs7jqnrHGFi-g/view?usp=share_link) |
  | EfficientViT-B3 | 288x288 | 84.2 | 96.7 | 49M  | 6.5G  | 141ms  | 5.63ms | [link](https://drive.google.com/file/d/1KfwbGtlyFgslNr4LIHERv6aCfkItEvRk/view?usp=share_link) |
</details>

### Cityscapes

<p align="left">
<img src="assets/files/city_results.png"  width="450">
</p>

| Model         |  Resolution | Cityscapes mIoU | Params |  MACs |  Jetson Orin Latency (bs1) | A100 Throughput (bs1) | Checkpoint |
|----------------------|:----------:|:----------:|:---------:|:------------:|:---------:|:---------:|:------------:|
| EfficientViT-L1 | 1024x2048 | 82.7 | 40M | 282G | 45.9ms  | 122 image/s | [link](https://www.dropbox.com/scl/fi/7s1ou7bdsbwwvq9bmjjuv/l1.pt?rlkey=m1ysvjkhrb0pb7uuyoir92p77&dl=0) |
| EfficientViT-L2 | 1024x2048 | 83.2 | 53M | 396G | 60.0ms  | 102 image/s | [link](https://www.dropbox.com/scl/fi/erehtq6j1daushirhbzc5/l2.pt?rlkey=068makdxpdh469ueps2trhte1&dl=0) |

<details>
  <summary>EfficientViT B series</summary>

  | Model         |  Resolution | Cityscapes mIoU | Params |  MACs |  Jetson Nano (bs1) | Jetson Orin (bs1) | Checkpoint |
  |----------------------|:----------:|:----------:|:---------:|:------------:|:---------:|:---------:|:------------:|
  | EfficientViT-B0 | 1024x2048 | 75.7 | 0.7M | 4.4G | 275ms  | 9.9ms  | [link](https://drive.google.com/file/d/1Ix1Dh3xlpaf0Wzh01Xmo-hAYkoXt1EAD/view?usp=sharing) |
  | EfficientViT-B1 | 1024x2048 | 80.5 | 4.8M | 25G  | 819ms  | 24.3ms | [link](https://drive.google.com/file/d/1jNjLFtIUNvu5MwSupgFHLc-2kmFLiu67/view?usp=sharing) |
  | EfficientViT-B2 | 1024x2048 | 82.1 | 15M  | 74G  | 1676ms | 46.5ms | [link](https://drive.google.com/file/d/1bwGjzVQOg_ygML8F9JhsIj-ntn-cuWmB/view?usp=sharing) |
  | EfficientViT-B3 | 1024x2048 | 83.0 | 40M  | 179G | 3192ms | 81.8ms | [link](https://drive.google.com/file/d/19aiy3qrKqx1n8zzy_ewYn4-Z3LM4bkn4/view?usp=sharing) |

</details>

### ADE20K

<p align="left">
<img src="assets/files/ade_results.png"  width="450">
</p>

| Model         |  Resolution | ADE20K mIoU | Params |  MACs |  Jetson Orin Latency (bs1) | A100 Throughput (bs16) | Checkpoint |
|----------------------|:----------:|:----------:|:---------:|:------------:|:---------:|:---------:|:------------:|
| EfficientViT-L1 | 512x512 | 49.1 | 40M | 36G | 7.2ms  | 947 image/s | [link](https://www.dropbox.com/scl/fi/kn4g04fgme9uctaazsgct/l1.pt?rlkey=x1052if1ae7795yafp3urib5r&dl=0) |
| EfficientViT-L2 | 512x512 | 50.7 | 51M | 45G | 9.0ms | 758 image/s | [link](https://www.dropbox.com/scl/fi/565wb47z1f5re9jckr42t/l2.pt?rlkey=ojffxngf6iv0oiost6c2tskul&dl=0) |

<details>
  <summary>EfficientViT B series</summary>

  | Model         |  Resolution | ADE20K mIoU | Params |  MACs |  Jetson Nano (bs1) | Jetson Orin (bs1) | Checkpoint |
  |----------------------|:----------:|:----------:|:---------:|:------------:|:---------:|:---------:|:------------:|
  | EfficientViT-B1 | 512x512 | 42.8 | 4.8M | 3.1G | 110ms | 4.0ms  | [link](https://drive.google.com/file/d/13YNtEJ-mRnAhu0fIs2EnAP-3TmSneRAC/view?usp=sharing) |
  | EfficientViT-B2 | 512x512 | 45.9 | 15M  | 9.1G | 212ms | 7.3ms  | [link](https://drive.google.com/file/d/1k5sWY6aJ1FCtMt4GRTZqSFlJ-u_TSHzc/view?usp=sharing) |
  | EfficientViT-B3 | 512x512 | 49.0 | 39M  | 22G  | 411ms | 12.5ms | [link](https://drive.google.com/file/d/1ghpTf9GTTj_8mn5QJh-7cLK1_wL3pKWr/view?usp=sharing) |

</details>

## Usage
```python
# segment anything
from efficientvit.sam_model_zoo import create_sam_model

efficientvit_sam = create_sam_model(
  name="l1", weight_url="assets/checkpoints/sam/l1.pt",
)
efficientvit_sam = efficientvit_sam.cuda().eval()
```

```python
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor

efficientvit_sam_predictor = EfficientViTSamPredictor(efficientvit_sam)
```

```python
from efficientvit.models.efficientvit.sam import EfficientViTSamAutomaticMaskGenerator

efficientvit_mask_generator = EfficientViTSamAutomaticMaskGenerator(efficientvit_sam)

```

```python
# classification
from efficientvit.cls_model_zoo import create_cls_model

model = create_cls_model(
  name="l2", weight_url="assets/checkpoints/cls/l2-r384.pt"
)
```

```python
# semantic segmentation
from efficientvit.seg_model_zoo import create_seg_model

model = create_seg_model(
  name="l2", dataset="cityscapes", weight_url="assets/checkpoints/seg/cityscapes/l2.pt"
)

model = create_seg_model(
  name="l2", dataset="ade20k", weight_url="assets/checkpoints/seg/ade20k/l2.pt"
)
```

## Evaluation

Please run `eval_sam_coco.py`, `eval_cls_model.py` or `eval_seg_model.py` to evaluate our models.

Examples: [segment anything](assets/files/eval_sam_model.sh), [classification](assets/files/eval_cls_model.sh), [segmentation](assets/files/eval_seg_model.sh)

## Visualization

Please run `demo_sam_model.py` to visualize our segment anything models.

Example:

```bash
# segment everything
python demo_sam_model.py --model l1 --mode all

# prompt with points
python demo_sam_model.py --model l1 --mode point

# prompt with box
python demo_sam_model.py --model l1 --mode box --box "[150,70,630,400]"

```

Please run `eval_seg_model.py` to visualize the outputs of our semantic segmentation models.

Example:

```bash
python eval_seg_model.py --dataset cityscapes --crop_size 1024 --model b3 --save_path demo/cityscapes/b3/
```

## Benchmarking with TFLite

To generate TFLite files, please refer to `tflite_export.py`. It requires the TinyNN package.

```bash
pip install git+https://github.com/alibaba/TinyNeuralNetwork.git
```

Example:

```bash
python tflite_export.py --export_path model.tflite --task seg --dataset ade20k --model b3 --resolution 512 512
```

## Benchmarking with TensorRT

To generate onnx files, please refer to `onnx_export.py`.

## Training

Please see [TRAINING.md](TRAINING.md) for detailed training instructions.

## Contact

Han Cai: <hancai@mit.edu>

## TODO

- [x] ImageNet Pretrained models
- [x] Segmentation Pretrained models
- [x] ImageNet training code
- [ ] EfficientViT L series, designed for cloud
- [ ] EfficientViT for segment anything
- [ ] EfficientViT for super-resolution
- [ ] Segmentation training code

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
