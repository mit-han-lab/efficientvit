# Segment Anything

## Datasets

[COCO2017](https://cocodataset.org/#download) and [LVIS annotations](https://www.lvisdataset.org/dataset).

To conduct box-prompted instance segmentation, you must first obtain the *source_json_file* of detected bounding boxes. Follow the instructions of [ViTDet](https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet), [YOLOv8](https://github.com/ultralytics/ultralytics), and [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) to get the *source_json_file*. You can also download our [pre-generated files](https://huggingface.co/han-cai/efficientvit-sam/tree/main/source_json_file).

<details>
<summary>Expected directory structure:</summary>

```python
coco
├── train2017
├── val2017
├── annotations
│   ├── instances_val2017.json
│   ├── lvis_v1_val.json
|── source_json_file
│   ├── coco_groundingdino.json
│   ├── coco_vitdet.json
│   ├── coco_yolov8.json
│   ├── lvis_vitdet.json
```

</details>

## Pretrained Models

Latency/Throughput is measured on NVIDIA Jetson AGX Orin, and NVIDIA A100 GPU with TensorRT, fp16. Data transfer time is included.

<p align="left">
<img src="../assets/files/sam_zero_shot_coco_mAP.png"  width="450">
</p>

| Model         |  Resolution | COCO mAP | LVIS mAP | Params |  MACs | Jetson Orin Latency (bs1) | A100 Throughput (bs16) | Checkpoint |
|----------------------|:----------:|:----------:|:---------:|:------------:|:---------:|:---------:|:------------:|:------------:|
| EfficientViT-SAM-L0 | 512x512 | 45.7 | 41.8 | 34.8M  | 35G | 8.2ms  | 762 images/s | [link](https://huggingface.co/han-cai/efficientvit-sam/resolve/main/l0.pt) |
| EfficientViT-SAM-L1 | 512x512 | 46.2 | 42.1 | 47.7M | 49G |  10.2ms | 638 images/s | [link](https://huggingface.co/han-cai/efficientvit-sam/resolve/main/l1.pt) |
| EfficientViT-SAM-L2 | 512x512 | 46.6 | 42.7 | 61.3M | 69G |  12.9ms | 538 images/s  | [link](https://huggingface.co/han-cai/efficientvit-sam/resolve/main/l2.pt) |
| EfficientViT-SAM-XL0 | 1024x1024 | 47.5 | 43.9 | 117.0M | 185G | 22.5ms  | 278 images/s | [link](https://huggingface.co/han-cai/efficientvit-sam/resolve/main/xl0.pt) |
| EfficientViT-SAM-XL1 | 1024x1024 | 47.8 | 44.4 | 203.3M | 322G | 37.2ms  | 182 images/s | [link](https://huggingface.co/han-cai/efficientvit-sam/resolve/main/xl1.pt) |
<p align="center">
<b> Table1: Summary of All EfficientViT-SAM Variants.</b> COCO mAP and LVIS mAP are measured using ViTDet's predicted bounding boxes as the prompt. End-to-end Jetson Orin latency and A100 throughput are measured with TensorRT and fp16.
</p>

## Usage

```python
# segment anything
from efficientvit.sam_model_zoo import create_sam_model

efficientvit_sam = create_sam_model(
  name="xl1", weight_url="assets/checkpoints/sam/xl1.pt",
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

## Evaluation

Note: For LVIS evaluation, please manually install the [lvis](https://github.com/lvis-dataset/lvis-api) package (check this [issue](https://github.com/lvis-dataset/lvis-api/issues/37) for more details).

### Box-Prompted Zero-Shot Instance Segmentation

#### Ground Truth Bounding Box

```python
# COCO
torchrun --nproc_per_node=8 eval_sam_model.py --dataset coco --image_root coco/val2017 --annotation_json_file coco/annotations/instances_val2017.json --model xl1 --weight_url assets/checkpoints/sam/xl1.pt --prompt_type box

# expected results: all=79.927, large=83.748, medium=82.210, small=75.833
```

```python
# LVIS
torchrun --nproc_per_node=8 eval_sam_model.py --dataset lvis --image_root coco --annotation_json_file coco/annotations/lvis_v1_val.json --model xl1 --weight_url assets/checkpoints/sam/xl1.pt --prompt_type box

# expected results: all=79.886, large=91.577, medium=88.447, small=74.412
```

#### Detected Bounding Box

```python
# COCO
torchrun --nproc_per_node=8 eval_sam_model.py --dataset coco --image_root coco/val2017 --annotation_json_file coco/annotations/instances_val2017.json --model xl1 --weight_url assets/checkpoints/sam/xl1.pt --prompt_type box_from_detector --source_json_file coco/source_json_file/coco_vitdet.json

# expected results: 
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.478
```

```python
# LVIS
torchrun --nproc_per_node=8 eval_sam_model.py --dataset lvis --image_root coco --annotation_json_file coco/annotations/lvis_v1_val.json --model xl1 --weight_url assets/checkpoints/sam/xl1.pt --prompt_type box_from_detector --source_json_file coco/source_json_file/lvis_vitdet.json

# expected results: 
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=300 catIds=all] = 0.444
```

### Point-Prompted Zero-Shot Instance Segmentation

```python
# COCO
torchrun --nproc_per_node=8 eval_sam_model.py --dataset coco --image_root coco/val2017 --annotation_json_file coco/annotations/instances_val2017.json --model xl1 --weight_url assets/checkpoints/sam/xl1.pt --prompt_type point --num_click 1

# expected results: all=59.757, large=62.132, medium=63.837, small=55.029
```

```python
# LVIS
torchrun --nproc_per_node=8 eval_sam_model.py --dataset lvis --image_root coco --annotation_json_file coco/annotations/lvis_v1_val.json --model xl1 --weight_url assets/checkpoints/sam/xl1.pt --prompt_type point --num_click 1

# expected results: all=56.624, large=72.442, medium=71.796, small=47.750
```

## Visualization

Please run `demo_sam_model.py` to visualize our segment anything models.

Example:

```bash
# segment everything
python demo_sam_model.py --model xl1 --mode all

# prompt with points
python demo_sam_model.py --model xl1 --mode point

# prompt with box
python demo_sam_model.py --model xl1 --mode box --box "[150,70,640,400]"

```

## Deployment

### ONNX Export

```python
# Export Encoder
python deployment/sam/onnx/export_encoder.py --model xl1 --weight_url assets/checkpoints/sam/xl1.pt --output assets/export_models/sam/onnx/xl1_encoder.onnx 
```

```python
# Export Decoder
python deployment/sam/onnx/export_decoder.py --model xl1 --weight_url assets/checkpoints/sam/xl1.pt --output assets/export_models/sam/onnx/xl1_decoder.onnx --return-single-mask
```

```python
# ONNX Inference
python -m deployment.sam.onnx.inference --model xl1 --encoder_model assets/export_models/sam/onnx/xl1_encoder.onnx --decoder_model assets/export_models/sam/onnx/xl1_decoder.onnx --mode point
```

### TensorRT Export

```python
# Export Encoder
trtexec --onnx=assets/export_models/sam/onnx/xl1_encoder.onnx --minShapes=input_image:1x3x1024x1024 --optShapes=input_image:4x3x1024x1024 --maxShapes=input_image:4x3x1024x1024 --saveEngine=assets/export_models/sam/tensorrt/xl1_encoder.engine
```

```python
# Export Decoder
trtexec --onnx=assets/export_models/sam/onnx/xl1_decoder.onnx --minShapes=point_coords:1x1x2,point_labels:1x1 --optShapes=point_coords:16x2x2,point_labels:16x2 --maxShapes=point_coords:16x2x2,point_labels:16x2 --fp16 --saveEngine=assets/export_models/sam/tensorrt/xl1_decoder.engine
```

```python
# TensorRT Inference
python -m deployment.sam.tensorrt.inference --model xl1 --encoder_engine assets/export_models/sam/tensorrt/xl1_encoder.engine --decoder_engine assets/export_models/sam/tensorrt/xl1_decoder.engine --mode point
```

## Citation

If EfficientViT or EfficientViT-SAM is useful or relevant to your research, please kindly recognize our contributions by citing our papers:

```
@article{cai2022efficientvit,
  title={Efficientvit: Enhanced linear attention for high-resolution low-computation visual recognition},
  author={Cai, Han and Gan, Chuang and Han, Song},
  journal={arXiv preprint arXiv:2205.14756},
  year={2022}
}

@article{zhang2024efficientvit,
  title={EfficientViT-SAM: Accelerated Segment Anything Model Without Performance Loss},
  author={Zhang, Zhuoyang and Cai, Han and Han, Song},
  journal={arXiv preprint arXiv:2402.05008},
  year={2024}
}
```
