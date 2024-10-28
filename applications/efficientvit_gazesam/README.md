# EfficientViT-GazeSAM

EfficientViT-GazeSAM is a gaze-prompted image segmentation model capable of running in real time with TensorRT on an NVIDIA RTX 4070.  GazeSAM is comprised of a face detection component (ProxylessGaze), gaze detection component (L2CS-Net), an object detection component (YOLO-NAS), a depth estimation component (Depth-Anything), and an image segmentation component (EfficientViT).

![GazeSAM demo](https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/gazesam/efficientvit_gazesam_demo.gif)

## Installation and Setup

Prior to following the runtime-specific instructions below, please make sure to follow the conda environment creation and package installation instructions for this repo.

```bash
# install extra packages
pip install -r extra_requirements.txt
```

### TensorRT (recommended mode for real-time performance on RTX 4070)

1. Ensure the following packages are installed.

    a. [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)

    b. [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)

    c. `python -m pip install cuda-python`.

2. Please follow the engine creation instructions within the `models` directory [here](models/README.md).  You can choose between the default version (FP32 + FP16 engines) and the optimized version (FP32, FP16, and INT8 engines). The optimized version is approximately 5ms faster per frame (on an RTX 4070) but both will run in real-time.

### ONNX

1. `python -m pip install onnxruntime-gpu`

    Note: if you run into ONNXRuntime issues, you can try uninstalling `onnxruntime` and `onnxruntime-gpu`, then reinstalling `onnxruntime-gpu`.

2. Download the ONNX model components [here](https://huggingface.co/mit-han-lab/efficientvit-sam/tree/main/gazesam/onnx) and save them to the `models/onnx` directory (make sure to create the `onnx` subfolder).

### PyTorch

1. Setup EfficientViT-SAM model [[guide](../efficientvit_sam/README.md#pretrained-efficientvit-sam-models)]

2. Setup depth estimation model

    a. Download the Depth-Anything [repo](https://github.com/LiheYoung/Depth-Anything) and save it as a subfolder within this current directory.

    b. `cp models/create_pytorch/dpt_replacement.py Depth-Anything/depth_anything/dpt.py`. This prepends the torchhub local download path with "Depth-Anything".

    c. Download the Depth-Anything-Base checkpoint [here](https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints/depth_anything_vitb14.pth).  Save it within the `models/pytorch` directory (make sure to create the `pytorch` subfolder).

3. Setup gaze estimation model

    a. Download the L2CS-Net pickle file [here](https://drive.google.com/drive/folders/1qDzyzXO6iaYIMDJDSyfKeqBx8O74mF8s).  Save it within the `models/pytorch` directory (make sure to create the `pytorch` subfolder).

4. Download the ONNX model components [here](https://huggingface.co/mit-han-lab/efficientvit-sam/tree/main/gazesam/onnx).  Save the files within the `models/onnx` directory (make sure to create the `onnx` subfolder).

## Usage

GazeSAM can process webcam and video file inputs. To run with webcam, run `python gazesam_demo.py --webcam`.  To run with input video, `python gazesam_demo.py --video <path>`.

 By default, we run with TensorRT (use the `runtime` flag to change this, but note that only TensorRT mode will produce results in real-time).  Results are saved by default to the `output_videos` directory (modifiable via the `output-dir` flag).

 If you generated engines using the optimized script, set `--precision-mode optimized`.  Modes described [here](models/README.md). You can download the example video [here](https://huggingface.co/mit-han-lab/efficientvit-sam/blob/main/gazesam/example.mp4).

Input video + default engines example: `python gazesam_demo.py --video input_videos/example.mp4 --precision-mode default`

Webcam + optimized engines example: `python gazesam_demo.py --webcam --precision-mode optimized`

## Contributor

[Nicole Stiles](https://github.com/ncstiles)

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
