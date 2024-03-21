# EfficientViT-SAM Demo

## Install

This demo is compatible with three runtimes: PyTorch, ONNX, and TensorRT.  The demo allows experimentation with all five versions of our model: l0, l1, l2, xl0, and xl1.  

Please follow the [Getting Started](../../README.md#getting-started) instructions, then follow the runtime-specific instructions below.

### PyTorch-specific installation instructions
1. Please download the model checkpoint files listed [here](../../applications/sam.md#pretrained-models) and save them to the `assets/checkpoints` directory.

### ONNX-specific installation instructions
1. Create ONNX models

    Option 1. Download the ONNX models from [Huggingface](https://huggingface.co/han-cai/efficientvit-sam/tree/main) and save them to the `assets/export_models/sam/onnx` directory.

    Option 2. Generate models locally.  Ensure you have downloaded the checkpoint files (installation instructions in PyTorch section above) before running the command below.

        chmod +x demo/sam/generate_models.sh

        ./demo/sam/generate_models.sh --onnx l0 l1 l2 xl0 xl1

### TensorRT-specific installation instructions

1. Please ensure you have CUDA working properly and the following packages installed.

    a. [tensorrt](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)

    b. [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)

2. Create TensorRT engines

    Ensure you have downloaded the ONNX models before running the script (installation instructions outlined in the ONNX section above).
    Engines will be saved to the `assets/export_models/sam/tensorrt` directory.

        chmod +x demo/sam/generate_models.sh

        ./demo/sam/generate_models.sh --tensorrt l0 l1 l2 xl0 xl1


## Usage
To launch the interactive demo, run the command below.  If no runtime is specified, PyTorch is used by default.

For users with CPUs, we recommend using the PyTorch or ONNX runtime.  

For users with GPU access, we recommend using the PyTorch runtime to start and explore the TensorRT runtime if looking for faster inference.

    python -m demo.sam.gradio_web_server --runtime [pytorch | onnx | tensorrt]

### Usage notes

We offer point-prompted, box-prompted, mixed prompt, and automatically generated image segmentation modes.

A left click creates a positive foreground prompt.  A right click (two-fingered click using Mac trackpad) creates a negative background prompt that works to exclude objects from the segmentation.

For our point-prompted mode and mixed prompt mode, all prompts will go towards segmenting a single object.  For example, if multiple points are added to the image, all points are assumed to be segmenting the same object.  In our mixed prompt mode, we enforce this by only taking into account the last drawn box prompt (with no restriction on the number of point prompts).

For our box-prompted mode, each box goes towards segmenting a separate object.

Our full image segmentation mode automatically generates segmentations over the full image.  The segmentation parameters in the slider will change the number and granularity of the final masks displayed.  Feel free to play around with these parameters and reset them to the defaults with the "reset segmentation parameters" button at the bottom.  The segmentation parameters are generally tuned towards our xl models.  When experimenting with our l-series models, you may get better full image segmentation results by changing the slider values.

The models listed in the dropdown are based on the files you have stored in the default installation directories.  Please ensure you save your models using the instructions above.

Either use the images shown in the example bar at the bottom of the demo by clicking on the image, uploading your own image, or clicking the webcam icon to take a picture. 


### Contributor

[Nicole Stiles](https://github.com/ncstiles)
