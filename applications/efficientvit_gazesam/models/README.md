# Engine Creation Instructions

## Generating TensorRT Engines

### Default

1. `cd models; mkdir -p onnx`

2. Download the component ONNX files
    - Listed [here](https://huggingface.co/mit-han-lab/efficientvit-sam/tree/main/gazesam/onnx); save them to the `onnx` directory within this folder.  

3. Run `bash create_default_engines.sh`
    - Models generated with FP32 precision: image encoder
    - Models generated with FP16 precision: image decoder, depth estimation model, face detection model, gaze estimation model, object detection model

### Optimized

1. `cd models; mkdir -p tensorrt/int8/caches`

2. Download INT8 calibration caches
    - Listed [here](https://huggingface.co/mit-han-lab/efficientvit-sam/tree/main/gazesam/int8_calib_caches); save them to `tensorrt/int8/caches`.
    - Depending on your download method, the filenames may contain the `gazesam_int8_calib_caches_` prefix.
        To remove this prefix, run `rename 's/^gazesam_int8_calib_caches_//' gazesam_int8_calib_caches_*.cache` (while `cd`'ed into `tensorrt/int8/caches`).

3. Run `bash create_optimized_engines.sh`
    - Models generated with FP32 precision: image encoder
    - Models generated with FP16 precision: image decoder, depth estimation model, face detection model
    - Models generated with INT8 precision: gaze estimation model, object detection model

## Generating ONNX models

Note that default ONNX models are [available](https://huggingface.co/mit-han-lab/efficientvit-sam/tree/main/gazesam/onnx), so this section is likely not going to be relevant to you unless you'd like to generate your own ONNXes.  If you plan to generate an ONNX model and later use it to compile an engine, please remember to replace our defaults with your new file!

Instructions below indicate how to recreate our ONNX models.

### Face detection model

- Downloaded directly from [ProxylessNAS](https://github.com/mit-han-lab/proxylessnas/tree/master/proxyless_gaze/deployment/onnx/models).

### Gaze estimation model

- L2CS-Net model, downloaded by running this [script](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/398_L2CS-Net/download.sh) and choosing the `l2cs_net_1x3x448x448.onnx` variation. 

### Depth estimation model

- Depth-Anything-M model, downloaded by following these [instructions](https://github.com/spacewalk01/depth-anything-tensorrt?tab=readme-ov-file#-model-preparation).  We use `vitb_14` by default.

### YOLO object detection model

- `python create_onnx/create_yolo.py --model-size [s | m | l] --runtime [trt | onnx]`.  
- Set the runtime flag to trt (controls the NMS format) if you plan to compile a TensorRT engine from it.  We use the `yolo-nas-m` model.

### EfficientViT encoder model

- Download [`efficientvit-sam-l0.pt`](../../efficientvit_sam/README.md#pretrained-efficientvit-sam-models)

- ```
    python applications/efficientvit_sam/deployment/onnx/export_encoder.py \
    --model efficientvit-sam-l0 \
    --output demo/gazesam/models/onnx/evit_encoder_l0.onnx 
    ```

### EfficientVIT decoder model

- ```
    python applications/efficientvit_gazesam/models/create_onnx/create_evit_decoder.py \
    --output demo/gazesam/models/onnx/evit_decoder_l0.onnx \
    --model-type efficientvit-sam-l0 \
    --opset 17 \
    --return-single-mask
    ```
