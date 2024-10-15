#!/bin/bash

# Define default paths for ONNX files
EVIT_ENCODER="onnx/evit_encoder_l0.onnx"
EVIT_DECODER="onnx/evit_decoder_l0.onnx"
DEPTH="onnx/depth_m.onnx"
YOLO="onnx/yolo_m.onnx"
FACE="onnx/face_detection.onnx"
GAZE="onnx/gaze_estimation.onnx"

# Calibration caches for int8 quantization
YOLO_CACHE="tensorrt/int8/caches/yolo_m.cache"
GAZE_CACHE="tensorrt/int8/caches/gaze_estimation.cache"

FP32_PRECISION="fp32"
FP16_PRECISION="fp16"
INT8_PRECISION="int8"

LOGDIR="tensorrt/logs"
current_time=$(date "+%Y%m%d%H%M%S")
LOGFILE="optimized_engine_creation_${current_time}.log"

declare -a ONNX_FILES=($EVIT_ENCODER $EVIT_DECODER $DEPTH $YOLO $FACE $GAZE)
declare -a CACHE_FILES=($YOLO_CACHE $GAZE_CACHE)

file_exists() {
    local file="$1"
    [[ -f "$file" ]]
}

# Check if all required ONNXes and caches exist
all_files_exist=true
for file in "${ONNX_FILES[@]}" "${CACHE_FILES[@]}"; do
    if ! file_exists "$file"; then
        if [[ $file == *.cache ]]; then
            echo "$file does not exist."
            echo "Check cache file is actually saved to the tensorrt/int8/caches dir".
            echo "Also check you've removed the potential 'caches_' prefix from the downloaded cache filename.  Check README in this dir for more details."
            echo ""
        else
            echo "Error: $file does not exist."
        fi
        all_files_exist=false
    fi
done

if ! $all_files_exist; then
    exit 1
fi

convert_to_engine() {
    local model_desc="$1"
    local onnx_path="$2"
    local precision="$3"
    local cache="$4"

    local extra_flags=""
    if [ "$precision" == "fp16" ]; then
        extra_flags="--fp16"
    elif [ "$precision" == "int8" ]; then
        extra_flags="--int8 --calib=$cache --verbose"
    fi

    if [ -n "$onnx_path" ]; then
        local filename=$(basename "$onnx_path" .onnx)
        local engine_filename="${filename}_${precision}.engine"
        echo "Converting $filename ONNX model at $onnx_path to TensorRT..."

        if [ "$model_desc" == "EfficientViT decoder" ]; then
            trtexec --onnx="$onnx_path" \
                    --saveEngine="tensorrt/${precision}/${engine_filename}" \
                    --minShapes=boxes:1x1x4 \
                    --optShapes=boxes:2x1x4 \
                    --maxShapes=boxes:10x1x4 \
                    $extra_flags \
                    &>> "${LOGDIR}/${LOGFILE}"
        elif [ "$precision" == "int8" ]; then
            trtexec --onnx="$onnx_path" \
                    --saveEngine="tensorrt/${precision}/${engine_filename}" \
                    $extra_flags \
                    &>> "${LOGDIR}/${LOGFILE}"
        
        else
            trtexec --onnx="$onnx_path" \
                    --saveEngine="tensorrt/${precision}/${engine_filename}" \
                    $extra_flags \
                    &>> "${LOGDIR}/${LOGFILE}"
        fi
    fi
}

mkdir -p tensorrt/$FP32_PRECISION
mkdir -p tensorrt/$FP16_PRECISION
mkdir -p tensorrt/$INT8_PRECISION
mkdir -p $LOGDIR

#int8
convert_to_engine "YOLO" "$YOLO" "$INT8_PRECISION" "$YOLO_CACHE"
convert_to_engine "Gaze estimation" "$GAZE" "$INT8_PRECISION" "$GAZE_CACHE"
#fp16
convert_to_engine "Face detection" "$FACE" "$FP16_PRECISION"
convert_to_engine "Depth estimation" "$DEPTH" "$FP16_PRECISION"
convert_to_engine "EfficientViT decoder" "$EVIT_DECODER" "$FP16_PRECISION"
#fp32
convert_to_engine "EfficientViT encoder" "$EVIT_ENCODER" "$FP32_PRECISION"
