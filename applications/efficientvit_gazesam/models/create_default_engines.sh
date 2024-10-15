#!/bin/bash

# Define default paths for ONNX files
EVIT_ENCODER_DEFAULT="onnx/evit_encoder_l0.onnx"
EVIT_DECODER_DEFAULT="onnx/evit_decoder_l0.onnx"
DEPTH_DEFAULT="onnx/depth_m.onnx"
YOLO_DEFAULT="onnx/yolo_m.onnx"
FACE_DEFAULT="onnx/face_detection.onnx"
GAZE_DEFAULT="onnx/gaze_estimation.onnx"
FP32_PRECISION="fp32"
FP16_PRECISION="fp16"

LOGDIR="tensorrt/logs"
current_time=$(date "+%Y%m%d%H%M%S")
LOGFILE="default_engine_creation_${current_time}.log"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --evit_encoder=<path>   Path to EfficientViT encoder ONNX file"
    echo "  --evit_decoder=<path>   Path to EfficientViT decoder ONNX file"
    echo "  --depth=<path>          Path to depth estimation ONNX file"
    echo "  --yolo=<path>           Path to YOLO ONNX file"
    echo "  --face=<path>           Path to face detection ONNX file"
    echo "  --gaze=<path>           Path to gaze estimation ONNX file"
    echo "  --all                   Convert all ONNX files to TensorRT engines with default paths"
    echo "  --fp16                  Convert models to FP16 precision"
    echo "  -h, --help              Display this help message"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --evit_encoder=*)
            EVIT_ENCODER="${1#*=}"
            shift
            ;;
        --evit_decoder=*)
            EVIT_DECODER="${1#*=}"
            shift
            ;;
        --depth=*)
            DEPTH="${1#*=}"
            shift
            ;;
        --yolo=*)
            YOLO="${1#*=}"
            shift
            ;;
        --face=*)
            FACE="${1#*=}"
            shift
            ;;
        --gaze=*)
            GAZE="${1#*=}"
            shift
            ;;
        --all)
            ALL=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if [ -z "$EVIT_ENCODER" ] && [ -z "$EVIT_DECODER" ] && [ -z "$DEPTH" ] && [ -z "$YOLO" ] && [ -z "$FACE" ] && [ -z "$GAZE" ]; then
    ALL=true
fi

# If --all option is provided, set all paths to default values
if [ "$ALL" = true ]; then
    echo "Converting all ONNX files to TensorRT engines with default paths..."
    EVIT_ENCODER="$EVIT_ENCODER_DEFAULT"
    EVIT_DECODER="$EVIT_DECODER_DEFAULT"
    DEPTH="$DEPTH_DEFAULT"
    YOLO="$YOLO_DEFAULT"
    FACE="$FACE_DEFAULT"
    GAZE="$GAZE_DEFAULT"
fi


declare -a ONNX_FILES=($EVIT_ENCODER $EVIT_DECODER $DEPTH $YOLO $FACE $GAZE)

file_exists() {
    local file="$1"
    [[ -f "$file" ]]
}

# Check all ONNXes exist
all_files_exist=true
for file in "${ONNX_FILES[@]}"; do
    if ! file_exists "$file"; then
        echo "Error: $file does not exist."
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

    local precision_flag=""
    if [ "$precision" == "fp16" ]; then
        precision_flag="--fp16"
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
                    $precision_flag \
                    &>> "${LOGDIR}/${LOGFILE}"
        else
            trtexec --onnx="$onnx_path" \
                    --saveEngine="tensorrt/${precision}/${engine_filename}" \
                    $precision_flag \
                    &>> "${LOGDIR}/${LOGFILE}"
        fi
    fi
}

mkdir -p tensorrt/$FP32_PRECISION
mkdir -p tensorrt/$FP16_PRECISION
mkdir -p $LOGDIR

#fp32
convert_to_engine "EfficientViT encoder" "$EVIT_ENCODER" "$FP32_PRECISION"
#fp16
convert_to_engine "EfficientViT decoder" "$EVIT_DECODER" "$FP16_PRECISION"
convert_to_engine "Depth estimation" "$DEPTH" "$FP16_PRECISION"
convert_to_engine "Face detection" "$FACE" "$FP16_PRECISION"
convert_to_engine "YOLO" "$YOLO" "$FP16_PRECISION"
convert_to_engine "Gaze estimation" "$GAZE" "$FP16_PRECISION"
