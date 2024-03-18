usage() {
    echo "Usage: $0 [-a|--all | -o|--onnx | -t|--tensorrt] [model_name(s)]"
    echo "Options:"
    echo "  -a, --all           Generate all models (ONNX models and TensorRT engines)"
    echo "  -o, --onnx          Generate only ONNX models"
    echo "  -t, --tensorrt      Generate only TensorRT engines"
    echo "Model Names:          l0, l1, l2, xl0, xl1"
    exit 1
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
fi

all_models=false
onnx_only=false
tensorrt_only=false

# determine whether to generate only ONNX models, only TensorRT engines, or both
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -a|--all )
            all_models=true
            shift
            ;;
        -o|--onnx )
            onnx_only=true
            shift
            ;;
        -t|--tensorrt )
            tensorrt_only=true
            shift
            ;;
        * )
            break
            ;;
    esac
done

# confirm at least one mode is specifed
if ! $all_models && ! $onnx_only && ! $tensorrt_only; then
    echo "Error: You must specify one of -a, -o, or -t options."
    usage
fi

# confirm mode compatibility
if [ "$all_models" = true ] && ( [ "$onnx_only" = true ] || [ "$tensorrt_only" = true ] ); then
    echo "Error: Options -a, -o, and -t cannot be used together."
    usage
fi

# Check if at least one model type is provided
if [ "$#" -eq 0 ]; then
    echo "Error: At least one model type (l0, l1, l2, xl0, or xl1) must be specified."
    usage
fi

# confirm model name validity
valid_models=("l0" "l1" "l2" "xl0" "xl1")
for model_name in "$@"; do
    if [[ ! " ${valid_models[@]} " =~ " ${model_name} " ]]; then
        echo "Error: Invalid model name '${model_name}'. Please choose from: ${valid_models[*]}"
        usage
    fi
done

generate_onnx_models() {
    local model_name=$1

    echo "\nCreating ${model_name} ONNX encoder"
    python deployment/sam/onnx/export_encoder.py \
        --model $model_name \
        --weight_url assets/checkpoints/sam/${model_name}.pt \
        --output assets/export_models/sam/onnx/${model_name}_encoder.onnx 

    echo "\nCreating ${model_name} ONNX decoder"
    python deployment/sam/onnx/export_decoder.py \
        --model $model_name \
        --weight_url assets/checkpoints/sam/${model_name}.pt \
        --output assets/export_models/sam/onnx/${model_name}_decoder.onnx \
        --return-single-mask
}

# l-series models use 512x512 image while xl-series models use 1024x1024 image
get_side_len() {
    local model_name="$1"
    case $model_name in
        l0|l1|l2)
            echo "512"
            ;;
        xl0|xl1)
            echo "1024"
            ;;
    esac
}

generate_tensorrt_engines() {
	local model_name=$1
	local side_len=$(get_side_len "$model_name")

	echo "\nCreating ${model_name} TensorRT encoder with side length ${side_len}"
	trtexec --onnx=assets/export_models/sam/onnx/${model_name}_encoder.onnx \
		--minShapes=input_image:1x3x${side_len}x${side_len} \
		--optShapes=input_image:1x3x${side_len}x${side_len} \
		--maxShapes=input_image:4x3x${side_len}x${side_len} \
		--saveEngine=assets/export_models/sam/tensorrt/${model_name}_encoder.engine

	echo "\nCreating ${model_name} TensorRT point decoder"
	trtexec --onnx=assets/export_models/sam/onnx/${model_name}_decoder.onnx \
		--minShapes=point_coords:1x1x2,point_labels:1x1 \
		--optShapes=point_coords:1x16x2,point_labels:1x16 \
		--maxShapes=point_coords:1x16x2,point_labels:1x16 \
		--fp16 \
		--saveEngine=assets/export_models/sam/tensorrt/${model_name}_point_decoder.engine

	echo "\nCreating ${model_name} TensorRT box decoder"
	trtexec --onnx=assets/export_models/sam/onnx/${model_name}_decoder.onnx \
		--minShapes=point_coords:1x1x2,point_labels:1x1 \
		--optShapes=point_coords:16x2x2,point_labels:16x2 \
		--maxShapes=point_coords:16x2x2,point_labels:16x2 \
		--fp16 \
		--saveEngine=assets/export_models/sam/tensorrt/${model_name}_box_decoder.engine

	echo "\nCreating ${model_name} TensorRT full image segmentation decoder"
	trtexec --onnx=assets/export_models/sam/onnx/${model_name}_decoder.onnx \
		--minShapes=point_coords:1x1x2,point_labels:1x1 \
		--optShapes=point_coords:64x1x2,point_labels:64x1 \
		--maxShapes=point_coords:128x1x2,point_labels:128x1 \
		--fp16 \
		--saveEngine=assets/export_models/sam/tensorrt/${model_name}_full_img_decoder.engine
}

if [ "$all_models" = true ] || [ "$onnx_only" = true ]; then
    mkdir -p assets/export_models/sam/onnx

    for model_name in "$@"; do
        generate_onnx_models "$model_name"
    done
fi

if [ "$all_models" = true ] || [ "$tensorrt_only" = true ]; then
    mkdir -p assets/export_models/sam/tensorrt

    for model_name in "$@"; do
        generate_tensorrt_engines "$model_name"
    done
fi