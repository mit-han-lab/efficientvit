# Cityscapes: mIoU = 75.653
python applications/efficientvit_seg/eval_efficientvit_seg_model.py --dataset cityscapes --model efficientvit-seg-b0-cityscapes

# Cityscapes: mIoU = 80.547
python applications/efficientvit_seg/eval_efficientvit_seg_model.py --dataset cityscapes --model efficientvit-seg-b1-cityscapes

# Cityscapes: mIoU = 82.073
python applications/efficientvit_seg/eval_efficientvit_seg_model.py --dataset cityscapes --model efficientvit-seg-b2-cityscapes

# Cityscapes: mIoU = 83.016
python applications/efficientvit_seg/eval_efficientvit_seg_model.py --dataset cityscapes --model efficientvit-seg-b3-cityscapes

# Cityscapes: mIoU = 82.716
python applications/efficientvit_seg/eval_efficientvit_seg_model.py --dataset cityscapes --model efficientvit-seg-l1-cityscapes

# Cityscapes: mIoU = 83.228
python applications/efficientvit_seg/eval_efficientvit_seg_model.py --dataset cityscapes --model efficientvit-seg-l2-cityscapes

# ADE20K: mIoU = 42.840
python applications/efficientvit_seg/eval_efficientvit_seg_model.py --dataset ade20k --crop_size 512 --model efficientvit-seg-b1-ade20k --path ~/dataset/ade20k/images/validation

# ADE20K: mIoU = 45.941
python applications/efficientvit_seg/eval_efficientvit_seg_model.py --dataset ade20k --crop_size 512 --model efficientvit-seg-b2-ade20k --path ~/dataset/ade20k/images/validation

# ADE20K: mIoU = 49.013
python applications/efficientvit_seg/eval_efficientvit_seg_model.py --dataset ade20k --crop_size 512 --model efficientvit-seg-b3-ade20k --path ~/dataset/ade20k/images/validation

# ADE20K: mIoU = 49.191
python applications/efficientvit_seg/eval_efficientvit_seg_model.py --dataset ade20k --crop_size 512 --model efficientvit-seg-l1-ade20k --path ~/dataset/ade20k/images/validation

# ADE20K: mIoU = 50.702
python applications/efficientvit_seg/eval_efficientvit_seg_model.py --dataset ade20k --crop_size 512 --model efficientvit-seg-l2-ade20k --path ~/dataset/ade20k/images/validation
