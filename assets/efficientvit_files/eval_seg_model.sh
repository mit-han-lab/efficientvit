# Cityscapes: mIoU = 75.7
python eval_seg_model.py --dataset cityscapes --model b0

# Cityscapes: mIoU = 80.5
python eval_seg_model.py --dataset cityscapes --model b1

# Cityscapes: mIoU = 82.1
python eval_seg_model.py --dataset cityscapes --model b2

# Cityscapes: mIoU = 83.0
python eval_seg_model.py --dataset cityscapes --model b3

# Cityscapes: mIoU = 82.7
python eval_seg_model.py --dataset cityscapes --model l1

# Cityscapes: mIoU = 83.2
python eval_seg_model.py --dataset cityscapes --model l2

# ADE20K: mIoU = 42.8
python eval_seg_model.py --dataset ade20k --crop_size 512 --model b1 --path /dataset/ade20k/images/validation

# ADE20K: mIoU = 45.9
python eval_seg_model.py --dataset ade20k --crop_size 512 --model b2 --path /dataset/ade20k/images/validation

# ADE20K: mIoU = 49.0
python eval_seg_model.py --dataset ade20k --crop_size 512 --model b3 --path /dataset/ade20k/images/validation

# ADE20K: mIoU = 49.1
python eval_seg_model.py --dataset ade20k --crop_size 512 --model l1 --path /dataset/ade20k/images/validation

# ADE20K: mIoU = 50.7
python eval_seg_model.py --dataset ade20k --crop_size 512 --model l2 --path /dataset/ade20k/images/validation
