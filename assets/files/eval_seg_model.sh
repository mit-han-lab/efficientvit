# Cityscapes: mIoU = 75.653
python eval_seg_model.py --dataset cityscapes --model b0

# Cityscapes: mIoU = 80.547
python eval_seg_model.py --dataset cityscapes --model b1

# Cityscapes: mIoU = 82.073
python eval_seg_model.py --dataset cityscapes --model b2

# Cityscapes: mIoU = 83.016
python eval_seg_model.py --dataset cityscapes --model b3

# Cityscapes: mIoU = 82.716
python eval_seg_model.py --dataset cityscapes --model l1

# Cityscapes: mIoU = 83.228
python eval_seg_model.py --dataset cityscapes --model l2

# ADE20K: mIoU = 42.840
python eval_seg_model.py --dataset ade20k --crop_size 512 --model b1 --path /dataset/ade20k/images/validation

# ADE20K: mIoU = 45.941
python eval_seg_model.py --dataset ade20k --crop_size 512 --model b2 --path /dataset/ade20k/images/validation

# ADE20K: mIoU = 49.013
python eval_seg_model.py --dataset ade20k --crop_size 512 --model b3 --path /dataset/ade20k/images/validation

# ADE20K: mIoU = 49.191
python eval_seg_model.py --dataset ade20k --crop_size 512 --model l1 --path /dataset/ade20k/images/validation

# ADE20K: mIoU = 50.702
python eval_seg_model.py --dataset ade20k --crop_size 512 --model l2 --path /dataset/ade20k/images/validation
