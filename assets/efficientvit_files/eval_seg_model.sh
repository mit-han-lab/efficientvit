# Cityscapes: mIoU = 75.5
python eval_seg_model.py --dataset cityscapes --crop_size 960 --model b0-r960

# Cityscapes: mIoU = 80.1
python eval_seg_model.py --dataset cityscapes --crop_size 896 --model b1-r896

# Cityscapes: mIoU = 82.1
python eval_seg_model.py --dataset cityscapes --crop_size 1024 --model b2-r1024

# Cityscapes: mIoU = 83.2
python eval_seg_model.py --dataset cityscapes --crop_size 1184 --model b3-r1184

# ADE20K: mIoU = 42.7
python eval_seg_model.py --dataset ade20k --path /dataset/ade20k-full/images/validation --crop_size 480 --model b1-r480

# ADE20K: mIoU = 45.1
python eval_seg_model.py --dataset ade20k --path /dataset/ade20k-full/images/validation --crop_size 416 --model b2-r416

# ADE20K: mIoU = 49.0
python eval_seg_model.py --dataset ade20k --path /dataset/ade20k-full/images/validation --crop_size 512 --model b3-r512
