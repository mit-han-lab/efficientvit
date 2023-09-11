# Top1 Acc=79.4, Top5 Acc=94.3
python eval_cls_model.py --model b1-r224 --image_size 224

# Top1 Acc=79.9, Top5 Acc=94.7
python eval_cls_model.py --model b1-r256 --image_size 256

# Top1 Acc=80.4, Top5 Acc=95.0
python eval_cls_model.py --model b1-r288 --image_size 288

# Top1 Acc=82.1, Top5 Acc=95.8
python eval_cls_model.py --model b2-r224 --image_size 224 --crop_ratio 1.0

# Top1 Acc=82.7, Top5 Acc=96.1
python eval_cls_model.py --model b2-r256 --image_size 256 --crop_ratio 1.0

# Top1 Acc=83.1, Top5 Acc=96.3
python eval_cls_model.py --model b2-r288 --image_size 288 --crop_ratio 1.0

# Top1 Acc=83.5, Top5 Acc=96.4
python eval_cls_model.py --model b3-r224 --image_size 224 --crop_ratio 1.0

# Top1 Acc=83.8, Top5 Acc=96.5
python eval_cls_model.py --model b3-r256 --image_size 256 --crop_ratio 1.0

# Top1 Acc=84.2, Top5 Acc=96.7
python eval_cls_model.py --model b3-r288 --image_size 288 --crop_ratio 1.0

# Top1 Acc=84.5, Top5 Acc=96.9
python eval_cls_model.py --model l1-r224 --image_size 224 --crop_ratio 1.0

# Top1 Acc=85.0, Top5 Acc=97.1
python eval_cls_model.py --model l2-r224 --image_size 224 --crop_ratio 1.0

# Top1 Acc=85.4, Top5 Acc=97.2
python eval_cls_model.py --model l2-r256 --image_size 256 --crop_ratio 1.0

# Top1 Acc=85.6, Top5 Acc=97.4
python eval_cls_model.py --model l2-r288 --image_size 288 --crop_ratio 1.0

# Top1 Acc=85.8, Top5 Acc=97.4
python eval_cls_model.py --model l2-r320 --image_size 320 --crop_ratio 1.0

# Top1 Acc=85.9, Top5 Acc=97.5
python eval_cls_model.py --model l2-r352 --image_size 352 --crop_ratio 1.0

# Top1 Acc=86.0, Top5 Acc=97.5
python eval_cls_model.py --model l2-r384 --image_size 384 --crop_ratio 1.0
