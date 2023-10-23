# Training Guides

## ImageNet

Please refer to `train_cls_model.py` for training models on imagenet.

Single-Node Training Examples:

```bash
torchpack dist-run -np 8 \
python train_cls_model.py configs/cls/imagenet/b1.yaml \
    --data_provider.image_size "[128,160,192,224,256,288]" \
    --run_config.eval_image_size "[288]" \
    --path .exp/cls/imagenet/b1_r288/
```

### EfficientViT L Series

```bash
torchpack dist-run -np 16 -H $server1:8,$server2:8 \
python train_cls_model.py configs/cls/imagenet/l1.yaml --fp16 \
    --path .exp/cls/imagenet/l1_r224/
```

```bash
torchpack dist-run -np 16 -H $server1:8,$server2:8 \
python train_cls_model.py configs/cls/imagenet/l2.yaml --fp16 \
    --path .exp/cls/imagenet/l2_r224/
```

```bash
torchpack dist-run -np 16 -H $server1:8,$server2:8 \
python train_cls_model.py configs/cls/imagenet/l3.yaml --fp16 \
    --path .exp/cls/imagenet/l3_r224/
```

### EfficientViT B Series

```bash
torchpack dist-run -np 16 -H $server1:8,$server2:8 \
python train_cls_model.py configs/cls/imagenet/b1.yaml \
    --path .exp/cls/imagenet/b1_r224/
```

```bash
torchpack dist-run -np 16 -H $server1:8,$server2:8 \
python train_cls_model.py configs/cls/imagenet/b1.yaml \
    --data_provider.image_size "[128,160,192,224,256,288]" \
    --run_config.eval_image_size "[288]" \
    --path .exp/cls/imagenet/b1_r288/
```

```bash
torchpack dist-run -np 16 -H $server1:8,$server2:8 \
python train_cls_model.py configs/cls/imagenet/b2.yaml \
    --path .exp/cls/imagenet/b2_r224/
```

```bash
torchpack dist-run -np 16 -H $server1:8,$server2:8 \
python train_cls_model.py configs/cls/imagenet/b2.yaml \
    --data_provider.image_size "[128,160,192,224,256,288]" \
    --run_config.eval_image_size "[288]" \
    --data_provider.data_aug "{n:1,m:5}" \
    --path .exp/cls/imagenet/b2_r288/
```

```bash
torchpack dist-run -np 16 -H $server1:8,$server2:8 \
python train_cls_model.py configs/cls/imagenet/b3.yaml \
    --path .exp/cls/imagenet/b3_r224/
```
