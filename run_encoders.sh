#!/usr/bin/env bash

endocers=(
    "mit_b2"
    "mit_b3"
    "mit_b4"
    "timm-efficientnet-b4"
    "timm-efficientnet-b3"
    "efficientnet-b5"
    "efficientnet-b4"
    "se_resnext101_32x4d"
    "se_resnext50_32x4d"
    "xception"    
)

for encoder in "${endocers[@]}";
do
    python train_unet.py --encoder $encoder
    # python train_segformer.py --encoder $encoder
done