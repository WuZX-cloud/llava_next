#!/bin/bash


#最后一位 v6：增加了vis_proj, num_heads改为从4改为8

bash qwen_3d/scripts/train_3d.sh  \
    -t "gradient_1_gpu_4-v4-fusion_add-m53"  \
    -n no_norm  \
    -m fusion_add  \
    -w True  \
    -d nf \
    -f 4 \
    -e True \
    -a False \
    -c 4,5,6,7 \
    -p 29504 \
    -s 0

# -a eval--是否eval模式   -e enable_3d 是否开启3d
