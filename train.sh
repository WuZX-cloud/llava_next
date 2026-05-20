#!/bin/bash


bash qwen_3d/scripts/train_3d.sh  \
    -t "gradient_1_gpu_4-v4-no_3d-m53"  \
    -n no_norm  \
    -m no_3d  \
    -w True  \
    -d nf \
    -f 4 \
    -e False \
    -a False \
    -c 0,1,2,3 \
    -p 29505 \
    -s 0

# -a eval--是否eval模式   -e enable_3d 是否开启3d
