#!/bin/bash


bash qwen_3d/scripts/train_3d.sh  \
    -t "without_3d_gradient_1_gpu_4-v4-2"  \
    -n no_norm  \
    -m concat_text  \
    -w True  \
    -d nf \
    -f 4 \
    -e False \
    -a False \
    -c 4,5,6,7

# -a eval--是否eval模式
