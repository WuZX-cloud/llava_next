#!/bin/bash

# m5:text as q, vis as k,v  依据最大最小值强制归一化到0~1
# m5-2 ：原来是依据最大最小值强制归一化到0~1，现调整为经过线形层自己学习。  m5-2的gate倒是Gate Stats: Min=0.6562, Max=0.9883, Mean=0.6590感觉是好了些，但是聚焦的地方都比较地离散且不可解释
# m5-2-s ：把gate 的loss加上试试， 加loss效果不好
# m5-3： 去掉线性层了，同时把sum改为了mean，替换sigmoid为softplus
bash qwen_3d/scripts/train_3d.sh  \
    -t "gradient_1_gpu_4-v4-text_cross_attention-question_embedding-0-method5-3-get_param_data"  \
    -n no_norm  \
    -m text_cross_attention  \
    -w True  \
    -d nf \
    -f 4 \
    -e True \
    -a False \
    -c 4,5 \
    -p 29503 \
    -s 0

# -a eval--是否eval模式   -e enable_3d 是否开启3d
