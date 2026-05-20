#!/bin/bash

TARGET_DIR="output_3d/merger_save_all_lora-new_label-after-multi_points-concat_text"
TARGET_FILE="adapter_model.safetensors"

while true; do
    if [ -f "${TARGET_DIR}/${TARGET_FILE}" ]; then
        sleep 120
        echo "$(date): Found ${TARGET_FILE}, executing script..."

        # ===== 在这里写你要执行的 bash 命令 =====
        # 例如：
        # bash run.sh
        # 或：
        # sh your_script.sh
        # 或直接写命令：
        # python train.py
        bash train_bash/qwen_3d/train_3d_bash2.sh

        break  # 如果执行一次后就退出循环，保留；否则删掉这一行
    else
        echo "$(date): ${TARGET_FILE} not found, sleep 120s..."
        sleep 120
    fi
done


# bash  train_bash/qwen_3d/train_3d_bash.sh
