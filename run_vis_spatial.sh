#!/bin/bash
# 可视化空间推理 vs 非空间任务的 3D 位置编码热力图对比
# 使用 text_cross_attention 模型

export PYTHONPATH=$(pwd)

# 选择一个训练好的 text_cross_attention 模型
MODEL_PATH=output_3d/merger_save_all_lora-gradient_1_gpu_4-v4-text_cross_attention-question_embedding-0-method5-3

GPU=${1:-7}
TEST_JSON=${2:-processed_data_with_depth_npy/test/test_data_multi_view_follow.json}
OUTPUT_DIR=${3:-vis_spatial_comparison_all}

CUDA_VISIBLE_DEVICES=${GPU} python vis_spatial_heatmap.py \
    --model_path ${MODEL_PATH} \
    --base_model_path models/Qwen2.5-VL-7B-Instruct \
    --test_json ${TEST_JSON} \
    --video_mapping processed_data_with_depth_npy/processed_mapping.json \
    --output_dir ${OUTPUT_DIR} \
    --num_spatial 15 \
    --num_non_spatial 5 \
    --vis_method mean \
    --enable_3d True \
    --merge_type text_cross_attention \
    --is_w2c False \
    --type_3d sincos \
    --norm_type no_norm \
    --grid_n 7 \
    --num_3d_freqs 4 \
    --gate_mode ${4:-softplus_mean} \
    --record True
