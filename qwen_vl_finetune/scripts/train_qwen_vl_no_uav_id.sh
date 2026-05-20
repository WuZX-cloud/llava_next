#!/bin/bash

# ==================== 单机多卡训练脚本(适配你的数据格式) ====================

# 训练脚本: train_qwen_vl.sh
# 使用方法: bash train_qwen_vl.sh

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# 设置GPU数量
# export CUDA_VISIBLE_DEVICES=0  # 使用4张卡

# NUM_GPUS=1
lora_type=merger_save_all_lora
eval=False


shuffle_uav_id=True  # 去掉uav id，打乱顺序，消除指代

while getopts "l:e:" opt; do
  case $opt in
    l) lora_type="$OPTARG" ;;
    e) eval="$OPTARG" ;;
    *)
      echo "用法:"
      echo "bash train_qwen_vl.sh -l lora_type"
      exit 1
      ;;
  esac
done



# 基础配置
MODEL_NAME="models/Qwen2.5-VL-7B-Instruct"
TRAIN_DATA="./processed_data_with_depth_npy/train/train_data_multi_view_follow_no_uavid.json"  # 你的训练数据
VIDEO_MAPPING="./processed_data_with_depth_npy/total_info_processed.json"  # video到图片路径的映射
OUTPUT_DIR="./output/${lora_type}-new_label-finetune_no_uavid2-gradient_1_gpu_4"

# 系统提示词(可自定义)
# SYSTEM_PROMPT="You are an intelligent multi-UAV perception and reasoning assistant.Each UAV provides an image from its own viewpoint.UAVs are identified as UAV1, UAV2, UAV3, etc. Your task is to analyze the images, reason across UAV viewpoints,and answer questions about the environment, objects, and their relationships. Always ground your answers in the provided UAV images and explicitly mention UAV IDs when relevant."
SYSTEM_PROMPT=None

# 训练超参数 
EPOCHS=3
PER_DEVICE_BATCH_SIZE=1  # 多图建议设为1
GRADIENT_ACCUMULATION=1   # 实际batch_size = 1 * 8 * 4 = 32
LEARNING_RATE=2e-4
MAX_LENGTH=4096
DEEPSPEED_STAGE=2  # 可选: 1, 2, 3

# LoRA参数
LORA_R=8
LORA_ALPHA=16

export PYTHONPATH=$(pwd)

# 使用DeepSpeed启动
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed qwen_vl_finetune/train/train_multimodal.py \
    --train_data $TRAIN_DATA \
    --video_mapping $VIDEO_MAPPING \
    --system_prompt "$SYSTEM_PROMPT" \
    --output_dir $OUTPUT_DIR \
    --model_name $MODEL_NAME \
    --epochs $EPOCHS \
    --batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation $GRADIENT_ACCUMULATION \
    --fixed_image_size 224 224 \
    --lr $LEARNING_RATE \
    --deepspeed_stage $DEEPSPEED_STAGE \
    --lora_r $LORA_R \
    --max_length $MAX_LENGTH \
    --lora_type $lora_type \
    --eval ${eval}  \
    --shuffle_uav_id ${shuffle_uav_id}
    # --use_flash_attention


# # ==================== 多机多卡训练脚本 ====================

# # 如果使用多机训练,使用以下脚本
# # 在主节点运行: bash train_qwen_vl_multinode.sh

# <<'MULTINODE_SCRIPT'
# #!/bin/bash

# # 多机配置
# MASTER_ADDR="192.168.1.100"  # 主节点IP
# MASTER_PORT=29500
# NNODES=2  # 节点数量
# NODE_RANK=0  # 当前节点编号(主节点为0)
# NPROC_PER_NODE=4  # 每个节点的GPU数量

# # 训练配置
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
# TRAIN_DATA="./data/train_multimodal.json"
# OUTPUT_DIR="./output/qwen_vl_multinode"

# deepspeed --num_gpus=$NPROC_PER_NODE \
#     --num_nodes=$NNODES \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     train_multimodal.py \
#     --train_data $TRAIN_DATA \
#     --output_dir $OUTPUT_DIR \
#     --model_name $MODEL_NAME \
#     --epochs 3 \
#     --batch_size 1 \
#     --gradient_accumulation 8 \
#     --lr 2e-4 \
#     --deepspeed_stage 2
# MULTINODE_SCRIPT


# # ==================== 推理测试脚本 ====================

# # inference.sh
# <<'INFERENCE_SCRIPT'
# #!/bin/bash

# python <<'EOF'
# import sys
# sys.path.append('.')
# from train_multimodal import inference_multimodal

# # 多图推理示例
# result = inference_multimodal(
#     model_path="./output/qwen_vl_multimodal_lora",
#     images=[
#         "test_images/img1.jpg",
#         "test_images/img2.jpg",
#         "test_images/img3.jpg"
#     ],
#     user_messages=[
#         [
#             {"type": "image", "image_id": 0},
#             {"type": "text", "text": "描述第一张图的内容"}
#         ],
#         [
#             {"type": "text", "text": "第二张图"},
#             {"type": "image", "image_id": 1},
#             {"type": "text", "text": "和第一张有什么不同?"}
#         ],
#         [
#             {"type": "image", "image_id": 2},
#             {"type": "text", "text": "最后这张图片呢?"}
#         ]
#     ],
#     system_prompt="你是一个专业的图像分析助手,请提供详细准确的描述"
# )

# print("=" * 50)
# print("模型输出:")
# print("=" * 50)
# print(result)
# EOF
# INFERENCE_SCRIPT


# # ==================== 环境检查脚本 ====================

# # check_environment.sh
# <<'CHECK_ENV'
# #!/bin/bash

# echo "检查训练环境..."
# echo "===================="

# # 检查Python版本
# python --version

# # 检查CUDA
# nvcc --version
# nvidia-smi

# # 检查PyTorch
# python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# # 检查必要的包
# python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
# python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
# python -c "import peft; print(f'PEFT: {peft.__version__}')"

# echo "===================="
# echo "环境检查完成"
# CHECK_ENV


# # ==================== 监控脚本 ====================

# # monitor.sh - 实时监控GPU使用情况
# <<'MONITOR'
# #!/bin/bash

# watch -n 1 'nvidia-smi; echo ""; echo "进程信息:"; ps aux | grep python | grep train'
# MONITOR


# # ==================== 转换checkpoint为完整模型 ====================

# # merge_lora.sh - 将LoRA权重合并到基础模型
# <<'MERGE_LORA'
# #!/bin/bash

# python <<'EOF'
# from peft import PeftModel
# from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
# import torch

# # 加载基础模型和LoRA
# base_model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     trust_remote_code=True
# )

# lora_model = PeftModel.from_pretrained(
#     base_model,
#     "./output/qwen_vl_multimodal_lora"
# )

# # 合并权重
# merged_model = lora_model.merge_and_unload()

# # 保存完整模型
# output_path = "./output/qwen_vl_merged"
# merged_model.save_pretrained(output_path)

# # 保存processor
# processor = AutoProcessor.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     trust_remote_code=True
# )
# processor.save_pretrained(output_path)

# print(f"模型已合并并保存到: {output_path}")
# EOF
# MERGE_LORA