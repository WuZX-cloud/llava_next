

# test_data=./processed_ACB/test/test_data_follow.json
# video_mapping=./processed_ACB/total_info_processed.json
# # system_prompt="You are an intelligent multi-UAV perception and reasoning assistant.Each UAV provides an image from its own viewpoint.UAVs are identified as UAV1, UAV2, UAV3, etc. Your task is to analyze the images, reason across UAV viewpoints,and answer questions about the environment, objects, and their relationships. Always ground your answers in the provided UAV images and explicitly mention UAV IDs when relevant."
# system_prompt=None
# output_dir=./output/3/test_results
# output_name=test

# model_path=./output_qwen_3d
# model_base=models/Qwen2.5-VL-7B-Instruct


# CUDA_VISIBLE_DEVICES=7 python -m qwen_3d.eval.batch_eval \
#             --test_data ${test_data} \
#             --video_mapping ${video_mapping} \
#             --system_prompt "${system_prompt}" \
#             --output_dir ${output_dir} \
#             --output_name ${output_name} \
#             --model_path ${model_path} \
#             --model_base ${model_base}

export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=7 python qwen_3d/eval/qwen_eval.py \
    --mode batch \
    --model_path ./output_qwen_3d \
    --base_model_path models/Qwen2.5-VL-7B-Instruct \
    --test_json processed_ACB/test/test_data_follow.json \
    --video_mapping qwen_3d/processed_data/processed_mapping.json \
    --output output_qwen_3d/test_results_with_3d.json \
    --enable_3d  \
    --limit 2  # 先测试10个样本
