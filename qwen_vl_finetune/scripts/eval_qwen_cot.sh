

test_data=./processed_data_with_depth_npy/test/test_data_multi_view_cot2.json
# test_data=./processed_data_with_depth_npy/test/test_data_multi_view_cot_scene_object.json
video_mapping=./processed_data_with_depth_npy/total_info_processed.json
# system_prompt="You are an intelligent multi-UAV perception and reasoning assistant.Each UAV provides an image from its own viewpoint.UAVs are identified as UAV1, UAV2, UAV3, etc. Your task is to analyze the images, reason across UAV viewpoints,and answer questions about the environment, objects, and their relationships. Always ground your answers in the provided UAV images and explicitly mention UAV IDs when relevant."
system_prompt=None
output_dir=./output/no-finetune-cot/test4       # output/merger_save_all_lora-finetune-gradient_1_gpu_4-cot/test
output_name=cot_test-new_T_top_p-sampled
model_path=models/Qwen2.5-VL-7B-Instruct  # output/merger_save_all_lora-finetune-gradient_1_gpu_4-cot


CUDA_VISIBLE_DEVICES=5 python -m qwen_vl_finetune.eval.eval_multi \
            --test_data ${test_data} \
            --video_mapping ${video_mapping} \
            --system_prompt "${system_prompt}" \
            --output_dir ${output_dir} \
            --output_name ${output_name} \
            --model_path ${model_path}

# sleep 5
# python qwen_vl_finetune/eval/acc_get.py --json_file "${output_dir}/${output_name}.json"