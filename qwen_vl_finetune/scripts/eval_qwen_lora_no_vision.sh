


lora_type=merger_save_all_lora-new_label-finetune_no_vision-gradient_1_gpu_4

shuffle_uav_id=False

while getopts "l:" opt; do
  case $opt in
    l) lora_type="$OPTARG" ;;
    *)
      echo "用法:"
      echo "bash train_qwen_vl.sh -l lora_type"
      exit 1
      ;;
  esac
done
# merger_save_all_lora-new_label-finetune_no_uavid-gradient_1_gpu_4
# test_data=./processed_data_with_depth_npy/test/test_data_multi_view_follow_no_uavid.json
test_data=./processed_data_with_depth_npy/test/test_data_multi_view_follow_no_vision.json
video_mapping=./processed_data_with_depth_npy/total_info_processed.json
# system_prompt="You are an intelligent multi-UAV perception and reasoning assistant.Each UAV provides an image from its own viewpoint.UAVs are identified as UAV1, UAV2, UAV3, etc. Your task is to analyze the images, reason across UAV viewpoints,and answer questions about the environment, objects, and their relationships. Always ground your answers in the provided UAV images and explicitly mention UAV IDs when relevant."
system_prompt=None

output_dir=./output/${lora_type}/test_results-no-sample2
output_name=${lora_type}

model_path=./output/${lora_type}
model_base=models/Qwen2.5-VL-7B-Instruct


CUDA_VISIBLE_DEVICES=4 python -m qwen_vl_finetune.eval.eval_multi \
            --test_data ${test_data} \
            --video_mapping ${video_mapping} \
            --system_prompt "${system_prompt}" \
            --output_dir ${output_dir} \
            --output_name ${output_name} \
            --model_path ${model_path} \
            --model_base ${model_base}  \
            --shuffle_uav_id ${shuffle_uav_id}


python qwen_vl_finetune/eval/acc_get.py --json_file "${output_dir}/${output_name}.json"