

# test_data=./processed_ACB/test/test_data_follow.json
# video_mapping=./processed_ACB/total_info_processed.json
# # system_prompt="You are an intelligent multi-UAV perception and reasoning assistant.Each UAV provides an image from its own viewpoint.UAVs are identified as UAV1, UAV2, UAV3, etc. Your task is to analyze the images, reason across UAV viewpoints,and answer questions about the environment, objects, and their relationships. Always ground your answers in the provided UAV images and explicitly mention UAV IDs when relevant."
# system_prompt=None
# output_dir=./output/3/test_results
# output_name=test

# model_path=./output_qwen_3d
# model_base=models/Qwen2.5-VL-7B-Instruct

gpu=0
# lora_type=merger_save_all_lora-gradient_1_gpu_4-v4-3.30-cot-text_cross_attention
lora_type=merger_save_all_lora-gradient_1_gpu_4-v4-4.7-cot-8epoch-text_cross_attention
norm_type=none
grid_n=7
merge_type=text_cross_attention
is_w2c=True
type_3d=nf # or nf
# test_json=processed_data_latest/test/test_data_multi_view_follow.json
# test_json=processed_data_latest/test/sim5_sim6_merged_test_data_multi_view_follow.json
test_json=processed_data_with_depth_npy/test/test_data_multi_view_cot2.json
# video_mapping=processed_data_latest/processed_mapping.json
video_mapping=processed_data_with_depth_npy/processed_mapping.json
num_3d_freqs=4

enable_3d=True
record=False
output_d=""

# bash qwen_3d/scripts/eval_3d.sh -l merger_save_all_lora-new_label-after-w2c-nf-no_norm-concat_text -n no_norm -m concat_text -w True -d nf

# bash qwen_3d/scripts/eval_3d.sh -l merger_save_all_lora-new_label_after_sim5-6_no_norm_w2c_nf_concat_text -n no_norm -m concat_text -w True -d nf


while getopts "t:l:n:m:w:d:f:e:g:r:o:" opt; do
  case $opt in
    t) test_json="$OPTARG" ;;
    l) lora_type="$OPTARG" ;;
    n) norm_type="$OPTARG" ;;
    m) merge_type="$OPTARG" ;;
    w) is_w2c="$OPTARG" ;;
    d) type_3d="$OPTARG" ;;
    f) num_3d_freqs="$OPTARG" ;;
    e) enable_3d="$OPTARG" ;;
    g) gpu="$OPTARG" ;;
    r) record="$OPTARG" ;;
    o) output_d="$OPTARG" ;;
    *)
      echo "用法:"
      echo "bash train_qwen_vl.sh -l lora_type -n norm_type  -m merge_type -w is_w2c -d type_3d -f num_3d_freqs -e enable_3d"
      exit 1
      ;;
  esac
done



model_path=output_3d/${lora_type}
# output_dir=output_3d/${lora_type}/test_results-multi-view-no-sample
output_dir=output_3d/${lora_type}/test_results-right_param-new_T-top_p-sampled-cot2-${output_d}
# 获取当前时间戳
timestamp=$(date +"%Y%m%d_%H%M%S")
output_name=${lora_type}-cot2-new_T-top_p-test-sampled-test_result-${timestamp}



export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=${gpu} python qwen_3d/eval/eval_3d.py \
    --mode batch \
    --model_path ${model_path} \
    --base_model_path models/Qwen2.5-VL-7B-Instruct \
    --test_json  ${test_json} \
    --video_mapping ${video_mapping} \
    --output_dir ${output_dir} \
    --output_name ${output_name} \
    --enable_3d  ${enable_3d}\
    --lora_enable \
    --norm_type ${norm_type} \
    --grid_n ${grid_n} \
    --system_prompt None \
    --merge_type ${merge_type} \
    --is_w2c ${is_w2c}  \
    --type_3d ${type_3d} \
    --num_3d_freqs ${num_3d_freqs}  \
    --record ${record}
    # --limit 1  # 先测试10个样本


# python qwen_vl_finetune/eval/acc_get.py --json_file "${output_dir}/${output_name}.json"
