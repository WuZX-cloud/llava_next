
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"


lora_type=merger_save_all_lora
type_name=new_label-after
norm_type=no_norm       # no_norm batch_norm     multi_points
grid_n=7
merge_type=text_guide   # direct_add vision_guide text_guide deep_text_guide concat_vision concat_text
is_w2c=False

while getopts "l:t:n:g:m:w:" opt; do
  case $opt in
    l) lora_type="$OPTARG" ;;
    t) type_name="$OPTARG" ;;
    n) norm_type="$OPTARG" ;;
    g) grid_n="$OPTARG" ;;
    m) merge_type="$OPTARG" ;;
    w) is_w2c="$OPTARG" ;;
    *)
      echo "用法:"
      echo "bash train_3d.sh -l lora_type -t type_name -n norm_type -g grid_n -m merge_type -w is_w2c"
      exit 1
      ;;
  esac
done

# bash qwen_3d/scripts/train_3d.sh -m new_label_test -e False -n norm -g 7

SYSTEM_PROMPT=None
output_dir=output_3d/${lora_type}-${type_name}-${norm_type}-${merge_type}-testtest


export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=6,7 deepspeed --master_port=29501 qwen_3d/train/train_3d.py  \
      --model_name models/Qwen2.5-VL-7B-Instruct \
      --train_data ./processed_ACB/train/train_data_multi_view_follow.json \
      --video_mapping ./qwen_3d/processed_data/processed_mapping.json \
      --vision_output_layer visual.merger \
      --system_prompt "$SYSTEM_PROMPT" \
      --output_dir ${output_dir} \
      --epochs 3 \
      --batch_size 1 \
      --gradient_accumulation 8 \
      --fixed_image_size 224 224 \
      --lr 2e-4 \
      --lora_r 8 \
      --deepspeed_stage 2 \
      --lora_enable \
      --lora_type ${lora_type}  \
      --enable_3d \
      --norm_type ${norm_type} \
      --grid_n ${grid_n} \
      --merge_type ${merge_type} \
      --is_w2c ${is_w2c}


# if [ "$enable_text_guide" == "True" ]; then
#     CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed --master_port=29502 qwen_3d/train/train_3d.py  \
#       --model_name models/Qwen2.5-VL-7B-Instruct \
#       --train_data ./processed_ACB/train/train_data_multi_view_follow.json \
#       --video_mapping ./qwen_3d/processed_data/processed_mapping.json \
#       --vision_output_layer visual.merger \
#       --system_prompt "$SYSTEM_PROMPT" \
#       --output_dir ${output_dir} \
#       --epochs 3 \
#       --batch_size 1 \
#       --gradient_accumulation 8 \
#       --fixed_image_size 224 224 \
#       --lr 2e-4 \
#       --lora_r 8 \
#       --deepspeed_stage 2 \
#       --lora_enable \
#       --lora_type ${lora_type}  \
#       --enable_3d \
#       --norm_type ${norm_type} \
#       --grid_n ${grid_n} \
#       --enable_text_guide 
# else
#     CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed --master_port=29502 qwen_3d/train/train_3d.py  \
#       --model_name models/Qwen2.5-VL-7B-Instruct \
#       --train_data ./processed_ACB/train/train_data_multi_view_follow.json \
#       --video_mapping ./qwen_3d/processed_data/processed_mapping.json \
#       --vision_output_layer visual.merger \
#       --system_prompt "$SYSTEM_PROMPT" \
#       --output_dir ${output_dir} \
#       --epochs 3 \
#       --batch_size 1 \
#       --gradient_accumulation 8 \
#       --fixed_image_size 224 224 \
#       --lr 2e-4 \
#       --lora_r 8 \
#       --deepspeed_stage 2 \
#       --lora_type ${lora_type}  \
#       --lora_enable \
#       --norm_type ${norm_type} \
#       --grid_n ${grid_n} \
#       --enable_3d 
# fi


    