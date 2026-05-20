
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"


lora_type=merger_save_all_lora
type_name=new_label-after-no-uavid
norm_type=no_norm       # no_norm batch_norm     multi_points
grid_n=7
merge_type=text_cross_attention   # direct_add vision_guide text_guide deep_text_guide concat_vision concat_text
is_w2c=True
type_3d=nf  # sincos or nf
enable_3d=True

train_data=./processed_data_with_depth_npy/train/train_data_multi_view_follow_no_uavid.json
# train_data=./processed_data_latest/train/train_data_multi_view_follow.json
video_mapping=./processed_data_with_depth_npy/processed_mapping.json
num_3d_freqs=4

cuda_devices=4,5,6,7
lambda_sparse=0
eval=False

port=29502

coords_type=none
sigma=0


while getopts "l:t:n:g:m:w:d:f:e:c:a:p:s:o:i:" opt; do
  case $opt in
    l) lora_type="$OPTARG" ;;
    t) type_name="$OPTARG" ;;
    n) norm_type="$OPTARG" ;;
    g) grid_n="$OPTARG" ;;
    m) merge_type="$OPTARG" ;;
    w) is_w2c="$OPTARG" ;;
    d) type_3d="$OPTARG" ;;
    f) num_3d_freqs="$OPTARG" ;;
    e) enable_3d="$OPTARG" ;;
    c) cuda_devices="$OPTARG" ;;
    a) eval="$OPTARG" ;;
    p) port="$OPTARG" ;;
    s) lambda_sparse="$OPTARG" ;;
    o) coords_type="$OPTARG" ;;
    i) sigma="$OPTARG" ;;
    *)
      echo "用法:"
      echo "bash train_3d.sh -l lora_type -t type_name -n norm_type -g grid_n -m merge_type -w is_w2c -d type_3d -f num_3d_freqs -e enable_3d"
      exit 1
      ;;
  esac
done

# bash qwen_3d/scripts/train_3d2.sh -t new_label_after_sim5-6_no_norm_w2c_nf -n no_norm -m concat_text -w True -d nf

SYSTEM_PROMPT=None
output_dir=output_3d/${lora_type}-${type_name}


export PYTHONPATH=$(pwd)

# CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port=29504 qwen_3d/train/train_3d.py  \
CUDA_VISIBLE_DEVICES="${cuda_devices}" deepspeed --master_port=${port} qwen_3d/train/train_3d.py  \
      --model_name models/Qwen2.5-VL-7B-Instruct \
      --train_data ${train_data} \
      --video_mapping ${video_mapping} \
      --vision_output_layer visual.merger \
      --system_prompt "$SYSTEM_PROMPT" \
      --output_dir ${output_dir} \
      --epochs 3 \
      --batch_size 1 \
      --gradient_accumulation 1 \
      --fixed_image_size 224 224 \
      --lr 2e-4 \
      --lora_r 8 \
      --deepspeed_stage 2 \
      --lora_enable \
      --lora_type ${lora_type}  \
      --norm_type ${norm_type} \
      --grid_n ${grid_n} \
      --merge_type ${merge_type} \
      --is_w2c ${is_w2c}  \
      --enable_3d ${enable_3d}  \
      --type_3d ${type_3d} \
      --num_3d_freqs ${num_3d_freqs}  \
      --eval ${eval}  \
      --lambda_sparse ${lambda_sparse} \
      --coords_type ${coords_type}  \
      --sigma ${sigma}


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


    