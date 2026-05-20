

from safetensors.torch import load_file
import os

pretrained_lora_path="output_3d/merger_save_all_lora-gradient_1_gpu_4-v4-3.30-cot-text_cross_attention"
# pretrained_lora_path="output_3d/merger_save_all_lora-gradient_1_gpu_4-v4-4.1-cot-text_cross_attention"

# pretrained_lora_path="output_3d/merger_save_all_lora-gradient_1_gpu_4-v4-3.31-text_cross_attention-none-sigma-0"


safetensors_path = os.path.join(pretrained_lora_path, "adapter_model.safetensors")

target_fragment = "position_3d_encoder.coord_projector.0.weight"
# target_fragment = "merger"
device = 'cpu'

# safetensors_path="output_3d/merger_save_all_lora-gradient_1_gpu_4-v4-4.1-cot-text_cross_attention/checkpoint-10185/adapter_model.safetensors"

if os.path.exists(safetensors_path):
    saved_state_dict = load_file(safetensors_path)
    print("load 成功")
    disk_tensor = None
    disk_key = ""
else:
    print('不存在')

for k in saved_state_dict.keys():
    print(k)
    if target_fragment in k:
        disk_key = k
        # 读取前5位，并转换到与内存张量相同的设备和精度
        disk_tensor = saved_state_dict[k].flatten()[:5].to(
            device=device, 
            dtype=device
        )
        break
        
if disk_tensor is not None:
    print(f"📦 [磁盘] 找到参数: {disk_key}")
    print(f"数值前 5 位: {disk_tensor.tolist()}")