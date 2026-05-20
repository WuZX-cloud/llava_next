# extract_and_fix_lora_adapter.py
import os
from collections import defaultdict
import torch
from safetensors import safe_open
from safetensors.torch import save_file

input_path  = "output_3d/merger_save_all_lora-gradient_1_gpu_4-v4-3.30-cot-text_cross_attention/adapter_model.safetensors"
output_path = "output_3d/merger_save_all_lora-gradient_1_gpu_4-v4-3.30-cot-text_cross_attention/adapter_model_clean.safetensors"

# 1. 读取所有 key
with safe_open(input_path, framework="pt") as f:
    all_keys = list(f.keys())

# 2. 过滤 LoRA 相关 key
def is_lora_key(k):
    return any(x in k for x in ["lora_A", "lora_B", "lora_embedding", "modules_to_save"])

lora_keys = [k for k in all_keys if is_lora_key(k)]

# 3. 统计
key_types = defaultdict(int)
for k in lora_keys:
    if   "lora_A"          in k: key_types["lora_A"] += 1
    elif "lora_B"          in k: key_types["lora_B"] += 1
    elif "modules_to_save" in k: key_types["modules_to_save"] += 1
    else:                         key_types["other"] += 1

print(f"Total keys : {len(all_keys)}")
print(f"LoRA  keys : {len(lora_keys)}")
for t, cnt in key_types.items():
    print(f"  {t}: {cnt}")

assert key_types["lora_A"] == key_types["lora_B"], \
    f"lora_A({key_types['lora_A']}) != lora_B({key_types['lora_B']}), 权重不完整!"

# 4. 提取 + 修复 key 名称
def fix_key(k):
    k = k.replace("lora_A.default.weight", "lora_A.weight")
    k = k.replace("lora_B.default.weight", "lora_B.weight")
    k = k.replace(".modules_to_save.default.", ".")
    return k

print("\nExtracting & fixing keys...")
dtype_counter = defaultdict(int)
state_dict = {}
with safe_open(input_path, framework="pt") as f:
    for k in lora_keys:
        tensor = f.get_tensor(k)
        dtype_counter[str(tensor.dtype)] += 1      # 统计原始 dtype
        tensor = tensor.to(torch.bfloat16)          # 转 bf16
        state_dict[fix_key(k)] = tensor

print("原始 dtype 分布:")
for dtype, cnt in dtype_counter.items():
    print(f"  {dtype}: {cnt}")

# 5. 保存
save_file(state_dict, output_path)

size_mb = os.path.getsize(output_path) / 1024 / 1024
print(f"\nSaved → {output_path}")
print(f"Size  : {size_mb:.1f} MB")

# 6. 打印样例确认格式和 dtype
print("\nKey 样例 (前5):")
for k, v in list(state_dict.items())[:5]:
    print(f"  {k}  {v.dtype}  {v.shape}")