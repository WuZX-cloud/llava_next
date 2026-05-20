# convert_to_safetensors.py
import sys
import json
import os

import torch
from safetensors.torch import save_file

from collections import defaultdict
from safetensors import safe_open



if len(sys.argv) != 3:
    print("Usage: python convert_to_safetensors.py <input_dir_or_bin> <output.safetensors>")
    sys.exit(1)

input_path = sys.argv[1]
output_safetensors = sys.argv[2]

if os.path.isdir(input_path):
    # 分片情况：读取 index.json 合并所有分片
    index_file = os.path.join(input_path, "pytorch_model.bin.index.json")
    print(f"Loading sharded model from {input_path} ...")
    
    with open(index_file, "r") as f:
        index = json.load(f)
    
    shard_files = sorted(set(index["weight_map"].values()))
    state_dict = {}
    for shard in shard_files:
        shard_path = os.path.join(input_path, shard)
        print(f"  Loading {shard} ...")
        shard_dict = torch.load(shard_path, map_location="cpu", weights_only=True)
        state_dict.update(shard_dict)
else:
    # 单文件情况
    print(f"Loading {input_path} ...")
    state_dict = torch.load(input_path, map_location="cpu", weights_only=True)

print(f"Saving to {output_safetensors} ...")
save_file(state_dict, output_safetensors)

input_path = output_safetensors
output_path = output_safetensors

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


print("Done!")