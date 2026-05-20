import torch
from safetensors.torch import load_file

def compare_safetensors(file_path1, file_path2):
    print(f"正在加载文件 1: {file_path1}")
    state_dict1 = load_file(file_path1)
    
    print(f"正在加载文件 2: {file_path2}")
    state_dict2 = load_file(file_path2)

    # 1. 对比参数层名称 (Keys) 是否一致
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    if keys1 != keys2:
        print("\n❌ 参数层名称 (Keys) 不一致！")
        print(f"仅在文件 1 中存在的层: {len(keys1 - keys2)} {list(keys1 - keys2)[0:5]}")
        print(f"仅在文件 2 中存在的层: {len(keys2 - keys1)}  {list(keys2 - keys1)[0:5]}")
        # return False
    else:
        print("\n✅ 所有的参数层名称 (Keys) 完全匹配。")

    # 2. 逐层对比具体的参数数值 (Values)
    all_match = True
    mismatch_count = 0
    
    print("\n正在对比具体参数数值...")
    print(f"共同的key 数量： {len(keys1 & keys2)}")
    for key in keys1 & keys2:
        tensor1 = state_dict1[key]
        tensor2 = state_dict2[key]

        # print(f"层 '{key}' dtype 不一致: {tensor1.dtype} vs {tensor2.dtype}")

        # 检查形状是否一致
        if tensor1.shape != tensor2.shape:
            print(f"❌ 层 '{key}' 形状不一致: {tensor1.shape} vs {tensor2.shape}")
            all_match = False
            mismatch_count += 1
            continue

        # 检查 dtype 是否一致
        if tensor1.dtype != tensor2.dtype:
            print(f"❌ 层 '{key}' dtype 不一致: {tensor1.dtype} vs {tensor2.dtype}")
            all_match = False
            mismatch_count += 1
            continue

        

        # 使用 torch.equal 进行绝对相等的对比
        # if not torch.equal(tensor1, tensor2):
        #     print(f"❌ 层 '{key}' 数值不一致！")
        #     # 可选：计算最大误差值，有助于判断是否是因为精度损失导致的微小差异
        #     max_diff = torch.max(torch.abs(tensor1 - tensor2)).item()
        #     print(f"   最大绝对误差: {max_diff}")
        #     all_match = False
        #     mismatch_count += 1

    if all_match:
        print("\n🎉 结论：两个文件中的所有参数形状和数值完全一致！")
    else:
        print(f"\n⚠️ 结论：发现 {mismatch_count} 层参数存在差异。")

# # 替换为你实际的文件路径
file1 = "output_3d/merger_save_all_lora-gradient_1_gpu_4-v4-3.31-text_cross_attention-none-sigma-0/adapter_model.safetensors"
# file2 = "output_3d/merger_save_all_lora-gradient_1_gpu_4-v4-3.30-cot-text_cross_attention/adapter_model.safetensors"
file2="output_3d/merger_save_all_lora-gradient_1_gpu_4-v4-3.30-cot-text_cross_attention/adapter_model_clean.safetensors"

compare_safetensors(file1, file2)


# from safetensors import safe_open

# file2="output_3d/merger_save_all_lora-gradient_1_gpu_4-v4-3.30-cot-text_cross_attention/adapter_model_clean.safetensors"

# with safe_open(file1, framework="pt") as f:
#     keys = list(f.keys())
    
# lora_keys = [k for k in keys if "lora_A" in k or "lora_B" in k]
# merger_keys = [k for k in keys if "merger" in k]
# print(f"Total: {len(keys)}, LoRA: {len(lora_keys)}, Merger: {len(merger_keys)}")