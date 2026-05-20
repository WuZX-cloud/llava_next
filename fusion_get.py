from safetensors.torch import load_file

# 1. 指定你的 safetensors 文件路径
file_path = "output_3d/merger_save_all_lora-gradient_1_gpu_4-v4-2-no_norm-fusion_add-num_3d_freqs-4/adapter_model.safetensors"

# 2. 加载权重文件到一个字典中
tensors = load_file(file_path)

# 3. (可选) 查看所有的参数键名 (Keys)
# 这步很重要，可以帮你确认 self.fusion 在保存时的确切名称
print("文件中的所有参数名:")
for key in tensors.keys():
    # 过滤出包含 'fusion' 的键名，方便查找
    if "fusion" in key:
        print(f"找到相关的 Key: {key}")

print("-" * 50)

# 4. 读取并打印特定参数的值
# 请将下面的 "exact_fusion_key_name" 替换为你在上面步骤 3 中找到的实际键名
# 比如可能是 "base_model.model.fusion" 或者 "lora_fusion" 等
target_key = "base_model.model.visual.merger.position_3d_encoder.fusion" 

if target_key in tensors:
    fusion_tensor = tensors[target_key]
    print(f"参数 '{target_key}' 的形状 (Shape): {fusion_tensor.shape}")
    print(f"参数 '{target_key}' 的值:\n{fusion_tensor}")
else:
    print(f"警告: 在文件中没有找到名为 '{target_key}' 的参数。请检查上面的 Key 列表。")