
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image

from transformers import AutoConfig
from qwen_vl_utils import process_vision_info

# 1. 准备数据
model_path = "models/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_path, min_pixels=256*28*28, max_pixels=256*28*28)
image = Image.new('RGB', (224, 224), color='red')



messages=[{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Describe this."}]}]


# 应用chat template
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
# print('text:')
# print(text)

# 处理输入
image_inputs, _ = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    return_tensors="pt",
    padding=True
)

inputs["labels"] = inputs["input_ids"].clone() # 简单复制，全量计算 Loss

# 2. 跑原生模型
print("Running Original...")
orig_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float32, device_map="cpu")
orig_model.to(torch.bfloat16)
with torch.no_grad():
    orig_out = orig_model(**inputs)
    print(f"Original Loss: {orig_out.loss.item()}")

# 3. 跑您的模型 (模拟 enable_3d=False)
print("Running Custom...")
# 确保这里用您的类，并且 args 设为 enable_3d=False
from qwen_3d.model.model import Qwen2_5_VLWith3D 
class Args: enable_3d = False; merge_type='concat'; norm_type='layernorm'; grid_n=2; type_3d='nf'; num_3d_freqs=10

args=Args
# 2. 实例化修改后的模型
# 1. 加载配置
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
print(f"✓ 配置加载完成: hidden_size={config.vision_config.hidden_size}")

# 2. 实例化修改后的模型
model = Qwen2_5_VLWith3D(config, args=args)
model.to(torch.bfloat16)

# 3. 加载预训练权重
print(f"正在加载预训练权重...")
base_model = orig_model

base_state_dict = base_model.state_dict()
if args.enable_3d:
    # 【核心修改】获取权重字典并进行 Key 重命名
    
    new_state_dict = {}
    
    for k, v in base_state_dict.items():
        # 检测是否是 merger 层的参数
        if k.startswith("visual.merger."):
            # 将 key 从 "visual.merger.xxx" 替换为 "visual.merger.original_merger.xxx"
            new_k = k.replace("visual.merger.", "visual.merger.original_merger.")
            new_state_dict[new_k] = v
        else:
            # 其他参数保持不变
            new_state_dict[k] = v
            
    # 使用修改后的字典加载
else:
    new_state_dict=base_state_dict
ret = model.load_state_dict(new_state_dict, strict=False)

print(f"✓ 预训练权重加载完成")
print(ret)
print("Missing keys:", ret.missing_keys)
print("Unexpected keys:", ret.unexpected_keys)
# 模拟权重加载后的替换（如果您是在 init 外面替换的）
# my_model.enable_3d_encoding(Args()) 
my_model = model

with torch.no_grad():
    my_out = my_model(**inputs)
    print(f"Wrapper Loss:  {my_out.loss.item()}")

# 4. 核心对比
diff = (orig_out.logits - my_out.logits).abs().max().item()
print(f"Logits Max Diff: {diff}")

if diff > 1e-3:
    print("❌ 模型计算不一致！问题出在 MergerWrapper 或 Forward 参数传递。")
else:
    print("✅ 模型计算一致。问题只能出在训练时的 Loss 计算（Label Masking）或 DataCollator。")

# 4. 判断
print(f"Loss Diff: {abs(orig_out.loss.item() - my_out.loss.item())}")