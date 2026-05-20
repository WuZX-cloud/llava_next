"""
Qwen2.5-VL多图图文交叉微调 + DeepSpeed多卡训练
支持:
1. 多图输入
2. 图文交叉对话
3. DeepSpeed ZeRO优化
4. 系统提示词
"""

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from PIL import Image
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
import os
from qwen_vl_utils import process_vision_info

import random

import logging

logger = logging.getLogger(__name__)

# ==================== 多图图文交叉数据准备 ====================

def load_video_image_mapping(mapping_json_path: str) -> Dict:
    """
    加载video到image路径的映射,并从嵌套字典中提取图片路径
    
    原始映射文件格式:
    {
        "Real_2_UAVs/MDMT_when2col_UAV1_1": {
            "Real_2_UAVs/Samples/UAV1/23-00000001-UAV1.jpg": {...},
            "Real_2_UAVs/Samples/UAV2/23-00000001-UAV2.jpg": {...},
            "intrinsic": [...],
            ...
        }
    }
    
    返回格式:
    {
        "Real_2_UAVs/MDMT_when2col_UAV1_1": [
            "Real_2_UAVs/Samples/UAV1/23-00000001-UAV1.jpg",
            "Real_2_UAVs/Samples/UAV2/23-00000001-UAV2.jpg"
        ]
    }
    """
    with open(mapping_json_path, 'r', encoding='utf-8') as f:
        raw_mapping = json.load(f)
    
    # 处理后的映射
    processed_mapping = {}
    
    # 支持的图像格式
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    
    for video_id, video_data in raw_mapping.items():
        if not isinstance(video_data, dict):
            print(f"Warning: video_id '{video_id}' 的值不是字典,跳过")
            continue
        
        prefix = "./processed_ACB"
        # 提取图片路径
        image_paths = []
        for key, value in video_data.items():
            # 检查是否包含"UAV"且以图像格式结尾
            if "UAV" in key and key.lower().endswith(image_extensions):

                image_paths.append(f"{prefix}/{key}")
        
        # 排序确保顺序一致(按UAV编号排序)
        image_paths.sort()
        
        if image_paths:
            processed_mapping[video_id] = image_paths
        else:
            print(f"Warning: video_id '{video_id}' 中未找到图片路径")
    
    print(f"成功加载 {len(processed_mapping)} 个video的映射")
    
    # 打印示例
    if processed_mapping:
        first_video_id = list(processed_mapping.keys())[0]
        print(f"示例映射: {first_video_id}")
        print(f"  图片数量: {len(processed_mapping[first_video_id])}")
        for i, img_path in enumerate(processed_mapping[first_video_id][:3]):
            print(f"  [{i}] {img_path}")
    
    return processed_mapping


def prepare_multimodal_dataset(
    json_path: str, 
    video_mapping_path: str,
    system_prompt: str = "你是一个专业的多模态分析助手,能够理解和分析多个无人机视角的图像。"
):
    """
    准备支持你的数据格式的数据集
    
    你的数据格式:
    [
        {
            "video": "Real_2_UAVs/MDMT_OB_UAV2_001",
            "conversations": [
                {
                    "from": "human",
                    "value": "UAV1 image:\n<image>\nUAV2 image:\n<image>\nQuestion:..."
                },
                {
                    "from": "gpt",
                    "value": "The white van is..."
                }
            ]
        }
    ]
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 加载video到image路径的映射
    video_mapping = load_video_image_mapping(video_mapping_path)
    
    processed_data = []
    
    for item in data:
        video_id = item['video']
        conversations = item['conversations']
        
        # 获取该video对应的图片路径列表(已排序)
        if video_id not in video_mapping:
            print(f"Warning: video_id '{video_id}' not found in mapping, skipping...")
            continue
        
        image_paths = video_mapping[video_id]
        
        # 转换为标准格式
        processed_item = {
            "video_id": video_id,
            "images": image_paths,  # 图片路径列表
            "conversations": convert_conversations(conversations, system_prompt)
        }
        
        processed_data.append(processed_item)
    
    return Dataset.from_list(processed_data)


def convert_conversations(conversations: List[Dict], system_prompt: str) -> List[Dict]:
    """
    将你的对话格式转换为标准格式
    
    from "human" -> role "user"
    from "gpt" -> role "assistant"
    添加系统提示词
    """
    converted = []
    
    # 首先添加系统提示词
    if system_prompt:
        converted.append({
            "role": "system",
            "content": system_prompt
        })
    
    for conv in conversations:
        if conv['from'] == 'human':
            converted.append({
                "role": "user",
                "content": conv['value']  # 保持原始文本,包含<image>占位符
            })
        elif conv['from'] == 'gpt':
            converted.append({
                "role": "assistant",
                "content": conv['value']
            })
    
    return converted

from typing import Tuple
@dataclass
class MultimodalDataCollator:
    """支持<image>占位符格式的数据整理器 - 使用官方qwen_vl_utils"""
    processor: AutoProcessor
    max_length: int = 4096
    debug_truncation: bool = False   # 👈 新增
    fixed_image_size: Tuple[int, int] = (224, 224)
    shuffle_uav_id: bool = False
    
    def __call__(self, features: List[Dict]) -> Dict:
        batch_messages = []
        
        for item in features:
            # 加载所有图片
            images = []
            if 'images' in item and item['images']:
                for img_path in item['images']:
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize(self.fixed_image_size)
                        images.append(img)
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
                        # 创建空白图片作为占位
                        images.append(Image.new('RGB', self.fixed_image_size, color='white'))
            
            # 将对话转换为官方消息格式
            if self.shuffle_uav_id :
                random.shuffle(images)   # 去除uav id指代性，所以打乱顺序

            messages = self.convert_to_official_format(
                item['conversations'], 
                images
            )
            
            batch_messages.append(messages)

        # 使用官方方式处理
        texts = []
        all_image_inputs = []
        all_video_inputs = []
        
        for messages in batch_messages:
            # 使用apply_chat_template格式化文本
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
            
            # 使用process_vision_info处理图像
            image_inputs, video_inputs = process_vision_info(messages)
            if image_inputs:
                all_image_inputs.extend(image_inputs)
            if video_inputs:
                all_video_inputs.extend(video_inputs)
        
        # print(f"image_inpus shape is {len(all_image_inputs)}  , type is {type(all_image_inputs[0])}")
        # exit(0)

        # 使用processor处理
        inputs = self.processor(
            text=texts,
            images=all_image_inputs if all_image_inputs else None,
            videos=all_video_inputs if all_video_inputs else None,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        )
        # print(f'inputs keys is {inputs.keys()}')
        # for key in inputs.keys():
        #     print(f"{key} type is {type(inputs[key])} , shape is {inputs[key].shape}")
        # exit(0)

        
 # 准备labels
        # inputs["labels"] = inputs["input_ids"].clone()

        
        # 将padding部分的label设为-100(不计算loss)
        # inputs["labels"][inputs["labels"] == self.processor.tokenizer.pad_token_id] = -100
        # print('inputs:')
        # print(self.processor.tokenizer.decode(inputs["input_ids"][0]))
        # print('labels:')
        # print(self.processor.tokenizer.decode(inputs["labels"][0]))
        # exit(0)

        input_ids = inputs["input_ids"]
        labels = input_ids.clone()
        
        # 获取特殊 token id
        tokenizer = self.processor.tokenizer
        pad_id = tokenizer.pad_token_id
        image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        # 注意：不同版本的tokenizer可能对 \n 处理不同，这里假设 processor 已经处理好了 chat 模板

        # =====================================================
        # 核心逻辑：只保留 Assistant 的回答，Mask 掉其他所有内容
        # =====================================================
        
        # 先将所有 label 设为 -100 (默认不学习)
        ignore_index = -100
        new_labels = torch.full_like(labels, ignore_index)
        
        # 遍历 batch 中的每一条数据
        for i in range(input_ids.shape[0]):
            # 找到 input_ids 中的 token 序列
            ids = input_ids[i]
            
            # 我们需要找到 Assistant 回答的片段
            # Qwen 的格式通常是: ...<|im_start|>assistant\n答案<|im_end|>
            # 我们要让 new_labels 在 "答案<|im_end|>" 的位置等于 input_ids，其他位置维持 -100
            
            # 简单状态机：
            # 扫描整个序列，找到 <|im_start|> + assistant
            # 标记开始记录，直到 <|im_end|>
            
            # 将 tensor 转为 list 方便查找 (或者使用 vector search，这里为了逻辑清晰用循环)
            # 注意：实际生产中为了速度可以用 tensor 操作，但这里数据量不大，python循环可接受
            
            seq_len = len(ids)
            j = 0
            while j < seq_len:
                # 检查是否是 <|im_start|>
                if ids[j] == im_start_id:
                    # 检查下一个 token 是否是 assistant 对应的 id
                    # 注意：tokenizer 可能会把 "assistant" 编码成一个或多个 token
                    # 简单起见，我们解码这一小段来看看
                    # 或者更通用的方法：找到 <|im_start|> 后，看它后面是不是 assistant
                    
                    # 为了兼容性，我们用一个简化的逻辑：
                    # 1. 找到 assistant 的 token 序列 (这通常由 apply_chat_template 保证)
                    # 2. 这里我们假设 assistant 的角色标记后紧跟着就是回复
                    
                    # 更稳健的方法是利用 labels 原始值（我们知道 labels 原始是 input_ids）
                    # Qwen2.5-VL 的 processor 处理完后，文本里会有 "<|im_start|>assistant"
                    # 我们直接找 "<|im_start|>assistant" 的结束位置
                    pass
                j += 1
            
            # 【实战中最稳妥的方法】：
            # 不用复杂的 token 匹配，直接利用 apply_chat_template 的掩码逻辑太复杂
            # 我们用一个简单的启发式方法：
            
            # 1. 我们的目标：Mask 掉 <|im_start|>assistant 之前的所有内容
            # 2. Mask 掉所有的 Image Pad
            
            # 恢复 labels 为 input_ids
            current_label = labels[i].clone()
            
            # Mask Images (必须做)
            current_label[current_label == image_pad_id] = ignore_index
            current_label[current_label == tokenizer.convert_tokens_to_ids("<|vision_start|>")] = ignore_index
            current_label[current_label == tokenizer.convert_tokens_to_ids("<|vision_end|>")] = ignore_index
            
            # Mask Padding (必须做)
            current_label[current_label == pad_id] = ignore_index
            
            # Mask User Prompt (Instruction Tuning 关键)
            # 我们寻找 "<|im_start|>assistant" 的 token 序列
            # 然后把这之前的所有 token 设为 -100
            
            # 将 ids 转为 list 查找 assistant 的位置
            # 注意：这需要你知道 assistant 被 encode 成了什么。
            # 我们可以反向查找：找到最后一个 <|im_start|>，通常这就是 assistant 的开始
            # (前提是单轮对话，如果是多轮对话需要更复杂的逻辑)
            
            # 查找所有 <|im_start|> 的索引
            im_start_indices = (ids == im_start_id).nonzero(as_tuple=True)[0]
            
            if len(im_start_indices) > 0:
                # 假设最后一个 <|im_start|> 是 assistant 的开始 (单轮对话适用)
                last_start_idx = im_start_indices[-1]
                
                # 验证一下这是不是 assistant (解码看看，或者检查 token id)
                # 这里的逻辑是：mask 掉 last_start_idx 之前的所有内容
                # 还要 mask 掉 "<|im_start|>assistant\n" 本身，只保留答案
                
                # 我们可以保守一点：只 mask 掉 last_start_idx 之前的内容
                # 这样至少 User 的问题被 mask 了
                current_label[:last_start_idx+2] = ignore_index # +2 是为了覆盖 assistant 标签
            
            new_labels[i] = current_label

        inputs["labels"] = new_labels

        if self.debug_truncation:
            truncated_indices = []

            for i in range(len(texts)):
                orig_len = self.max_length
                trunc_len = inputs["attention_mask"][i].sum().item()
                print(f"inputs len is :{trunc_len}")

                if orig_len == trunc_len:
                    truncated_indices.append((i, orig_len, trunc_len))

            if truncated_indices:
                print("⚠️ Detected truncated samples:")
                for idx, o, t in truncated_indices:
                    print(f"  Sample {idx}: {o} -> {t}")
            else:
                print("✅ No samples truncated in this batch")

        # --- 调试代码：查看真实的 Label ---
        # print('inputs:')
        # print(self.processor.tokenizer.decode(inputs["input_ids"][0]))

        # print('labels (Debug View):')
        # # 创建一个临时的 list，把 -100 变成 pad_token_id 以便 decode 能显示出来
        # debug_labels = inputs["labels"][0].clone()
        # # 将 -100 替换为 0 (或者 tokenizer.pad_token_id)，这样 decode 就不会忽略它了
        # debug_labels[debug_labels == -100] = self.processor.tokenizer.pad_token_id 
        # print(self.processor.tokenizer.decode(debug_labels))
        # exit(0)
        
        return inputs
    
    def convert_to_official_format(
        self, 
        conversations: List[Dict], 
        images: List[Image.Image]
    ) -> List[Dict]:
        """
        将对话转换为Qwen2.5-VL官方格式
        
        官方格式:
        [
            {
                "role": "system",
                "content": "系统提示词"
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": <PIL.Image>},
                    {"type": "text", "text": "问题"}
                ]
            },
            {
                "role": "assistant", 
                "content": "回答"
            }
        ]
        """
        messages = []
        
        for conv in conversations:
            role = conv['role']
            content = conv['content']
            
            if role == 'system':
                # 系统提示词
                messages.append({
                    "role": "system",
                    "content": content
                })
            elif role == 'assistant':
                # assistant回复(纯文本)
                messages.append({
                    "role": "assistant",
                    "content": content
                })
            elif role == 'user':
                # user消息,处理<image>占位符
                if '<image>' not in content:
                    # 纯文本
                    messages.append({
                        "role": "user",
                        "content": content
                    })
                else:
                    # 包含图片,解析成官方格式
                    formatted_content = self.parse_user_content_with_images(
                        content, 
                        images
                    )
                    messages.append({
                        "role": "user",
                        "content": formatted_content
                    })
        
        return messages
    
    def parse_user_content_with_images(
        self, 
        text: str, 
        images: List[Image.Image]
    ) -> List[Dict]:
        """
        解析包含<image>占位符的文本,转换为官方格式
        
        输入: "UAV1:\n<image>\nUAV2:\n<image>\nQuestion: ..."
        输出: [
            {"type": "text", "text": "UAV1:\n"},
            {"type": "image", "image": <PIL.Image>},
            {"type": "text", "text": "\nUAV2:\n"},
            {"type": "image", "image": <PIL.Image>},
            {"type": "text", "text": "\nQuestion: ..."}
        ]
        """
        content_parts = []
        current_image_idx = 0
        
        # 按<image>分割文本
        parts = text.split('<image>')
        
        for i, part in enumerate(parts):
            # 添加文本部分(如果非空)
            if part:
                content_parts.append({
                    "type": "text",
                    "text": part
                })
            
            # 添加图片(除了最后一个分割部分)
            if i < len(parts) - 1 and current_image_idx < len(images):
                content_parts.append({
                    "type": "image",
                    "image": images[current_image_idx]
                })
                current_image_idx += 1
        
        return content_parts


# ==================== 模型配置 ====================

def setup_model_with_lora(
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    use_flash_attention: bool = False,
    args = None
):
    """设置LoRA模型 - 使用官方推荐方式"""
    
    # 根据官方示例加载模型
    if use_flash_attention:
        # 推荐在多图和视频场景使用flash_attention_2
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        )
    else:
        # 默认加载方式
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        print("model 加载成功")

    # print(model)

    
    # LoRA配置
    # lora_config = LoraConfig(
    #     task_type=TaskType.CAUSAL_LM,
    #     r=lora_r,
    #     lora_alpha=lora_alpha,
    #     lora_dropout=lora_dropout,
    #     # target_modules=[
    #     #     "q_proj", "k_proj", "v_proj", "o_proj",
    #     #     "gate_proj", "up_proj", "down_proj"
    #     # ],
    #     # target_modules=[
    #     #     "q_proj", "k_proj", "v_proj", "o_proj",  # attention层
    #     #     "gate_proj", "up_proj", "down_proj",      # MLP层
    #     # ],
    #     target_modules=r".*model\.layers\..*proj", 
    #     modules_to_save=["visual"], 
    #     bias="none",
    #     inference_mode=False
    # )

    from qwen_3d.config.lora_config import LORA_CONFIG
    lora_config = LORA_CONFIG[args.lora_type]
    
    model = get_peft_model(model, lora_config)

    # ================= 修复开始 =================
    # 解决 ModulesToSaveWrapper 丢失 dtype 属性导致的报错
    # 检查 visual 是否被 wrap 成了 ModulesToSaveWrapper
    from peft.utils.other import ModulesToSaveWrapper
    
    # 获取模型中的 visual 模块（根据你的模型层级可能略有不同，通常是 model.visual 或 model.base_model.model.visual）
    # Qwen2.5-VL 在 PEFT 模型中通常可以通过 model.visual 直接访问到（经过了层层转发）
    
    if hasattr(model, "visual"):
        if isinstance(model.visual, ModulesToSaveWrapper):
            # 将原始模块的 dtype 赋值给 wrapper
            # 注意：在 DeepSpeed Stage 3 下，参数可能被分区，但 original_module 通常还能访问到配置
            if hasattr(model.visual.original_module, "dtype"):
                model.visual.dtype = model.visual.original_module.dtype
            else:
                # 如果获取不到，根据你加载模型的精度强制指定 (例如 bfloat16)
                model.visual.dtype = torch.bfloat16 
                
            print(f"已修复 visual 模块的 dtype: {model.visual.dtype}")
            
    # 如果 visual 在更深层级（例如 model.base_model.model.visual）
    elif hasattr(model.base_model.model, "visual"):
         if isinstance(model.base_model.model.visual, ModulesToSaveWrapper):
            model.base_model.model.visual.dtype = model.base_model.model.visual.original_module.dtype
            print(f"已修复 base_model.visual 模块的 dtype")
    # ================= 修复结束 =================

    model.print_trainable_parameters()



    if args.local_rank == 0 or args.local_rank == -1:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

        # 方法3：统计可训练参数数量
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        if all_params > 0:
            ratio = trainable_params / all_params
            logger.info(f"Total parameters : {all_params}, trainable parameters: {trainable_params:,}, 占比 {ratio:.2%}")
        else:
            logger.info(f"Total trainable parameters: {trainable_params:,}")


    
    # 启用梯度检查点以节省显存
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    
    return model


# ==================== DeepSpeed训练配置 ====================

def create_deepspeed_config(output_dir: str, stage: int = 2) -> str:
    """
    创建DeepSpeed配置文件
    stage: 1, 2, 3 对应ZeRO的不同优化级别
    - Stage 1: 优化器状态分片
    - Stage 2: 优化器状态+梯度分片 (推荐)
    - Stage 3: 优化器状态+梯度+模型参数分片 (最省显存)
    """
    
    ds_config = {
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": stage,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            } if stage == 3 else {},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "gather_16bit_weights_on_model_save": True
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 10,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False
    }
    
    config_path = os.path.join(output_dir, "ds_config.json")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(ds_config, f, indent=2)
    
    return config_path


def train_multimodal_model(
    model,
    model_name,
    train_dataset,
    args,
    output_dir: str = "./qwen_vl_multimodal",
    num_epochs: int = 3,
    per_device_batch_size: int = 1,  # 多图时建议设为1
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-4,
    warmup_ratio: float = 0.1,
    max_length: int = 2048,
    deepspeed_stage: int = 2,
    save_steps: int = 500,
    logging_steps: int = 10,
):
    """
    使用DeepSpeed训练多模态模型
    """
    
    # 加载processor
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # 创建DeepSpeed配置
    ds_config_path = create_deepspeed_config(output_dir, stage=deepspeed_stage)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        seed=42,
        data_seed=42,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=1,
        bf16=True,  #         
        deepspeed=ds_config_path,  # 启用DeepSpeed
        report_to="tensorboard",
        remove_unused_columns=False,
        dataloader_num_workers=8,
        dataloader_pin_memory=False, # 暂时用false
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        save_safetensors=True,
        max_grad_norm=1.0
    )
    shuffle_uav_id = False
    if args.shuffle_uav_id == 'True':
        shuffle_uav_id = True
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=MultimodalDataCollator(
            processor=processor,
            max_length=max_length,
            fixed_image_size=args.fixed_image_size,
            shuffle_uav_id = shuffle_uav_id
        )
    )
    
    # 开始训练
    print("=" * 50)
    print("开始DeepSpeed多卡训练...")
    print(f"训练样本数: {len(train_dataset)}")
    print(f"DeepSpeed Stage: {deepspeed_stage}")
    print(f"总batch size: {per_device_batch_size * gradient_accumulation_steps * torch.cuda.device_count()}")
    print("=" * 50)
    
    trainer.train()
    
    # 保存模型
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    
    print(f"\n模型已保存到: {output_dir}")
    
    return trainer


def train_multimodal_model_eval(
    model,
    model_name,
    train_dataset,
    args,
    output_dir: str = "./qwen_vl_multimodal",
    num_epochs: int = 3,
    per_device_batch_size: int = 1,  # 多图时建议设为1
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-4,
    warmup_ratio: float = 0.1,
    max_length: int = 2048,
    deepspeed_stage: int = 2,
    save_steps: int = 500,
    logging_steps: int = 10,
):
    from torch.utils.data import random_split
    # 加载processor
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # 创建DeepSpeed配置
    ds_config_path = create_deepspeed_config(output_dir, stage=deepspeed_stage)

    # -------------------------------------------------------------------------
    # 修改 1: 数据集切分 (从 train_dataset 中拆分 0.1 作为 eval)
    # -------------------------------------------------------------------------
    # 计算切分长度
    dataset_len = len(train_dataset)
    eval_len = int(dataset_len * 0.1)
    train_len = dataset_len - eval_len

    # 使用 random_split 进行切分 (设置 generator 保证复现性)
    # 注意：如果你的 train_dataset 是 HuggingFace 的 Dataset 对象，
    # 也可以直接用 train_dataset = train_dataset.train_test_split(test_size=0.1)
    final_train_dataset, final_eval_dataset = random_split(
        train_dataset, 
        [train_len, eval_len], 
        generator=torch.Generator().manual_seed(42)
    )

    print(f"原始数据集: {dataset_len}")
    print(f"切分后 - 训练集: {len(final_train_dataset)}, 验证集: {len(final_eval_dataset)}")

    # -------------------------------------------------------------------------
    # 修改 2: TrainingArguments 配置
    # -------------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        seed=42,
        data_seed=42,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        
        # === 核心修改部分开始 ===
        # 1. 设置评估策略为 steps
        evaluation_strategy="steps", 
        per_device_eval_batch_size=per_device_batch_size, 
        # 2. 每 10 step 评估一次
        eval_steps=500,
        
        # 3. 保存策略必须与评估策略一致（这里也设为 steps）
        save_strategy="steps",
        # 4. 每 10 step 保存一次 checkpoint (为了能从中选出最好的)
        save_steps=500, 
        
        # 5. 训练结束时自动加载效果最好的模型权重
        load_best_model_at_end=True,
        # 6. 指定用于比较“最好”的指标 (默认就是 loss，这里显式写出来更清晰)
        metric_for_best_model="loss",
        # 7. loss 是越小越好，所以设为 False
        greater_is_better=False,
        # === 核心修改部分结束 ===

        save_total_limit=1, # 只保留最好的那一个 checkpoint（以及最新的）
        bf16=True,
        deepspeed=ds_config_path,
        report_to="tensorboard",
        remove_unused_columns=False,
        dataloader_num_workers=16,
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        save_safetensors=True,
        max_grad_norm=1.0,
        label_names=["labels"]
    )

    # -------------------------------------------------------------------------
    # 修改 3: 初始化 Trainer 时传入 eval_dataset
    # -------------------------------------------------------------------------
    # 公共的数据整理器
    data_collator = MultimodalDataCollator(
            processor=processor,
            max_length=max_length,
            fixed_image_size=args.fixed_image_size
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=final_train_dataset, # 使用切分后的训练集
        eval_dataset=final_eval_dataset,   # 传入切分后的验证集
        data_collator=data_collator
    )

    print("=" * 60)
    # print(f"开始训练 (3D={'启用' if enable_3d else '禁用'})")
    print(f"训练集大小: {len(final_train_dataset)}")
    print(f"验证集大小: {len(final_eval_dataset)}")
    print(f"总batch size: {per_device_batch_size * gradient_accumulation_steps * torch.cuda.device_count()}")
    print("=" * 60)

    trainer.train()

    # 保存模型
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    
    print(f"\n模型已保存到: {output_dir}")
    
    return trainer



# ==================== 主函数 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True, help="训练数据JSON路径")
    parser.add_argument("--video_mapping", type=str, required=True, help="video到image路径的映射JSON")
    parser.add_argument("--system_prompt", type=str, 
                       default=None,
                       help="系统提示词")
    parser.add_argument("--output_dir", type=str, default="./qwen_vl_output")
    parser.add_argument("--lora_type", type=str, default="all_lora")
    
    parser.add_argument("--shuffle_uav_id", type=str, default="False")

    parser.add_argument("--eval", type=str, default="False")
    parser.add_argument("--model_name", type=str, default="models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--deepspeed_stage", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--use_flash_attention", action="store_true", 
                       help="使用flash_attention_2(推荐用于多图场景)")
    parser.add_argument("--min_pixels", type=int, default=None,
                       help="最小像素数(默认256*28*28)")
    parser.add_argument("--max_pixels", type=int, default=None,
                       help="最大像素数(默认1280*28*28)")
    parser.add_argument("--fixed_image_size", type=int, nargs=2, default=[224, 224],
                       help="固定图像尺寸 (H W)")

    parser.add_argument(
            "--local_rank",
            type=int,
            default=-1,
            help="Local rank for distributed training"
        )
    
    args = parser.parse_args()

    import os

    os.makedirs(args.output_dir, exist_ok=True)

    if args.local_rank == 0 or args.local_rank == -1:
        # 清理 root logger 已有的 handler
        for h in logging.root.handlers[:]:
            logging.root.removeHandler(h)
            
        logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(f'{args.output_dir}/model_log.log', encoding='utf-8', mode='w')
                    # logging.StreamHandler()
                ]
            )
    
    # 准备数据
    print("加载数据集...")
    train_dataset = prepare_multimodal_dataset(
        json_path=args.train_data,
        video_mapping_path=args.video_mapping,
        system_prompt=args.system_prompt
    )
    
    print(f"数据集大小: {len(train_dataset)}")
    if len(train_dataset) > 0:
        print(f"示例数据: video_id={train_dataset[0]['video_id']}, images={len(train_dataset[0]['images'])}")
    
    # 设置模型
    print("加载模型...")
    model = setup_model_with_lora(
        model_name=args.model_name,
        lora_r=args.lora_r,
        use_flash_attention=args.use_flash_attention,
        args=args
    )
    
    # 训练
    if args.eval == 'True':
        print("-"*10 +"使用eval训练中验证，最后保存最优模型 "+"-"*10 )
        trainer = train_multimodal_model_eval(
            model=model,
            model_name=args.model_name,
            train_dataset=train_dataset,
            args=args,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            per_device_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.lr,
            deepspeed_stage=args.deepspeed_stage,
            max_length=args.max_length
        )
    else :
        trainer = train_multimodal_model(
            model=model,
            model_name=args.model_name,
            train_dataset=train_dataset,
            args=args,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            per_device_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.lr,
            deepspeed_stage=args.deepspeed_stage,
            max_length=args.max_length
        )
    
    
    print("\n训练完成!")

    