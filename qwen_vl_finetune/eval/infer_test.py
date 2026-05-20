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



# ==================== 推理示例 ====================


def inference_multimodal(
    model_path: str,
    images: List[str],
    user_query: str,
    system_prompt: str = "你是一个专业的多模态分析助手",
    max_new_tokens: int = 512,
    device: int = 0
):
    """
    多图推理 - 使用官方方式
    
    Args:
        model_path: 模型路径
        images: 图片路径列表
        user_query: 用户问题(可包含<image>占位符)
        system_prompt: 系统提示词
        max_new_tokens: 最大生成token数
    """
    
    # 加载模型和processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    print(f"use {device}")
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # 加载图片
    loaded_images = []
    for img_path in images:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((448, 448))  # 强制缩小到固定尺寸
        loaded_images.append(img)
    
    # 构建消息 - 官方格式
    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]
    
    # 处理user_query中的<image>占位符
    if '<image>' in user_query:
        # 解析图文混合内容
        content_parts = []
        parts = user_query.split('<image>')
        current_img_idx = 0
        
        for i, part in enumerate(parts):
            if part:
                content_parts.append({"type": "text", "text": part})
            
            if i < len(parts) - 1 and current_img_idx < len(loaded_images):
                content_parts.append({
                    "type": "image",
                    "image": loaded_images[current_img_idx]
                })
                current_img_idx += 1
        
        messages.append({
            "role": "user",
            "content": content_parts
        })
    else:
        # 纯文本或所有图片在前
        content = []
        for img in loaded_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": user_query})
        
        messages.append({
            "role": "user",
            "content": content
        })
    
    # 按照官方示例处理
    # 1. apply_chat_template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 2. process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)
    
    # 3. processor处理
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    model.to(device)
    print(model.device)

    print(processor.tokenizer.decode(inputs["input_ids"][0]))

    inputs = inputs.to(model.device)

    
    # 4. 生成
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    # 5. 解码(只取新生成的部分)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return output_text



if __name__ == "__main__":
    response = inference_multimodal(
        model_path="models/Qwen2.5-VL-7B-Instruct",
        images=["1.jpg"],
        user_query="UAV1:<image>里有什么？",
        device=1
    )
    print(response)