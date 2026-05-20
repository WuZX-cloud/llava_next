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

from ..train.train_multimodal import load_video_image_mapping

from tqdm import tqdm
import traceback
import logging
from peft import PeftModel

import random

logger = logging.getLogger(__name__)

def eval_single_sample(
    model,
    processor,
    images: List[str],
    user_query: str,
    system_prompt: str = None,
    max_new_tokens: int = 512,
    args = None
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
    
    sampled=False
    if args.sampled == 'True':
        sampled=True

    shuffle_uav_id=False
    if args.shuffle_uav_id == 'True':
        shuffle_uav_id=True
    
    # 加载图片
    loaded_images = []
    for img_path in images:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))  # 强制缩小到固定尺寸
        loaded_images.append(img)
    
    if shuffle_uav_id:
        random.shuffle(loaded_images)
    # 构建消息 - 官方格式
    messages = []
    if system_prompt:
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
        # 纯文本
        content = []
        # for img in loaded_images:
        #     content.append({"type": "image", "image": img})
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

    # # === DEBUG COLLATOR ===
    # texts = text
    # batch_messages = messages
    # all_images = loaded_images
    # print(f"\n[Debug Collator]")
    # print(f"  Batch size: {len(batch_messages)}")
    # print(f"  len(all_images): {len(all_images) if all_images else 0}")
    # if len(texts) > 0:
    #     print(f"  First Text Preview: {texts}") # 看看有没有 <|vision_start|> 或 <image>
    # if len(batch_messages) > 0:
    #     # 检查 convert_to_official_format 是否真的把 PIL 对象放进去了
    #     import json
    #     def safe_serialize(obj):
    #         if hasattr(obj, 'size'): return f"<PIL Image {obj.size}>"
    #         return str(obj)
    #     print(f"  First Message Structure: {batch_messages}") 
    # # ======================

    # print(processor.tokenizer.decode(inputs["input_ids"][0]))
    # exit(0)

    inputs = inputs.to(model.device)

    
    # 4. 生成
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=sampled,
        temperature=0.5,
        top_p=1.0,
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
    
    return output_text, user_query


def eval_model_batch(args):
    """批量推理主函数"""
    
    
    # 初始化模型
    if args.model_base is None:
    # 加载模型和processor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 ,
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            use_fast=True
        )
    else :
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_base,
            torch_dtype=torch.bfloat16 ,
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(
            model,
            args.model_path,
            is_trainable=False
        )
        processor = AutoProcessor.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            use_fast=True
        )
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

        model = model.merge_and_unload()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    model.eval()
    model.to(torch.bfloat16)
    print(f"use {device}")
    model.to(device)


    with open(args.test_data, 'r', encoding='utf-8') as f1:
        samples = json.load(f1)

    mapping = load_video_image_mapping(args.video_mapping)

    def get_query(conversations):
        for conv in conversations:
            if conv['from'] == 'human':
                return conv['value']
    
    
    # 验证JSON格式
    if not isinstance(samples, list):
        raise ValueError(f"Input JSON must be a list, got {type(samples)}")
    
    # 批量推理
    results = []
    for idx, sample in enumerate(tqdm(samples, desc="Processing samples")):
    # idx = 1030
    # sample = samples[idx]
    # while idx < len(samples):
        # idx += 1
        try:
            # logger.info(f"Processing sample {idx + 1}/{len(samples)}")
            images = mapping[sample['video']]
            user_query=get_query(sample['conversations'])
            system_prompt = args.system_prompt

            # question = user_query.replace("<video>\n", "")
            # option = "\n".join([f"{k}.{v}" for k, v in sample['options'].items()])
            # user_query = f"{question}\n{option}\nAnswer with the option's letter from the given choices directly."


            response, clean_value = eval_single_sample(
                model=model, 
                processor=processor, 
                images=images,
                user_query=user_query,
                system_prompt=system_prompt,
                max_new_tokens=args.max_length,
                args = args
                )
            
            result = {
                **sample,  # 保留原始样本信息
                'response': response
            }
            results.append(result)
            
            # print(f"\n[Sample {idx + 1}] Query: {clean_value}")
            # print(f"[Sample {idx + 1}] Response: {response}\n")
            
        except Exception as e:
            logger.error(f"Error processing sample {idx + 1}: {str(e)}")
            # 打印完整 traceback 到控制台
            traceback.print_exc()

            # 如果你希望也写入日志：
            logger.error(traceback.format_exc())
            result = {
                **sample,
                'response': None,
                'error': str(e)
            }
            results.append(result)
    
    # 保存结果
    output_path = f"{args.output_dir}/{args.output_name}.json"
    logger.info(f"Saving results to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Batch inference completed! Results saved to {output_path}")
    logger.info(f"Successfully processed: {sum(1 for r in results if 'error' not in r)}/{len(samples)}")

if __name__ == "__main__" :
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, required=True, help="训练数据JSON路径")
    parser.add_argument("--video_mapping", type=str, required=True, help="video到image路径的映射JSON")
    parser.add_argument("--system_prompt", type=str, 
                       default="你是一个专业的多模态分析助手,能够理解和分析多个无人机视角的图像。",
                       help="系统提示词")
    parser.add_argument("--output_dir", type=str, default="./qwen_vl_finetune/")
    parser.add_argument("--output_name", type=str, default="./qwen_vl_output")
    parser.add_argument("--model_path", type=str, default="models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--sampled", type=str, default="False")
    parser.add_argument("--shuffle_uav_id", type=str, default="False")


    args = parser.parse_args()


    log_dir = args.output_dir
    os.makedirs(log_dir, exist_ok=True)

    
    # 配置 logging（只在主程序配置一次）
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{args.output_dir}/{args.output_name}.log", encoding='utf-8', mode='w'),  # 输出到文件
            # logging.StreamHandler()  # 同时输出到控制台（可选）
        ]
    )

    eval_model_batch(args)





