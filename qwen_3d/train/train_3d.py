"""
Qwen2.5-VL + 3D Position Encoding 完整实现
方案B: 修改模型类,固定分辨率,保持patch-pixel对应关系
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from PIL import Image
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import os
from qwen_vl_utils import process_vision_info

import traceback

from qwen_3d.model.threeD import ThreeDPositionEncoding, DepthTo3DCoordinates
from qwen_3d.model.model import Qwen2_5_VLWith3D

import argparse

import logging

logger = logging.getLogger(__name__)

global local_rank





def load_camera_params(video_mapping_path: str) -> Dict:
    """
    加载相机参数 (内参+外参)
    
    处理后的映射文件格式:
    {
        "video_id": {
            "image_paths": ["path1.jpg", "path2.jpg"],
            "depth_paths": ["depth1.png", "depth2.png"],
            "intrinsics": [K1, K2],  # 每个UAV的内参
            "extrinsics": [T1, T2],  # 每个UAV的外参
            "num_uavs": 2
        }
    }
    """
    with open(video_mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    
    print(f"加载相机参数: {len(mapping)} 个video")
    return mapping


def prepare_multimodal_dataset_with_3d(
    json_path: str,
    video_mapping_path: str,
    system_prompt: str = None,
    patch_size: int = 14,
    fixed_image_size: Tuple[int, int] = (448, 448)
):
    """
    准备包含3D信息的数据集 (Lazy Loading版本)
    
    只存储路径和元数据,不预先计算3D坐标
    在DataCollator中按需加载和计算
    
    Args:
        json_path: 训练数据JSON
        video_mapping_path: 处理后的mapping
        system_prompt: 系统提示词
        patch_size: ViT patch大小
        fixed_image_size: 固定图像尺寸
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    # train_data=train_data[:10] # 临时测试
    
    # 加载处理后的mapping
    video_mapping = load_camera_params(video_mapping_path)
    
    processed_data = []
    skipped_count = 0
    
    print(f"\n处理训练数据: {len(train_data)} 个样本")
    print(f"模式: Lazy Loading (只存储路径,不预先计算3D坐标)")
    
    for idx, item in enumerate(train_data):
        video_id = item['video']
        
        if video_id not in video_mapping:
            if (idx + 1) % 100 == 0 or skipped_count < 10:
                print(f"[{idx+1}/{len(train_data)}] 跳过 {video_id}: 不在mapping中")
            skipped_count += 1
            continue
        
        video_data = video_mapping[video_id]
        
        # 验证文件路径存在性 (快速检查,不加载实际数据)
        image_paths = video_data['image_paths']
        depth_paths = video_data['depth_paths']
        
        # 简单验证:至少检查第一个文件
        if not os.path.exists(image_paths[0]):
            if skipped_count < 10:
                print(f"[{idx+1}/{len(train_data)}] 跳过 {video_id}: RGB图不存在 {image_paths[0]}")
            skipped_count += 1
            continue
        
        if not os.path.exists(depth_paths[0]):
            if skipped_count < 10:
                print(f"[{idx+1}/{len(train_data)}] 跳过 {video_id}: 深度图不存在 {depth_paths[0]}")
            skipped_count += 1
            continue
        
        # 构建数据项 (只存储路径和元数据)
        processed_item = {
            "video_id": video_id,
            "image_paths": image_paths,      # RGB图路径列表
            "depth_paths": depth_paths,      # 深度图路径列表
            "intrinsics": video_data['intrinsics'],  # 内参列表
            "extrinsics": video_data['extrinsics'],  # 外参列表
            "conversations": convert_conversations(item['conversations'], system_prompt),
            "num_views": len(image_paths),
            # 3D计算所需的元数据
            "patch_size": patch_size,
            "fixed_image_size": fixed_image_size
        }
        
        processed_data.append(processed_item)
        
        # 定期打印进度
        if (idx + 1) % 500 == 0:
            print(f"  进度: {idx+1}/{len(train_data)} (跳过: {skipped_count})")
    
    print(f"\n✓ 处理完成:")
    print(f"  - 成功: {len(processed_data)} 个样本")
    print(f"  - 跳过: {skipped_count} 个样本")
    print(f"  - 内存占用: 仅路径和元数据 (~{len(processed_data) * 0.001:.2f} MB)")
    
    if len(processed_data) > 0:
        example = processed_data[0]
        print(f"\n示例数据:")
        print(f"  - video_id: {example['video_id']}")
        print(f"  - 视角数: {example['num_views']}")
        print(f"  - RGB路径: {example['image_paths'][0]}")
        print(f"  - 深度路径: {example['depth_paths'][0]}")
        logger.info(f"===========示例数据============")
        logger.info(example)
    
    return Dataset.from_list(processed_data)


def convert_conversations(conversations: List[Dict], system_prompt: str) -> List[Dict]:
    """转换对话格式"""
    converted = []
    if system_prompt:
        converted.append({"role": "system", "content": system_prompt})
    
    for conv in conversations:
        role = "user" if conv['from'] == 'human' else "assistant"
        converted.append({"role": role, "content": conv['value']})
    
    return converted


# ==================== 修改后的DataCollator ====================

@dataclass
class MultimodalDataCollatorWith3D:
    """
    支持3D坐标的数据整理器 (Lazy Loading版本)
    在collate时动态加载图像和计算3D坐标
    """
    processor: AutoProcessor
    max_length: int = 4096
    enable_3d: bool = True
    args: Optional[argparse.Namespace] = None     # 👈 就是它
    
    def __post_init__(self):
        """初始化3D坐标生成器 (如果启用3D)"""
        if self.enable_3d:
            # 创建一个共享的3D坐标转换器
            # 从第一个batch推断patch_size和fixed_image_size
            self._coord_generator = None
    
    def _get_or_create_coord_generator(self, patch_size: int, fixed_image_size: Tuple[int, int]):
        """延迟初始化坐标生成器"""
        if self._coord_generator is None:
            is_w2c = False
            if args.is_w2c == "True":
                is_w2c = True
            self._coord_generator = DepthTo3DCoordinates(
                patch_size=patch_size,
                fixed_image_size=fixed_image_size,
                type=self.args.norm_type,
                grid_n=self.args.grid_n,
                is_w2c=is_w2c
            )
        return self._coord_generator
    
    def _load_and_compute_3d(self, item: Dict) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        加载RGB和深度图,计算3D坐标
        
        Args:
            item: 包含路径和相机参数的数据项
        
        Returns:
            resized_rgbs: resize后的RGB图列表
            coords_3d: (num_views * num_patches, 3) 3D坐标
        """
        # 获取坐标生成器
        coord_gen = self._get_or_create_coord_generator(
            patch_size=item['patch_size'],
            fixed_image_size=tuple(item['fixed_image_size'])
        )
        
        # 加载图像和深度图
        # rgb_list = []
        # depth_list = []
        
        # for rgb_path, depth_path in zip(item['image_paths'], item['depth_paths']):
            # 加载RGB
            # rgb = Image.open(rgb_path).convert('RGB')
            # rgb_list.append(rgb)
            
            # # 加载深度图
            # if depth_path.endswith('.npy'):
            #     depth = np.load(depth_path)
            # else:
            #     depth = np.array(Image.open(depth_path))
            #     # uint16深度图转换为float (mm → m)
            #     if depth.dtype == np.uint16:
            #         depth = depth.astype(np.float32) / 1000.0
            
            # depth_list.append(depth)
        
        depth_list = item['depth_paths']
        rgb_list = item['image_paths']

        # 转换内外参为numpy数组
        intrinsics = [np.array(K, dtype=np.float32) for K in item['intrinsics']]
        extrinsics = [np.array(T, dtype=np.float32) for T in item['extrinsics']]
        
        # 生成3D坐标
        coords_3d, resized_rgbs = coord_gen.process_multi_view(
            depth_list=depth_list,
            intrinsic_list=intrinsics,
            extrinsic_list=extrinsics,
            rgb_list=rgb_list
        )

        coords_3d = torch.as_tensor(coords_3d)

        if args.coords_type == "shuffled":
            num_views = coords_3d.shape[0] // 256
            coords_3d = self.make_shuffled_3d(coords_3d, num_views)
        elif args.coords_type == "shuffled_all":
            coords_3d = self.make_global_shuffled_3d(coords_3d)
        elif args.coords_type == "noisy":
            coords_3d = self.make_noisy_3d(coords_3d, args.sigma)
        
        return resized_rgbs, coords_3d
    
    def __call__(self, features: List[Dict]) -> Dict:
        """
        动态加载和处理batch数据
        """
        batch_messages = []
        batch_coords_3d = []
        load_errors = []
        
        for idx, item in enumerate(features):
            try:
                # 1. 动态加载RGB和计算3D坐标
                if self.enable_3d:
                    # print("启动了动态加载RGB和计算3D坐标")
                    resized_rgbs, coords_3d = self._load_and_compute_3d(item)
                    batch_coords_3d.append(torch.as_tensor(coords_3d).to(dtype=torch.bfloat16))
                else:
                    # 如果不启用3D,只加载RGB
                    resized_rgbs = []
                    for rgb_path in item['image_paths']:
                        rgb = Image.open(rgb_path).convert('RGB')
                        # 需要resize到固定尺寸
                        rgb = rgb.resize(
                            (item['fixed_image_size'][0], item['fixed_image_size'][1])
                        )
                        resized_rgbs.append(rgb)
                
                # 2. 转换为PIL Image
                # images = [Image.fromarray(rgb) for rgb in resized_rgbs]
                images = resized_rgbs
                # print(f"images len is : {len(images)}")
                # 3. 转换为Qwen格式
                messages = self.convert_to_official_format(item['conversations'], images)
                batch_messages.append(messages)
                
            except Exception as e:
                traceback.print_exc()
                # 记录错误但继续处理其他样本
                load_errors.append((idx, item.get('video_id', 'unknown'), str(e)))
                # 添加空消息占位
                batch_messages.append([
                    {"role": "user", "content": "Error loading data"},
                    {"role": "assistant", "content": "Error"}
                ])
                if self.enable_3d:
                    # 添加零坐标占位
                    batch_coords_3d.append(torch.zeros(1, 3))
        
        # 打印加载错误 (如果有)
        if load_errors:
            print(f"\n⚠️  Batch中 {len(load_errors)} 个样本加载失败:")
            for idx, video_id, error in load_errors[:3]:  # 只显示前3个
                print(f"  [{idx}] {video_id}: {error}")
        
        # 收集所有的纯问题文本
        batch_question_texts = []
        
        for item in batch_messages:
            # item['conversations'] 通常是List[Dict]
            # 我们假设最后一轮是用户提问
            # 格式: [{'role': 'user', 'content': '...'}, {'role': 'assistant', ...}]
            
            last_user_msg = ""
            for msg in item:
                if msg['role'] == 'user':
                    # content 可能是字符串，也可能是 list (包含 image/text)
                    content = msg['content']
                    if isinstance(content, str):
                        last_user_msg = content
                    elif isinstance(content, list):
                        # 拼接所有 text 部分
                        last_user_msg = "".join([x['text'] for x in content if x['type'] == 'text'])
            
            # 提取纯问题
            q_text = self._extract_question_from_text(last_user_msg)
            
            # 重要：Qwen 的 tokenizer 不会自动加特殊 token 吗？
            # 建议加上 prompt template 的一部分或者纯文本，对于提取语义来说纯文本够了
            batch_question_texts.append(q_text)

        

        # 4. 使用processor处理文本和图像
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in batch_messages
        ]
        
        all_images = []
        all_videos = []
        for msg in batch_messages:
            img_inputs, video_inputs = process_vision_info(msg)
            if img_inputs:
                all_images.extend(img_inputs)
            if video_inputs:
                all_videos.extend(video_inputs)
        
        # # === DEBUG COLLATOR ===
        # print(f"\n[Debug Collator]")
        # print(f"  Batch size: {len(batch_messages)}")
        # print(f"  len(all_images): {len(all_images) if all_images else 0}")
        # if len(texts) > 0:
        #     print(f"  First Text Preview: {texts[0][:200]}") # 看看有没有 <|vision_start|> 或 <image>
        # if len(batch_messages) > 0:
        #     # 检查 convert_to_official_format 是否真的把 PIL 对象放进去了
        #     import json
        #     def safe_serialize(obj):
        #         if hasattr(obj, 'size'): return f"<PIL Image {obj.size}>"
        #         return str(obj)
        #     print(f"  First Message Structure: {batch_messages[0]}") 
        # # ======================
        # exit(0)

        inputs = self.processor(
            text=texts,
            images=all_images if all_images else None,
            videos=all_videos if all_videos else None,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        )
        # print(f'inputs keys is {inputs.keys()}')
        # for key in inputs.keys():
        #     print(f"{key} type is {type(inputs[key])} , shape is {inputs[key].shape}")
        # exit(0)
        
        # 5. 准备labels
        # inputs["labels"] = inputs["input_ids"].clone()
        # # print(self.processor.tokenizer.decode(inputs["input_ids"][0]))
        # # exit(0)
        # inputs["labels"][inputs["labels"] == self.processor.tokenizer.pad_token_id] = -100

         # ================== 修复开始 ==================
        # 1. 复制 input_ids 作为 labels
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

        # # --- 调试代码：查看真实的 Label ---
        # print('inputs:')
        # print(self.processor.tokenizer.decode(inputs["input_ids"][0]))

        # print('labels (Debug View):')
        # # 创建一个临时的 list，把 -100 变成 pad_token_id 以便 decode 能显示出来
        # debug_labels = inputs["labels"][0].clone()
        # # 将 -100 替换为 0 (或者 tokenizer.pad_token_id)，这样 decode 就不会忽略它了
        # debug_labels[debug_labels == -100] = self.processor.tokenizer.pad_token_id 
        # print(self.processor.tokenizer.decode(debug_labels))
        # exit(0)

        # ================== 修复结束 ==================
        
        # 6. 添加3D坐标 (padding到统一长度)
        if self.enable_3d and batch_coords_3d:
            max_patches = max(c.shape[0] for c in batch_coords_3d)
            padded_coords = []
            coords_mask = torch.zeros(len(batch_coords_3d), max_patches)
            
            for i, coords in enumerate(batch_coords_3d):
                # 记录真实坐标的数量
                coords_mask[i, :coords.shape[0]] = 1
                
                # Padding
                pad_size = max_patches - coords.shape[0]
                if pad_size > 0:
                    coords = F.pad(coords, (0, 0, 0, pad_size), value=0)
                padded_coords.append(coords)
            
            inputs["coords_3d"] = torch.stack(padded_coords)
            inputs["coords_3d_mask"] = coords_mask
            

            # ================== 新增：单独 Tokenize 问题 ==================
            # 使用 tokenizer 处理纯文本
            # padding=True: 补齐到这一批里最长的问题长度 (通常很短，只有20-30 token)
            # add_special_tokens=False: 我们不需要 <|im_start|> 那些，只需要语义
            # return_attention_mask=True: 必须拿到 mask
            
            q_encodings = self.processor.tokenizer(
                batch_question_texts,
                padding=True,
                truncation=True,
                max_length=128, # 问题通常不会太长，限制一下节省显存
                return_tensors="pt",
                add_special_tokens=False # 纯语义编码
            )
            
            inputs["question_input_ids"] = q_encodings["input_ids"]
            # inputs["question_attention_mask"] = q_encodings["attention_mask"]
            
            # DEBUG: 打印看看提取对不对 (调试完删除)
            # print(f"Extracted Question: {batch_question_texts[0]}")
            # ============================================================

            


        return inputs
    
    def convert_to_official_format(self, conversations, images):
        """转换为Qwen官方格式"""
        messages = []
        img_idx = 0
        
        for conv in conversations:
            if conv['role'] in ['system', 'assistant']:
                messages.append(conv)
            elif conv['role'] == 'user':
                content = conv['content']
                if '<image>' not in content and images is None:
                    messages.append(conv)
                elif '<image>' in content and images is not None:
                    # 解析图文交叉
                    parts = []
                    segments = content.split('<image>')
                    
                    for i, segment in enumerate(segments):
                        # 添加文本部分
                        if segment:
                            parts.append({"type": "text", "text": segment})
                        
                        # 添加图片 (除了最后一个分割)
                        if i < len(segments) - 1 and img_idx < len(images):
                            parts.append({"type": "image", "image": images[img_idx]})
                            img_idx += 1
                    
                    messages.append({"role": "user", "content": parts})
                else :
                    parts = []
                    for img in images:
                        parts.append({"type": "image", "image": img})
                    parts.append({"type": "text", "text": content})
                    messages.append({"role": "user", "content": parts})
                    
        return messages

    def _extract_question_from_text(self, full_text):
        """
        从原始文本中利用正则提取问题。
        假设结构：Context \n Question \n Options (A. xxx)
        """
        import re
        # 1. 尝试找到选项 A. 或 A) 的位置作为结束点
        # 匹配 "\n A." 或 "\n A)" 或 "\nA."
        opt_pattern = r"\n\s*[A][\.\)]"
        opt_match = re.search(opt_pattern, full_text)
        
        if opt_match:
            end_idx = opt_match.start()
            # 截取到选项之前的内容
            text_before_opt = full_text[:end_idx]
            
            # 2. 在剩下的文本中，找到最后一个换行符 "\n"
            # 通常问题就在最后一个换行符之后
            last_newline = text_before_opt.rfind('\n')
            
            if last_newline != -1:
                question = text_before_opt[last_newline+1:].strip()
            else:
                # 如果没有换行符，可能整段都是问题
                question = text_before_opt.strip()
        else:
            # 没找到选项（非选择题），找最后一个问号？
            # 或者简单点：取最后一行
            lines = full_text.strip().split('\n')
            if len(lines) > 0:
                question = lines[-1].strip()
            else:
                question = full_text.strip()
                
        # 兜底：如果提取出来太短（比如空字符串），就返回原文本的后半部分
        if len(question) < 3:
            question = full_text[-50:] 
            
        return question
    

    def make_shuffled_3d(self, coords_3d, num_views):
        """
        coords_3d: [N, 3]
        num_views: m
        假设 N = n * m，且按视角顺序拼接


        保留了每个视角自己的数值分布
        不会把不同视角的几何 token 混到一起
        但已经破坏了 3D token 与视觉 token 的逐点对应关系
        对照含义很清楚
        """

        N, C = coords_3d.shape
        assert C == 3
        assert N % num_views == 0, f"N={N} cannot be evenly divided by num_views={num_views}"

        n = N // num_views
        coords_view = coords_3d.view(num_views, n, 3).clone()

        for v in range(num_views):
            perm = torch.randperm(n, device=coords_3d.device)
            coords_view[v] = coords_view[v][perm]

        coords_shuffled = coords_view.view(N, 3)
        return coords_shuffled

    def make_global_shuffled_3d(self, coords_3d):
        """
        【全局打乱】
        打破了视角的边界，把所有视角的几何 token 完全混到一个大池子里。
        不仅破坏了 3D token 与视觉 token 的逐点对应关系，
        也彻底破坏了每个视角原本独立的数值分布（属于 View A 的点可能跑到了 View B 的位置）。
        """
        N, C = coords_3d.shape
        # 核心：直接生成长度为 N 的全局随机索引
        # 因为传入的 coords_3d 已经是展平的全局状态，所以无需再做 reshape
        global_perm = torch.randperm(N, device=coords_3d.device)

        # 直接应用全局打乱
        coords_shuffled = coords_3d[global_perm]
        
        return coords_shuffled

    def make_noisy_3d(self, coords_3d, sigma=0.05):
        """
        coords_3d: [N, 3]
        sigma: 噪声强度，建议先试 0.05
        """
        scale = coords_3d.std()
        noise = torch.randn_like(coords_3d) * (sigma * scale)
        coords_noisy = coords_3d + noise

        return coords_noisy
    
# ==================== 模型设置与训练 ====================

from transformers import Trainer
import torch


class CustomTrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        model = self.model
        
        # 设定不同的学习率
        base_lr = self.args.learning_rate
        merger_lr = 10 * base_lr
        
        # 定义不需要权重衰减的参数
        no_decay = ["bias", "LayerNorm.weight"]
        
        # --- 修正开始：创建 4 个桶来存放参数 ---
        # 1. Merger 模块 + 需要衰减
        # 2. Merger 模块 + 不需要衰减
        # 3. Base 模块 + 需要衰减
        # 4. Base 模块 + 不需要衰减
        merger_decay_params = []
        merger_no_decay_params = []
        base_decay_params = []
        base_no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # 判断是否属于 merger
            is_merger = "merger" in name
            # 判断是否不衰减
            is_no_decay = any(nd in name for nd in no_decay)

            if is_merger:
                if is_no_decay:
                    merger_no_decay_params.append(param)
                else:
                    merger_decay_params.append(param)
            else:
                if is_no_decay:
                    base_no_decay_params.append(param)
                else:
                    base_decay_params.append(param)

        # 构建最终的参数组列表 (最多只有 4 组)
        optimizer_grouped_parameters = [
            {
                "params": merger_decay_params,
                "weight_decay": self.args.weight_decay,
                "lr": merger_lr,
            },
            {
                "params": merger_no_decay_params,
                "weight_decay": 0.0,
                "lr": merger_lr,
            },
            {
                "params": base_decay_params,
                "weight_decay": self.args.weight_decay,
                "lr": base_lr,
            },
            {
                "params": base_no_decay_params,
                "weight_decay": 0.0,
                "lr": base_lr,
            },
        ]
        # --- 修正结束 ---

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        
        return self.optimizer


def setup_model_with_3d_and_lora(
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    enable_3d: bool = True,
    lora_enable: bool = True,
    lora_r: int = 64,
    lora_alpha: int = 16,
    args=None
):
    """
    加载模型并配置3D模块+LoRA
    
    Args:
        vision_output_layer: 对于Qwen2.5-VL,推荐使用:
            - "visual.merger" (默认,最佳选择)
            - "visual.blocks.31" (次优,最后一个transformer block)
    """
    from transformers import AutoConfig
    
    print(f"\n{'='*60}")
    print(f"正在加载 Qwen2.5-VL + 3D Position Encoding")
    print(f"{'='*60}")
    
    # 1. 加载配置
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    print(f"✓ 配置加载完成: hidden_size={config.vision_config.hidden_size}")
    
    # 2. 实例化修改后的模型
    model = Qwen2_5_VLWith3D(config, args=args)
    model.to(torch.bfloat16)
    
    # 3. 加载预训练权重
    print(f"正在加载预训练权重...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

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
    # print(model)
    # exit(0)

    # # 2. 获取 Base 模型中 Merger 的某一个特定参数值（取前5个数）
    # base_tensor = base_model.visual.merger.mlp[0].weight.data.flatten()[:5]
    # print(f"【Base  权重】前5位: {base_tensor.tolist()}")

    # # 3. 获取你的 Custom 模型中 对应位置 的参数值
    # # 注意路径：visual.merger -> original_merger -> mlp...
    # custom_tensor = model.visual.merger.original_merger.mlp[0].weight.data.flatten()[:5]
    # print(f"【MyModel 权重】前5位: {custom_tensor.tolist()}")

    # 4. 判定
    # if torch.allclose(base_tensor, custom_tensor, atol=1e-5):
    #     print("✅ 结论：权重加载成功！数值完全一致。")
    #     # 如果走到这里，问题肯定在第二步（参数传递）
    # else:
    #     print("❌ 结论：权重加载失败！现在的权重是随机初始化的。")
    #     print(f"   (怀疑 Std=0.0233 确实是随机分布)")

    # 2. 提取 Base 模型 第0层 Block 的 Attention 权重 (前5个数)
    base_w = base_model.visual.blocks[0].attn.qkv.weight.data.flatten()[:5]

    # 3. 提取 你的模型 第0层 Block 的 Attention 权重
    my_w = model.visual.blocks[0].attn.qkv.weight.data.flatten()[:5]

    print(f"Base Block[0] W: {base_w.tolist()}")
    print(f"My   Block[0] W: {my_w.tolist()}")

    # 4. 判断
    if torch.allclose(base_w, my_w, atol=1e-5):
        print("✅ 奇迹：权重居然是一样的？（那只能是输入图片预处理错了）")
    else:
        print("❌ 确诊：Vision Encoder 权重不匹配！你在用随机权重跑模型。")

    base_w = base_model.model.layers[0].self_attn.q_proj.weight.data.flatten()[:5]

    # 3. 提取 你的模型 第0层 Block 的 Attention 权重
    my_w = model.model.layers[0].self_attn.q_proj.weight.data.flatten()[:5]

    print(f"Base Block[0] W: {base_w.tolist()}")
    print(f"My   Block[0] W: {my_w.tolist()}")

    # 4. 判断
    if torch.allclose(base_w, my_w, atol=1e-5):
        print("✅ 奇迹：权重居然是一样的？（那只能是输入图片预处理错了）")
    else:
        print("❌ 确诊：Vision Encoder 权重不匹配！你在用随机权重跑模型。")
    
    del base_model
    for param in model.parameters():
        param.requires_grad = False

    
    # 5. 配置LoRA (只对LLM的attention和MLP层)
    if lora_enable:
        print(f"\n配置LoRA (r={lora_r}, alpha={lora_alpha})...")

        # lora_config = LoraConfig(
        #     task_type=TaskType.CAUSAL_LM,
        #     r=lora_r,
        #     lora_alpha=lora_alpha,
        #     lora_dropout=0.1,
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
        logger.info(f"lora type is {args.lora_type}")
        lora_config = LORA_CONFIG[args.lora_type]
        # lora_config = LoraConfig(
        #         task_type=TaskType.CAUSAL_LM,
        #         r=8,
        #         lora_alpha=16,
        #         lora_dropout=0.1,
        #         target_modules=[
        #             # --- 1. 语言模型部分 (LLM) ---
        #             # 对应 model.layers 下的 attention 和 mlp
        #             "q_proj", "k_proj", "v_proj", "o_proj", 
        #             "gate_proj", "up_proj", "down_proj",
                    
        #             # --- 2. 视觉模型部分 (Visual) ---
        #             # 对应 visual.blocks 下的 attention (输入是 qkv)
        #             "qkv", 
        #             # 对应 visual.blocks 下的 attention output
        #             # 使用 "attn.proj" 可以精准匹配线性层，避开 patch_embed 下的 Conv3d "proj"
        #             "attn.proj",  
                    
        #             # 视觉部分的 MLP 层名字和 LLM 一样 (gate_proj 等)，
        #             # 所以上面的 "gate_proj" 等配置会自动覆盖视觉部分的 MLP，不需要重复写
        #         ],
        #         # --- 3. Merger 层全量微调 ---
        #         # 你的结构树显示名字是 visual.merger，这里配置正确
        #         modules_to_save=["visual.merger"],
        #         bias="none",
        #     )
        
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
    else :
        print("跳过lora")

    # 6. 冻结视觉编码器 (visual.*)
    print(f"\n设置训练参数...")
    # for name, param in model.named_parameters():
    #     if 'visual' in name and 'position_3d' not in name:
    #         param.requires_grad = False
    
    # 7. 确保3D模块可训练
    # if enable_3d:
    #     for name, param in model.named_parameters():
    #         if 'position_3d_encoder' in name:
    #             param.requires_grad = True
    
    # 8. 打印可训练参数统计
    if lora_enable:
        model.print_trainable_parameters()
    else :
        total_params = sum(
            p.numel() for n, p in model.named_parameters() 
            if p.requires_grad
        )
        print(f"\n总可训练参数: {total_params:,} ({total_params/1e6:.2f}M)")

    # # 额外统计3D模块的参数
    # if enable_3d:
    #     total_3d_params = sum(
    #         p.numel() for n, p in model.named_parameters() 
    #         if 'position_3d_encoder' in n and p.requires_grad
    #     )
    #     print(f"\n3D模块可训练参数: {total_3d_params:,} ({total_3d_params/1e6:.2f}M)")
    if args.local_rank == 0 or args.local_rank == -1:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

        # 方法3：统计可训练参数数量
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        if all_params > 0:
            ratio = trainable_params / all_params
            logger.info(f"Total parameters : {all_params:,}, trainable parameters: {trainable_params:,}, 占比 {ratio:.2%}")
        else:
            logger.info(f"Total trainable parameters: {trainable_params:,}")
    
    # 9. 启用梯度检查点
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    
    print(f"{'='*60}")
    print(f"✓ 模型配置完成!")
    print(f"{'='*60}\n")
    
    return model

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


def train_with_3d(
    model,
    model_name: str,
    train_dataset: Dataset,
    output_dir: str = "./qwen_vl_3d_output",
    num_epochs: int = 3,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    max_length: int = 2048,
    deepspeed_stage: int = 2,
    enable_3d: bool = True,
    args= None
):
    """训练支持3D的模型"""
    
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    # 创建DeepSpeed配置
    if deepspeed_stage == 2:
        ds_config = {
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": deepspeed_stage,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "overlap_comm": True,
            "contiguous_gradients": True
        },
        "gradient_accumulation_steps": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto"
        }
    elif deepspeed_stage ==3:
        ds_config = {
                "bf16": {"enabled": True},
                "zero_optimization": {
                    "stage": deepspeed_stage,
                    "offload_optimizer": {"device": "cpu", "pin_memory": True},
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "stage3_gather_16bit_weights_on_model_save": True
                },
                "gradient_accumulation_steps": "auto",
                "train_batch_size": "auto",
                "train_micro_batch_size_per_gpu": "auto"
            }
    # ds_config = {
    #     "bf16": {"enabled": True}}
    
    ds_config_path = os.path.join(output_dir, "ds_config.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(ds_config_path, 'w') as f:
        json.dump(ds_config, f, indent=2)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        seed=42,
        data_seed=42,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=500,
        save_total_limit=1,
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
        label_names=["labels"]  # 👈 关键
    )
    
    new_trainer = False
    if new_trainer:
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=MultimodalDataCollatorWith3D(
                processor=processor,
                max_length=max_length,
                enable_3d=enable_3d,
                args=args
            )
        )
    else:
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=MultimodalDataCollatorWith3D(
                processor=processor,
                max_length=max_length,
                enable_3d=enable_3d,
                args=args
            )
        )
    
    print("=" * 60)
    print(f"开始训练 (3D={'启用' if enable_3d else '禁用'})")
    print(f"数据集大小: {len(train_dataset)}")
    print(f"总batch size: {per_device_batch_size * gradient_accumulation_steps * torch.cuda.device_count()}")
    print("=" * 60)
    
    trainer.train()
    
    # # 保存
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    
    # print(f"\n✓ 模型已保存到: {output_dir}")
    # ===== 修改后的保存逻辑 =====
    print("保存模型...")


def train_with_3d_eval(
    model,
    model_name: str,
    train_dataset: Dataset,
    output_dir: str = "./qwen_vl_3d_output",
    num_epochs: int = 3,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    max_length: int = 2048,
    deepspeed_stage: int = 2,
    enable_3d: bool = True,
    args= None
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
        warmup_ratio=0.1,
        logging_steps=10,
        
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
    data_collator = MultimodalDataCollatorWith3D(
                processor=processor,
                max_length=max_length,
                enable_3d=enable_3d,
                args=args
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





import random
import numpy as np
import torch

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 保证cudnn算法的可复现性 (会牺牲一点速度)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==================== 主函数 ====================

if __name__ == "__main__":
    
    set_global_seed(42)
    
    parser = argparse.ArgumentParser(description="Qwen2.5-VL + 3D Position Encoding 训练")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--video_mapping", type=str, required=True,
                       help="处理后的mapping文件 (来自convert_acb_data.py的输出)")
    parser.add_argument("--output_dir", type=str, default="./qwen_vl_3d_output")
    parser.add_argument("--lora_type", type=str, default="merger_save_all_lora")

    parser.add_argument("--norm_type", type=str, default="norm", help="depth to 3d norm type")

    parser.add_argument("--grid_n", type=int, default=3)
    parser.add_argument("--num_3d_freqs", type=int, default=10)


    parser.add_argument("--system_prompt", type=str, 
                       default=None,
                       help="系统提示词")
    parser.add_argument("--model_name", type=str, default="models/Qwen2.5-VL-7B-Instruct")

    parser.add_argument("--eval", type=str, default="False", help="是否启用eval模式")
    
    # 3D相关参数
    parser.add_argument("--enable_3d", type=str, default="True", help="启用3D位置编码")
    parser.add_argument("--merge_type", type=str, default="direct_add",
                       help="3d embedding 的融合方式")
    parser.add_argument("--gate_mode", type=str, default="softplus_mean",
                       choices=["softplus_mean", "sigmoid_max", "raw_sigmoid"],
                       help="Gate 计算模式: softplus_mean(旧), sigmoid_max(推荐新), raw_sigmoid(无softmax)")

    parser.add_argument("--coords_type", type=str, default="none",
                       help="3d坐标的消融类型, ['shuffled', 'noisy']")
    parser.add_argument("--sigma", type=float, default=0, help="noisy 的误差sigma系数")
    
    parser.add_argument("--is_w2c", type=str, default="False",
                       help="外参是否是w2c矩阵")
    parser.add_argument("--type_3d", type=str, default="sincos",
                       help="3d坐标的编码方式")
    parser.add_argument("--vision_output_layer", type=str, default="visual.merger",
                       help="Hook层,默认visual.merger")
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--fixed_image_size", type=int, nargs=2, default=[224, 224],
                       help="固定图像尺寸 (H W)")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda_sparse", type=float, default=0)
    parser.add_argument("--lora_enable", action="store_true", help="启用lora微调")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--deepspeed_stage", type=int, default=2)
    parser.add_argument("--local_rank", type=int, default=-1)
    
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
    
    # 1. 准备数据
    print("\n[1/3] 加载数据集...")
    train_dataset = prepare_multimodal_dataset_with_3d(
        json_path=args.train_data,
        video_mapping_path=args.video_mapping,
        system_prompt=args.system_prompt,
        patch_size=args.patch_size,
        fixed_image_size=tuple(args.fixed_image_size)
    )
    
    # 2. 设置模型
    print("\n[2/3] 加载模型...")
    model = setup_model_with_3d_and_lora(
        model_name=args.model_name,
        enable_3d=args.enable_3d,
        lora_enable=args.lora_enable,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        args= args
    )
    
    enable_3d = False
    if args.enable_3d == "True":
        enable_3d = True

    # 3. 训练
    print("\n[3/3] 开始训练...")
    if args.eval == 'True':
        print("-"*10 +"使用eval训练中验证，最后保存最优模型 "+"-"*10 )
        trainer = train_with_3d_eval(
            model=model,
            model_name=args.model_name,
            train_dataset=train_dataset,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            per_device_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.lr,
            max_length=args.max_length,
            deepspeed_stage=args.deepspeed_stage,
            enable_3d=enable_3d,
            args=args
        )
    else:
        trainer = train_with_3d(
            model=model,
            model_name=args.model_name,
            train_dataset=train_dataset,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            per_device_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.lr,
            max_length=args.max_length,
            deepspeed_stage=args.deepspeed_stage,
            enable_3d=enable_3d,
            args=args
        )
    
    print("\n✓ 训练完成!")
    print(f"  - 模型路径: {args.output_dir}")
    print(f"  - 3D编码: {'已启用' if args.enable_3d else '未启用'}")
    if args.vision_output_layer:
        print(f"  - Hook层: {args.vision_output_layer}")