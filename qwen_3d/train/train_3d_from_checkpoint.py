"""
Qwen2.5-VL + 3D Position Encoding 完整实现
方案B: 修改模型类,固定分辨率,保持patch-pixel对应关系
包含：场景二（加载已有 LoRA 权重作为新起点继续训练）
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
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
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

from safetensors.torch import load_file

import gc

logger = logging.getLogger(__name__)

global local_rank


def load_camera_params(video_mapping_path: str) -> Dict:
    """
    加载相机参数 (内参+外参)
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
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

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
        
        image_paths = video_data['image_paths']
        depth_paths = video_data['depth_paths']
        
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
        
        processed_item = {
            "video_id": video_id,
            "image_paths": image_paths,      
            "depth_paths": depth_paths,      
            "intrinsics": video_data['intrinsics'],  
            "extrinsics": video_data['extrinsics'],  
            "conversations": convert_conversations(item['conversations'], system_prompt),
            "num_views": len(image_paths),
            "patch_size": patch_size,
            "fixed_image_size": fixed_image_size
        }
        
        processed_data.append(processed_item)
        
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
    processor: AutoProcessor
    max_length: int = 4096
    enable_3d: bool = True
    args: Optional[argparse.Namespace] = None     
    
    def __post_init__(self):
        if self.enable_3d:
            self._coord_generator = None
            
    def _get_or_create_coord_generator(self, patch_size: int, fixed_image_size: Tuple[int, int]):
        if self._coord_generator is None:
            is_w2c = False
            if self.args.is_w2c == "True":
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
        coord_gen = self._get_or_create_coord_generator(
            patch_size=item['patch_size'],
            fixed_image_size=tuple(item['fixed_image_size'])
        )
        
        depth_list = item['depth_paths']
        rgb_list = item['image_paths']

        intrinsics = [np.array(K, dtype=np.float32) for K in item['intrinsics']]
        extrinsics = [np.array(T, dtype=np.float32) for T in item['extrinsics']]
        
        coords_3d, resized_rgbs = coord_gen.process_multi_view(
            depth_list=depth_list,
            intrinsic_list=intrinsics,
            extrinsic_list=extrinsics,
            rgb_list=rgb_list
        )

        coords_3d = torch.as_tensor(coords_3d)

        if self.args.coords_type == "shuffled":
            num_views = coords_3d.shape[0] // 256
            coords_3d = self.make_shuffled_3d(coords_3d, num_views)
        elif self.args.coords_type == "shuffled_all":
            coords_3d = self.make_global_shuffled_3d(coords_3d)
        elif self.args.coords_type == "noisy":
            coords_3d = self.make_noisy_3d(coords_3d, self.args.sigma)
        
        return resized_rgbs, coords_3d
    
    def __call__(self, features: List[Dict]) -> Dict:
        batch_messages = []
        batch_coords_3d = []
        load_errors = []
        
        for idx, item in enumerate(features):
            try:
                if self.enable_3d:
                    resized_rgbs, coords_3d = self._load_and_compute_3d(item)
                    batch_coords_3d.append(torch.as_tensor(coords_3d).to(dtype=torch.bfloat16))
                else:
                    resized_rgbs = []
                    for rgb_path in item['image_paths']:
                        rgb = Image.open(rgb_path).convert('RGB')
                        rgb = rgb.resize(
                            (item['fixed_image_size'][0], item['fixed_image_size'][1])
                        )
                        resized_rgbs.append(rgb)
                
                images = resized_rgbs
                messages = self.convert_to_official_format(item['conversations'], images)
                batch_messages.append(messages)
                
            except Exception as e:
                traceback.print_exc()
                load_errors.append((idx, item.get('video_id', 'unknown'), str(e)))
                batch_messages.append([
                    {"role": "user", "content": "Error loading data"},
                    {"role": "assistant", "content": "Error"}
                ])
                if self.enable_3d:
                    batch_coords_3d.append(torch.zeros(1, 3))
        
        if load_errors:
            print(f"\n⚠️  Batch中 {len(load_errors)} 个样本加载失败:")
            for idx, video_id, error in load_errors[:3]:
                print(f"  [{idx}] {video_id}: {error}")
        
        batch_question_texts = []
        
        for item in batch_messages:
            last_user_msg = ""
            for msg in item:
                if msg['role'] == 'user':
                    content = msg['content']
                    if isinstance(content, str):
                        last_user_msg = content
                    elif isinstance(content, list):
                        last_user_msg = "".join([x['text'] for x in content if x['type'] == 'text'])
            
            q_text = self._extract_question_from_text(last_user_msg)
            batch_question_texts.append(q_text)

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
        
        inputs = self.processor(
            text=texts,
            images=all_images if all_images else None,
            videos=all_videos if all_videos else None,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        )
        
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()
        
        tokenizer = self.processor.tokenizer
        pad_id = tokenizer.pad_token_id
        image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        
        ignore_index = -100
        new_labels = torch.full_like(labels, ignore_index)
        
        for i in range(input_ids.shape[0]):
            ids = input_ids[i]
            current_label = labels[i].clone()
            
            current_label[current_label == image_pad_id] = ignore_index
            current_label[current_label == tokenizer.convert_tokens_to_ids("<|vision_start|>")] = ignore_index
            current_label[current_label == tokenizer.convert_tokens_to_ids("<|vision_end|>")] = ignore_index
            current_label[current_label == pad_id] = ignore_index
            
            im_start_indices = (ids == im_start_id).nonzero(as_tuple=True)[0]
            
            if len(im_start_indices) > 0:
                last_start_idx = im_start_indices[-1]
                current_label[:last_start_idx+2] = ignore_index 
            
            new_labels[i] = current_label

        inputs["labels"] = new_labels
        
        if self.enable_3d and batch_coords_3d:
            max_patches = max(c.shape[0] for c in batch_coords_3d)
            padded_coords = []
            coords_mask = torch.zeros(len(batch_coords_3d), max_patches)
            
            for i, coords in enumerate(batch_coords_3d):
                coords_mask[i, :coords.shape[0]] = 1
                pad_size = max_patches - coords.shape[0]
                if pad_size > 0:
                    coords = F.pad(coords, (0, 0, 0, pad_size), value=0)
                padded_coords.append(coords)
            
            inputs["coords_3d"] = torch.stack(padded_coords)
            inputs["coords_3d_mask"] = coords_mask
            
            q_encodings = self.processor.tokenizer(
                batch_question_texts,
                padding=True,
                truncation=True,
                max_length=128, 
                return_tensors="pt",
                add_special_tokens=False 
            )
            
            inputs["question_input_ids"] = q_encodings["input_ids"]
            
        return inputs
    
    def convert_to_official_format(self, conversations, images):
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
                    parts = []
                    segments = content.split('<image>')
                    
                    for i, segment in enumerate(segments):
                        if segment:
                            parts.append({"type": "text", "text": segment})
                        
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
        import re
        opt_pattern = r"\n\s*[A][\.\)]"
        opt_match = re.search(opt_pattern, full_text)
        
        if opt_match:
            end_idx = opt_match.start()
            text_before_opt = full_text[:end_idx]
            last_newline = text_before_opt.rfind('\n')
            
            if last_newline != -1:
                question = text_before_opt[last_newline+1:].strip()
            else:
                question = text_before_opt.strip()
        else:
            lines = full_text.strip().split('\n')
            if len(lines) > 0:
                question = lines[-1].strip()
            else:
                question = full_text.strip()
                
        if len(question) < 3:
            question = full_text[-50:] 
            
        return question
    

    def make_shuffled_3d(self, coords_3d, num_views):
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
        N, C = coords_3d.shape
        global_perm = torch.randperm(N, device=coords_3d.device)
        coords_shuffled = coords_3d[global_perm]
        return coords_shuffled

    def make_noisy_3d(self, coords_3d, sigma=0.05):
        scale = coords_3d.std()
        noise = torch.randn_like(coords_3d) * (sigma * scale)
        coords_noisy = coords_3d + noise
        return coords_noisy
    
# ==================== 模型设置与训练 ====================

class CustomTrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        model = self.model
        
        base_lr = self.args.learning_rate
        merger_lr = 10 * base_lr
        
        no_decay = ["bias", "LayerNorm.weight"]
        
        merger_decay_params = []
        merger_no_decay_params = []
        base_decay_params = []
        base_no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            is_merger = "merger" in name
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
    from transformers import AutoConfig
    
    print(f"\n{'='*60}")
    print(f"正在加载 Qwen2.5-VL + 3D Position Encoding")
    print(f"{'='*60}")
    
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    print(f"✓ 配置加载完成: hidden_size={config.vision_config.hidden_size}")
    
    model = Qwen2_5_VLWith3D(config, args=args)
    model.to(torch.bfloat16)
    
    print(f"正在加载预训练权重...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"":"cpu"}
    )

    base_state_dict = base_model.state_dict()
    if args.enable_3d:
        new_state_dict = {}
        for k, v in base_state_dict.items():
            if k.startswith("visual.merger."):
                new_k = k.replace("visual.merger.", "visual.merger.original_merger.")
                new_state_dict[new_k] = v
            else:
                new_state_dict[k] = v
    else:
        new_state_dict=base_state_dict
        
    ret = model.load_state_dict(new_state_dict, strict=False)
    
    print(f"✓ 预训练权重加载完成--ok-pretrained_load")
    print("Missing keys:", ret.missing_keys)
    print("Unexpected keys:", ret.unexpected_keys)

    # === 修改点 2：暴力清理内存和显存 ===
    del base_model
    del base_state_dict  # <--- 删掉旧字典
    del new_state_dict   # <--- 删掉新字典

    # gc.collect()         # <--- 强制 Python 回收内存
    # torch.cuda.empty_cache() # <--- 强制 PyTorch 清空显存碎片
    # ====================================
    for param in model.parameters():
        param.requires_grad = False
    
    # 5. 配置LoRA (支持场景二: 从已有 LoRA 权重继续训练)
    if lora_enable:
        print(f"\n配置LoRA (r={lora_r}, alpha={lora_alpha})...")
        
        # === 核心修改点：检测是否传入了已有的 LoRA 权重路径 ===
        if hasattr(args, 'pretrained_lora_path') and args.pretrained_lora_path:
            print(f"==================================================")
            print(f"🚀 [场景二] 检测到已有 LoRA 权重，正在加载作为新起点: from old checkpoints as new start")
            print(f"路径: {args.pretrained_lora_path}")
            print(f"==================================================")
            
            # 使用 PeftModel.from_pretrained 加载已有权重，并将其设为可训练状态
            model = PeftModel.from_pretrained(model, args.pretrained_lora_path, is_trainable=True,torch_device="cpu")
            

            # print("\n" + "="*50)
            # print("开始验证 3D Position Encoder 权重加载情况...")
            # try:
            #     # 核心特征字符串
            #     target_fragment = "position_3d_encoder.coord_projector.0.weight"
                
            #     # 1. 从当前显存模型中提取权重
            #     mem_tensor = None
            #     mem_full_name = ""
            #     for name, param in model.named_parameters():
            #         if target_fragment in name:
            #             mem_tensor = param.data.flatten()[:5]
            #             mem_full_name = name
            #             break
                        
            #     if mem_tensor is None:
            #         print(f"⚠️ [内存] 未找到包含 '{target_fragment}' 的参数！请检查模型结构。")
            #     else:
            #         print(f"🔍 [内存] 找到参数: {mem_full_name}")
            #         print(f"数值前 5 位: {mem_tensor.tolist()}")
                    
            #         # 2. 从磁盘 (safetensors) 读取权重并对比
            #         if hasattr(args, 'pretrained_lora_path') and args.pretrained_lora_path:

            #             from safetensors.torch import load_file
                        
            #             safetensors_path = os.path.join(args.pretrained_lora_path, "adapter_model.safetensors")
                        
            #             if os.path.exists(safetensors_path):
            #                 saved_state_dict = load_file(safetensors_path)
                            
            #                 disk_tensor = None
            #                 disk_key = ""
            #                 for k in saved_state_dict.keys():
            #                     if target_fragment in k:
            #                         disk_key = k
            #                         # 读取前5位，并转换到与内存张量相同的设备和精度
            #                         disk_tensor = saved_state_dict[k].flatten()[:5].to(
            #                             device=mem_tensor.device, 
            #                             dtype=mem_tensor.dtype
            #                         )
            #                         break
                                    
            #                 if disk_tensor is not None:
            #                     print(f"📦 [磁盘] 找到参数: {disk_key}")
            #                     print(f"数值前 5 位: {disk_tensor.tolist()}")
                                
            #                     # 3. 严格对比
            #                     if torch.allclose(disk_tensor, mem_tensor, atol=1e-5):
            #                         print("✅ 完美匹配！结论：3D 坐标投影层 (coord_projector) 权重加载成功。")
            #                     else:
            #                         print("❌ 严重警告：匹配失败！当前模型加载的数值与磁盘不一致。")
            #                 else:
            #                     print(f"⚠️ [磁盘] 在 {safetensors_path} 中未找到包含该特征词的 key。")
            #                     print(f"当前文件包含的 keys 示例: {list(saved_state_dict.keys())[:5]}")
            #             else:
            #                 print(f"⚠️ 找不到文件: {safetensors_path}")
            #         else:
            #             print("⚠️ 未提供 --pretrained_lora_path，跳过磁盘对比。")
                        
            # except Exception as e:
            #     print(f"检查权重时发生错误: {e}")
            # print("="*50 + "\n")
            
        else:
            from qwen_3d.config.lora_config import LORA_CONFIG
            logger.info(f"lora type is {args.lora_type}")
            lora_config = LORA_CONFIG[args.lora_type]
            model = get_peft_model(model, lora_config)
            
        # ================= 修复开始 =================
        from peft.utils.other import ModulesToSaveWrapper
        
        if hasattr(model, "visual"):
            if isinstance(model.visual, ModulesToSaveWrapper):
                if hasattr(model.visual.original_module, "dtype"):
                    model.visual.dtype = model.visual.original_module.dtype
                else:
                    model.visual.dtype = torch.bfloat16 
                    
                print(f"已修复 visual 模块的 dtype: {model.visual.dtype}")
                
        elif hasattr(model.base_model.model, "visual"):
            if isinstance(model.base_model.model.visual, ModulesToSaveWrapper):
                model.base_model.model.visual.dtype = model.base_model.model.visual.original_module.dtype
                print(f"已修复 base_model.visual 模块的 dtype")
        # ================= 修复结束 =================

        # 检查 merger 中随便一个层的权重前5位
        # 注意：PEFT 会把 modules_to_save 挂在 modules_to_save.default 下面
        try:
            sample_weight = model.base_model.model.visual.merger.modules_to_save.default.original_merger.mlp[0].weight.data.flatten()[:5]
            print(f"🔍 检查 Merger 加载后的权重: {sample_weight.tolist()}")
        except Exception as e:
            pass
    else :
        print("跳过lora")

    print(f"\n设置训练参数...")
    
    if lora_enable:
        model.print_trainable_parameters()
    else :
        total_params = sum(
            p.numel() for n, p in model.named_parameters() 
            if p.requires_grad
        )
        print(f"\n总可训练参数: {total_params:,} ({total_params/1e6:.2f}M)")

    if args.local_rank == 0 or args.local_rank == -1:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        if all_params > 0:
            ratio = trainable_params / all_params
            logger.info(f"Total parameters : {all_params:,}, trainable parameters: {trainable_params:,}, 占比 {ratio:.2%}")
        else:
            logger.info(f"Total trainable parameters: {trainable_params:,}")
    
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    
    print(f"{'='*60}")
    print(f"✓ 模型配置完成!")
    print(f"{'='*60}\n")
    
    return model

def create_deepspeed_config(output_dir: str, stage: int = 2) -> str:
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
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
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
    
    ds_config_path = os.path.join(output_dir, "ds_config.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(ds_config_path, 'w') as f:
        json.dump(ds_config, f, indent=2)
    
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
        label_names=["labels"] 
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
    
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    
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
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    ds_config_path = create_deepspeed_config(output_dir, stage=deepspeed_stage)

    dataset_len = len(train_dataset)
    eval_len = int(dataset_len * 0.1)
    train_len = dataset_len - eval_len

    final_train_dataset, final_eval_dataset = random_split(
        train_dataset, 
        [train_len, eval_len], 
        generator=torch.Generator().manual_seed(42)
    )

    print(f"原始数据集: {dataset_len}")
    print(f"切分后 - 训练集: {len(final_train_dataset)}, 验证集: {len(final_eval_dataset)}")

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
        
        evaluation_strategy="steps", 
        per_device_eval_batch_size=per_device_batch_size, 
        eval_steps=500,
        
        save_strategy="steps",
        save_steps=500, 
        
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,

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
        label_names=["labels"]
    )

    data_collator = MultimodalDataCollatorWith3D(
                processor=processor,
                max_length=max_length,
                enable_3d=enable_3d,
                args=args
            )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=final_train_dataset, 
        eval_dataset=final_eval_dataset,   
        data_collator=data_collator
    )

    print("=" * 60)
    print(f"训练集大小: {len(final_train_dataset)}")
    print(f"验证集大小: {len(final_eval_dataset)}")
    print(f"总batch size: {per_device_batch_size * gradient_accumulation_steps * torch.cuda.device_count()}")
    print("=" * 60)

    trainer.train()

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
    
    # === 新增：用于场景二的参数，接收之前训练好的 LoRA 权重路径 ===
    parser.add_argument("--pretrained_lora_path", type=str, default=None,
                        help="预训练的LoRA权重路径，用于基于已有LoRA继续新一阶段的训练")
    
    args = parser.parse_args()


    os.makedirs(args.output_dir, exist_ok=True)

    if args.local_rank == 0 or args.local_rank == -1:
        for h in logging.root.handlers[:]:
            logging.root.removeHandler(h)
            
        logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(f'{args.output_dir}/model_log.log', encoding='utf-8', mode='w')
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