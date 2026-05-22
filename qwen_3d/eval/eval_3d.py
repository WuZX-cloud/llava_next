"""
Qwen2.5-VL + 3D Position Encoding 推理脚本
支持单样本推理、批量推理、可视化
"""

from ctypes import resize
from email import message
import torch
import numpy as np
from PIL import Image
import json
import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import argparse
import traceback

from transformers import AutoProcessor, AutoConfig
from peft import PeftModel

from qwen_3d.model.model import Qwen2_5_VLWith3D
from qwen_3d.model.threeD import DepthTo3DCoordinates


from transformers import (
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info



class Qwen3DInference:
    """
    3D增强的Qwen2.5-VL推理器
    """
    def __init__(
        self,
        model_path: str,
        base_model_path: Optional[str] = None,
        device: str = "cuda",
        enable_3d: bool = True,
        lora_enable: bool = False,  # <--- 新增参数
        args=None
    ):
        """
        Args:
            model_path: 训练后的模型路径
            base_model_path: 基础模型路径
            device: 运行设备
            enable_3d: 是否启用3D编码
            lora_enable: 是否加载LoRA权重
        """
        self.device = device

        self.enable_3d = False
        enable_3d=False
        if args.enable_3d == "True":
            self.enable_3d = True
            enable_3d = True

        if args.sampled == "True":
            self.sampled = True
        else:
            self.sampled=False

        self.processor = None
        self.model = None
        
        print(f"\n{'='*60}")
        print(f"加载 Qwen2.5-VL + 3D 推理模型")
        print(f"模式: {'LoRA微调' if lora_enable else '全量微调 (Full Finetune)'}")
        print(f"{'='*60}")
        
        # 1. 加载processor
        print(f"加载 Processor: {model_path}")
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print(f"✓ Processor 加载完成")

        # ---------------------------------------------------------
        # 情况一：全量微调 (直接加载整个模型)
        # ---------------------------------------------------------
        if not lora_enable:
            print(f"正在加载全量微调模型 (from_pretrained)...")
            try:
                self.model = Qwen2_5_VLWith3D.from_pretrained(
                    model_path,              # 直接指向包含 model.safetensors 的目录
                    enable_3d=enable_3d,     # 确保结构正确初始化
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map="cpu"         # 先加载到CPU
                )
                print(f"✓ 全量模型加载成功")
            except Exception as e:
                print(f"❌ 全量加载失败，请检查路径下是否有完整权重文件。错误: {e}")
                raise e

        # ---------------------------------------------------------
        # 情况二：LoRA + Base Model + 手动补丁 (保留你原有的逻辑)
        # ---------------------------------------------------------
        else:
            print(f"正在以 LoRA 模式加载...")
            
            print(f"  - 基础模型: {base_model_path}")
            
            # 1. 实例化结构
            config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
            self.model = Qwen2_5_VLWith3D(config, args=args)
            # 开启record
            if args.record == "True":
                self.model.visual.merger.position_3d_encoder.record = True

            self.model.to(torch.bfloat16)

            # 2. 加载基础权重
            print("  - 加载 Base Model 权重...")
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            self.model.load_state_dict(base_model.state_dict(), strict=False)
            del base_model

            # 3. 加载LoRA权重
            print(f"  - 加载 LoRA 权重...")
            self.model = PeftModel.from_pretrained(self.model, model_path)

            # ================= 修复开始 =================
            # 解决 ModulesToSaveWrapper 丢失 dtype 属性导致的报错
            # 检查 visual 是否被 wrap 成了 ModulesToSaveWrapper
            from peft.utils.other import ModulesToSaveWrapper
            
            # 获取模型中的 visual 模块（根据你的模型层级可能略有不同，通常是 model.visual 或 model.base_model.model.visual）
            # Qwen2.5-VL 在 PEFT 模型中通常可以通过 model.visual 直接访问到（经过了层层转发）
            model = self.model
            
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

            self.model = model
            # ================= 修复结束 =================
            

        # ---------------------------------------------------------
        # 通用后续处理
        # ---------------------------------------------------------
        self.model.eval()
        self.model.to(device)
        
        print(f"✓ 模型加载完成")
        print(f"{'='*60}\n")
        

        # 初始化3D坐标生成器
        if enable_3d:
            is_w2c = False
            if args.is_w2c == "True":
                is_w2c=True
            self.coord_generator = DepthTo3DCoordinates(
                patch_size=14,
                fixed_image_size=args.fixed_image_size,
                type=args.norm_type,
                grid_n=args.grid_n,
                is_w2c=is_w2c
            )
    
    def load_and_compute_3d(
        self,
        rgb_paths: List[str],
        depth_paths: List[str],
        intrinsics: List[np.ndarray],
        extrinsics: List[np.ndarray]
    ) -> Tuple[List[Image.Image], Optional[torch.Tensor]]:
        """
        加载图像并计算3D坐标
        """
        if not self.enable_3d:
            # 不启用3D,只加载RGB
            images = [Image.open(p).convert('RGB') for p in rgb_paths]
            return images, None
        
        # 加载RGB和深度
        rgb_list = []
        depth_list = []
        
        for rgb_path, depth_path in zip(rgb_paths, depth_paths):
            rgb = np.array(Image.open(rgb_path).convert('RGB'))
            rgb_list.append(rgb)
            
            if depth_path.endswith('.npy'):
                depth = np.load(depth_path)
            else:
                depth = np.array(Image.open(depth_path))
                if depth.dtype == np.uint16:
                    depth = depth.astype(np.float32) / 1000.0
            
            depth_list.append(depth)
        
        # 生成3D坐标
        coords_3d, resized_rgbs = self.coord_generator.process_multi_view(
            depth_list=depth_list,
            intrinsic_list=intrinsics,
            extrinsic_list=extrinsics,
            rgb_list=rgb_list
        )
        
        # 转换为PIL图像
        images = resized_rgbs
        coords_3d_tensor = torch.from_numpy(coords_3d).float()
        
        return images, coords_3d_tensor
    

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

    def save_gate_heatmaps_from_tensor(
        self,
        gate_tensor,          # 改动这里：直接传 tensor
        original_images, 
        output_dir="vis_results", 
        method='mean'
    ):
        # 移除原来获取 model.cached_gate 的代码
        # gate_data = model.cached_gate.squeeze(0) 
        
        # 直接使用传入的 tensor
        gate_data = gate_tensor.squeeze(0).float()
        total_tokens, C = gate_data.shape
        
        # 2. 基础参数检查
        num_imgs = len(original_images)
        tokens_per_img = 256 # 固定值
        grid_size = 16       # sqrt(256)
        
        # 校验：Token总数是否等于 图片数 * 256
        assert total_tokens == num_imgs * tokens_per_img, \
            f"数据不匹配！Gate有 {total_tokens} 个Token，但传入了 {num_imgs} 张图 (应为 {num_imgs * tokens_per_img})。"

        # 3. 计算激活值强度 (Aggregation) -> [Total_N]
        if method == 'l2':
            # 计算 L2 范数 (能量大小)
            activations = torch.norm(gate_data, p=2, dim=1).numpy()
        else:
            # 计算平均值 (默认)
            activations = torch.mean(gate_data, dim=1).numpy()
            
        gate=activations
        print(f"Gate Stats: Min={gate.min().item():.4f}, Max={gate.max().item():.4f}, Mean={gate.mean().item():.4f}")

        # 4. 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        print(f"开始处理 {num_imgs} 张图片的热力图...")

        # 5. 循环处理每一张图
        for idx in range(num_imgs):
            # --- A. 切片：拿到当前这张图对应的 256 个 gate 值 ---
            start_pos = idx * tokens_per_img
            end_pos = (idx + 1) * tokens_per_img
            
            # 取出 256 个值
            token_vals = activations[start_pos : end_pos] 
            
            # --- B. 重塑：变成 16x16 的网格 ---
            heatmap = token_vals.reshape(grid_size, grid_size)
            
            # --- C. 归一化：为了画图好看，把数值拉伸到 0-255 ---
            # 减去最小值，除以最大值，这样最亮的地方就是红色，最暗是蓝色
            min_v, max_v = heatmap.min(), heatmap.max()
            if max_v - min_v > 1e-8:
                heatmap_norm = (heatmap - min_v) / (max_v - min_v)
            else:
                heatmap_norm = heatmap # 避免除以0
                
            heatmap_uint8 = np.uint8(255 * heatmap_norm)

            # --- D. 放大：从 16x16 插值放大到 224x224 ---
            # cv2.INTER_CUBIC 可以让热力图看起来圆润平滑，不像马赛克
            heatmap_resized = cv2.resize(heatmap_uint8, (224, 224), interpolation=cv2.INTER_CUBIC)

            # --- E. 伪彩色映射 (Heatmap) ---
            # COLORMAP_JET: 蓝(低) -> 青 -> 黄 -> 红(高)
            heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

            # --- F. 叠加到原图 ---
            orig_img = original_images[idx]
            # 确保原图是 uint8 格式
            if orig_img.dtype != np.uint8:
                orig_img = np.array(orig_img, dtype=np.uint8)
                
            # 0.6 是原图权重，0.4 是热力图权重
            overlay = cv2.addWeighted(orig_img, 1.0, heatmap_color, 0.3, 0)

            # --- G. 保存 ---
            save_path = os.path.join(output_dir, f"gate_vis_{idx}.jpg")
            cv2.imwrite(save_path, overlay)
            
        print(f"处理完成，结果保存在: {output_dir}")


    @torch.no_grad()
    def generate(
        self,
        images: List[Image.Image],
        prompt: str,
        coords_3d: Optional[torch.Tensor] = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.5,
        top_p: float = 1.0,
        do_sample: bool = False,
        groud_truth=None
    ) -> str:
        """
        生成回复
        """
        
        messages = self.convert_to_official_format(
                prompt, 
                images
            )
        # print('message:')
        # print(messages)

        # 收集所有的纯问题文本
        batch_question_texts = []
        batch_messages = [messages]
        
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
        
        # 应用chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # print('text:')
        # print(text)
        
        # 处理输入
        image_inputs, _ = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            return_tensors="pt",
            padding=True
        )
        
        
        
        # 添加3D坐标
        if self.enable_3d and coords_3d is not None:
            coords_3d = coords_3d.unsqueeze(0).to(self.device)  # (1, N, 3)
            inputs["coords_3d"] = coords_3d
            
            # 创建mask
            coords_mask = torch.ones(1, coords_3d.shape[1], device=self.device)
            inputs["coords_3d_mask"] = coords_mask
            # print("coords_3d is Not None")

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
        else :
            if self.enable_3d:
                print("coords_3d is None")
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 生成
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id
        )
        # print(self.processor.tokenizer.decode(inputs["input_ids"][0]))
        # print('output:')
        # print(self.processor.tokenizer.decode(outputs[0]))
        # 解码
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        # print('generated_ids:')
        # print(self.processor.tokenizer.decode(generated_ids))
        response = self.processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
        print("groud_truth:")
        print(groud_truth)
        print('response:')
        print(response)
        # exit(0)

        images_np = [np.array(img) for img in images]
        # 2. 从 context 获取刚才存下的 gate
        # 注意：Coords3DContext 是类共享状态，实例化一个新的或者用 self.model.coords_context 是一样的
        current_gate = self.model.coords_context.latest_gate
        # 3. 运行可视化
        if current_gate is not None:
            if response == groud_truth:
                a = input("output_dir:")
                print("开始可视化")
                print(prompt)
                # 这里我们稍微改一下你的 save_gate_heatmaps 函数接口，让它直接接受 tensor 而不是 model
                self.save_gate_heatmaps_from_tensor(
                    gate_tensor=current_gate,   # 直接传数据
                    original_images=images_np,
                    output_dir=f"output_heat_map/{a}",
                    # method="l2"
                    method="mean"
                )
                
                # 可选：画完后清理，防止污染下一次推理
                self.model.coords_context.clear()
                
        else :
            print("current_gate is None")

        return response.strip()

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
    

    def inference_from_paths(
        self,
        rgb_paths: List[str],
        depth_paths: Optional[List[str]],
        intrinsics: Optional[List[np.ndarray]],
        extrinsics: Optional[List[np.ndarray]],
        prompt: str,
        groud_truth=None,
        **generate_kwargs
    ) -> str:
        """
        从文件路径推理
        """
        # 加载图像和计算3D
        if self.enable_3d and depth_paths is not None:
            # print("生成coords_3d")
            images, coords_3d = self.load_and_compute_3d(
                rgb_paths, depth_paths, intrinsics, extrinsics
            )
            # print("计算了 coords_3d")
        else:
            images = [Image.open(p).convert('RGB') for p in rgb_paths]
            coords_3d = None
        
        conversations = prompt

        # 生成
        response = self.generate(images=images, prompt=conversations, do_sample=self.sampled, coords_3d=coords_3d, groud_truth=groud_truth)
        
        return response

import torch
import numpy as np
import cv2
import os



def inference_single_sample(
    model_path: str,
    rgb_paths: List[str],
    prompt: str,
    depth_paths: Optional[List[str]] = None,
    camera_params_path: Optional[str] = None,
    enable_3d: bool = True,
    base_model_path: Optional[str] = None,
    lora_enable: bool = False
):
    """
    单样本推理示例
    """
    print("\n" + "="*70)
    print("单样本推理测试")
    print("="*70)
    
    # 加载相机参数
    intrinsics = None
    extrinsics = None
    
    if enable_3d and camera_params_path:
        with open(camera_params_path, 'r') as f:
            params = json.load(f)
            intrinsics = [np.array(k, dtype=np.float32) for k in params['intrinsics']]
            extrinsics = [np.array(t, dtype=np.float32) for t in params['extrinsics']]
    
    # 初始化推理器
    
    inferencer = Qwen3DInference(
        model_path=model_path,
        base_model_path=base_model_path,
        enable_3d=enable_3d,
        lora_enable=lora_enable
    )
    
    # 推理
    print(f"\n输入:")
    print(f"  图像数: {len(rgb_paths)}")
    for i, path in enumerate(rgb_paths):
        print(f"    [{i+1}] {path}")
    print(f"  问题: {prompt}")
    
    if enable_3d and depth_paths:
        print(f"  深度图: {len(depth_paths)} 张")
        print(f"  3D编码: ✓ 启用")
    else:
        print(f"  3D编码: ✗ 禁用")
    
    print(f"\n生成中...")
    
    response = inferencer.inference_from_paths(
        rgb_paths=rgb_paths,
        depth_paths=depth_paths,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        prompt=prompt,
        max_new_tokens=4096
    )
    
    print(f"\n{'='*70}")
    print(f"模型回复:")
    print(f"{'='*70}")
    print(response)
    print(f"{'='*70}\n")
    
    return response


def inference_batch(
    model_path: str,
    test_json: str,
    video_mapping_path: str,
    output_path: str = None,
    enable_3d: bool = True,
    base_model_path: Optional[str] = None,
    limit: Optional[int] = None,
    lora_enable: bool = False,
    args=None

):
    """
    批量推理
    """
    print("\n" + "="*70)
    print("批量推理测试")
    print("="*70)
    
    # 加载数据
    with open(test_json, 'r') as f:
        test_data = json.load(f)
    
    with open(video_mapping_path, 'r') as f:
        video_mapping = json.load(f)
    
    if limit:
        # 简单过滤一下，只测试有3张图的样本(根据你的逻辑可调)
        test_data = [x for x in test_data if len(video_mapping[x["video"]]["image_paths"]) == 3]
        test_data = test_data[:limit]
        print(f"限制推理数量: {limit}")
    
    print(f"测试样本数: {len(test_data)}")

    def get_device(min_free_gb=4):
        if not torch.cuda.is_available():
            return torch.device("cpu")

        # 当前 GPU
        device_id = torch.cuda.current_device()

        # 查询显存
        free, total = torch.cuda.mem_get_info(device_id)
        free_gb = free / 1024**3
        total_gb = total / 1024**3

        print(f"GPU total: {total_gb:.2f} GB, free: {free_gb:.2f} GB")

        if free_gb >= min_free_gb:
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    device = get_device(min_free_gb=20)

    print(f"using device : {device}")
    
    # 初始化推理器
    inferencer = Qwen3DInference(
        model_path=model_path,
        base_model_path=base_model_path,
        device=device,
        enable_3d=enable_3d,
        lora_enable=lora_enable,
        args=args
    )
    
    # 批量推理
    results = []
    result2 = []
    
    for idx, item in enumerate(test_data):
        video_id = item['video']
        
        if video_id not in video_mapping:
            print(f"[{idx+1}/{len(test_data)}] 跳过 {video_id}: 不在mapping中")
            continue
        
        video_data = video_mapping[video_id]
        
        # 提取问题
        conversations = item['conversations']
        question = None
        ground_truth = None
        conversation = []

        if args.system_prompt:
            conversation.append(
                {"role": "system", "content": args.system_prompt}
            )
        
        for conv in conversations:
            if conv['from'] == 'human':
                question = conv['value']
                conversation.append(
                    {'role': 'user', 'content': question}
                )
            elif conv['from'] == 'gpt':
                ground_truth = conv['value']
        
        if not question:
            continue
        
        # 推理
        try:
            response = inferencer.inference_from_paths(
                rgb_paths=video_data['image_paths'],
                depth_paths=video_data.get('depth_paths') if enable_3d else None,
                intrinsics=[np.array(k, dtype=np.float32) for k in video_data.get('intrinsics', [])] if enable_3d else None,
                extrinsics=[np.array(t, dtype=np.float32) for t in video_data.get('extrinsics', [])] if enable_3d else None,
                prompt=conversation,
                max_new_tokens=4096,
                groud_truth=ground_truth
            )
            
            results.append({
                "video_id": video_id,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": response,
                "images": video_data['image_paths']
            })
            
            # 实时保存，防止中断
            result2.append({
                **item,
                'response': response
            })
            
            print(f"[{idx+1}/{len(test_data)}] ✓ {video_id}")
            
        except Exception as e:
            print(f"[{idx+1}/{len(test_data)}] ✗ {video_id}: {e}")
            traceback.print_exc()
            results.append({
                "video_id": video_id,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": f"ERROR: {str(e)}",
                "images": video_data['image_paths']
            })
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.output_name}.json")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result2, f, indent=2, ensure_ascii=False)
    # with open(output_path.replace(".json", "2.json"), 'w', encoding='utf-8') as f:
    #     json.dump(result2, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 推理完成!")
    print(f"  成功: {len([r for r in results if not r['prediction'].startswith('ERROR')])} 个")
    print(f"  失败: {len([r for r in results if r['prediction'].startswith('ERROR')])} 个")
    print(f"  结果已保存: {output_path}")
    
    return results


def evaluate_results(results_path: str, output_metrics_path: Optional[str] = None):
    """
    评估推理结果
    """
    print("\n" + "="*70)
    print("评估推理结果")
    print("="*70)
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    total = len(results)
    errors = len([r for r in results if r['prediction'].startswith('ERROR')])
    success = total - errors
    
    print(f"\n基础统计:")
    print(f"  总样本数: {total}")
    print(f"  成功推理: {success} ({success/total*100:.1f}%)")
    print(f"  推理失败: {errors} ({errors/total*100:.1f}%)")
    
    if success > 0:
        match_scores = []
        for r in results:
            if not r['prediction'].startswith('ERROR'):
                pred_words = set(r['prediction'].lower().split())
                gt_words = set(r['ground_truth'].lower().split())
                
                if len(gt_words) > 0:
                    overlap = len(pred_words & gt_words)
                    score = overlap / len(gt_words)
                    match_scores.append(score)
        
        if match_scores:
            avg_match = np.mean(match_scores)
            print(f"\n词级别匹配:")
            print(f"  平均重叠率: {avg_match*100:.2f}%")
    
    if output_metrics_path:
        metrics = {
            "total_samples": total,
            "successful": success,
            "failed": errors,
            "success_rate": success / total if total > 0 else 0,
            "avg_word_match": float(np.mean(match_scores)) if match_scores else 0
        }
        
        with open(output_metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✓ 指标已保存: {output_metrics_path}")
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL + 3D 推理测试")
    parser.add_argument("--mode", type=str, required=True,
                       choices=["single", "batch", "evaluate"],
                       help="推理模式: single(单样本), batch(批量), evaluate(评估)")
    
    # 模型相关
    parser.add_argument("--model_path", type=str, required=True,
                       help="训练后的模型路径")
    parser.add_argument("--base_model_path", type=str, default=None,
                       help="基础模型路径 (如果None,从adapter_config读取)")
    parser.add_argument("--enable_3d", type=str, default="True",
                       help="启用3D位置编码")

    # parser.add_argument("--enable_text_guide", action="store_true", help="启用文本指导3D位置编码")
    parser.add_argument("--merge_type", type=str, default="direct_add",
                       help="3d embedding 的融合方式")
    parser.add_argument("--gate_mode", type=str, default="softplus_mean",
                       choices=["softplus_mean", "sigmoid_max", "raw_sigmoid"],
                       help="Gate 计算模式: softplus_mean(旧), sigmoid_max(推荐新), raw_sigmoid(无softmax)")
    parser.add_argument("--is_w2c", type=str, default="False",
                       help="外参是否是w2c矩阵")
    parser.add_argument("--type_3d", type=str, default="sincos",
                       help="3d坐标的编码方式")

    parser.add_argument("--record", type=str, default="False",
                       help="是否可视化记录")
    
    parser.add_argument("--sampled", type=str, default="False",
                       help="是否采样输出")

    parser.add_argument("--norm_type", type=str, default="norm", help="depth to 3d norm type")
    parser.add_argument("--grid_n", type=int, default=3)
    parser.add_argument("--num_3d_freqs", type=int, default=10)

    parser.add_argument("--system_prompt", type=str, 
                       default=None,
                       help="系统提示词")
    
    # === 新增参数: 是否开启LoRA ===
    parser.add_argument("--lora_enable", action="store_true",
                       help="是否使用LoRA加载模式。如果不加此参数，默认使用全量加载模式。")
    
    # 单样本推理参数
    parser.add_argument("--rgb_paths", type=str, nargs="+",
                       help="RGB图像路径 (单样本模式)")
    parser.add_argument("--depth_paths", type=str, nargs="+",
                       help="深度图路径 (单样本模式)")
    parser.add_argument("--camera_params", type=str,
                       help="相机参数JSON路径 (单样本模式)")
    parser.add_argument("--prompt", type=str,
                       help="问题文本 (单样本模式)")
    
    # 批量推理参数
    parser.add_argument("--test_json", type=str,
                       help="测试数据JSON (批量模式)")
    parser.add_argument("--video_mapping", type=str,
                       help="video映射文件 (批量模式)")
    parser.add_argument("--output_dir", type=str, default="inference_results",
                       help="输出路径")
    parser.add_argument("--output_name", type=str, default="inference_results",
                        help="输出文件名")
    parser.add_argument("--limit", type=int, default=None,
                       help="限制推理数量 (用于快速测试)")

    parser.add_argument("--fixed_image_size", type=int, nargs=2, default=[224, 224],
                       help="固定图像尺寸 (H W)")
    
    # 评估参数
    parser.add_argument("--results_path", type=str,
                       help="推理结果路径 (评估模式)")
    parser.add_argument("--metrics_output", type=str,
                       help="指标输出路径 (评估模式)")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        # 单样本推理
        if not args.rgb_paths or not args.prompt:
            parser.error("single模式需要--rgb_paths和--prompt")
        
        inference_single_sample(
            model_path=args.model_path,
            rgb_paths=args.rgb_paths,
            prompt=args.prompt,
            depth_paths=args.depth_paths,
            camera_params_path=args.camera_params,
            enable_3d=args.enable_3d,
            base_model_path=args.base_model_path,
            lora_enable=args.lora_enable  # 传递参数
        )
    
    elif args.mode == "batch":
        # 批量推理
        if not args.test_json or not args.video_mapping:
            parser.error("batch模式需要--test_json和--video_mapping")
        
        inference_batch(
            model_path=args.model_path,
            test_json=args.test_json,
            video_mapping_path=args.video_mapping,
            enable_3d=args.enable_3d,
            base_model_path=args.base_model_path,
            limit=args.limit,
            lora_enable=args.lora_enable, # 传递参数
            args=args
        )
    
    elif args.mode == "evaluate":
        # 评估结果
        if not args.results_path:
            parser.error("evaluate模式需要--results_path")
        
        evaluate_results(
            results_path=args.results_path,
            output_metrics_path=args.metrics_output
        )


if __name__ == "__main__":
    main()