"""
Qwen2.5-VL + 3D Position Encoding 推理脚本
支持单样本推理、批量推理、可视化
"""

import torch
import numpy as np
from PIL import Image
import json
import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import argparse

from transformers import AutoProcessor
from peft import PeftModel
from qwen_3d.train_qwen_3d2 import (
    Qwen2_5_VLWith3D,
    DepthTo3DCoordinates,
)
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
)

import traceback

class Qwen3DInference:
    """
    3D增强的Qwen2.5-VL推理器
    """
    def __init__(
        self,
        model_path: str,
        base_model_path: Optional[str] = None,
        device: str = "cuda",
        enable_3d: bool = True
    ):
        """
        Args:
            model_path: 训练后的模型路径 (包含LoRA权重)
            base_model_path: 基础模型路径 (如果None,从model_path读取)
            device: 运行设备
            enable_3d: 是否启用3D编码
        """
        self.device = device
        self.enable_3d = enable_3d
        
        print(f"\n{'='*60}")
        print(f"加载 Qwen2.5-VL + 3D 推理模型")
        print(f"{'='*60}")
        
        # 1. 加载processor
        print(f"加载 Processor...")
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print(f"✓ Processor 加载完成")
        
        # 2. 加载模型
        print(f"加载模型...")
        
        # 读取训练配置
        config_path = os.path.join(model_path, "training_args.bin")
        if os.path.exists(config_path):
            import torch
            train_args = torch.load(config_path)
            print(f"  - 训练配置已加载")
        
        # 确定基础模型路径
        if base_model_path is None:
            # 尝试从adapter_config.json读取
            adapter_config_path = os.path.join(model_path, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                    base_model_path = adapter_config.get("base_model_name_or_path")
        
        if base_model_path is None:
            raise ValueError("无法确定base_model_path,请手动指定")
        
        print(f"  - 基础模型: {base_model_path}")
        
        # 加载基础模型
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
        self.model = Qwen2_5_VLWith3D(config, enable_3d=enable_3d)
        self.model.to(torch.bfloat16)

        # 加载基础权重
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        self.model.load_state_dict(base_model.state_dict(), strict=False)
        del base_model
        
        # 注册3D hook
        if enable_3d:
            self.model.register_3d_hook("visual.merger")
            print(f"  - 3D Position Encoding 已启用")

        # 加载LoRA权重
        print(f"  - 加载 LoRA 权重...")
        self.model = PeftModel.from_pretrained(self.model, model_path)
        # self.model = self.model.merge_and_unload()  # 合并LoRA到主模型
        
        # 加载3D模块 ⭐ 新增
        old = None
        for k in self.model.state_dict().keys():
            if "position_3d_encoder" in k:
                print(k)
                old = k
                break

        # for n, p in self.model.named_parameters():
        #     if "position_3d_encoder" in n:
        #         print("训练前:", n, p.mean().item())
        position_3d_state = torch.load(f"{model_path}/position_3d_encoder.pt")
        new = None
        for k in position_3d_state.keys():
            if "position_3d_encoder" in k:
                print(k)
                new = k
                break
        old = old.split("position_3d_encoder")[0]
        new = new.split("position_3d_encoder")[0]
        # print(old)
        # print(new)
        # exit()
        new_state_dict = {}

        for k, v in position_3d_state.items():
            new_key = k.replace(new, old)
            new_state_dict[new_key] = v
        ret = self.model.load_state_dict(new_state_dict, strict=False)

        # print("Missing keys:", ret.missing_keys)
        print("Unexpected keys:", ret.unexpected_keys)
        # for n, p in self.model.named_parameters():
        #     if "position_3d_encoder" in n:
        #         print("训练后:", n, p.mean().item())
        # ✅ 3D模块权重正确加载
        # exit(-1)
        self.model.eval()
        self.model.to(device)
        
        print(f"✓ 模型加载完成")
        print(f"{'='*60}\n")
        
        # 3. 初始化3D坐标生成器
        if enable_3d:
            self.coord_generator = DepthTo3DCoordinates(
                patch_size=14,
                fixed_image_size=(448, 448)
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
        
        Returns:
            images: PIL图像列表
            coords_3d: (num_views * num_patches, 3) 或 None
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
        images = [Image.fromarray(rgb) for rgb in resized_rgbs]
        coords_3d_tensor = torch.from_numpy(coords_3d).float()
        
        return images, coords_3d_tensor
    
    @torch.no_grad()
    def generate(
        self,
        images: List[Image.Image],
        prompt: str,
        coords_3d: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        生成回复
        
        Args:
            images: 图像列表
            prompt: 文本提示
            coords_3d: 3D坐标 (可选)
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_p: nucleus采样参数
            do_sample: 是否采样
        
        Returns:
            生成的文本
        """
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": []
            }
        ]
        
        # 添加图像
        for img in images:
            messages[0]["content"].append({
                "type": "image",
                "image": img
            })
        
        # 添加文本
        messages[0]["content"].append({
            "type": "text",
            "text": prompt
        })
        
        # 应用chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 处理输入
        from qwen_vl_utils import process_vision_info
        image_inputs, _ = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            return_tensors="pt",
            padding=True
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 添加3D坐标
        if self.enable_3d and coords_3d is not None:
            coords_3d = coords_3d.unsqueeze(0).to(self.device)  # (1, N, 3)
            inputs["coords_3d"] = coords_3d
            
            # 创建mask
            coords_mask = torch.ones(1, coords_3d.shape[1], device=self.device)
            inputs["coords_3d_mask"] = coords_mask
        

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
        
        # 解码
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def inference_from_paths(
        self,
        rgb_paths: List[str],
        depth_paths: Optional[List[str]],
        intrinsics: Optional[List[np.ndarray]],
        extrinsics: Optional[List[np.ndarray]],
        prompt: str,
        **generate_kwargs
    ) -> str:
        """
        从文件路径推理
        
        Args:
            rgb_paths: RGB图路径列表
            depth_paths: 深度图路径列表 (如果enable_3d=False可为None)
            intrinsics: 内参列表
            extrinsics: 外参列表
            prompt: 提示文本
        
        Returns:
            生成的回复
        """
        # 加载图像和计算3D
        if self.enable_3d and depth_paths is not None:
            print("生成coords_3d")
            images, coords_3d = self.load_and_compute_3d(
                rgb_paths, depth_paths, intrinsics, extrinsics
            )
        else:
            images = [Image.open(p).convert('RGB') for p in rgb_paths]
            coords_3d = None
        
        # 生成
        response = self.generate(images, prompt, coords_3d, **generate_kwargs)
        
        return response


def inference_single_sample(
    model_path: str,
    rgb_paths: List[str],
    prompt: str,
    depth_paths: Optional[List[str]] = None,
    camera_params_path: Optional[str] = None,
    enable_3d: bool = True,
    base_model_path: Optional[str] = None
):
    """
    单样本推理示例
    
    Args:
        model_path: 训练后的模型路径
        rgb_paths: RGB图像路径列表
        prompt: 问题
        depth_paths: 深度图路径列表
        camera_params_path: 相机参数JSON路径
        enable_3d: 是否启用3D
        base_model_path: 基础模型路径
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
        enable_3d=enable_3d
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
        max_new_tokens=256,
        temperature=0.7
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
    output_path: str,
    enable_3d: bool = True,
    base_model_path: Optional[str] = None,
    limit: Optional[int] = None
):
    """
    批量推理
    
    Args:
        model_path: 训练后的模型路径
        test_json: 测试数据JSON路径
        video_mapping_path: video映射文件路径
        output_path: 输出结果路径
        enable_3d: 是否启用3D
        base_model_path: 基础模型路径
        limit: 只推理前N个样本 (用于快速测试)
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
        # test_data = test_data[:limit]
        test_data = [x for x in test_data if len(video_mapping[x["video"]]["image_paths"]) == 3]
        test_data = test_data[:limit]
        print(f"限制推理数量: {limit}")
    
    print(f"测试样本数: {len(test_data)}")
    
    # 初始化推理器
    inferencer = Qwen3DInference(
        model_path=model_path,
        base_model_path=base_model_path,
        enable_3d=enable_3d
    )
    
    # 批量推理
    results = []
    
    for idx, item in enumerate(test_data):
        video_id = item['video']
        
        if video_id not in video_mapping:
            print(f"[{idx+1}/{len(test_data)}] 跳过 {video_id}: 不在mapping中")
            continue
        
        video_data = video_mapping[video_id]
        
        # 提取问题 (假设第一轮对话是human的问题)
        conversations = item['conversations']
        question = None
        ground_truth = None
        
        for conv in conversations:
            if conv['from'] == 'human':
                # 移除<image>标记,只保留文本
                question = conv['value'].replace('<image>', '').strip()
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
                prompt=question,
                max_new_tokens=256,
                temperature=0.7
            )
            
            results.append({
                "video_id": video_id,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": response,
                "images": video_data['image_paths']
            })
            result = {
                **item,  # 保留原始样本信息
                'response': response
            }
            
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
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(output_path.replace(".json", "2.json"), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 推理完成!")
    print(f"  成功: {len([r for r in results if not r['prediction'].startswith('ERROR')])} 个")
    print(f"  失败: {len([r for r in results if r['prediction'].startswith('ERROR')])} 个")
    print(f"  结果已保存: {output_path}")
    
    # 打印几个示例
    print(f"\n示例结果 (前3个):")
    for i, result in enumerate(results[:3]):
        print(f"\n[{i+1}] {result['video_id']}")
        print(f"  Q: {result['question'][:100]}...")
        print(f"  GT: {result['ground_truth'][:100]}...")
        print(f"  Pred: {result['prediction'][:100]}...")
    
    return results


def evaluate_results(results_path: str, output_metrics_path: Optional[str] = None):
    """
    评估推理结果
    
    简单计算一些基础指标
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
    
    # 简单的词级别匹配率
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
    
    # 保存指标
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
    parser.add_argument("--enable_3d", action="store_true",
                       help="启用3D位置编码")
    
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
    parser.add_argument("--output", type=str, default="inference_results.json",
                       help="输出路径")
    parser.add_argument("--limit", type=int, default=None,
                       help="限制推理数量 (用于快速测试)")
    
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
            base_model_path=args.base_model_path
        )
    
    elif args.mode == "batch":
        # 批量推理
        if not args.test_json or not args.video_mapping:
            parser.error("batch模式需要--test_json和--video_mapping")
        
        inference_batch(
            model_path=args.model_path,
            test_json=args.test_json,
            video_mapping_path=args.video_mapping,
            output_path=args.output,
            enable_3d=args.enable_3d,
            base_model_path=args.base_model_path,
            limit=args.limit
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