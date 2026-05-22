"""
针对空间推理任务的 3D 位置编码热力图可视化脚本
自动筛选 Object Grounding / Object Matching / Who to Collaborate 等
需要强空间感知能力的样本，批量生成对比热力图
"""

import torch
import numpy as np
from PIL import Image
import json
import os
import cv2
import argparse
from pathlib import Path

from transformers import AutoProcessor, AutoConfig
from peft import PeftModel
from qwen_3d.model.model import Qwen2_5_VLWith3D
from qwen_3d.model.threeD import DepthTo3DCoordinates
from qwen_3d.model.coords3dcontext import Coords3DContext
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


SPATIAL_TYPES = ["Object Grounding", "Object Matching", "Who to Collaborate"]
NON_SPATIAL_TYPES = ["Scene Description", "Quality Assessment"]

# 确保覆盖不同UAV数量的样本
DATASET_PRIORITY = ["Real_2_UAVs", "Sim_3_UAVs", "Sim5_VQA_UAVs", "Sim6_VQA_UAVs"]


def save_heatmap(gate_tensor, original_images, output_dir, sample_id, method='mean'):
    """保存热力图，不需要交互输入"""
    gate_data = gate_tensor.squeeze(0).float()
    total_tokens, C = gate_data.shape

    num_imgs = len(original_images)
    tokens_per_img = 256
    grid_size = 16

    if total_tokens != num_imgs * tokens_per_img:
        print(f"  [WARN] Token mismatch: gate has {total_tokens}, expected {num_imgs * tokens_per_img}. Skipping.")
        return False

    if method == 'l2':
        activations = torch.norm(gate_data, p=2, dim=1).numpy()
    else:
        activations = torch.mean(gate_data, dim=1).numpy()

    print(f"  Gate Stats: Min={activations.min():.6f}, Max={activations.max():.6f}, Mean={activations.mean():.6f}, Std={activations.std():.6f}")

    os.makedirs(output_dir, exist_ok=True)

    for idx in range(num_imgs):
        start_pos = idx * tokens_per_img
        end_pos = (idx + 1) * tokens_per_img
        token_vals = activations[start_pos:end_pos]
        heatmap = token_vals.reshape(grid_size, grid_size)

        min_v, max_v = heatmap.min(), heatmap.max()
        if max_v - min_v > 1e-8:
            heatmap_norm = (heatmap - min_v) / (max_v - min_v)
        else:
            heatmap_norm = np.zeros_like(heatmap)

        heatmap_uint8 = np.uint8(255 * heatmap_norm)
        heatmap_resized = cv2.resize(heatmap_uint8, (224, 224), interpolation=cv2.INTER_CUBIC)
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

        orig_img = original_images[idx]
        if isinstance(orig_img, Image.Image):
            orig_img = np.array(orig_img)
        orig_img = cv2.resize(orig_img, (224, 224))
        if orig_img.shape[2] == 3:
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)

        overlay = cv2.addWeighted(orig_img, 0.6, heatmap_color, 0.4, 0)

        # 保存叠加图和纯热力图
        cv2.imwrite(os.path.join(output_dir, f"{sample_id}_uav{idx+1}_overlay.jpg"), overlay)
        cv2.imwrite(os.path.join(output_dir, f"{sample_id}_uav{idx+1}_heatmap.jpg"), heatmap_color)
        cv2.imwrite(os.path.join(output_dir, f"{sample_id}_uav{idx+1}_original.jpg"), orig_img)

    return True


def run_visualization(args):
    # Load data
    with open(args.test_json, 'r') as f:
        test_data = json.load(f)
    with open(args.video_mapping, 'r') as f:
        video_mapping = json.load(f)

    # Filter samples by question type, ensuring diversity across UAV counts
    spatial_all = [x for x in test_data if x.get("question_type") in SPATIAL_TYPES]
    non_spatial_all = [x for x in test_data if x.get("question_type") in NON_SPATIAL_TYPES]

    print(f"Spatial reasoning samples (total): {len(spatial_all)}")
    print(f"Non-spatial samples (total): {len(non_spatial_all)}")

    # Pick samples from each dataset to ensure UAV count diversity
    def pick_diverse(samples, n):
        picked = []
        for ds in DATASET_PRIORITY:
            ds_samples = [x for x in samples if x.get("dataset") == ds]
            per_ds = max(1, n // len(DATASET_PRIORITY))
            picked.extend(ds_samples[:per_ds])
        # Fill remaining
        remaining = [x for x in samples if x not in picked]
        picked.extend(remaining[:max(0, n - len(picked))])
        return picked[:n]

    spatial_samples = pick_diverse(spatial_all, args.num_spatial)
    non_spatial_samples = pick_diverse(non_spatial_all, args.num_non_spatial)
    all_samples = spatial_samples + non_spatial_samples

    print(f"\nWill visualize {len(spatial_samples)} spatial + {len(non_spatial_samples)} non-spatial = {len(all_samples)} total")

    # Load model
    print("\n" + "=" * 60)
    print("Loading model...")
    print("=" * 60)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    config = AutoConfig.from_pretrained(args.base_model_path, trust_remote_code=True)
    model = Qwen2_5_VLWith3D(config, args=args)
    model.visual.merger.position_3d_encoder.record = True
    model.to(torch.bfloat16)

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.load_state_dict(base_model.state_dict(), strict=False)
    del base_model

    model = PeftModel.from_pretrained(model, args.model_path)

    from peft.utils.other import ModulesToSaveWrapper
    if hasattr(model, "visual") and isinstance(model.visual, ModulesToSaveWrapper):
        model.visual.dtype = torch.bfloat16

    model.eval()
    model.to(args.device)
    print("Model loaded.\n")

    # 3D coord generator
    coord_generator = DepthTo3DCoordinates(
        patch_size=14,
        fixed_image_size=args.fixed_image_size,
        type=args.norm_type,
        grid_n=args.grid_n,
        is_w2c=(args.is_w2c == "True")
    )

    # Run inference and visualize
    output_base = args.output_dir
    os.makedirs(output_base, exist_ok=True)

    summary = []

    for idx, item in enumerate(all_samples):
        video_id = item['video']
        q_type = item.get('question_type', 'unknown')
        is_spatial = q_type in SPATIAL_TYPES
        category = "spatial" if is_spatial else "non_spatial"

        if video_id not in video_mapping:
            print(f"[{idx+1}] Skip {video_id}: not in mapping")
            continue

        video_data = video_mapping[video_id]
        dataset_name = item.get("dataset", "unknown")
        num_uavs = len(video_data.get('image_paths', []))

        # Get question and ground truth
        conversations = item['conversations']
        question = None
        ground_truth = None
        conv_messages = []

        for conv in conversations:
            if conv['from'] == 'human':
                question = conv['value']
                conv_messages.append({'role': 'user', 'content': question})
            elif conv['from'] == 'gpt':
                ground_truth = conv['value']

        if not question:
            continue

        print(f"\n[{idx+1}/{len(all_samples)}] {category.upper()} | Type: {q_type} | UAVs: {num_uavs} | Dataset: {dataset_name} | Video: {video_id}")

        try:
            # Load images and compute 3D coords
            rgb_paths = video_data['image_paths']
            depth_paths = video_data.get('depth_paths')
            intrinsics = [np.array(k, dtype=np.float32) for k in video_data.get('intrinsics', [])]
            extrinsics = [np.array(t, dtype=np.float32) for t in video_data.get('extrinsics', [])]

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

            coords_3d, resized_rgbs = coord_generator.process_multi_view(
                depth_list=depth_list,
                intrinsic_list=intrinsics,
                extrinsic_list=extrinsics,
                rgb_list=rgb_list
            )
            coords_3d_tensor = torch.from_numpy(coords_3d).float()
            images = resized_rgbs

            # Build messages for model
            messages = []
            for conv in conversations:
                if conv['from'] == 'human':
                    text = conv['value']
                    parts = text.split('<image>')
                    content_parts = []
                    img_idx = 0
                    for i, part in enumerate(parts):
                        if part:
                            content_parts.append({"type": "text", "text": part})
                        if i < len(parts) - 1 and img_idx < len(images):
                            content_parts.append({"type": "image", "image": images[img_idx]})
                            img_idx += 1
                    messages.append({"role": "user", "content": content_parts})
                elif conv['from'] == 'gpt':
                    pass  # Don't include assistant for generation

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, return_tensors="pt", padding=True)

            # Add 3D coords
            coords_3d_input = coords_3d_tensor.unsqueeze(0).to(args.device)
            inputs["coords_3d"] = coords_3d_input
            inputs["coords_3d_mask"] = torch.ones(1, coords_3d_input.shape[1], device=args.device)

            # Extract question text for text-guided encoding
            import re
            full_text = question.replace('<image>', '')
            opt_pattern = r"\n\s*[A][\.\)]"
            opt_match = re.search(opt_pattern, full_text)
            if opt_match:
                text_before = full_text[:opt_match.start()]
                last_nl = text_before.rfind('\n')
                q_text = text_before[last_nl+1:].strip() if last_nl != -1 else text_before.strip()
            else:
                lines = full_text.strip().split('\n')
                q_text = lines[-1].strip() if lines else full_text.strip()
            if len(q_text) < 3:
                q_text = full_text[-50:]

            q_enc = processor.tokenizer(
                [q_text], padding=True, truncation=True, max_length=128,
                return_tensors="pt", add_special_tokens=False
            )
            inputs["question_input_ids"] = q_enc["input_ids"]

            inputs = {k: v.to(args.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=128, do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id
                )

            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            response = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            print(f"  GT: {ground_truth} | Pred: {response} | {'CORRECT' if response == ground_truth else 'WRONG'}")

            # Get gate and visualize
            current_gate = Coords3DContext._shared_state["latest_gate"]
            if current_gate is not None:
                sample_dir = os.path.join(output_base, category, f"{q_type.replace(' ', '_')}_{idx}")
                success = save_heatmap(current_gate, images, sample_dir, f"sample", method=args.vis_method)
                if success:
                    # Save metadata
                    meta = {
                        "video_id": video_id,
                        "question_type": q_type,
                        "category": category,
                        "dataset": dataset_name,
                        "num_uavs": num_uavs,
                        "question_text": q_text,
                        "ground_truth": ground_truth,
                        "prediction": response,
                        "correct": response == ground_truth
                    }
                    with open(os.path.join(sample_dir, "meta.json"), 'w') as f:
                        json.dump(meta, f, indent=2, ensure_ascii=False)
                    summary.append(meta)
                    print(f"  Saved to: {sample_dir}")
            else:
                print(f"  [WARN] No gate recorded!")

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()

    # Save summary
    summary_path = os.path.join(output_base, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*60}")
    print(f"Done! {len(summary)} samples visualized.")
    print(f"Spatial: {len([s for s in summary if s['category']=='spatial'])}")
    print(f"Non-spatial: {len([s for s in summary if s['category']=='non_spatial'])}")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, default="models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--test_json", type=str, default="processed_data_with_depth_npy/test/sim5_sim6_merged_test_data_multi_view_follow.json")
    parser.add_argument("--video_mapping", type=str, default="processed_data_with_depth_npy/processed_mapping.json")
    parser.add_argument("--output_dir", type=str, default="vis_spatial_comparison")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_spatial", type=int, default=10, help="Number of spatial samples to visualize")
    parser.add_argument("--num_non_spatial", type=int, default=5, help="Number of non-spatial samples for comparison")
    parser.add_argument("--vis_method", type=str, default="mean", choices=["mean", "l2"])

    # Model config args (match eval_3d.sh)
    parser.add_argument("--enable_3d", type=str, default="True")
    parser.add_argument("--merge_type", type=str, default="text_cross_attention")
    parser.add_argument("--is_w2c", type=str, default="False")
    parser.add_argument("--type_3d", type=str, default="sincos")
    parser.add_argument("--norm_type", type=str, default="no_norm")
    parser.add_argument("--grid_n", type=int, default=7)
    parser.add_argument("--num_3d_freqs", type=int, default=4)
    parser.add_argument("--fixed_image_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--record", type=str, default="True")

    args = parser.parse_args()
    run_visualization(args)
