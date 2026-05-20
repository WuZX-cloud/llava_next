"""
将ACB数据集的mapping格式转换为训练所需格式
自动提取每个video的RGB、深度图、内参、外参
"""

import json
import os
from collections import defaultdict
from typing import Dict, List
import numpy as np


def extract_video_data_from_mapping(mapping_path: str, prefix: str = "./processed_ACB") -> Dict:
    """
    从原始mapping提取每个video的完整信息
    
    Args:
        mapping_path: 原始mapping.json路径
        prefix: 数据根目录前缀
    
    Returns:
        processed_mapping: {
            "video_id": {
                "image_paths": ["path1.jpg", "path2.jpg"],
                "depth_paths": ["path1-depth.png", "path2-depth.png"],
                "intrinsics": [K1, K2],  # 每个UAV的内参
                "extrinsics": [T1, T2],  # 每个UAV的外参(pose)
                "num_uavs": 2
            }
        }
    """
    with open(mapping_path, 'r', encoding='utf-8') as f:
        raw_mapping = json.load(f)
    
    print(f"加载原始mapping: {len(raw_mapping)} 个条目")
    
    # 按video_id组织数据
    video_data = defaultdict(lambda: {
        'images': [],
        'depths': [],
        'intrinsics': [],
        'extrinsics': [],
        'uav_names': []
    })
    
    # 遍历所有条目
    for video_id, content in raw_mapping.items():
        for img_path, img_data in content.items():
            # 跳过非图片键
            if not isinstance(img_data, dict) or 'pose' not in img_data:
                continue
            
            # 提取UAV名称 (从路径中提取,如UAV1, UAV2)
            if 'UAV' in img_path:
                uav_name = img_path.split('/')[-2]  # 例如: "UAV1"
            else:
                continue
            
            # RGB图路径
            rgb_path = f"{prefix}/{img_path}"
            
            # 深度图路径
            depth_path = f"{prefix}/{img_data['depth']}"
            
            # 内参
            # intrinsic = content['intrinsic']
            intrinsic = img_data['intrinsic']
            
            # 外参 (pose)
            extrinsic = img_data['pose']
            
            # 添加到对应video
            video_data[video_id]['images'].append(rgb_path)
            video_data[video_id]['depths'].append(depth_path)
            video_data[video_id]['intrinsics'].append(intrinsic)
            video_data[video_id]['extrinsics'].append(extrinsic)
            video_data[video_id]['uav_names'].append(uav_name)
    
    # 转换为最终格式并排序
    processed_mapping = {}
    for video_id, data in video_data.items():
        # 按UAV名称排序,确保顺序一致 (UAV1, UAV2, ...)
        sorted_indices = sorted(
            range(len(data['uav_names'])),
            key=lambda i: data['uav_names'][i]
        )
        
        processed_mapping[video_id] = {
            'image_paths': [data['images'][i] for i in sorted_indices],
            'depth_paths': [data['depths'][i] for i in sorted_indices],
            'intrinsics': [data['intrinsics'][i] for i in sorted_indices],
            'extrinsics': [data['extrinsics'][i] for i in sorted_indices],
            'num_uavs': len(sorted_indices)
        }
    
    print(f"✓ 处理完成: {len(processed_mapping)} 个video")
    
    # 打印示例
    if processed_mapping:
        first_video = list(processed_mapping.keys())[0]
        first_data = processed_mapping[first_video]
        print(f"\n示例 - {first_video}:")
        print(f"  UAV数量: {first_data['num_uavs']}")
        print(f"  RGB图: {first_data['image_paths'][0]}")
        print(f"  深度图: {first_data['depth_paths'][0]}")
        print(f"  内参: {np.array(first_data['intrinsics'][0]).shape}")
        print(f"  外参: {np.array(first_data['extrinsics'][0]).shape}")
    
    return processed_mapping


def save_processed_mapping(processed_mapping: Dict, output_path: str):
    """保存处理后的mapping"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_mapping, f, indent=2, ensure_ascii=False)
    print(f"✓ 已保存到: {output_path}")


def verify_data_integrity(processed_mapping: Dict) -> Dict[str, List[str]]:
    """
    验证数据完整性
    返回缺失文件的报告
    """
    print("\n" + "="*60)
    print("验证数据完整性...")
    print("="*60)
    
    issues = defaultdict(list)
    
    for video_id, data in processed_mapping.items():
        # 检查文件数量一致性
        n_imgs = len(data['image_paths'])
        n_depths = len(data['depth_paths'])
        n_intrinsics = len(data['intrinsics'])
        n_extrinsics = len(data['extrinsics'])
        
        if not (n_imgs == n_depths == n_intrinsics == n_extrinsics):
            issues[video_id].append(
                f"数量不一致: images={n_imgs}, depths={n_depths}, "
                f"intrinsics={n_intrinsics}, extrinsics={n_extrinsics}"
            )
        
        # 检查文件是否存在
        for img_path in data['image_paths']:
            if not os.path.exists(img_path):
                issues[video_id].append(f"RGB图缺失: {img_path}")
        
        for depth_path in data['depth_paths']:
            if not os.path.exists(depth_path):
                issues[video_id].append(f"深度图缺失: {depth_path}")
        
        # 验证矩阵维度
        for i, K in enumerate(data['intrinsics']):
            K_array = np.array(K)
            if K_array.shape != (3, 3):
                issues[video_id].append(f"内参{i}维度错误: {K_array.shape}")
        
        for i, T in enumerate(data['extrinsics']):
            T_array = np.array(T)
            if T_array.shape != (4, 4):
                issues[video_id].append(f"外参{i}维度错误: {T_array.shape}")
    
    # 打印报告
    if issues:
        print(f"\n⚠️  发现 {len(issues)} 个video存在问题:")
        for video_id, problem_list in list(issues.items())[:5]:  # 只显示前5个
            print(f"\n  {video_id}:")
            for problem in problem_list[:3]:  # 每个video只显示前3个问题
                print(f"    - {problem}")
        if len(issues) > 5:
            print(f"\n  ... 还有 {len(issues)-5} 个video存在问题")
    else:
        print("✓ 所有数据完整!")
    
    return dict(issues)


def filter_valid_videos(
    processed_mapping: Dict,
    train_data: List[Dict],
    issues: Dict[str, List[str]]
) -> List[Dict]:
    """
    过滤掉有问题的video,只保留有效的训练样本
    """
    valid_train_data = []
    skipped = []
    
    for item in train_data:
        video_id = item['video']
        
        # 检查是否在mapping中
        if video_id not in processed_mapping:
            skipped.append((video_id, "不在mapping中"))
            continue
        
        # 检查是否有问题
        if video_id in issues:
            skipped.append((video_id, issues[video_id][0]))  # 只显示第一个问题
            continue
        
        valid_train_data.append(item)
    
    print(f"\n训练数据过滤:")
    print(f"  原始样本: {len(train_data)}")
    print(f"  有效样本: {len(valid_train_data)}")
    print(f"  跳过样本: {len(skipped)}")
    
    if skipped:
        print(f"\n前5个被跳过的样本:")
        for video_id, reason in skipped[:5]:
            print(f"  - {video_id}: {reason}")
    
    return valid_train_data


def create_training_config(
    processed_mapping: Dict,
    patch_size: int = 14,
    fixed_image_size: tuple = (448, 448)
) -> Dict:
    """
    创建训练配置文件
    """
    # 统计数据集信息
    total_videos = len(processed_mapping)
    uav_counts = [data['num_uavs'] for data in processed_mapping.values()]
    avg_uavs = np.mean(uav_counts)
    
    # 计算patch数量
    num_patches_per_view = (fixed_image_size[0] // patch_size) * (fixed_image_size[1] // patch_size)
    
    config = {
        "dataset": {
            "total_videos": total_videos,
            "avg_uavs_per_video": float(avg_uavs),
            "image_size": list(fixed_image_size),
            "patch_size": patch_size,
            "patches_per_view": num_patches_per_view
        },
        "training": {
            "vision_output_layer": "visual.merger",
            "enable_3d": True,
            "lora_r": 64,
            "learning_rate": 2e-4,
            "batch_size": 1,
            "gradient_accumulation": 8
        }
    }
    
    return config


def main():
    """主函数: 完整的数据处理流程"""
    import argparse
    # python qwen_3d/convert_acb_data.py  --mapping processed_ACB/total_info_processed.json --train_data processed_ACB/test/sim5_sim6_merged_test_data_multi_view_follow.json --output_dir processed_data2
    parser = argparse.ArgumentParser(description="转换ACB数据集格式")
    parser.add_argument("--mapping", type=str, required=True,
                       help="原始mapping.json路径")
    parser.add_argument("--train_data", type=str, required=True,
                       help="训练数据JSON路径")
    parser.add_argument("--output_dir", type=str, default="./processed_data",
                       help="输出目录")
    parser.add_argument("--prefix", type=str, default="./processed_ACB",
                       help="数据根目录前缀")
    parser.add_argument("--verify", action="store_true",
                       help="验证数据完整性")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print(" ACB数据集格式转换")
    print("="*70)
    
    # 1. 处理mapping
    print("\n[1/5] 处理mapping...")
    processed_mapping = extract_video_data_from_mapping(
        mapping_path=args.mapping,
        prefix=args.prefix
    )
    
    # 保存处理后的mapping
    output_mapping_path = os.path.join(args.output_dir, "processed_mapping.json")
    save_processed_mapping(processed_mapping, output_mapping_path)
    
    # 2. 加载训练数据
    print("\n[2/5] 加载训练数据...")
    with open(args.train_data, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    print(f"✓ 加载 {len(train_data)} 个训练样本")
    
    # 3. 验证数据 (可选)
    issues = {}
    if args.verify:
        print("\n[3/5] 验证数据完整性...")
        issues = verify_data_integrity(processed_mapping)
    else:
        print("\n[3/5] 跳过数据验证 (使用--verify启用)")
    
    # 4. 过滤有效样本
    print("\n[4/5] 过滤有效训练样本...")
    valid_train_data = filter_valid_videos(
        processed_mapping,
        train_data,
        issues
    )
    
    # 保存过滤后的训练数据
    output_train_path = os.path.join(args.output_dir, "train_filtered.json")
    with open(output_train_path, 'w', encoding='utf-8') as f:
        json.dump(valid_train_data, f, indent=2, ensure_ascii=False)
    print(f"✓ 已保存到: {output_train_path}")
    
    # 5. 生成训练配置
    print("\n[5/5] 生成训练配置...")
    config = create_training_config(processed_mapping)
    config_path = os.path.join(args.output_dir, "training_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f"✓ 已保存到: {config_path}")
    
    # 总结
    print("\n" + "="*70)
    print(" 处理完成!")
    print("="*70)
    print(f"\n输出文件:")
    print(f"  1. {output_mapping_path}")
    print(f"     - 包含所有video的RGB、深度、内外参信息")
    print(f"  2. {output_train_path}")
    print(f"     - 过滤后的有效训练样本")
    print(f"  3. {config_path}")
    print(f"     - 推荐的训练配置")
    
    print(f"\n数据统计:")
    print(f"  总video数: {len(processed_mapping)}")
    print(f"  有效训练样本: {len(valid_train_data)}")
    if issues:
        print(f"  有问题的video: {len(issues)}")
    
    print(f"\n下一步:")
    print(f"  python train_qwen_3d.py \\")
    print(f"    --train_data {output_train_path} \\")
    print(f"    --video_mapping {output_mapping_path} \\")
    print(f"    --depth_root {args.prefix} \\")
    print(f"    --rgb_root {args.prefix} \\")
    print(f"    --enable_3d \\")
    print(f"    --vision_output_layer visual.merger \\")
    print(f"    --output_dir ./output_qwen_3d")


if __name__ == "__main__":
    main()