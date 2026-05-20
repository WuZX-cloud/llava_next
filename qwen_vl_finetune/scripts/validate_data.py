"""
数据验证脚本 - 验证训练数据格式和图片路径
使用方法: python validate_data.py --train_data data/train_data.json --video_mapping data/video_mapping.json
"""

import json
import os
from pathlib import Path
from PIL import Image
import argparse
from typing import Dict, List, Tuple


def load_json(file_path: str) -> dict:
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 加载文件失败 {file_path}: {e}")
        return None


def extract_image_paths_from_mapping(raw_mapping: Dict) -> Tuple[Dict, List[str]]:
    """
    从嵌套字典格式的映射中提取图片路径
    
    输入格式:
    {
        "video_id": {
            "path/to/UAV1.jpg": {...},
            "path/to/UAV2.jpg": {...},
            "intrinsic": [...],
            ...
        }
    }
    
    返回:
    (processed_mapping, warnings)
    """
    processed_mapping = {}
    warnings = []
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    
    for video_id, video_data in raw_mapping.items():
        if not isinstance(video_data, dict):
            warnings.append(f"video_id '{video_id}' 的值不是字典,跳过")
            continue
        
        # 路径前缀
        prefix = "processed_ACB"

        # 提取图片路径
        image_paths = []
        for key in video_data.keys():
            # 检查是否包含"UAV"且以图像格式结尾
            if "UAV" in key and key.lower().endswith(image_extensions):
                image_paths.append(f"{prefix}/{key}")
        
        # 排序确保顺序一致
        image_paths.sort()
        
        if image_paths:
            processed_mapping[video_id] = image_paths
        else:
            warnings.append(f"video_id '{video_id}' 中未找到有效的图片路径")
    
    return processed_mapping, warnings


def validate_video_mapping(raw_mapping: Dict) -> Tuple[bool, List[str]]:
    """验证video映射文件"""
    errors = []
    
    if not isinstance(raw_mapping, dict):
        errors.append("video_mapping必须是字典格式")
        return False, errors
    
    # 提取图片路径
    processed_mapping, warnings = extract_image_paths_from_mapping(raw_mapping)
    
    # 将warnings添加到errors中(作为提示信息)
    for warning in warnings:
        print(f"⚠️  {warning}")
    
    # 验证提取的图片路径
    for video_id, image_paths in processed_mapping.items():
        for i, img_path in enumerate(image_paths):
            if not os.path.exists(img_path):
                errors.append(f"图片不存在: {img_path} (video: {video_id}, index: {i})")
            else:
                # 尝试打开图片验证格式
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception as e:
                    errors.append(f"图片损坏或格式错误: {img_path} - {str(e)}")
    
    if not processed_mapping:
        errors.append("未能从映射文件中提取到任何有效的图片路径")
    
    return len(errors) == 0, errors


def validate_train_data(
    train_data: List[Dict], 
    raw_video_mapping: Dict
) -> Tuple[bool, List[str], Dict]:
    """验证训练数据"""
    errors = []
    stats = {
        "total_samples": len(train_data),
        "total_conversations": 0,
        "total_images": 0,
        "missing_videos": [],
        "image_count_distribution": {}
    }
    
    # 先提取处理后的映射
    video_mapping, _ = extract_image_paths_from_mapping(raw_video_mapping)
    
    for idx, item in enumerate(train_data):
        # 检查必需字段
        if "video" not in item:
            errors.append(f"样本 {idx}: 缺少 'video' 字段")
            continue
        
        if "conversations" not in item:
            errors.append(f"样本 {idx}: 缺少 'conversations' 字段")
            continue
        
        video_id = item["video"]
        conversations = item["conversations"]
        
        # 检查video_id是否在映射中
        if video_id not in video_mapping:
            errors.append(f"样本 {idx}: video_id '{video_id}' 不在video_mapping中或未包含有效图片")
            stats["missing_videos"].append(video_id)
            continue
        
        # 统计对话数
        stats["total_conversations"] += len(conversations)
        
        # 检查对话格式
        for conv_idx, conv in enumerate(conversations):
            if "from" not in conv:
                errors.append(f"样本 {idx}, 对话 {conv_idx}: 缺少 'from' 字段")
                continue
            
            if "value" not in conv:
                errors.append(f"样本 {idx}, 对话 {conv_idx}: 缺少 'value' 字段")
                continue
            
            if conv["from"] not in ["human", "gpt"]:
                errors.append(f"样本 {idx}, 对话 {conv_idx}: 'from' 必须是 'human' 或 'gpt'")
            
            # 统计<image>标记数量
            if conv["from"] == "human":
                value = conv["value"]
                image_count = value.count("<image>")
                
                # 检查<image>数量是否匹配映射中的图片数量
                available_images = len(video_mapping[video_id])
                if image_count != available_images:
                    errors.append(
                        f"样本 {idx}: <image>标记数量({image_count}) "
                        f"与映射中的图片数量({available_images})不匹配"
                    )
                
                stats["total_images"] += image_count
                
                # 统计图片数量分布
                if image_count not in stats["image_count_distribution"]:
                    stats["image_count_distribution"][image_count] = 0
                stats["image_count_distribution"][image_count] += 1
    
    return len(errors) == 0, errors, stats


def print_validation_report(
    mapping_valid: bool,
    mapping_errors: List[str],
    data_valid: bool,
    data_errors: List[str],
    stats: Dict
):
    """打印验证报告"""
    
    print("\n" + "="*60)
    print("数据验证报告")
    print("="*60)
    
    # Video Mapping验证结果
    print("\n📋 Video Mapping验证:")
    if mapping_valid:
        print("✅ 通过 - 所有图片路径有效")
    else:
        print(f"❌ 失败 - 发现 {len(mapping_errors)} 个错误:")
        for error in mapping_errors[:10]:  # 只显示前10个错误
            print(f"   • {error}")
        if len(mapping_errors) > 10:
            print(f"   ... 还有 {len(mapping_errors) - 10} 个错误")
    
    # 训练数据验证结果
    print("\n📋 训练数据验证:")
    if data_valid:
        print("✅ 通过 - 数据格式正确")
    else:
        print(f"❌ 失败 - 发现 {len(data_errors)} 个错误:")
        for error in data_errors[:10]:
            print(f"   • {error}")
        if len(data_errors) > 10:
            print(f"   ... 还有 {len(data_errors) - 10} 个错误")
    
    # 统计信息
    print("\n📊 数据统计:")
    print(f"   • 总样本数: {stats['total_samples']}")
    print(f"   • 总对话数: {stats['total_conversations']}")
    print(f"   • 总图片数: {stats['total_images']}")
    print(f"   • 平均每样本对话数: {stats['total_conversations']/max(stats['total_samples'], 1):.2f}")
    print(f"   • 平均每样本图片数: {stats['total_images']/max(stats['total_samples'], 1):.2f}")
    
    if stats['image_count_distribution']:
        print("\n   图片数量分布:")
        for count, freq in sorted(stats['image_count_distribution'].items()):
            print(f"     {count} 张图片: {freq} 个样本")
    
    if stats['missing_videos']:
        print(f"\n⚠️  缺失的video_id ({len(stats['missing_videos'])} 个):")
        for vid in stats['missing_videos'][:5]:
            print(f"   • {vid}")
        if len(stats['missing_videos']) > 5:
            print(f"   ... 还有 {len(stats['missing_videos']) - 5} 个")
    
    # 总结
    print("\n" + "="*60)
    if mapping_valid and data_valid:
        print("✅ 验证通过! 数据可以用于训练")
    else:
        print("❌ 验证失败! 请修复上述错误后重试")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="验证Qwen2.5-VL训练数据")
    parser.add_argument("--train_data", type=str, required=True, help="训练数据JSON路径")
    parser.add_argument("--video_mapping", type=str, required=True, help="video映射JSON路径")
    
    args = parser.parse_args()
    
    # 加载数据
    print("加载数据文件...")
    raw_video_mapping = load_json(args.video_mapping)
    train_data = load_json(args.train_data)
    
    if raw_video_mapping is None or train_data is None:
        print("❌ 数据加载失败,退出验证")
        return
    
    # 提取图片路径
    print("\n提取图片路径...")
    video_mapping, extraction_warnings = extract_image_paths_from_mapping(raw_video_mapping)
    
    print(f"✅ 加载完成")
    print(f"   • Video映射(原始): {len(raw_video_mapping)} 个video")
    print(f"   • Video映射(提取): {len(video_mapping)} 个video包含有效图片")
    print(f"   • 训练样本: {len(train_data)} 个")
    
    if extraction_warnings:
        print(f"\n⚠️  提取过程中的警告 ({len(extraction_warnings)} 个):")
        for warning in extraction_warnings[:5]:
            print(f"   • {warning}")
        if len(extraction_warnings) > 5:
            print(f"   ... 还有 {len(extraction_warnings) - 5} 个警告")
    
    # 显示几个提取示例
    if video_mapping:
        print("\n📝 提取的图片路径示例:")
        for i, (video_id, img_paths) in enumerate(list(video_mapping.items())[:3]):
            print(f"   [{i+1}] {video_id}")
            for j, img_path in enumerate(img_paths):
                print(f"       • UAV{j+1}: {img_path}")
    
    # 验证video映射
    print("\n正在验证video映射...")
    mapping_valid, mapping_errors = validate_video_mapping(raw_video_mapping)
    
    # 验证训练数据
    print("正在验证训练数据...")
    data_valid, data_errors, stats = validate_train_data(train_data, raw_video_mapping)
    
    # 打印报告
    print_validation_report(
        mapping_valid, mapping_errors,
        data_valid, data_errors,
        stats
    )


if __name__ == "__main__":
    main()