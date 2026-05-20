"""
测试图片路径提取功能
使用方法: python test_extraction.py --video_mapping data/video_mapping.json
"""

import json
import argparse
from typing import Dict, List, Tuple


def extract_image_paths(raw_mapping: Dict) -> Tuple[Dict, List[str]]:
    """
    从嵌套字典格式的映射中提取图片路径
    """
    processed_mapping = {}
    warnings = []
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    
    for video_id, video_data in raw_mapping.items():
        if not isinstance(video_data, dict):
            warnings.append(f"video_id '{video_id}' 的值不是字典,跳过")
            continue
        
        # 提取图片路径
        image_paths = []
        for key in video_data.keys():
            # 检查是否包含"UAV"且以图像格式结尾
            if "UAV" in key and key.lower().endswith(image_extensions):
                image_paths.append(key)
        
        # 排序确保顺序一致
        image_paths.sort()
        
        if image_paths:
            processed_mapping[video_id] = image_paths
        else:
            warnings.append(f"video_id '{video_id}' 中未找到有效的图片路径")
    
    return processed_mapping, warnings


def main():
    parser = argparse.ArgumentParser(description="测试映射文件的图片路径提取")
    parser.add_argument("--video_mapping", type=str, required=True, help="video映射JSON路径")
    parser.add_argument("--show_all", action="store_true", help="显示所有提取结果")
    
    args = parser.parse_args()
    
    # 加载映射文件
    print(f"加载映射文件: {args.video_mapping}")
    try:
        with open(args.video_mapping, 'r', encoding='utf-8') as f:
            raw_mapping = json.load(f)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return
    
    print(f"✅ 成功加载,包含 {len(raw_mapping)} 个video")
    
    # 提取图片路径
    print("\n开始提取图片路径...")
    processed_mapping, warnings = extract_image_paths(raw_mapping)
    
    # 显示统计信息
    print("\n" + "="*70)
    print("提取结果统计")
    print("="*70)
    print(f"• 原始video数量: {len(raw_mapping)}")
    print(f"• 包含有效图片的video数量: {len(processed_mapping)}")
    print(f"• 警告数量: {len(warnings)}")
    
    # 统计图片数量分布
    image_count_dist = {}
    total_images = 0
    for video_id, img_paths in processed_mapping.items():
        count = len(img_paths)
        total_images += count
        if count not in image_count_dist:
            image_count_dist[count] = 0
        image_count_dist[count] += 1
    
    print(f"• 总图片数: {total_images}")
    print(f"• 平均每个video的图片数: {total_images/max(len(processed_mapping), 1):.2f}")
    
    print("\n图片数量分布:")
    for count, freq in sorted(image_count_dist.items()):
        print(f"  {count} 张图片: {freq} 个video")
    
    # 显示警告
    if warnings:
        print("\n⚠️  警告信息:")
        for warning in warnings[:10]:
            print(f"  • {warning}")
        if len(warnings) > 10:
            print(f"  ... 还有 {len(warnings) - 10} 个警告")
    
    # 显示示例
    print("\n" + "="*70)
    print("提取示例 (前5个)")
    print("="*70)
    
    show_count = len(processed_mapping) if args.show_all else min(5, len(processed_mapping))
    
    for i, (video_id, img_paths) in enumerate(list(processed_mapping.items())[:show_count]):
        print(f"\n[{i+1}] Video ID: {video_id}")
        print(f"    图片数量: {len(img_paths)}")
        for j, img_path in enumerate(img_paths):
            print(f"    [{j}] {img_path}")
    
    if not args.show_all and len(processed_mapping) > 5:
        print(f"\n... 还有 {len(processed_mapping) - 5} 个video")
        print("使用 --show_all 参数查看所有结果")
    
    # 保存提取结果(可选)
    save_choice = input("\n是否保存提取的映射结果? (y/n): ").lower()
    if save_choice == 'y':
        output_file = args.video_mapping.replace('.json', '_processed.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_mapping, f, indent=2, ensure_ascii=False)
        print(f"✅ 已保存到: {output_file}")
    
    print("\n" + "="*70)
    print("测试完成!")
    print("="*70)


if __name__ == "__main__":
    main()