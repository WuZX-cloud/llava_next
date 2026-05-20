"""
测试Qwen2.5-VL + 3D Position Encoding集成
验证数据加载、模型forward和训练流程
"""

import torch
import numpy as np
from PIL import Image
import json
import os


def test_depth_to_3d_conversion():
    """测试1: 深度图转3D坐标"""
    print("\n" + "="*60)
    print("测试1: 深度图 → 3D坐标转换")
    print("="*60)
    
    from train_qwen_3d import DepthTo3DCoordinates
    
    # 初始化转换器
    converter = DepthTo3DCoordinates(
        patch_size=14,
        fixed_image_size=(448, 448)
    )
    
    # 创建模拟数据
    H, W = 1080, 1920  # 原始分辨率
    depth = np.random.rand(H, W) * 10.0  # 模拟深度值 0-10米
    rgb = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    
    # 模拟相机参数
    intrinsic = np.array([
        [1000, 0, W/2],
        [0, 1000, H/2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    extrinsic = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 5],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    
    # 测试单视角
    print(f"原始图像尺寸: {H}x{W}")
    print(f"目标尺寸: 448x448")
    print(f"Patch大小: 14x14")
    print(f"预期patch数量: {converter.num_patches}")
    
    # Resize
    depth_resized, rgb_resized = converter.resize_depth_and_rgb(depth, rgb)
    print(f"✓ Resize完成: depth={depth_resized.shape}, rgb={rgb_resized.shape}")
    
    # 转换为3D
    coords_3d = converter.depth_to_3d_world(depth_resized, intrinsic, extrinsic)
    print(f"✓ 3D坐标生成: shape={coords_3d.shape}")
    print(f"  - 坐标范围: x=[{coords_3d[:,0].min():.2f}, {coords_3d[:,0].max():.2f}]")
    print(f"  - 坐标范围: y=[{coords_3d[:,1].min():.2f}, {coords_3d[:,1].max():.2f}]")
    print(f"  - 坐标范围: z=[{coords_3d[:,2].min():.2f}, {coords_3d[:,2].max():.2f}]")
    
    # 测试多视角
    print(f"\n测试多视角合并...")
    coords_multi, _ = converter.process_multi_view(
        depth_list=[depth, depth],
        intrinsic_list=[intrinsic, intrinsic],
        extrinsic_list=[extrinsic, extrinsic],
        rgb_list=[rgb, rgb]
    )
    print(f"✓ 多视角合并完成: shape={coords_multi.shape}")
    print(f"  - 预期: {2 * converter.num_patches} = {coords_multi.shape[0]}")
    
    return coords_3d


def test_3d_position_encoding():
    """测试2: 3D位置编码模块"""
    print("\n" + "="*60)
    print("测试2: 3D Position Encoding模块")
    print("="*60)
    
    from train_qwen_3d import ThreeDPositionEncoding
    
    # 初始化模块
    encoder = ThreeDPositionEncoding(
        hidden_size=3584,
        num_3d_freqs=10
    )
    
    # 模拟输入
    B, N, D = 2, 1024, 3584  # batch_size=2, num_patches=1024
    vision_tokens = torch.randn(B, N, D)
    coords_3d = torch.randn(B, N, 3)  # (x, y, z)
    
    print(f"输入: vision_tokens={vision_tokens.shape}, coords_3d={coords_3d.shape}")
    
    # Forward
    enhanced = encoder(vision_tokens, coords_3d)
    
    print(f"✓ 输出: enhanced={enhanced.shape}")
    print(f"✓ 融合权重 alpha={encoder.fusion_alpha.item():.4f}")
    
    # 验证residual connection
    diff = (enhanced - vision_tokens).abs().mean()
    print(f"✓ 平均变化量: {diff.item():.6f} (应该较小,说明是residual)")
    
    return encoder


def test_model_loading():
    """测试3: 模型加载和Hook注册"""
    print("\n" + "="*60)
    print("测试3: 模型加载 + Hook注册")
    print("="*60)
    
    from train_qwen_3d import setup_model_with_3d_and_lora
    
    try:
        model = setup_model_with_3d_and_lora(
            model_name="models/Qwen2.5-VL-7B-Instruct",
            enable_3d=True,
            lora_r=8,  # 测试时用小一点的rank
            vision_output_layer="visual.merger"
        )
        
        print(f"✓ 模型加载成功")
        print(f"✓ 模型类型: {type(model).__name__}")
        
        # 检查3D模块
        if hasattr(model.base_model, 'position_3d_encoder'):
            print(f"✓ 3D模块存在")
        
        # 检查hook
        if hasattr(model.base_model, '_coords_3d_cache'):
            print(f"✓ Hook缓存已初始化")
        
        return model
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_data_collator():
    """测试4: DataCollator处理3D坐标"""
    print("\n" + "="*60)
    print("测试4: DataCollator + 3D坐标")
    print("="*60)
    
    from train_qwen_3d import MultimodalDataCollatorWith3D
    from transformers import AutoProcessor
    
    # 加载processor
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        trust_remote_code=True
    )
    
    collator = MultimodalDataCollatorWith3D(
        processor=processor,
        enable_3d=True
    )
    
    # 模拟一个batch的数据
    features = [
        {
            "video_id": "test_001",
            "resized_images": [
                np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8),
                np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8)
            ],
            "coords_3d": np.random.randn(2048, 3).astype(np.float32),  # 2视角 x 1024patches
            "conversations": [
                {"role": "system", "content": "Test system"},
                {"role": "user", "content": "<image>\n<image>\nQuestion?"},
                {"role": "assistant", "content": "Answer."}
            ]
        }
    ]
    
    try:
        batch = collator(features)
        
        print(f"✓ Batch处理成功")
        print(f"  - input_ids: {batch['input_ids'].shape}")
        print(f"  - labels: {batch['labels'].shape}")
        if 'coords_3d' in batch:
            print(f"  - coords_3d: {batch['coords_3d'].shape}")
            print(f"  - coords_3d_mask: {batch['coords_3d_mask'].shape}")
        
        return batch
        
    except Exception as e:
        print(f"❌ Batch处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(model, batch):
    """测试5: 完整的forward pass"""
    print("\n" + "="*60)
    print("测试5: Forward Pass (带3D坐标)")
    print("="*60)
    
    if model is None or batch is None:
        print("跳过: 模型或数据未准备好")
        return
    
    try:
        model.eval()
        with torch.no_grad():
            # 准备输入
            inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch.get("attention_mask"),
                "pixel_values": batch.get("pixel_values"),
                "image_grid_thw": batch.get("image_grid_thw"),
                "coords_3d": batch.get("coords_3d"),
                "coords_3d_mask": batch.get("coords_3d_mask")
            }
            
            # 移除None值
            inputs = {k: v for k, v in inputs.items() if v is not None}
            
            print(f"输入keys: {list(inputs.keys())}")
            
            # Forward
            outputs = model(**inputs)
            
            print(f"✓ Forward成功")
            print(f"  - loss: {outputs.loss if hasattr(outputs, 'loss') else 'N/A'}")
            print(f"  - logits: {outputs.logits.shape if hasattr(outputs, 'logits') else 'N/A'}")
            
    except Exception as e:
        print(f"❌ Forward失败: {e}")
        import traceback
        traceback.print_exc()


def create_minimal_test_data():
    """创建最小测试数据集"""
    print("\n" + "="*60)
    print("创建最小测试数据")
    print("="*60)
    
    # 创建测试目录
    os.makedirs("test_data", exist_ok=True)
    
    # 1. 创建训练JSON
    train_data = [
        {
            "video": "test_video_001",
            "conversations": [
                {"from": "human", "value": "UAV1:\n<image>\nUAV2:\n<image>\nDescribe the scene."},
                {"from": "gpt", "value": "This is a test scene with two UAV views."}
            ]
        }
    ]
    
    with open("test_data/train.json", "w") as f:
        json.dump(train_data, f, indent=2)
    
    # 2. 创建video映射 (带相机参数)
    video_mapping = {
        "test_video_001": {
            "Real_2_UAVs/Samples/UAV1/test_001-UAV1.jpg": {},
            "Real_2_UAVs/Samples/UAV2/test_001-UAV2.jpg": {},
            "intrinsic": [
                [1000, 0, 960],
                [0, 1000, 540],
                [0, 0, 1]
            ],
            "extrinsic_list": [
                [[1,0,0,0], [0,1,0,0], [0,0,1,5], [0,0,0,1]],
                [[1,0,0,10], [0,1,0,0], [0,0,1,5], [0,0,0,1]]
            ]
        }
    }
    
    with open("test_data/video_mapping.json", "w") as f:
        json.dump(video_mapping, f, indent=2)
    
    # 3. 创建测试图像和深度图
    os.makedirs("test_data/rgb", exist_ok=True)
    os.makedirs("test_data/depth", exist_ok=True)
    
    for uav in ["UAV1", "UAV2"]:
        # RGB
        rgb_path = f"test_data/rgb/test_001-{uav}.jpg"
        rgb_img = Image.fromarray(
            np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        )
        rgb_img.save(rgb_path)
        
        # Depth
        depth_path = f"test_data/depth/test_001-{uav}_depth.npy"
        depth = np.random.rand(1080, 1920) * 10.0
        np.save(depth_path, depth)
    
    print(f"✓ 测试数据已创建在: test_data/")
    print(f"  - train.json")
    print(f"  - video_mapping.json")
    print(f"  - rgb/ (2张图)")
    print(f"  - depth/ (2张深度图)")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print(" Qwen2.5-VL + 3D Position Encoding - 完整测试套件")
    print("="*70)
    
    # 测试1: 深度转3D
    coords_3d = test_depth_to_3d_conversion()
    
    # 测试2: 3D编码模块
    encoder = test_3d_position_encoding()
    
    # 测试3: 模型加载 (可选,需要模型文件)
    print("\n提示: 测试3-5需要下载Qwen模型,跳过可按Ctrl+C")
    try:
        model = test_model_loading()
        
        # 测试4: DataCollator
        batch = test_data_collator()
        
        # 测试5: Forward pass
        test_forward_pass(model, batch)
        
    except KeyboardInterrupt:
        print("\n用户跳过模型测试")
    except Exception as e:
        print(f"\n模型测试跳过: {e}")
    
    # 创建测试数据
    create_minimal_test_data()
    
    print("\n" + "="*70)
    print(" 测试完成!")
    print("="*70)
    print("\n下一步:")
    print("1. 准备你的真实数据 (RGB + 深度图 + 相机参数)")
    print("2. 修改video_mapping.json添加intrinsic和extrinsic_list")
    print("3. 运行训练命令:")
    print("   python train_qwen_3d.py \\")
    print("     --train_data your_train.json \\")
    print("     --video_mapping your_mapping.json \\")
    print("     --depth_root ./depths \\")
    print("     --rgb_root ./rgbs \\")
    print("     --enable_3d \\")
    print("     --vision_output_layer visual.merger")


if __name__ == "__main__":
    run_all_tests()