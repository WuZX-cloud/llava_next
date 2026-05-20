import traceback
import numpy as np
import json
from qwen_3d.model.threeD import DepthTo3DCoordinates

import numpy as np
from PIL import Image

class GlobalCoordStats:
    def __init__(self):
        self.total_points = 0
        self.sample_point_counts = []

        self.global_min = np.array([np.inf, np.inf, np.inf])
        self.global_max = np.array([-np.inf, -np.inf, -np.inf])

        self.sum = np.zeros(3)
        self.sum_sq = np.zeros(3)

        # 为了画分布（可控制数量）
        self.sampled_points = []

    def update(self, coords, max_store=20000):
        """
        coords: [N, 3]
        """
        if coords.size == 0:
            return

        coords = coords[np.isfinite(coords).all(axis=1)]

        N = coords.shape[0]
        self.total_points += N
        self.sample_point_counts.append(N)

        self.global_min = np.minimum(self.global_min, coords.min(axis=0))
        self.global_max = np.maximum(self.global_max, coords.max(axis=0))

        self.sum += coords.sum(axis=0)
        self.sum_sq += (coords ** 2).sum(axis=0)

        # reservoir sampling（用于可视化）
        if len(self.sampled_points) < max_store:
            self.sampled_points.append(coords)




# train_data=./processed_data_latest/train/sim5_sim6_merged_train_data_multi_view_follow.json
# video_mapping=./processed_data_latest/processed_mapping.json

test_json = 'processed_data_latest/train/train_data_multi_view_follow.json'
# test_json = 
video_mapping_path = 'processed_data_with_depth_npy/processed_mapping.json'
norm_type = 'no_norm'
grid_n = 7
is_w2c=True

# 加载数据
with open(test_json, 'r') as f:
    test_data = json.load(f)

with open(video_mapping_path, 'r') as f:
    video_mapping = json.load(f)


print(f"测试样本数: {len(test_data)}")

stats = GlobalCoordStats()

# 初始化推理器
coord_generator = DepthTo3DCoordinates(
                patch_size=14,
                fixed_image_size=(224,224),
                type=norm_type,
                grid_n=grid_n,
                is_w2c=is_w2c
            )

# 批量推理
results = []
result2 = []
coords_3ds = None

for idx, item in enumerate(test_data):
    video_id = item['video']
    
    if video_id not in video_mapping:
        print(f"[{idx+1}/{len(test_data)}] 跳过 {video_id}: 不在mapping中")
        continue
    
    video_data = video_mapping[video_id]

    # 加载图像和深度图
    rgb_list = []
    depth_list = []
    
    for rgb_path, depth_path in zip(video_data['image_paths'], video_data['depth_paths']):
        # 加载RGB
        rgb = Image.open(rgb_path).convert('RGB')
        rgb_list.append(rgb)
        
        # 加载深度图
        if depth_path.endswith('.npy'):
            depth = np.load(depth_path)
        else:
            depth = np.array(Image.open(depth_path))
            # uint16深度图转换为float (mm → m)
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) / 1000.0
                # print('已处理factor')
        
        depth_list.append(depth)
    
    
    # 推理
    try:
        # coords_3ds, resize_rgbs = coord_generator.process_multi_view(
        #     rgb_list=video_data['image_paths'],
        #     depth_list=video_data.get('depth_paths') ,
        #     intrinsic_list=[np.array(k, dtype=np.float32) for k in video_data.get('intrinsics', [])] ,
        #     extrinsic_list=[np.array(t, dtype=np.float32) for t in video_data.get('extrinsics', [])] ,
        # )
        coords_3ds, resize_rgbs = coord_generator.process_multi_view(
            rgb_list=rgb_list,
            depth_list=depth_list,
            intrinsic_list=[np.array(k, dtype=np.float32) for k in video_data.get('intrinsics')] ,
            extrinsic_list=[np.array(t, dtype=np.float32) for t in video_data.get('extrinsics')] ,
        )
        
        stats.update(coords_3ds)
        
        print(f"[{idx+1}/{len(test_data)}] ✓ {video_id}")
        
    except Exception as e:
        print(f"[{idx+1}/{len(test_data)}] ✗ {video_id}: {e}")
        traceback.print_exc()
        

# 保存结果
# import numpy as np

    # coords = coords_3ds  # [N, 3]

    # print("=== Basic Info ===")
    # print(f"Num points: {coords.shape[0]}")
    # print(f"dtype: {coords.dtype}")

    # mins = coords.min(axis=0)
    # maxs = coords.max(axis=0)
    # means = coords.mean(axis=0)
    # stds = coords.std(axis=0)

    # print(f"X range: {mins[0]:.3f} ~ {maxs[0]:.3f}")
    # print(f"Y range: {mins[1]:.3f} ~ {maxs[1]:.3f}")
    # print(f"Z range: {mins[2]:.3f} ~ {maxs[2]:.3f}")

    # print(f"Mean (x,y,z): {means}")
    # print(f"Std  (x,y,z): {stds}")

    # from mpl_toolkits.mplot3d import Axes3D
    # import matplotlib.pyplot as plt

    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # # 如果点太多，可以先采样
    # N = coords.shape[0]
    # if N > 50000:
    #     idx = np.random.choice(N, 50000, replace=False)
    #     plot_coords = coords[idx]
    # else:
    #     plot_coords = coords[0:256]

    # ax.scatter(
    #     plot_coords[:, 0],
    #     plot_coords[:, 1],
    #     plot_coords[:, 2],
    #     s=1
    # )

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title(f'3D Point Cloud (N={plot_coords.shape[0]})')

    # plt.savefig('z_points_3d-inv-1.png', dpi=300)
    # break

import numpy as np

mean = stats.sum / stats.total_points
std = np.sqrt(stats.sum_sq / stats.total_points - mean ** 2)

print("\n===== GLOBAL STATS =====")
print(f"Total samples: {len(stats.sample_point_counts)}")
print(f"Total points: {stats.total_points}")

print(f"Points per sample:")
print(f"  mean: {np.mean(stats.sample_point_counts):.0f}")
print(f"  min : {np.min(stats.sample_point_counts)}")
print(f"  max : {np.max(stats.sample_point_counts)}")

print(f"\nGlobal XYZ range:")
print(f"  min: {stats.global_min}")
print(f"  max: {stats.global_max}")

print(f"\nGlobal mean: {mean}")
print(f"Global std : {std}")



"""
test:
Global XYZ range:
  min: [-3.58012773 -2.43584739 -2.71288031]
  max: [3.7174879  2.98634433 6.75728317]

Global mean: [ 0.08823171 -0.00630502  0.69780148]
Global std : [0.52242162 0.35675377 0.47802588]
"""

"""
train:
===== GLOBAL STATS =====
Total samples: 13579
Total points: 11779328
Points per sample:
  mean: 867
  min : 512
  max : 1536

Global XYZ range:
  min: [-6.26221362 -2.71244273 -3.35342978]
  max: [6.35044325 5.56052641 6.94661428]

Global mean: [ 0.06678534 -0.00212686  0.70942926]
Global std : [0.51186429 0.35375278 0.49298788]
"""