
from cgitb import text
from re import A
import torch
import torch.nn as nn

import numpy as np
from typing import List, Dict, Optional, Tuple

import numpy as np
from PIL import Image
from typing import Tuple, List, Optional, Union
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm


class DepthTo3DCoordinates_0:
    """
    从深度图和相机参数生成3D坐标
    处理固定分辨率下的patch-pixel对应关系
    """
    def __init__(self, patch_size: int = 14, fixed_image_size: Tuple[int, int] = (448, 448)):
        """
        Args:
            patch_size: ViT的patch大小 (通常是14)
            fixed_image_size: 固定的图像输入尺寸 (H, W)
        """
        self.patch_size = patch_size
        self.fixed_image_size = fixed_image_size
        
        # 计算patch grid尺寸
        self.grid_h = fixed_image_size[0] // patch_size
        self.grid_w = fixed_image_size[1] // patch_size
        self.num_patches = self.grid_h * self.grid_w
        
        print(f"[DepthTo3D] Patch Grid: {self.grid_h}x{self.grid_w} = {self.num_patches} patches")
    
    def resize_depth_and_rgb(
        self, 
        depth: np.ndarray, 
        rgb: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将深度图和RGB图resize到固定尺寸
        
        Args:
            depth: (H, W) 原始深度图
            rgb: (H, W, 3) 原始RGB图
        
        Returns:
            depth_resized: (fixed_h, fixed_w)
            rgb_resized: (fixed_h, fixed_w, 3)
        """
        from PIL import Image
        
        # Depth使用nearest插值保持深度值准确性
        depth_img = Image.fromarray(depth)
        depth_resized = np.array(
            depth_img.resize((self.fixed_image_size[1], self.fixed_image_size[0]), 
                           resample=Image.NEAREST)
        )
        
        # RGB使用双线性插值
        if isinstance(rgb, np.ndarray):
            rgb_img = Image.fromarray(rgb)
        else:
            rgb_img = rgb
        rgb_resized = np.array(
            rgb_img.resize((self.fixed_image_size[1], self.fixed_image_size[0]), 
                         resample=Image.BILINEAR)
        )
        
        return depth_resized, rgb_resized
    
    def patch_average_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        计算每个patch的平均深度值
        
        Args:
            depth: (H, W) 已resize到fixed_image_size的深度图
        
        Returns:
            patch_depths: (num_patches,) 每个patch的平均深度
        """
        H, W = depth.shape
        ps = self.patch_size
        
        # 重塑为 (grid_h, patch_size, grid_w, patch_size)
        patches = depth.reshape(self.grid_h, ps, self.grid_w, ps)
        
        # 转置并平均: (grid_h, grid_w, patch_size, patch_size) -> (grid_h, grid_w)
        patches = patches.transpose(0, 2, 1, 3)  # (grid_h, grid_w, ps, ps)
        patch_depths = patches.reshape(self.grid_h, self.grid_w, -1).mean(axis=-1)
        
        # 展平为1D
        return patch_depths.flatten()  # (num_patches,)
    
    def depth_to_3d_world(
        self,
        depth: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray
    ) -> np.ndarray:
        """
        将深度图转换为世界坐标系的3D点云
        
        Args:
            depth: (H, W) 深度图 (已resize)
            intrinsic: (3, 3) 相机内参矩阵
            extrinsic: (4, 4) 相机外参矩阵 [R|t]
        
        Returns:
            coords_3d: (num_patches, 3) 每个patch中心的世界坐标 (x,y,z)
        """
        H, W = depth.shape
        ps = self.patch_size
        
        # 1. 计算每个patch的中心像素坐标
        patch_centers_v = np.arange(self.grid_h) * ps + ps // 2  # (grid_h,)
        patch_centers_u = np.arange(self.grid_w) * ps + ps // 2  # (grid_w,)
        
        # 生成网格
        uu, vv = np.meshgrid(patch_centers_u, patch_centers_v)
        uu = uu.flatten()  # (num_patches,)
        vv = vv.flatten()  # (num_patches,)
        
        # 2. 获取每个patch的平均深度
        zz = self.patch_average_depth(depth)  # (num_patches,)
        
        # 3. 反投影到相机坐标系
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        
        x_cam = (uu - cx) * zz / fx
        y_cam = (vv - cy) * zz / fy
        z_cam = zz
        
        # 4. 转换到世界坐标系
        points_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(zz)], axis=1)  # (N, 4)
        points_world = (extrinsic @ points_cam.T).T[:, :3]  # (N, 3)
        
        return points_world
    
    def process_multi_view(
        self,
        depth_list: List[np.ndarray],
        intrinsic_list: List[np.ndarray],
        extrinsic_list: List[np.ndarray],
        rgb_list: Optional[List[np.ndarray]] = None
    ) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
        """
        处理多视角数据,生成所有patch的3D坐标
        
        Args:
            depth_list: 多个深度图
            intrinsic_list: 对应的内参
            extrinsic_list: 对应的外参
            rgb_list: 对应的RGB图 (用于返回resize后的)
        
        Returns:
            all_coords_3d: (num_views * num_patches, 3)
            resized_rgb_list: resize后的RGB列表 (如果提供)
        """
        all_coords = []
        resized_rgbs = [] if rgb_list else None
        
        for i, (depth, K, T) in enumerate(zip(depth_list, intrinsic_list, extrinsic_list)):
            # Resize depth和rgb
            if rgb_list:
                depth_resized, rgb_resized = self.resize_depth_and_rgb(depth, rgb_list[i])
                resized_rgbs.append(rgb_resized)
            else:
                depth_resized, _ = self.resize_depth_and_rgb(depth, depth)
            
            # 转换为3D坐标
            coords_3d = self.depth_to_3d_world(depth_resized, K, T)
            all_coords.append(coords_3d)
        
        # 合并所有视角
        all_coords_3d = np.concatenate(all_coords, axis=0)
        
        return all_coords_3d, resized_rgbs


    """
    专门处理无人机/室外场景的 Patch-to-3D 转换模块
    """
    def __init__(
        self, 
        patch_size: int = 14, 
        fixed_image_size: Tuple[int, int] = (448, 448),
        scene_bounds: Optional[dict] = None
    ):
        """
        Args:
            patch_size: ViT patch大小
            fixed_image_size: 模型输入的固定分辨率 (H, W)
            scene_bounds: 场景的物理边界，用于归一化。
                          格式: {'min': [x,y,z], 'max': [x,y,z]}
                          如果不提供，将根据当前batch的数据动态计算（但在推理时建议固定）。
        """
        self.patch_size = patch_size
        self.fixed_image_size = fixed_image_size # (H, W)
        
        # 计算 Patch Grid
        self.grid_h = fixed_image_size[0] // patch_size
        self.grid_w = fixed_image_size[1] // patch_size
        self.num_patches = self.grid_h * self.grid_w
        
        # 预计算像素网格 (uv coordinates)
        self._init_pixel_grid()
        
        # 场景边界 (用于归一化到 -1 ~ 1)
        # 对于无人机场景，建议先统计整个数据集的 XYZ 范围填入这里
        self.scene_bounds = scene_bounds 

        print(f"[Drone3D] Grid: {self.grid_h}x{self.grid_w}, Resolution: {fixed_image_size}")

    def _init_pixel_grid(self):
        """预计算每个 Patch 中心在 Resize 后图像上的像素坐标 (u, v)"""
        # Patch 中心坐标
        y_centers = np.arange(self.grid_h) * self.patch_size + self.patch_size // 2
        x_centers = np.arange(self.grid_w) * self.patch_size + self.patch_size // 2
        
        # 生成网格
        uu, vv = np.meshgrid(x_centers, y_centers)
        self.uu_flat = uu.flatten() # (num_patches,)
        self.vv_flat = vv.flatten() # (num_patches,)

    def resize_data(
        self, 
        depth: np.ndarray, 
        rgb: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], float, float]:
        """
        调整图像大小并返回缩放比例 (用于修正内参)
        """
        orig_h, orig_w = depth.shape
        target_h, target_w = self.fixed_image_size
        
        # 计算缩放因子 (Scale Factor)
        scale_w = target_w / orig_w
        scale_h = target_h / orig_h
        
        # Resize Depth (使用 NEAREST 保持深度值的物理意义，避免插值出虚假深度)
        depth_pil = Image.fromarray(depth)
        depth_resized = np.array(depth_pil.resize((target_w, target_h), resample=Image.NEAREST))
        
        # Resize RGB (使用 BILINEAR)
        rgb_resized = None
        if rgb is not None:
            if isinstance(rgb, np.ndarray):
                rgb_pil = Image.fromarray(rgb)
            else:
                rgb_pil = rgb
            rgb_resized = np.array(rgb_pil.resize((target_w, target_h), resample=Image.BILINEAR))
            
        return depth_resized, rgb_resized, scale_w, scale_h

    def get_patch_depth_median(self, depth: np.ndarray) -> np.ndarray:
        """
        计算 Patch 深度：使用中位数 (Median) 抗噪
        无人机视角下，一个 Patch 可能同时包含地面(远)和树顶(近)，平均值会产生错误的悬空点。
        """
        ps = self.patch_size
        # Reshape view: (grid_h, ps, grid_w, ps) -> (grid_h, grid_w, ps, ps)
        patches = depth.reshape(self.grid_h, ps, self.grid_w, ps).transpose(0, 2, 1, 3)
        
        # 展平 patch 内部像素: (grid_h, grid_w, ps*ps)
        patches_flat = patches.reshape(self.grid_h, self.grid_w, -1)
        
        # 计算中位数，忽略 0 或无效值 (可选)
        # 这里直接计算中位数
        patch_z = np.median(patches_flat, axis=-1) 
        
        return patch_z.flatten() # (num_patches,)

    def normalize_world_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        将巨大的无人机世界坐标归一化到 [-1, 1] 范围
        """
        if self.scene_bounds is None:
            # 如果没有提供全局边界，则使用当前 Batch 的动态边界 (不推荐用于训练，建议固定)
            # 或者使用一个经验值，例如无人机通常在 500m 范围内
            min_xyz = coords.min(axis=0)
            max_xyz = coords.max(axis=0)
            center = (min_xyz + max_xyz) / 2
            scale = (max_xyz - min_xyz).max() / 2 + 1e-6 # 保持长宽比
            
            return (coords - center) / scale
        else:
            # 使用预设的全局边界 (推荐)
            min_b = np.array(self.scene_bounds['min'])
            max_b = np.array(self.scene_bounds['max'])
            
            # 简单的 Min-Max 归一化到 [0, 1] -> [-1, 1]
            range_b = max_b - min_b
            range_b[range_b < 1e-6] = 1.0 # 避免除零
            
            norm_01 = (coords - min_b) / range_b
            norm_11 = norm_01 * 2.0 - 1.0
            
            # 截断越界值 (Clip)，防止某些异常点跑出范围
            return np.clip(norm_11, -1.2, 1.2)

    def process_single_view(
        self,
        depth: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray, # 假设是 Camera-to-World 矩阵
        rgb: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        
        # 1. Resize & 计算缩放比例
        depth_res, rgb_res, sw, sh = self.resize_data(depth, rgb)
        
        # 2. 修正内参矩阵
        fx = intrinsic[0, 0] * sw
        fy = intrinsic[1, 1] * sh
        cx = intrinsic[0, 2] * sw
        cy = intrinsic[1, 2] * sh
        
        # 3. 获取 Patch 深度 Z
        z_cam = self.get_patch_depth_median(depth_res) # (N,)
        
        # 4. 反投影 (Back-projection) -> Camera Coordinate System
        # x = (u - cx) * z / fx
        x_cam = (self.uu_flat - cx) * z_cam / fx
        y_cam = (self.vv_flat - cy) * z_cam / fy
        
        # 组合成 (N, 4) 的齐次坐标: [x, y, z, 1]
        ones = np.ones_like(z_cam)
        points_cam_homo = np.stack([x_cam, y_cam, z_cam, ones], axis=1) 
        
        # 5. 坐标变换 -> World Coordinate System
        # 假设 extrinsic 是 C2W (Camera Pose): P_world = T_c2w @ P_cam
        # points_cam_homo.T 是 (4, N)
        # 结果 (4, N) -> 转置回 (N, 4)
        points_world = (extrinsic @ points_cam_homo.T).T
        
        # 取前三个维度 (x, y, z)
        coords_world = points_world[:, :3]
        
        # 6. 归一化 (非常关键步骤)
        coords_norm = self.normalize_world_coords(coords_world)
        
        return coords_norm, rgb_res

    def process_batch(
        self,
        depth_list: List[np.ndarray],
        intrinsic_list: List[np.ndarray],
        extrinsic_list: List[np.ndarray],
        rgb_list: Optional[List[np.ndarray]] = None
    ) -> Tuple[torch.Tensor, Optional[List[np.ndarray]]]:
        """
        处理一个 Batch 的数据，返回 Tensor
        """
        batch_coords = []
        batch_rgbs = [] if rgb_list else None
        
        for i, (d, K, T) in enumerate(zip(depth_list, intrinsic_list, extrinsic_list)):
            r = rgb_list[i] if rgb_list else None
            c_norm, r_res = self.process_single_view(d, K, T, r)
            
            batch_coords.append(c_norm)
            if batch_rgbs is not None:
                batch_rgbs.append(r_res)
        
        # 转换为 Tensor: (B, N, 3)
        import torch
        coords_tensor = torch.tensor(np.array(batch_coords), dtype=torch.float32)
        
        return coords_tensor, batch_rgbs
# ==================== 3D Position Encoding Module ====================

import numpy as np
import torch
from PIL import Image
from typing import Tuple, List, Optional, Union

class DepthTo3DCoordinates:
    """
    从深度图和相机参数生成3D坐标 (优化版)
    集成功能：
    1. 自动处理 Resize 后的内参缩放
    2. 使用中位数 (Median) 提取 Patch 深度，抗噪
    3. 鲁棒动态归一化，将任意尺度的场景映射到 [-1, 1]
    """
    def __init__(self, patch_size: int = 14, fixed_image_size: Tuple[int, int] = (448, 448), type='norm', grid_n: int = 3, is_w2c: bool=False):
        """
        Args:
            patch_size: ViT的patch大小 (通常是14)
            fixed_image_size: 固定的图像输入尺寸 (H, W)
        """
        self.patch_size = patch_size
        self.fixed_image_size = fixed_image_size
        self.grid_n = grid_n
        self.is_w2c = is_w2c
        
        # 计算patch grid尺寸
        self.grid_h = fixed_image_size[0] // patch_size
        self.grid_w = fixed_image_size[1] // patch_size
        self.num_patches = self.grid_h * self.grid_w

        # 归一化坐标的类型
        self.type = type  # norm : 默认， batch_norm : batch级归一化, no_norm ：无归一化
        
        if self.type == 'multi_points' :

            self._init_subpixel_grid()
            print(f"[DepthToMultiPoint] Initialized: {fixed_image_size} -> {self.grid_h}x{self.grid_w} patches.")
            print(f"[DepthToMultiPoint] Sampling {grid_n}x{grid_n} points per patch.")
        
        else :
            # 预计算像素网格 (uv coordinates)，避免每次forward重复计算
            self._init_pixel_grid()
            print(f"[DepthTo3D] Initialized: {fixed_image_size} -> {self.grid_h}x{self.grid_w} patches")

        

    def _init_pixel_grid(self):
        """预计算 Patch 中心的像素坐标"""
        # 坐标对应 Resize 后的图像
        y_centers = np.arange(self.grid_h) * self.patch_size + self.patch_size // 2
        x_centers = np.arange(self.grid_w) * self.patch_size + self.patch_size // 2
        
        uu, vv = np.meshgrid(x_centers, y_centers)
        self.uu_flat = uu.flatten() # (num_patches,)
        self.vv_flat = vv.flatten() # (num_patches,)

    def _init_subpixel_grid(self):
        """
        预计算每个 Patch 内部 KxK 个点的像素坐标。
        结果保存在 self.uu_flat, self.vv_flat，形状为 (num_patches, points_per_patch)
        """
        # 1. 计算 Patch 内部的采样偏移量 (相对于 Patch 左上角)
        # 例如 patch_size=14, grid_n=3, step=4.66, offsets=[2.33, 7.0, 11.66]
        step = self.patch_size / self.grid_n
        offsets = np.arange(self.grid_n) * step + step / 2.0
        
        # 2. 生成所有 Patch 的左上角坐标
        patch_y = np.arange(self.grid_h) * self.patch_size
        patch_x = np.arange(self.grid_w) * self.patch_size
        
        # 3. 利用 Meshgrid 和 Broadcasting 生成全图所有采样点
        # grid_indices: (grid_h, grid_w)
        gy, gx = np.meshgrid(patch_y, patch_x, indexing='ij')
        
        # sub_offsets: (grid_n, grid_n)
        soy, sox = np.meshgrid(offsets, offsets, indexing='ij')
        
        # 核心广播操作:
        # gy: (H, W, 1, 1) + soy: (1, 1, n, n) -> (H, W, n, n)
        vv_grid = gy[:, :, None, None] + soy[None, None, :, :]
        uu_grid = gx[:, :, None, None] + sox[None, None, :, :]
        
        # 4. 展平为 (num_patches, points_per_patch)
        # (H, W, n, n) -> (H*W, n*n)
        self.uu_flat = uu_grid.reshape(self.num_patches, -1)
        self.vv_flat = vv_grid.reshape(self.num_patches, -1)
    
    def get_subsampled_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        根据预计算的 grid 坐标，从深度图中采样深度值。
        包含去噪逻辑：如果某个子点无效，用该 Patch 的中位数填充。
        """
        h, w = depth.shape
        # 将 float 坐标转为整数索引，并限制在图像范围内
        u_idx = np.clip(np.round(self.uu_flat).astype(int), 0, w - 1)
        v_idx = np.clip(np.round(self.vv_flat).astype(int), 0, h - 1)
        
        # 索引深度: (num_patches, points_per_patch)
        z_samples = depth[v_idx, u_idx]
        
        # --- 鲁棒性处理 ---
        # 计算每个 Patch 的中位数 (忽略 NaN 和 0)
        # 注意：如果整个 patch 都是 0，nanmedian 会报 warning 或返回 nan，需要处理
        with np.errstate(all='ignore'):
            # 将 0 视为 NaN 以便计算 median
            z_for_median = z_samples.copy()
            z_for_median[z_for_median <= 0] = np.nan
            patch_medians = np.nanmedian(z_for_median, axis=1, keepdims=True)
        
        # 将计算出的 NaN median (即整个 patch 无效) 替换为 0
        patch_medians = np.nan_to_num(patch_medians, nan=0.0)
        
        # 找出无效的采样点 (<=0 或 NaN)
        invalid_mask = (z_samples <= 0) | np.isnan(z_samples)
        
        # 用该 Patch 的中位数替换无效点
        # 这样即使某个像素是黑洞，只要 Patch 里有点，就能保持几何平面不崩塌
        z_final = np.where(invalid_mask, patch_medians, z_samples)
        
        return z_final

    def load_depth_from_png(self, path_or_arr: Union[str, np.ndarray], scale_factor: float = 1000.0) -> np.ndarray:
        """
        工具函数：读取 uint16 PNG 深度图并转为米
        """
        # if isinstance(path_or_arr, str):
        #     img = Image.open(path_or_arr)
        #     depth_raw = np.array(img)
        # else:
        #     depth_raw = path_or_arr
            
        # # 转换为 float32 并除以比例因子 (毫米 -> 米)
        # depth_m = depth_raw.astype(np.float32) / scale_factor
        saved_path = path_or_arr
        if '.png' in saved_path:
            depth_img = Image.open(saved_path)
            depth_raw = np.array(depth_img, dtype=np.float32)
            recovered_np = depth_raw / scale_factor
        elif '.npy' in saved_path:
            recovered_np = np.load(saved_path)
        else:
            raise "depth path is error"
        depth_m = recovered_np
        return depth_m

    def resize_data(self, depth: np.ndarray, rgb: Optional[np.ndarray] = None):
        """
        Resize 数据并返回缩放比例 (用于修正内参)
        """
        orig_h, orig_w = depth.shape
        target_h, target_w = self.fixed_image_size
        
        # 计算缩放因子 (Scale Factor)
        scale_w = target_w / orig_w
        scale_h = target_h / orig_h
        
        # Depth: 使用 NEAREST 保持物理意义，避免插值出虚假深度
        depth_pil = Image.fromarray(depth)
        depth_resized = np.array(depth_pil.resize((target_w, target_h), resample=Image.NEAREST))
        # depth_resized = np.array(depth_pil.resize((target_w, target_h), resample=Image.BICUBIC))
        
        # RGB: 使用 Bilinear
        rgb_resized = None
        if rgb is not None:
            if isinstance(rgb, np.ndarray):
                rgb_pil = Image.fromarray(rgb)
            else:
                rgb_pil = rgb
            rgb_resized = rgb_pil.resize((target_w, target_h), resample=Image.BILINEAR)
            
        return depth_resized, rgb_resized, scale_w, scale_h

    def get_patch_depth_median(self, depth: np.ndarray) -> np.ndarray:
        """
        核心改进：使用中位数 (Median) 而非平均值
        解决无人机视角下，一个 Patch 同时包含近处树冠和远处地面的问题
        """
        ps = self.patch_size
        # Reshape: (H, W) -> (grid_h, ps, grid_w, ps) -> (grid_h, grid_w, ps, ps)
        patches = depth.reshape(self.grid_h, ps, self.grid_w, ps).transpose(0, 2, 1, 3)
        
        # 展平 patch 内部: (grid_h, grid_w, ps*ps)
        patches_flat = patches.reshape(self.grid_h, self.grid_w, -1)
        
        # 计算中位数 (axis=-1)
        patch_z = np.median(patches_flat, axis=-1)
        
        return patch_z.flatten() # (num_patches,)

    def robust_dynamic_normalize(self, coords: np.ndarray) -> np.ndarray:
        """
        核心改进：鲁棒动态归一化 (Robust Dynamic Normalization)
        
        不依赖全局 min/max，而是利用当前点云的统计特性（百分位数）进行归一化。
        这能保证无论无人机在世界坐标的 (0,0,0) 还是 (50000, 50000, 200)，
        输出给模型的坐标永远分布在 [-1, 1] 附近。
        """
        # 1. 过滤无效深度 (防止天空/极远噪点破坏统计)
        # 假设 Z 是深度方向，或者简单地认为所有坐标都应有意义
        valid_mask = np.isfinite(coords).all(axis=1)
        if valid_mask.sum() == 0:
            return np.zeros_like(coords) # 防止全空崩溃

        valid_coords = coords[valid_mask]

        # 2. 计算统计边界 (使用 Percentile 2% - 98%)
        # 使用 min/max 容易被重建中的离群点（飞点）带偏
        p_min = np.percentile(valid_coords, 2, axis=0)
        p_max = np.percentile(valid_coords, 98, axis=0)
        
        # 3. 计算中心和缩放尺度
        center = (p_min + p_max) / 2.0
        
        # 使用各向同性缩放 (Isotropic Scaling): 取 XYZ 中最大的跨度作为分母
        # 这样可以保持物体的长宽比 (Aspect Ratio) 不变，球体不会变椭球
        max_range = (p_max - p_min).max()
        scale = max_range / 2.0
        
        if scale < 1e-6: scale = 1.0 # 防止除零

        # 4. 应用归一化
        coords_norm = (coords - center) / scale
        
        # 5. 截断 (Clip)
        # 因为用了 98% 分位点，剩下 2% 的噪点会超过 [-1, 1]，需要截断防止 Transformer 梯度爆炸
        coords_norm = np.clip(coords_norm, -1.2, 1.2)
        
        return coords_norm

    def process_single_view(
        self,
        depth_input: Union[np.ndarray, str],
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        rgb_input: Optional[Union[np.ndarray, str]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        处理单张视图
        """
        # 0. 加载深度图 (如果是路径)
        if isinstance(depth_input, str):
            depth_m = self.load_depth_from_png(depth_input)
        else:
            depth_m = depth_input # 假设已经是米单位，如果也是uint16需外部处理或调用load_depth_from_png

        # 0. 加载RGB (如果是路径)
        rgb_arr = None
        if rgb_input is not None:
            if isinstance(rgb_input, str):
                rgb_arr = np.array(Image.open(rgb_input))
            else:
                rgb_arr = rgb_input

        # 1. Resize 并获取缩放比例
        depth_res, rgb_res, sw, sh = self.resize_data(depth_m, rgb_arr)
        
        # 2. 修正内参 (关键步骤)
        # 原始 K: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        fx = intrinsic[0, 0] * sw
        fy = intrinsic[1, 1] * sh
        cx = intrinsic[0, 2] * sw
        cy = intrinsic[1, 2] * sh
        
        # 3. 获取 Patch 深度 (使用中位数)
        z_cam = self.get_patch_depth_median(depth_res)
        
        # 4. 反投影 (Back-projection) -> Camera Coords
        # x = (u - cx) * z / fx
        x_cam = (self.uu_flat - cx) * z_cam / fx
        y_cam = (self.vv_flat - cy) * z_cam / fy
        
        # 组合齐次坐标 (N, 4)
        points_cam_homo = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=1)
        
        # 5. 转换到世界坐标 (C2W)
        # 假设 extrinsic 是 C2W 矩阵 (Camera Pose)
        # Formula: P_world = T_c2w * P_cam
        # (4, 4) @ (4, N) -> (4, N) -> Transpose -> (N, 4)
        points_world = (extrinsic @ points_cam_homo.T).T[:, :3]

        if self.is_w2c:
            # 1. extrinsic 是 W2C（和方法一一致）
            # print('开启w2c转c2w')
            T_w2c = extrinsic  # (4, 4)

            # 2. 转为 C2W
            T_c2w = np.linalg.inv(T_w2c)
            points_world = (T_c2w @ points_cam_homo.T).T[:, :3]
        
        # 6. 归一化 (关键步骤：解决 Loss=40)
        coords_norm = self.robust_dynamic_normalize(points_world)

        ## 不单张归一化化，或者采用batch级归一化
        # coords_norm = points_world
        
        return coords_norm, rgb_res


    def process_single_view_no_norm(
        self,
        depth_input: Union[np.ndarray, str],
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        rgb_input: Optional[Union[np.ndarray, str]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        处理单张视图
        """
        # 0. 加载深度图 (如果是路径)
        if isinstance(depth_input, str):
            depth_m = self.load_depth_from_png(depth_input)
        else:
            depth_m = depth_input # 假设已经是米单位，如果也是uint16需外部处理或调用load_depth_from_png

        # 0. 加载RGB (如果是路径)
        rgb_arr = None
        if rgb_input is not None:
            if isinstance(rgb_input, str):
                rgb_arr = np.array(Image.open(rgb_input))
            else:
                rgb_arr = rgb_input

        # 1. Resize 并获取缩放比例
        depth_res, rgb_res, sw, sh = self.resize_data(depth_m, rgb_arr)
        
        # 2. 修正内参 (关键步骤)
        # 原始 K: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        fx = intrinsic[0, 0] * sw
        fy = intrinsic[1, 1] * sh
        cx = intrinsic[0, 2] * sw
        cy = intrinsic[1, 2] * sh
        
        # 3. 获取 Patch 深度 (使用中位数)
        z_cam = self.get_patch_depth_median(depth_res)
        
        # 4. 反投影 (Back-projection) -> Camera Coords
        # x = (u - cx) * z / fx
        x_cam = (self.uu_flat - cx) * z_cam / fx
        y_cam = (self.vv_flat - cy) * z_cam / fy
        
        # 组合齐次坐标 (N, 4)
        points_cam_homo = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=1)
        
        # 5. 转换到世界坐标 (C2W)
        # 假设 extrinsic 是 C2W 矩阵 (Camera Pose)
        # Formula: P_world = T_c2w * P_cam
        # (4, 4) @ (4, N) -> (4, N) -> Transpose -> (N, 4)
        points_world = (extrinsic @ points_cam_homo.T).T[:, :3]

        if self.is_w2c:
            # 1. extrinsic 是 W2C（和方法一一致）
            # print('开启w2c转c2w')
            T_w2c = extrinsic  # (4, 4)

            # 2. 转为 C2W
            T_c2w = np.linalg.inv(T_w2c)
            points_world = (T_c2w @ points_cam_homo.T).T[:, :3]
        
        # 6. 归一化 (关键步骤：解决 Loss=40)
        # coords_norm = self.robust_dynamic_normalize(points_world)

        ## 不单张归一化化，或者采用batch级归一化
        coords_norm = points_world
        
        return coords_norm, rgb_res

    def process_multi_view(
        self,
        depth_list: List[Union[str, np.ndarray]],
        intrinsic_list: List[np.ndarray],
        extrinsic_list: List[np.ndarray],
        rgb_list: Optional[List[Union[str, np.ndarray]]] = None
    ) -> Tuple[torch.Tensor, Optional[List[np.ndarray]]]:
        """
        处理一个 Batch，返回 Tensor
        
        Returns:
            coords_tensor: (B, N, 3) 归一化后的坐标
            rgb_results: List of resized RGB images
        """
        if self.type == 'no_norm':
            # print('depth type is no_norm')
            return self.process_multi_view_no_norm(
                depth_list=depth_list,
                intrinsic_list=intrinsic_list,
                extrinsic_list=extrinsic_list,
                rgb_list=rgb_list
                )
        if self.type == 'batch_norm':
            # print('depth type is batch_norm')
            return self.process_multi_view_batch_norm(
                depth_list=depth_list,
                intrinsic_list=intrinsic_list,
                extrinsic_list=extrinsic_list,
                rgb_list=rgb_list
            )
        if self.type == 'multi_points':
            # print(f'depth type is multi_points, grid is {self.grid_n}')
            return self.process_multi_view_multi_points(
                depth_list=depth_list,
                intrinsic_list=intrinsic_list,
                extrinsic_list=extrinsic_list,
                rgb_list=rgb_list
            )

        batch_coords = []
        batch_rgbs = [] if rgb_list else None
        
        for i in range(len(depth_list)):
            d_in = depth_list[i]
            K = intrinsic_list[i]
            T = extrinsic_list[i]
            r_in = rgb_list[i] if rgb_list else None
            
            # 处理单张
            c_norm, r_res = self.process_single_view(d_in, K, T, r_in)
            
            batch_coords.append(c_norm)
            if batch_rgbs is not None:
                batch_rgbs.append(r_res)
        
        # 转换为 Tensor
        coords_tensor = np.concatenate(batch_coords, axis=0)
        
        return coords_tensor, batch_rgbs

    def process_multi_view_batch_norm(
        self,
        depth_list: List[Union[str, np.ndarray]],
        intrinsic_list: List[np.ndarray],
        extrinsic_list: List[np.ndarray],
        rgb_list: Optional[List[Union[str, np.ndarray]]] = None
    ) -> Tuple[torch.Tensor, Optional[List[np.ndarray]]]:
        """
        处理一个 Batch，返回 Tensor
        
        Returns:
            coords_tensor: (B, N, 3) 归一化后的坐标
            rgb_results: List of resized RGB images
        """
        batch_coords = []
        batch_rgbs = [] if rgb_list else None
        
        for i in range(len(depth_list)):
            d_in = depth_list[i]
            K = intrinsic_list[i]
            T = extrinsic_list[i]
            r_in = rgb_list[i] if rgb_list else None
            
            # 处理单张
            c_norm, r_res = self.process_single_view_no_norm(d_in, K, T, r_in)
            
            batch_coords.append(c_norm)
            if batch_rgbs is not None:
                batch_rgbs.append(r_res)
        
        # 转换为 Tensor
        # coords_tensor = np.concatenate(batch_coords, axis=0)

        # batch级的归一化
        # Stack: (B, N, 3)
        all_raw_coords = np.stack(batch_coords, axis=0)
        
        # 2. 计算整个 Batch 的统计信息 (Global Statistics)
        # 展平所有 Batch 的所有点来计算 min/max
        flat_all = all_raw_coords.reshape(-1, 3)
        
        # 过滤无效点
        valid_mask = np.isfinite(flat_all).all(axis=1)
        if valid_mask.sum() == 0:
            # 极端情况处理
            return np.zeros_like(all_raw_coords), batch_rgbs, {}
            
        valid_data = flat_all[valid_mask]

        # 统一计算 Center 和 Scale
        p_min = np.percentile(valid_data, 2, axis=0)
        p_max = np.percentile(valid_data, 98, axis=0)
        
        center = (p_min + p_max) / 2.0
        max_range = (p_max - p_min).max()
        scale = max_range / 2.0
        if scale < 1e-6: scale = 1.0
        
        # 3. 应用统一归一化
        # 利用 broadcasting: (B, N, 3) - (3,) -> (B, N, 3)
        coords_norm = (all_raw_coords - center) / scale
        coords_norm = np.clip(coords_norm, -1.2, 1.2)
        
        # 4. 返回元数据 (Meta Info)，以便后续可能需要还原
        meta_info = {
            "center": center,
            "scale": scale,
            "min_bound": p_min,
            "max_bound": p_max
        }
        coords_norm = coords_norm.reshape(-1, 3)
        # print(f"coords_norm shape is {coords_norm.shape}")

        return coords_norm, batch_rgbs
        

    def process_multi_view_no_norm(
        self,
        depth_list: List[Union[str, np.ndarray]],
        intrinsic_list: List[np.ndarray],
        extrinsic_list: List[np.ndarray],
        rgb_list: Optional[List[Union[str, np.ndarray]]] = None
    ) -> Tuple[torch.Tensor, Optional[List[np.ndarray]]]:
        """
        处理一个 Batch，返回 Tensor
        
        Returns:
            coords_tensor: (B, N, 3) 归一化后的坐标
            rgb_results: List of resized RGB images
        """
        batch_coords = []
        batch_rgbs = [] if rgb_list else None
        
        for i in range(len(depth_list)):
            d_in = depth_list[i]
            K = intrinsic_list[i]
            T = extrinsic_list[i]
            r_in = rgb_list[i] if rgb_list else None
            
            # 处理单张
            c_norm, r_res = self.process_single_view_no_norm(d_in, K, T, r_in)
            
            batch_coords.append(c_norm)
            if batch_rgbs is not None:
                batch_rgbs.append(r_res)
        
        # 转换为 Tensor
        coords_tensor = np.concatenate(batch_coords, axis=0)

        coords_norm = coords_tensor

        return coords_norm, batch_rgbs
        

    def process_single_view_multi_points(
        self,
        depth_input: Union[np.ndarray, str],
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        rgb_input: Optional[Union[np.ndarray, str]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        处理单张视图
        Returns:
            coords_norm: (num_patches, points_per_patch, 3)
            rgb_res: Resized RGB image
        """
        # 0. 加载
        if isinstance(depth_input, str):
            depth_m = self.load_depth_from_png(depth_input)
        else:
            depth_m = depth_input
            
        rgb_arr = None
        if rgb_input is not None:
            if isinstance(rgb_input, str):
                rgb_arr = np.array(Image.open(rgb_input))
            else:
                rgb_arr = rgb_input

        # 1. Resize
        depth_res, rgb_res, sw, sh = self.resize_data(depth_m, rgb_arr)
        
        # 2. 修正内参
        fx = intrinsic[0, 0] * sw
        fy = intrinsic[1, 1] * sh
        cx = intrinsic[0, 2] * sw
        cy = intrinsic[1, 2] * sh
        
        # 3. 获取多点深度 (N, K*K)
        z_cam = self.get_subsampled_depth(depth_res)
        
        # 4. 反投影 (Back-projection) -> Camera Coords
        # (N, K*K)
        x_cam = (self.uu_flat - cx) * z_cam / fx
        y_cam = (self.vv_flat - cy) * z_cam / fy
        
        # 组合齐次坐标 Stack: (N, K*K, 4)
        ones = np.ones_like(z_cam)
        points_cam_homo = np.stack([x_cam, y_cam, z_cam, ones], axis=-1)
        
        # 5. 转换到世界坐标
        # Extrinsic (C2W): (4, 4)
        # 为了矩阵乘法方便，先 reshape 成 (Total_Points, 4)
        N_patches = points_cam_homo.shape[0]
        K_points = points_cam_homo.shape[1]
        
        flat_cam = points_cam_homo.reshape(-1, 4) # (N*K*K, 4)
        
        # Formula: P_world = T_c2w * P_cam
        # (N*K*K, 4) @ (4, 4).T -> (N*K*K, 4)
        flat_world = flat_cam @ extrinsic.T

        if self.is_w2c:
            # 1. extrinsic 是 W2C（和方法一一致）
            # print('开启w2c转c2w')
            T_w2c = extrinsic  # (4, 4)
            # 2. 转为 C2W
            T_c2w = np.linalg.inv(T_w2c)
            flat_world = flat_cam @ T_c2w.T
        
        # 取前3维并 Reshape 回 (N, K*K, 3)
        points_world = flat_world[:, :3].reshape(N_patches, K_points, 3)
        
        # 6. 归一化
        # coords_norm = self.robust_dynamic_normalize(points_world)
        coords_norm = points_world
        
        return coords_norm, rgb_res

    def process_multi_view_multi_points(
        self,
        depth_list: List[Union[str, np.ndarray]],
        intrinsic_list: List[np.ndarray],
        extrinsic_list: List[np.ndarray],
        rgb_list: Optional[List[Union[str, np.ndarray]]] = None
    ) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
        """
        处理 Batch 数据
        Returns:
            coords_batch: (B, num_patches, points_per_patch, 3)
        """
        batch_coords = []
        batch_rgbs = [] if rgb_list else None
        
        for i in range(len(depth_list)):
            d_in = depth_list[i]
            K = intrinsic_list[i]
            T = extrinsic_list[i]
            r_in = rgb_list[i] if rgb_list else None
            
            c_norm, r_res = self.process_single_view_multi_points(d_in, K, T, r_in)
            
            batch_coords.append(c_norm)
            if batch_rgbs is not None:
                batch_rgbs.append(r_res)
        
        # Stack to (B*N, K*K, 3)
        # coords_tensor = np.stack(batch_coords, axis=0)
        coords_tensor = np.concatenate(batch_coords, axis=0)
        
        return coords_tensor, batch_rgbs




class ThreeDPositionEncoding_0(nn.Module):
    """
    3D位置编码模块 - 为视觉patch token注入3D空间信息
    """
    def __init__(
        self,
        hidden_size: int = 3584,  # Qwen2.5-VL-7B
        num_3d_freqs: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_3d_freqs = num_3d_freqs
        
        # 3D坐标编码维度: xyz各用sin/cos
        self.coord_encoding_dim = 3 * 2 * num_3d_freqs
        
        # 坐标编码投影网络
        self.coord_projector = nn.Sequential(
            nn.Linear(self.coord_encoding_dim, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # 融合门控 (residual weight)
        # self.fusion_alpha = nn.Parameter(torch.tensor(0.1))
        self.fusion_alpha = nn.Parameter(torch.tensor([0.1]))
        
        print(f"[3D Position Encoding] 初始化完成: hidden_size={hidden_size}, freqs={num_3d_freqs}")
    
    def positional_encoding_3d(self, coords_3d: torch.Tensor) -> torch.Tensor:
        """
        3D坐标的正弦位置编码
        
        Args:
            coords_3d: (B, N, 3) - (x, y, z)坐标
        
        Returns:
            encoding: (B, N, coord_encoding_dim)
        """
        B, N, _ = coords_3d.shape
        device = coords_3d.device
        
        # 生成频率
        freqs = torch.pow(
            10000.0,
            -torch.arange(0, self.num_3d_freqs, device=device, dtype=torch.float32) / self.num_3d_freqs
        )
        
        encoded_list = []
        for dim in range(3):  # x, y, z
            coord = coords_3d[:, :, dim:dim+1]  # (B, N, 1)
            angles = coord * freqs.view(1, 1, -1)  # (B, N, num_freqs)
            encoded_list.extend([torch.sin(angles), torch.cos(angles)])
        
        encoding = torch.cat(encoded_list, dim=-1)  # (B, N, 3*2*num_freqs)
        return encoding
    
    def forward(
        self,
        vision_tokens: torch.Tensor,
        coords_3d: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        为vision token添加3D位置信息
        
        Args:
            vision_tokens: (B, N, hidden_size) - 视觉编码器输出
            coords_3d: (B, N, 3) - 对应的3D坐标
            attention_mask: (B, N) - mask掉padding的位置
        
        Returns:
            enhanced_tokens: (B, N, hidden_size)
        """
        if coords_3d is None:
            return vision_tokens
        
        # 1. 编码3D坐标
        pos_encoding = self.positional_encoding_3d(coords_3d)  # (B, N, coord_encoding_dim)
        pos_encoding = pos_encoding.to(dtype=vision_tokens.dtype)
        pos_features = self.coord_projector(pos_encoding)
        # pos_features = self.coord_projector(pos_encoding)      # (B, N, hidden_size)
        
        # 2. 加权融合 (residual connection with learnable weight)
        enhanced = vision_tokens + self.fusion_alpha * pos_features
        
        # 3. 如果有mask,将padding位置置零
        if attention_mask is not None:
            enhanced = enhanced * attention_mask.unsqueeze(-1)
        
        return enhanced



class TextAwareGating(nn.Module):
    def __init__(self, vision_dim, text_dim):
        super().__init__()
        
        # 1. 文本对齐层：把 Text (3584) 映射到 Vision (1280)
        self.text_proj = nn.Linear(text_dim, vision_dim)
        
        # 2. 视觉对齐层（可选）：把 Vision 映射到交互空间
        self.vision_proj = nn.Linear(vision_dim, vision_dim)
        
        # 3. 门控生成层
        # 输入是 (Vision * Text) 的结果，代表了“匹配特征”
        # 这种显式的乘法比 Concat 更容易让模型学会“相关性”
        self.gate_generator = nn.Sequential(
            nn.Linear(vision_dim, vision_dim // 2),
            nn.LayerNorm(vision_dim // 2), # 加 Norm 训练更稳
            nn.ReLU(),
            nn.Linear(vision_dim // 2, vision_dim),
            nn.Sigmoid() # 输出 0~1 的系数
            # nn.Tanh()
        )
        
        # 初始化 gate_generator 最后一层 bias 为负数 (如 -2)
        # 这样初始 gate 值会偏向 0 (Sigmoid(-2) ≈ 0.12)，起步更平滑
        nn.init.constant_(self.gate_generator[-2].bias, -2.0)

    def forward(self, vision_tokens, text_global):
        """
        vision_tokens: [B, N, C_vis]
        text_global:   [B, C_text] (聚合后的文本特征)
        """
        # 1. 对齐文本 [B, C_text] -> [B, C_vis]
        text_feat = self.text_proj(text_global) 
        
        # 2. 对齐视觉 [B, N, C_vis]
        vis_feat = self.vision_proj(vision_tokens)
        
        # 3. 扩展文本以匹配视觉 Token 数量 [B, 1, C_vis]
        text_feat_expanded = text_feat.unsqueeze(1)
        
        # 4. === 核心：显式计算交互/相似度 ===
        # 使用元素级乘法。如果视觉特征和文本特征在某个维度上都活跃（匹配），
        # 这里的数值就会很大。这相当于一种 Attention 机制。
        interaction = vis_feat * text_feat_expanded # [B, N, C_vis]
        
        # 5. 生成门控系数
        gate = self.gate_generator(interaction) # [B, N, C_vis]
        
        return gate

import torch
import torch.nn as nn
import torch.nn.functional as F

class RawScoreCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 定义线性投影层
        self.aq_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.ak_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.av_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.aproj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        query: [Batch, N_q, Dim] (Vision Tokens)
        key:   [Batch, N_k, Dim] (Text Sequence)
        value: [Batch, N_k, Dim] (Text Sequence)
        mask:  [Batch, N_k]      (0为Padding，1为有效) 注意这里逻辑可能和PyTorch官方相反，请根据你的数据调整
        """
        B, N_q, C = query.shape
        _, N_k, _ = key.shape

        # 1. 线性投影 + 分头 [B, N, Num_Heads, Head_Dim] -> [B, Num_Heads, N, Head_Dim]
        q = self.aq_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.ak_proj(key).reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.av_proj(value).reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 2. 计算原始分数 (Raw Scores / Logits)
        # q: [B, H, N_q, d]
        # k.T: [B, H, d, N_k]
        # attn_logits: [B, H, N_q, N_k]
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale

        # 3. 处理 Mask (如果有)
        if mask is not None:
            # 假设 mask 是 [B, N_k] (1有效，0无效)
            #我们需要把它广播到 [B, 1, 1, N_k]
            mask_expanded = mask.reshape(B, 1, 1, N_k)
            # 将无效位置的分数设为极小值 (-inf)，这样 Softmax 后就是 0
            attn_logits = attn_logits.masked_fill(mask_expanded == 0, float('-inf'))

        # =======================================================
        # 【关键点】这里截获了attn_logits (Softmax 之前)
        # 你可以在这里对 logits 进行处理，或者直接返回它
        # =======================================================

        # 4. Softmax + Weighted Sum (常规流程)
        attn_weights = F.softmax(attn_logits, dim=-1) # [B, H, N_q, N_k]
        attn_weights = self.attn_drop(attn_weights)
        
        x = (attn_weights @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.aproj(x)
        x = self.proj_drop(x)

        # 5. 返回结果
        # 我们额外返回 attn_logits (为了节省显存，可以只返回平均后的 logits)
        # 如果你想分析每个 Head，就直接返回 attn_logits
        mean_logits = attn_logits.mean(dim=1) # [B, N_q, N_k] 对多头取平均
        
        return x, attn_weights, mean_logits




class TextGuidedCrossAttentionGate(nn.Module):
    def __init__(self, vision_dim, text_dim, num_heads=8, gate_mode="softplus_mean"):
        super().__init__()

        self.gate_mode = gate_mode
        attn_dim = 768

        # 1. 文本映射层：将文本维度映射到视觉维度，以便进行 Attention
        self.text_proj = nn.Linear(text_dim, attn_dim)

        self.vis_proj = nn.Linear(vision_dim, attn_dim)

        # 2. 【关键新增】Pre-Norm 层，强制拉齐分布
        self.ln_v = nn.LayerNorm(vision_dim) # 给 vision_tokens 用
        self.ln_t = nn.LayerNorm(text_dim)   # 给 text_features 用

        # 2. Cross-Attention 组件
        # Query = Vision (图像想找什么?)
        # Key/Value = Text (文本提供了什么线索?)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )

        # gate_mode: sigmoid_max 和 raw_sigmoid 需要额外的可学习参数
        if gate_mode in ("sigmoid_max", "raw_sigmoid"):
            self.gate_linear = nn.Linear(1, 1)
            nn.init.constant_(self.gate_linear.bias, -3.0)
            nn.init.ones_(self.gate_linear.weight)
        
        # 3. 门控生成层 (MLP)
        # 输入是 Attention 的输出 (即图像从文本中“检索”到的相关信息)
        # self.gate_generator = nn.Sequential(
        #     nn.Linear(vision_dim, vision_dim // 2),
        #     nn.LayerNorm(vision_dim // 2),
        #     nn.ReLU(),
        #     nn.Linear(vision_dim // 2, vision_dim),
        #     nn.Sigmoid() 
        # )
        
        # # 4. 初始化策略：偏置设为负数，初始状态下 Gate 关闭 (Sparse Start)
        # nn.init.constant_(self.gate_generator[-2].bias, -2.0)
        # # 此外，将最后一层的 weight 初始化得很小，防止初始的大梯度破坏预训练特征
        # nn.init.normal_(self.gate_generator[-2].weight, std=0.001)

        # 在模型初始化时替换
        # self.cross_attn_logits = RawScoreCrossAttention(dim=vision_dim, num_heads=8)

        # self.gate_projector = nn.Linear(1, 1)

    def forward(self, vision_tokens, text_sequence, text_mask=None):
        """
        vision_tokens: [B, N, C_vis]  <- Query
        text_sequence: [B, S, C_text] <- Key, Value (注意：这里要传入完整的序列，不要 sum!)
        text_mask:     [B, S]         <- Padding Mask (可选，针对文本长度不一的情况)
        """
        B, N, C = vision_tokens.shape
        
        # 1. 映射文本维度 [B, S, C_text] -> [B, S, C_vis]
        # 1. 【关键】先做 Norm，防止 Dot Product 数值异常
        vis = self.vis_proj(self.ln_v(vision_tokens) )
        # q = self.ln_v(vision_tokens)
        text = self.text_proj(self.ln_t(text_sequence))

        # q = q * 5.0
        
        # 2. 执行 Cross-Attention
        # attn_output: [B, N, C_vis] -> 每个视觉 Token 加权融合了相关的文本信息
        # attn_weights: [B, N, S]    -> 可以可视化！看到每个 patch 关注哪个单词
        # attn_output, attn_weights = self.cross_attn(
        #     query=q,
        #     key=key_value,
        #     value=key_value,
        #     key_padding_mask=text_mask # 如果有 mask 最好传进来
        # )
        ################# method 5 ######################
        if self.gate_mode == "raw_sigmoid":
            # 方案 B：不用 nn.MultiheadAttention 的 softmax，直接算 raw scores
            import math
            raw_scores = torch.bmm(text, vis.transpose(1, 2)) / math.sqrt(vis.shape[-1])  # [B, S, N_vis]
            max_relevance, _ = raw_scores.max(dim=1)  # [B, N_vis]
            gate = torch.sigmoid(self.gate_linear(max_relevance.unsqueeze(-1)))  # [B, N_vis, 1]
            attn_weights = F.softmax(raw_scores, dim=-1)  # for visualization only
        else:
            attn_output, attn_weights = self.cross_attn(
                query=text,
                key=vis,
                value=vis,
                key_padding_mask=None
            )

            if self.gate_mode == "sigmoid_max":
                # 方案 A：取 max 保留峰值 + learned sigmoid
                attn_max, _ = attn_weights.max(dim=1)  # [B, N_vis]
                gate = torch.sigmoid(self.gate_linear(attn_max.unsqueeze(-1)))  # [B, N_vis, 1]
            else:
                # 旧方案：softplus_mean（默认）
                attn_sum = attn_weights.mean(dim=1)  # [B, N_vis]
                gate = F.softplus(attn_sum.unsqueeze(-1))  # [B, N_vis, 1]

        ################# method 5 ######################

        # 在 forward 里打印
        # 取第一个 batch，第一个 visual token 的关注分布
        # print(f"DEBUG Attn Weights Max: {attn_weights[0,0,:].max().item()}")
        # print(f"DEBUG Attn Weights Min: {attn_weights[0,0,:].min().item()}")
        
        # 3. 这里的 attn_output 应该加上残差，防止梯度消失
        # gate_input = attn_output

        
        # # # 4. 生成 Gate
        # gate = self.gate_generator(gate_input)



        # # ================= 修改开始：方案二 =================

        # # 步骤 A: 计算熵
        # # 加 1e-9 防止 log(0)
        # # entropy shape: [B, N]
        # entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-9), dim=-1)

        # # 步骤 B: 计算理论最大熵 (均匀分布时的熵)
        # # S = text length
        # max_entropy = torch.log(torch.tensor(attn_weights.size(-1), device=q.device))

        # # 步骤 C: 归一化并反转
        # # 熵越小(越有序)，gate 越大
        # gate_score = 1.0 - (entropy / max_entropy)
        
        # # 截断负值 (以防万一)
        # gate_score = torch.clamp(gate_score, min=0.0, max=1.0)

        # # 步骤 D: 调整形状 [B, N, 1]
        # gate = gate_score.unsqueeze(-1)

        # ================= 修改结束 =================


        # # ================= 修改开始：方案三 =================
    
        # S = attn_weights.size(-1) # 文本长度
        # avg_weight = 1.0 / S      # 平均注意力权重 (如果是均匀分布)

        # # 步骤 A: 设定阈值
        # # 只有超过平均水平 1.5 倍的连接才被视为“有效连接”
        # threshold = avg_weight * 1.5 

        # # 步骤 B: 创建 0/1 掩码
        # # 只有显著的权重保留，其他的视为噪声(0)
        # significant_mask = (attn_weights > threshold)

        # # 步骤 C: 累加显著权重
        # # 背景 Patch 通常比较平滑，没有超过阈值的点，sum 接近 0
        # # 物体 Patch 有尖峰，sum 接近该尖峰的值
        # gate_score = (attn_weights * significant_mask).sum(dim=-1)

        # # 步骤 D: 调整形状 [B, N, 1]
        # gate = gate_score.unsqueeze(-1)

        # # ================= 修改结束 =================



        # # 这里的 attn_logits 是未经过 Softmax 的原始分数
        # attn_output, attn_weights, attn_logits = self.cross_attn_logits(
        #     query=q, key=key_value, value=key_value, mask=text_mask
        # )
        
        # # -----------------------------------------------------------
        # # 使用 Logits 计算 Gate
        # # -----------------------------------------------------------
        
        # # 1. 过滤掉 Mask 部分 (设置 Mask 为 0 的位置 Logit 为负无穷或极小值)
        # # 虽然 attn_logits 内部已经 mask fill 过了，但如果是取出来的 mean_logits 可能需要再次确认
        # # 或者简单粗暴一点，只取最大值
        
        # # 2. 计算每个 Vision Token 对整个句子的 "最大激活值"
        # # 我们假设：如果一个 Patch 是"车"，它一定会和文本里的"车"产生一个极大的 Logit (比如 8.0)
        # # 如果是背景，它和所有词的 Logit 都很小 (比如都在 0.0 - 1.0 之间)
        # print(f"attn_logits is : {attn_logits}")
        # max_relevance_score, _ = attn_logits.max(dim=-1) # [B, N_vis]
        
        # # 3. 将这个 Score 映射到 [0, 1] 作为 Gate
        # # 因为 Logits 理论范围是 (-inf, inf)，我们需要一个激活函数
        # # 比如：我们认为 Logit > 3.0 就是显著相关
        
        # # 方案 A: Sigmoid (平滑映射)
        # # 加上 bias=-3.0，意味着 Logit=3.0 时 gate=0.5；Logit=5.0 时 gate≈0.88
        # gate_score = torch.sigmoid(max_relevance_score - 3.0) 
        
        # # 方案 B: ReLU + Normalize (硬阈值)
        # # gate_score = F.relu(max_relevance_score) / 10.0 # 假设最大可能值是10
        
        # gate = gate_score.unsqueeze(-1) # [B, N, 1]

        
        return gate, attn_weights

import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding3D(nn.Module):
    def __init__(self, num_freqs=10, max_radius=8.0):
        super().__init__()
        self.num_freqs = num_freqs
        self.max_radius = max_radius # 根据你的train/test数据统计，7.0左右是最大值，取8.0安全
        
        # 频率生成：2^0, 2^1, ... 2^(L-1)
        # 这种方式在NeRF中被称为 log-linear sampling
        self.freq_bands = 2.0 ** torch.linspace(0., num_freqs - 1, num_freqs)

    def forward(self, coords_3d: torch.Tensor) -> torch.Tensor:
        # coords_3d: [B, N, 3]
        
        # 1. 归一化 (Normalize) 到 [-1, 1] 附近
        # 这样能保证频率的物理意义在不同尺度下是一致的
        coords_norm = coords_3d / self.max_radius 
        
        device = coords_norm.device
        freqs = self.freq_bands.to(device) # [num_freqs]
        
        # 2. 扩展维度以便广播计算
        # coords: [B, N, 3, 1]
        # freqs:  [1, 1, 1, num_freqs]
        x = coords_norm.unsqueeze(-1)
        f = freqs.view(1, 1, 1, -1)
        
        # 3. 计算 x * freq * pi
        # 乘以 pi 是为了对齐 [-1, 1] 区间与正弦周期
        # 此时最高频分量将在 [-1, 1] 内震荡 2^(num_freqs-1) 次
        x = x * f * np.pi 
        
        # 4. 计算 sin 和 cos
        # 结果维度: [B, N, 3, num_freqs]
        sin_x = torch.sin(x)
        cos_x = torch.cos(x)
        
        # 5. 拼接所有特征
        # 最终维度: [B, N, 3 * num_freqs * 2]
        # 也就是每个点的特征维度是 3(xyz) * 频率数 * 2(sin/cos)
        encoded = torch.cat([sin_x, cos_x], dim=-1)
        
        # 展平最后两维: [B, N, 6 * num_freqs]
        encoded = encoded.view(encoded.shape[0], encoded.shape[1], -1)
        
        return encoded


class ThreeDPositionEncoding(nn.Module):
    """
    3D位置编码模块 - 为视觉patch token注入3D空间信息
    Modified for Zero-Initialization
    """
    def __init__(
        self,
        hidden_size: int = 3584,
        text_dim: int = 3584,
        num_3d_freqs: int = 10,
        dropout: float = 0.05,
        type : str = "norm",
        merge_type : str = "None",
        grid_n: int = 3,
        type_3d: str = "sincos",
        record: bool = False,
        gate_mode: str = "softplus_mean"
    ):
        super().__init__()

        self.merge_type = merge_type
        self.hidden_size = hidden_size
        self.num_3d_freqs = num_3d_freqs
        self.type = type
        self.text_dim = text_dim
        self.dropout = dropout
        self.grid_n = grid_n
        self.type_3d = type_3d

        self.record = record
        self.gate_mode = gate_mode
        
        # 3D坐标编码维度
        self.coord_encoding_dim = 3 * 2 * num_3d_freqs
    
        ## merger_type : direct_add vision_guide text_guide deep_text_guide concat_vision concat_text

        ## direct_add   vision_guide   text_guide  deep_text_guide concat_vision concat_text
        if self.type != "multi_points":
            self.coord_projector = nn.Sequential(
                nn.Linear(self.coord_encoding_dim, hidden_size), # 中间层可以不缩减，或者缩减
                nn.GELU(),
                nn.LayerNorm(hidden_size),  # 【恢复】保证输入到最后一层的数据是分布规整的
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size) # 最后一层
            )
            self.pos_output_norm = nn.LayerNorm(hidden_size)
            # self._init_weights()
            print(f"[3D Position Encoding] 初始化完成 ")

        ##  all need
        self.rmsnorm1 = Qwen2RMSNorm(hidden_size, eps=1e-6)
        self.rmsnorm2 = Qwen2RMSNorm(hidden_size, eps=1e-6)

        if self.merge_type == "fusion_add":
            self.fusion_add_init()

        if self.merge_type == "vision_guide":
            self.vision_guide_init()

        if self.merge_type == 'text_guide' or self.merge_type == 'deep_text_guide':
            self.text_guide_init() 

        if self.merge_type == "text_cross_attention":
            self.text_cross_init()
        
        if self.merge_type == "text_cross_scale":
            self.text_cross_scale_init()
        

        if self.merge_type == "concat_vision":
            self.concat_vision_init()

        if self.merge_type == "concat_text":
            self.concat_text_init()
        
        if self.merge_type == "concat_text2":
            self.concat_text2_init()

        if self.merge_type == "vision_text_guide":
            self.vision_text_init()

        if self.merge_type == "film_text_guide":
            self.film_guide_init()


        if self.type == "multi_points":
            self.multi_points_init()

        if self.type_3d == "nf":
            self.encoding_3d = PositionalEncoding3D(
                num_freqs=self.num_3d_freqs,
                max_radius=8
            )

    def multi_points_init(self):
        # --- 分支 1: 质心位置编码 (Global Position) ---
        # xyz * 2 (sin/cos) * freqs
        self.centroid_dim = 3 * 2 * self.num_3d_freqs 
        self.centroid_mlp = nn.Sequential(
            nn.Linear(self.centroid_dim, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.GELU()
        )
        
        # --- 分支 2: 局部几何编码 (Local Geometry) ---
        # 输入是 Flatten 后的相对坐标 (num_points * 3)
        # 这里使用 MLP 来捕捉各点之间的相对关系 (类似于 PointNet)
        self.num_points = self.grid_n * self.grid_n
        self.geometry_input_dim = self.num_points * 3 
        self.geometry_mlp = nn.Sequential(
            nn.Linear(self.geometry_input_dim, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.GELU()
        )
        
        # --- 融合层 ---
        # 将位置信息和几何信息融合
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(self.dropout)
        )
    
    def direct_add_init(self):
        pass

    def fusion_add_init(self):
        self.fusion = nn.Parameter(torch.tensor(0.))
        
    def vision_guide_init(self):
        hidden_size = self.hidden_size
        ## vision_guide
        self.gate_projector = nn.Linear(hidden_size, hidden_size)
        # 初始化 gate 为 0
        nn.init.zeros_(self.gate_projector.weight)
        nn.init.zeros_(self.gate_projector.bias)


    def concat_vision_init(self):
        input_dim = self.hidden_size * 2
        vision_dim = self.hidden_size
        hidden_dim = self.hidden_size
        
        # 2. 视觉对齐层（可选）：把 Vision 映射到交互空间
        self.vision_proj = nn.Linear(vision_dim, vision_dim)

        self.concat_vision_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(), # 或者 ReLU
            nn.Linear(hidden_dim, vision_dim), # 降维回 C
            # 可选：加一个 LayerNorm 有助于稳定训练
            # nn.LayerNorm(vision_dim) 
            nn.Dropout(self.dropout) 
        )

    def concat_text_init(self):
        input_dim = self.hidden_size * 3
        vision_dim = self.hidden_size
        hidden_dim = self.hidden_size * 2

        text_dim = self.text_dim

        self.text_aggregator = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.Tanh(),
            nn.Linear(text_dim, 1, bias=False)
        )

        self.text_proj = nn.Linear(text_dim, vision_dim)
        
        # 2. 视觉对齐层（可选）：把 Vision 映射到交互空间
        self.text_vision_proj = nn.Linear(vision_dim, vision_dim)

        self.concat_text_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(), # 或者 ReLU
            nn.Linear(hidden_dim, vision_dim), # 降维回 C
            # 可选：加一个 LayerNorm 有助于稳定训练
            # nn.LayerNorm(vision_dim) 
            nn.Dropout(self.dropout) 
        )
    def concat_text2_init(self):
        input_dim = self.hidden_size * 3
        vision_dim = self.hidden_size
        hidden_dim = self.hidden_size * 4

        text_dim = self.text_dim

        self.text_aggregator = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.Tanh(),
            nn.Linear(text_dim, 1, bias=False)
        )

        self.text_proj = nn.Linear(text_dim, vision_dim)
        
        # 2. 视觉对齐层（可选）：把 Vision 映射到交互空间
        self.text_vision_proj = nn.Linear(vision_dim, vision_dim)

        self.concat_text_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(), # 或者 ReLU
            nn.Linear(hidden_dim, vision_dim), # 降维回 C
            # 可选：加一个 LayerNorm 有助于稳定训练
            # nn.LayerNorm(vision_dim) 
            nn.Dropout(self.dropout) 
        )

    def text_guide_init(self):
        vision_dim = self.hidden_size
        num_3d_freqs = self.num_3d_freqs
        text_dim = self.text_dim
         # 1. 3D 特征提取 (输出维度要匹配 Vision)
        self.coord_encoding_dim = 3 * 2 * num_3d_freqs
        self.pos_mlp = nn.Sequential(
            nn.Linear(self.coord_encoding_dim, vision_dim),
            nn.LayerNorm(vision_dim),
            nn.GELU(),
            nn.Linear(vision_dim, vision_dim) 
        )
        # 零初始化最后一层
        nn.init.zeros_(self.pos_mlp[-1].weight)
        nn.init.zeros_(self.pos_mlp[-1].bias)

        # 2. 文本特征聚合 (Attention Pooling)
        # 输入是 [B, Seq, 3584]，聚合为 [B, 3584]
        self.text_aggregator = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.Tanh(),
            nn.Linear(text_dim, 1, bias=False)
        )

        # === 修改处：使用基于相似度的门控模块 ===
        self.gating_module = TextAwareGating(vision_dim, text_dim)

    def text_cross_init(self):
        vision_dim = self.hidden_size
        num_3d_freqs = self.num_3d_freqs
        text_dim = self.text_dim

        # === 修改处：使用基于相似度的门控模块 ===
        self.text_cross_gating_module = TextGuidedCrossAttentionGate(
            vision_dim, text_dim, gate_mode=self.gate_mode
        )


    def text_cross_scale_init(self):
        vision_dim = self.hidden_size
        num_3d_freqs = self.num_3d_freqs
        text_dim = self.text_dim

        # === 修改处：使用基于相似度的门控模块 ===
        self.text_scale_gating_module = TextGuidedCrossAttentionGate(
            vision_dim, text_dim, gate_mode=self.gate_mode
        )

        self.pos_scale = nn.Parameter(torch.ones(1) * 10.0) # 初始值设大一点，比如10

    def vision_text_init(self):
        vision_dim = self.hidden_size
        num_3d_freqs = self.num_3d_freqs
        text_dim = self.text_dim

        self.text_aggregator = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.Tanh(),
            nn.Linear(text_dim, 1, bias=False)
        )

        # self.text_proj = nn.Linear(text_dim, vision_dim)

        self.hybrid_gate_mlp = nn.Sequential(
            nn.Linear(vision_dim + text_dim, vision_dim), # 将拼接后的维度映射回 vision 维度
            nn.GELU(),# 可以选择加一个激活函数，如 nn.GELU(), 再加一层 Linear，或者直接一层
            nn.Linear(vision_dim, vision_dim) 
        )

    def film_guide_init(self):
        vision_dim = self.hidden_size
        text_dim = self.text_dim
        
        # 文本聚合器 (保持不变)
        self.text_aggregator = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.Tanh(),
            nn.Linear(text_dim, 1, bias=False)
        )

        # self.text_proj = nn.Linear(text_dim, vision_dim)

        # FiLM 生成器
        # 输入: 3D特征(vision_dim) + 文本特征(text_dim)
        # 输出: 2 * vision_dim (分别对应 gamma 和 beta)
        self.film_mlp = nn.Sequential(
            nn.Linear(vision_dim + text_dim, vision_dim),
            nn.GELU(),
            nn.Linear(vision_dim, vision_dim * 2) # 输出双倍维度
        )
        
        # 可选：初始化 film_mlp 的最后一层为 0，确保初始状态下不做改变
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)


    def _init_weights(self):
        """
        执行全零初始化策略
        """
        # 找到 projector 中的最后一个 Linear 层
        last_linear = self.coord_projector[-1]
        
        # 将其权重和偏置全部置为 0
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

        
        # 其他层保持默认初始化（通常是 Kaiming 或 Xavier）
        # 这样网络内部有梯度流，只是最后输出暂时被掐断为0


    def positional_encoding_3d(self, coords_3d: torch.Tensor) -> torch.Tensor:
        if self.type_3d == 'nf':
            return self.encoding_3d.forward(
                coords_3d=coords_3d
            )

        # ... (保持原有的编码逻辑不变) ...
        B, N, _ = coords_3d.shape
        device = coords_3d.device
        
        freqs = torch.pow(
            10000.0,
            -torch.arange(0, self.num_3d_freqs, device=device, dtype=torch.float32) / self.num_3d_freqs
        )
        
        encoded_list = []
        for dim in range(3):
            coord = coords_3d[:, :, dim:dim+1]
            angles = coord * freqs.view(1, 1, -1)
            encoded_list.extend([torch.sin(angles), torch.cos(angles)])
        
        return torch.cat(encoded_list, dim=-1)
    
    def forward(
        self,
        vision_tokens: torch.Tensor,
        coords_3d: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        if self.merge_type == "direct_add":
            return self.direct_add_forward(
                vision_tokens=vision_tokens,
                coords_3d=coords_3d,
                text_embeddings=text_embeddings,
                attention_mask=attention_mask
            )
        if self.merge_type == "fusion_add":
            return self.fusion_add_forward(
                vision_tokens=vision_tokens,
                coords_3d=coords_3d,
                text_embeddings=text_embeddings,
                attention_mask=attention_mask
            )

        if self.merge_type == "vision_guide":
            return self.vision_guide_forward(
                vision_tokens=vision_tokens,
                coords_3d=coords_3d,
                text_embeddings=text_embeddings,
                attention_mask=attention_mask
            )

        if self.merge_type == 'text_guide' or self.merge_type == 'deep_text_guide':
            return self.text_guide_forward(
                vision_tokens=vision_tokens,
                coords_3d=coords_3d,
                text_embeddings=text_embeddings,
                attention_mask=attention_mask
            )

        if self.merge_type == 'text_cross_attention':
            return self.text_cross_forward(
                vision_tokens=vision_tokens,
                coords_3d=coords_3d,
                text_embeddings=text_embeddings,
                attention_mask=attention_mask
            )
        if self.merge_type == 'text_cross_scale':
            return self.text_cross_scale_forward(
                vision_tokens=vision_tokens,
                coords_3d=coords_3d,
                text_embeddings=text_embeddings,
                attention_mask=attention_mask
            )
        
        if self.merge_type == "concat_vision":
            return self.concat_vision_forward(
                vision_tokens=vision_tokens,
                coords_3d=coords_3d,
                text_embeddings=text_embeddings,
                attention_mask=attention_mask
            )

        if self.merge_type == "concat_text":
            return self.concat_text_forward(
                vision_tokens=vision_tokens,
                coords_3d=coords_3d,
                text_embeddings=text_embeddings,
                attention_mask=attention_mask
            )
        if self.merge_type == "concat_text2":
            return self.concat_text2_forward(
                vision_tokens=vision_tokens,
                coords_3d=coords_3d,
                text_embeddings=text_embeddings,
                attention_mask=attention_mask
            )

        if self.merge_type == "vision_text_guide":
            return self.vision_text_guide_forward(
                vision_tokens=vision_tokens,
                coords_3d=coords_3d,
                text_embeddings=text_embeddings,
                attention_mask=attention_mask
            )

        if self.merge_type == "film_text_guide":
            return self.film_guide_forward(
                vision_tokens=vision_tokens,
                coords_3d=coords_3d,
                text_embeddings=text_embeddings,
                attention_mask=attention_mask
            )

        print('error with the merge_type')
        exit(-1)


    def multi_points_forward(
        self,
        vision_tokens: torch.Tensor,
        coords_3d: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            vision_tokens: (B, N, hidden_size)
            points_3d: (B, N, num_points, 3) - 从 DepthToMultiPoint3D 获得的坐标
            attention_mask: (B, N)
        """
        points_3d = coords_3d
        if points_3d is None:
            return vision_tokens
            
        B, N, K, _ = points_3d.shape
        assert K == self.num_points, f"Model expects {self.num_points} points, got {K}"
        
        # 1. 计算质心 (Centroid) -> 表示 "绝对位置"
        # (B, N, 3)
        centroid = torch.mean(points_3d, dim=2)
        
        # 2. 计算相对坐标 (Relative Coordinates) -> 表示 "几何形状"
        # 减去质心，消除绝对位置影响
        # (B, N, K, 3)
        relative_coords = points_3d - centroid.unsqueeze(2)
        
        # 3. 编码质心 (Branch 1)
        # 使用 Sinusoidal 编码处理坐标数值范围大的问题
        centroid_enc = self.positional_encoding_3d(centroid) # (B, N, centroid_dim)
        centroid_enc = centroid_enc.to(dtype=vision_tokens.dtype)
        global_feat = self.centroid_mlp(centroid_enc)         # (B, N, h/2)
        
        # 4. 编码几何 (Branch 2)
        # 将相对坐标展平: (B, N, K*3)
        # 因为我们是网格化采样，点的顺序是固定的 (左上 -> 右下)，
        # 所以全连接层可以学习到空间结构 (例如 point_0 和 point_2 的 Z 差值代表坡度)
        local_flat = relative_coords.view(B, N, -1) 
        local_feat = self.geometry_mlp(local_flat)            # (B, N, h/2)
        
        # 5. 拼接与融合
        combined = torch.cat([global_feat, local_feat], dim=-1) # (B, N, h)
        pos_features = self.fusion_mlp(combined)
        
        return pos_features


    def direct_add_forward(
        self,
        vision_tokens: torch.Tensor,
        coords_3d: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ):
        # 1. 编码
        if self.type == 'multi_points':
            pos_features = self.multi_points_forward(
                vision_tokens=vision_tokens,
                coords_3d=coords_3d,
                text_embeddings=text_embeddings,
                attention_mask=attention_mask
            )
        else:
            pos_encoding = self.positional_encoding_3d(coords_3d)
            pos_encoding = pos_encoding.to(dtype=vision_tokens.dtype)

            # 2. 投影 (此时如果是Step 0，pos_features 输出全是 0)
            pos_features = self.coord_projector(pos_encoding)

        enhanced = vision_tokens + pos_features
   
        if attention_mask is not None:
            enhanced = enhanced * attention_mask.unsqueeze(-1)
        
        enhanced = self.rmsnorm2(enhanced)

        return enhanced
    
    def fusion_add_forward(
        self,
        vision_tokens: torch.Tensor,
        coords_3d: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ):
        # 1. 编码
        if self.type == 'multi_points':
            pos_features = self.multi_points_forward(
                vision_tokens=vision_tokens,
                coords_3d=coords_3d,
                text_embeddings=text_embeddings,
                attention_mask=attention_mask
            )
        else:
            pos_encoding = self.positional_encoding_3d(coords_3d)
            pos_encoding = pos_encoding.to(dtype=vision_tokens.dtype)

            # 2. 投影 (此时如果是Step 0，pos_features 输出全是 0)
            pos_features = self.coord_projector(pos_encoding)

        pos_features = self.pos_output_norm(pos_features)

        enhanced = vision_tokens + self.fusion * pos_features
   
        if attention_mask is not None:
            enhanced = enhanced * attention_mask.unsqueeze(-1)
        
        enhanced = self.rmsnorm2(enhanced)

        return enhanced

    def vision_guide_forward(
        self,
        vision_tokens: torch.Tensor,
        coords_3d: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ):
        # 1. 编码
        if self.type == 'multi_points':
            pos_features = self.multi_points_forward(
                vision_tokens=vision_tokens,
                coords_3d=coords_3d,
                text_embeddings=text_embeddings,
                attention_mask=attention_mask
            )
        else:
            pos_encoding = self.positional_encoding_3d(coords_3d)
            pos_encoding = pos_encoding.to(dtype=vision_tokens.dtype)

            # 2. 投影 (此时如果是Step 0，pos_features 输出全是 0)
            pos_features = self.coord_projector(pos_encoding)

        gate = torch.sigmoid(self.gate_projector(vision_tokens))

        enhanced = vision_tokens + gate * pos_features
        # enhanced = vision_tokens + gate * pos_features

        if attention_mask is not None:
            enhanced = enhanced * attention_mask.unsqueeze(-1)
        
        enhanced = self.rmsnorm2(enhanced)

        return enhanced

    def concat_vision_forward(
        self,
        vision_tokens: torch.Tensor,
        coords_3d: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ):
        # 1. 编码
        if self.type == 'multi_points':
            pos_features = self.multi_points_forward(
                vision_tokens=vision_tokens,
                coords_3d=coords_3d,
                text_embeddings=text_embeddings,
                attention_mask=attention_mask
            )
        else:
            pos_encoding = self.positional_encoding_3d(coords_3d)
            pos_encoding = pos_encoding.to(dtype=vision_tokens.dtype)

            # 2. 投影 (此时如果是Step 0，pos_features 输出全是 0)
            pos_features = self.coord_projector(pos_encoding)

        vision_features = self.vision_proj(vision_tokens)

        cat_features = torch.cat([vision_features, pos_features], dim=-1)
        enhanced = self.concat_vision_mlp(cat_features)

        enhanced = self.rmsnorm2(enhanced)

        return enhanced
    
    def concat_text_forward(
        self,
        vision_tokens: torch.Tensor,
        coords_3d: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ):
        # 1. 编码
        if self.type == 'multi_points':
            pos_features = self.multi_points_forward(
                vision_tokens=vision_tokens,
                coords_3d=coords_3d,
                text_embeddings=text_embeddings,
                attention_mask=attention_mask
            )
        else:
            pos_encoding = self.positional_encoding_3d(coords_3d)
            pos_encoding = pos_encoding.to(dtype=vision_tokens.dtype)

            # 2. 投影 (此时如果是Step 0，pos_features 输出全是 0)
            pos_features = self.coord_projector(pos_encoding)

        # --- 文本特征处理 ---
        # 1. 计算 Attention 权重
        attn_scores = self.text_aggregator(text_embeddings) # [B, S, 1]
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # 2. 加权聚合
        text_global = torch.sum(text_embeddings * attn_weights, dim=1) # [B, 3584]
        B, N, _ = vision_tokens.shape
        text_features = self.text_proj(text_global)

        text_features = text_features.unsqueeze(1)   # (B, 1, C)
        text_features = text_features.expand(-1, N, -1)  # (B, N, C)


        vision_features = self.text_vision_proj(vision_tokens)

        cat_features = torch.cat([vision_features, pos_features, text_features], dim=-1)
        enhanced = self.concat_text_mlp(cat_features)

        enhanced = self.rmsnorm2(vision_tokens + enhanced)

        return enhanced

    def concat_text2_forward(
        self,
        vision_tokens: torch.Tensor,
        coords_3d: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ):
        # 1. 编码
        if self.type == 'multi_points':
            pos_features = self.multi_points_forward(
                vision_tokens=vision_tokens,
                coords_3d=coords_3d,
                text_embeddings=text_embeddings,
                attention_mask=attention_mask
            )
        else:
            pos_encoding = self.positional_encoding_3d(coords_3d)
            pos_encoding = pos_encoding.to(dtype=vision_tokens.dtype)

            # 2. 投影 (此时如果是Step 0，pos_features 输出全是 0)
            pos_features = self.coord_projector(pos_encoding)

        # --- 文本特征处理 ---
        # 1. 计算 Attention 权重
        attn_scores = self.text_aggregator(text_embeddings) # [B, S, 1]
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # 2. 加权聚合
        text_global = torch.sum(text_embeddings * attn_weights, dim=1) # [B, 3584]
        B, N, _ = vision_tokens.shape
        text_features = self.text_proj(text_global)

        text_features = text_features.unsqueeze(1)   # (B, 1, C)
        text_features = text_features.expand(-1, N, -1)  # (B, N, C)


        vision_features = self.text_vision_proj(vision_tokens)

        cat_features = torch.cat([vision_features, pos_features, text_features], dim=-1)
        enhanced = self.concat_text_mlp(cat_features)

        enhanced = self.rmsnorm2(vision_tokens + enhanced)

        return enhanced


    def text_guide_forward(
        self,
        vision_tokens: torch.Tensor,
        coords_3d: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ):
        # 1. 编码
        if self.type == 'multi_points':
            pos_features = self.multi_points_forward(
                vision_tokens=vision_tokens,
                coords_3d=coords_3d,
                text_embeddings=text_embeddings,
                attention_mask=attention_mask
            )
        else:
            pos_encoding = self.positional_encoding_3d(coords_3d)
            pos_encoding = pos_encoding.to(dtype=vision_tokens.dtype)

            # 2. 投影 (此时如果是Step 0，pos_features 输出全是 0)
            pos_features = self.coord_projector(pos_encoding)

        # --- 文本特征处理 ---
        # 1. 计算 Attention 权重
        attn_scores = self.text_aggregator(text_embeddings) # [B, S, 1]
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # 2. 加权聚合
        text_global = torch.sum(text_embeddings * attn_weights, dim=1) # [B, 3584]
        
        gate = self.gating_module(vision_tokens, text_global)

        # =========== [核心修改] 抓取 Gate 数据 ===========
        if self.record and hasattr(self, 'shared_context'):
            print(f"开启记录")
            # detach(): 切断梯度
            # cpu(): 移到 CPU
            # clone(): 确保后续计算不影响这里
             # 注意：建议在 Context 类里专门开个槽位放这个，比如 latest_gate
            self.shared_context.update_gate(gate)
        # ===============================================

        # --- 融合 ---
        enhanced = vision_tokens + gate * pos_features
        # enhanced = vision_tokens + gate * pos_features

        enhanced = self.rmsnorm2(enhanced)

        return enhanced

    def text_cross_forward(
        self,
        vision_tokens: torch.Tensor,
        coords_3d: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ):
        # print("当前type 是 text_cross_attention")
        # 1. 编码
        if self.type == 'multi_points':
            pos_features = self.multi_points_forward(
                vision_tokens=vision_tokens,
                coords_3d=coords_3d,
                text_embeddings=text_embeddings,
                attention_mask=attention_mask
            )
        else:
            pos_encoding = self.positional_encoding_3d(coords_3d)
            pos_encoding = pos_encoding.to(dtype=vision_tokens.dtype)

            # 2. 投影 (此时如果是Step 0，pos_features 输出全是 0)
            pos_features = self.coord_projector(pos_encoding)

        pos_features = self.pos_output_norm(pos_features)
        # 假设 attention_mask 是 [B, S]，而在 MultiheadAttention 中
        # key_padding_mask 需要 True 表示 padding (被忽略)，False 表示有效
        # 通常 Transformer 输出的 mask 是 1=有效, 0=Padding，需要取反
        # if attention_mask is not None:
        #     # 转换为 bool mask (True 为需要被 mask 掉的位置)
        #     key_padding_mask = (attention_mask == 0) 
        # else:
        #     key_padding_mask = None

        key_padding_mask = None
        
        # 在调用 self.gating_module 之前加入：
        # print(f"DEBUG Check Vision: Shape={vision_tokens.shape}")
        # # 检查第一个 token 和第二个 token 的距离
        # dist = torch.norm(vision_tokens[0, 0, :] - vision_tokens[0, 1, :])
        # print(f"DEBUG Check Diff: Token[0] vs Token[1] distance = {dist.item()}")

        # forward 返回 gate 和 文本注意力权重(用于调试)
        gate, text_attn_map = self.text_cross_gating_module(
            vision_tokens=vision_tokens,       # Query
            text_sequence=text_embeddings,     # Key/Value (完整序列)
            text_mask=key_padding_mask
        )


        # =========== [核心修改] 抓取 Gate 数据 ===========
        if self.record and hasattr(self, 'shared_context'):
            print(f"开启记录")
            # detach(): 切断梯度
            # cpu(): 移到 CPU
            # clone(): 确保后续计算不影响这里
             # 注意：建议在 Context 类里专门开个槽位放这个，比如 latest_gate
            self.shared_context.update_gate(gate)
        # ===============================================

        # =========== [新增] 计算稀疏 Loss 并上传 ===========
        if hasattr(self, 'shared_context'):
            # print("传递了 正则loss")
            # L1 正则：鼓励 gate 中大部分元素为 0
            # 这里的 gate 带有梯度，直接计算 mean
            sparsity_loss = torch.mean(torch.abs(gate))
            
            # 上传到 context (注意：不要 detach！)
            self.shared_context.add_aux_loss(sparsity_loss)
        # =================================================

        # --- 融合 ---
        enhanced = vision_tokens + gate * pos_features
        # enhanced = vision_tokens + gate * pos_features

        enhanced = self.rmsnorm2(enhanced)

        return enhanced

    def text_cross_scale_forward(
        self,
        vision_tokens: torch.Tensor,
        coords_3d: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ):
        # print("当前type 是 text_cross_attention")
        # 1. 编码
        if self.type == 'multi_points':
            pos_features = self.multi_points_forward(
                vision_tokens=vision_tokens,
                coords_3d=coords_3d,
                text_embeddings=text_embeddings,
                attention_mask=attention_mask
            )
        else:
            pos_encoding = self.positional_encoding_3d(coords_3d)
            pos_encoding = pos_encoding.to(dtype=vision_tokens.dtype)

            # 2. 投影 (此时如果是Step 0，pos_features 输出全是 0)
            pos_features = self.coord_projector(pos_encoding)

        # 假设 attention_mask 是 [B, S]，而在 MultiheadAttention 中
        # key_padding_mask 需要 True 表示 padding (被忽略)，False 表示有效
        # 通常 Transformer 输出的 mask 是 1=有效, 0=Padding，需要取反
        
        
        pos_features = self.pos_output_norm(pos_features)

        key_padding_mask = None
        # if hasattr(self, 'shared_context'):
        #     key_padding_mask = self.shared_context.question_mask
            # if key_padding_mask is None:
            #     key_padding_mask

        # forward 返回 gate 和 文本注意力权重(用于调试)
        gate, text_attn_map = self.text_scale_gating_module(
            vision_tokens=vision_tokens,       # Query
            text_sequence=text_embeddings,     # Key/Value (完整序列)
            text_mask=key_padding_mask
        )


        # =========== [核心修改] 抓取 Gate 数据 ===========
        if self.record and hasattr(self, 'shared_context'):
            print(f"开启记录")
            # detach(): 切断梯度
            # cpu(): 移到 CPU
            # clone(): 确保后续计算不影响这里
             # 注意：建议在 Context 类里专门开个槽位放这个，比如 latest_gate
            self.shared_context.update_gate(gate)
        # ===============================================

        # =========== [新增] 计算稀疏 Loss 并上传 ===========
        if hasattr(self, 'shared_context'):
            # print("传递了 正则loss")
            # L1 正则：鼓励 gate 中大部分元素为 0
            # 这里的 gate 带有梯度，直接计算 mean
            sparsity_loss = torch.mean(torch.abs(gate))
            
            # 上传到 context (注意：不要 detach！)
            self.shared_context.add_aux_loss(sparsity_loss)
        # =================================================

        # with torch.no_grad():
        #     vis_scale = vision_tokens.std() # 比如 1560 左右

        # 这是一个 Trick：让 3D 特征的波动幅度能匹配上视觉特征
        # enhanced = vision_tokens + gate * (pos_features * vis_scale)

        # --- 融合 ---
        # enhanced = vision_tokens + gate * pos_features
        enhanced = vision_tokens + gate * (pos_features * self.pos_scale)
        # print(f"pos_scale is {self.pos_scale}")
        # enhanced = vision_tokens + gate * pos_features

        enhanced = self.rmsnorm2(enhanced)

        return enhanced

    def vision_text_guide_forward(
        self,
        vision_tokens: torch.Tensor,
        coords_3d: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ):
        # 1. 编码 (位置特征提取，保持不变)
        if self.type == 'multi_points':
            pos_features = self.multi_points_forward(
                vision_tokens=vision_tokens,
                coords_3d=coords_3d,
                text_embeddings=text_embeddings,
                attention_mask=attention_mask
            )
        else:
            pos_encoding = self.positional_encoding_3d(coords_3d)
            pos_encoding = pos_encoding.to(dtype=vision_tokens.dtype)

            # 2. 投影
            pos_features = self.coord_projector(pos_encoding)

        # --- 文本特征处理 ---
        # 1. 计算 Attention 权重并聚合 (参考 concat_text_forward)
        attn_scores = self.text_aggregator(text_embeddings) # [B, S, 1]
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # 2. 加权聚合得到全局文本特征
        text_global = torch.sum(text_embeddings * attn_weights, dim=1) # [B, Text_Dim]

        # text_global = self.text_proj(text_global)
        
        # 3. 维度对齐与拼接
        B, N, C = vision_tokens.shape
        # 将全局文本特征扩展到 Vision 的序列长度 [B, 1, Text_Dim] -> [B, N, Text_Dim]
        text_features_expanded = text_global.unsqueeze(1).expand(-1, N, -1)

        
        # 将 Vision 和 Text 特征拼接 [B, N, Vision_Dim + Text_Dim]
        gate_input = torch.cat([vision_tokens, text_features_expanded], dim=-1)

        # 4. 通过 MLP 生成 Gate
        # 注意：需要在 __init__ 中定义 self.hybrid_gate_mlp
        # 输入维度: Vision_Dim + Text_Dim, 输出维度: Vision_Dim (即 pos_features 的维度)
        gate = torch.sigmoid(self.hybrid_gate_mlp(gate_input))

        # --- 融合 ---
        # Gate 作用于 pos_features，然后与 vision_tokens 残差连接
        enhanced = vision_tokens + gate * pos_features

        # Mask 处理 (参考 vision_guide_forward)
        if attention_mask is not None:
            enhanced = enhanced * attention_mask.unsqueeze(-1)

        enhanced = self.rmsnorm2(enhanced)

        return enhanced


    def film_guide_forward(
        self,
        vision_tokens: torch.Tensor,
        coords_3d: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ):
        # 1. 编码 (位置特征提取)
        if self.type == 'multi_points':
            pos_features = self.multi_points_forward(
                vision_tokens=vision_tokens,
                coords_3d=coords_3d,
                text_embeddings=text_embeddings,
                attention_mask=attention_mask
            )
        else:
            pos_encoding = self.positional_encoding_3d(coords_3d)
            pos_encoding = pos_encoding.to(dtype=vision_tokens.dtype)
            pos_features = self.coord_projector(pos_encoding)

        # --- 文本特征处理 ---
        attn_scores = self.text_aggregator(text_embeddings) 
        attn_weights = F.softmax(attn_scores, dim=1)
        text_global = torch.sum(text_embeddings * attn_weights, dim=1) # [B, Text_Dim]

        # text_global = self.text_proj(text_global)
        
        # --- 构造条件向量 (Condition) ---
        B, N, C = vision_tokens.shape
        # 扩展文本特征: [B, N, Text_Dim]
        text_features_expanded = text_global.unsqueeze(1).expand(-1, N, -1)
        
        # 将 3D几何信息 和 语义信息 结合作为条件
        # condition: [B, N, Vision_Dim + Text_Dim]
        condition = torch.cat([pos_features, text_features_expanded], dim=-1)

        # --- FiLM 调制 ---
        # 生成缩放系数(gamma) 和 偏置系数(beta)
        style = self.film_mlp(condition) # [B, N, 2 * C]
        gamma, beta = style.chunk(2, dim=-1) # split into [B, N, C] and [B, N, C]

        # 核心公式: x_new = x * (1 + gamma) + beta
        # 这里用 3D+Text 来调制 Vision
        enhanced = vision_tokens * (1 + gamma) + beta
        
        # 最后把原始的 pos_features 也以残差形式加进去(可选，增强几何感知)
        enhanced = enhanced + pos_features

        # Mask 处理
        if attention_mask is not None:
            enhanced = enhanced * attention_mask.unsqueeze(-1)

        enhanced = self.rmsnorm2(enhanced)

        return enhanced







import torch.nn.functional as F

class TextGuided3DPositionEncoding(nn.Module):
    def __init__(
        self,
        vision_dim: int = 1280,      # 对应 Qwen2.5 Visual Block 的输出维度
        text_dim: int = 3584,        # 对应 self.model.embed_tokens 的输出维度
        num_3d_freqs: int = 10,
        dropout: float = 0.1,
        type: str = "norm",
        grid_n: int =3
    ):
        super().__init__()
        # self.hidden_size = hidden_size
        self.num_3d_freqs = num_3d_freqs
        
        # 3D坐标编码维度
        self.coord_encoding_dim = 3 * 2 * num_3d_freqs

        self.type = type


        
        # 1. 3D 特征提取 (输出维度要匹配 Vision)
        self.coord_encoding_dim = 3 * 2 * num_3d_freqs
        self.pos_mlp = nn.Sequential(
            nn.Linear(self.coord_encoding_dim, vision_dim),
            nn.LayerNorm(vision_dim),
            nn.GELU(),
            nn.Linear(vision_dim, vision_dim) 
        )
        # 零初始化最后一层
        nn.init.zeros_(self.pos_mlp[-1].weight)
        nn.init.zeros_(self.pos_mlp[-1].bias)

        # 2. 文本特征聚合 (Attention Pooling)
        # 输入是 [B, Seq, 3584]，聚合为 [B, 3584]
        self.text_aggregator = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.Tanh(),
            nn.Linear(text_dim, 1, bias=False)
        )



        # === 修改处：使用基于相似度的门控模块 ===
        self.gating_module = TextAwareGating(vision_dim, text_dim)

        self.rmsnorm1 = Qwen2RMSNorm(vision_dim, eps=1e-6)
        self.rmsnorm2 = Qwen2RMSNorm(vision_dim, eps=1e-6)

        # --- 分支 1: 质心位置编码 (Global Position) ---
        hidden_size = vision_dim
        # xyz * 2 (sin/cos) * freqs
        self.centroid_dim = 3 * 2 * num_3d_freqs 
        self.centroid_mlp = nn.Sequential(
            nn.Linear(self.centroid_dim, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU()
        )
        
        # --- 分支 2: 局部几何编码 (Local Geometry) ---
        # 输入是 Flatten 后的相对坐标 (num_points * 3)
        # 这里使用 MLP 来捕捉各点之间的相对关系 (类似于 PointNet)
        self.num_points = grid_n * grid_n
        self.geometry_input_dim = self.num_points * 3 
        self.geometry_mlp = nn.Sequential(
            nn.Linear(self.geometry_input_dim, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU()
        )
        
        # --- 融合层 ---
        # 将位置信息和几何信息融合
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout)
        )

    def positional_encoding_3d(self, coords_3d: torch.Tensor) -> torch.Tensor:
        # ... (保持原有的编码逻辑不变) ...
        B, N, _ = coords_3d.shape
        device = coords_3d.device
        
        freqs = torch.pow(
            10000.0,
            -torch.arange(0, self.num_3d_freqs, device=device, dtype=torch.float32) / self.num_3d_freqs
        )
        
        encoded_list = []
        for dim in range(3):
            coord = coords_3d[:, :, dim:dim+1]
            angles = coord * freqs.view(1, 1, -1)
            encoded_list.extend([torch.sin(angles), torch.cos(angles)])
        
        return torch.cat(encoded_list, dim=-1)

    def positional_encoding_3d_nerf(self, coords_3d: torch.Tensor) -> torch.Tensor:
        """
        NeRF-style Positional Encoding for continuous coordinates in [-1, 1]
        """
        B, N, _ = coords_3d.shape
        device = coords_3d.device
        
        # 生成频率: [2^0, 2^1, ..., 2^(L-1)] * PI
        # 比如 L=10，频率覆盖从粗到细
        bands = torch.pow(2.0, torch.arange(self.num_3d_freqs, device=device, dtype=torch.float32))
        freqs = bands * torch.pi # [L]
        
        # coords: [B, N, 3, 1] * freqs: [1, 1, 1, L] -> [B, N, 3, L]
        # 广播乘法
        angles = coords_3d.unsqueeze(-1) * freqs.view(1, 1, 1, -1)
        
        # 计算 sin, cos
        # result: [B, N, 3, L, 2] -> flatten -> [B, N, 3 * L * 2]
        encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        
        # 展平最后两维
        return encoded.reshape(B, N, -1)

    def forward(self, vision_tokens, coords_3d, text_embeddings, attention_mask: Optional[torch.Tensor] = None):
        """
        vision_tokens: [B, N, 1280]
        coords_3d: [B, N, 3]
        text_embeddings: [B, Seq_Len, 3584]  <-- 从最顶层传下来的 raw embeddings
        """
        if coords_3d is None:
            return vision_tokens

        if self.type == "multi_points":
            return self.multi_points_forward(
                vision_tokens=vision_tokens,
                coords_3d=coords_3d,
                text_embeddings=text_embeddings,
                attention_mask=attention_mask
            )

        # --- 3D 特征 ---
        pos_encoding = self.positional_encoding_3d(coords_3d).to(dtype=vision_tokens.dtype)
        pos_features = self.pos_mlp(pos_encoding) # [B, N, 1280]

        # --- 文本特征处理 ---
        # 1. 计算 Attention 权重
        attn_scores = self.text_aggregator(text_embeddings) # [B, S, 1]
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # 2. 加权聚合
        text_global = torch.sum(text_embeddings * attn_weights, dim=1) # [B, 3584]
        
        # 3. 投影对齐
        # text_aligned = self.text_proj(text_global) # [B, 3584] -> [B, 1280]
        
        # # --- 门控生成 ---
        # text_expanded = text_aligned.unsqueeze(1).expand(-1, vision_tokens.size(1), -1)
        # combined = torch.cat([vision_tokens, text_expanded], dim=-1) # [B, N, 2560]
        # gate = self.gate_net(combined) # [B, N, 1280]


        gate = self.gating_module(vision_tokens, text_global)

        # --- 融合 ---
        enhanced = vision_tokens + gate * self.rmsnorm1(pos_features)
        # enhanced = vision_tokens + gate * pos_features

        enhanced = self.rmsnorm2(enhanced)

        return enhanced

    def multi_points_forward(
        self,
        vision_tokens: torch.Tensor,
        coords_3d: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            vision_tokens: (B, N, hidden_size)
            points_3d: (B, N, num_points, 3) - 从 DepthToMultiPoint3D 获得的坐标
            attention_mask: (B, N)
        """
        points_3d = coords_3d
        if points_3d is None:
            return vision_tokens
            
        B, N, K, _ = points_3d.shape
        assert K == self.num_points, f"Model expects {self.num_points} points, got {K}"
        
        # 1. 计算质心 (Centroid) -> 表示 "绝对位置"
        # (B, N, 3)
        centroid = torch.mean(points_3d, dim=2)
        
        # 2. 计算相对坐标 (Relative Coordinates) -> 表示 "几何形状"
        # 减去质心，消除绝对位置影响
        # (B, N, K, 3)
        relative_coords = points_3d - centroid.unsqueeze(2)
        
        # 3. 编码质心 (Branch 1)
        # 使用 Sinusoidal 编码处理坐标数值范围大的问题
        centroid_enc = self.positional_encoding_3d(centroid) # (B, N, centroid_dim)
        centroid_enc = centroid_enc.to(dtype=vision_tokens.dtype)
        global_feat = self.centroid_mlp(centroid_enc)         # (B, N, h/2)
        
        # 4. 编码几何 (Branch 2)
        # 将相对坐标展平: (B, N, K*3)
        # 因为我们是网格化采样，点的顺序是固定的 (左上 -> 右下)，
        # 所以全连接层可以学习到空间结构 (例如 point_0 和 point_2 的 Z 差值代表坡度)
        local_flat = relative_coords.view(B, N, -1) 
        local_feat = self.geometry_mlp(local_flat)            # (B, N, h/2)
        
        # 5. 拼接与融合
        combined = torch.cat([global_feat, local_feat], dim=-1) # (B, N, h)
        pos_features = self.fusion_mlp(combined)
        
        # 6. 残差连接注入到 Vision Tokens
        # --- 文本特征处理 ---
        # 1. 计算 Attention 权重
        attn_scores = self.text_aggregator(text_embeddings) # [B, S, 1]
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # 2. 加权聚合
        text_global = torch.sum(text_embeddings * attn_weights, dim=1) # [B, 3584]
        
        # 3. 投影对齐
        # text_aligned = self.text_proj(text_global) # [B, 3584] -> [B, 1280]
        
        # # --- 门控生成 ---
        # text_expanded = text_aligned.unsqueeze(1).expand(-1, vision_tokens.size(1), -1)
        # combined = torch.cat([vision_tokens, text_expanded], dim=-1) # [B, N, 2560]
        # gate = self.gate_net(combined) # [B, N, 1280]


        gate = self.gating_module(vision_tokens, text_global)

        # 融合：用 gate 控制 pos_features 的幅度
        # 即使 pos_features 很大，gate 刚开始是 0，也会被压住
        enhanced = vision_tokens + gate * self.rmsnorm1( pos_features)
        # enhanced = vision_tokens + gate * pos_features
        ## LN2

        # 3. 融合
        # enhanced = vision_tokens + self.fusion_alpha * pos_features


        ## LN1
        # enhanced = vision_tokens + pos_features
        ## LN1
        
        if attention_mask is not None:
            enhanced = enhanced * attention_mask.unsqueeze(-1)
        
        enhanced = self.rmsnorm2(enhanced)

        # print(f'type is multi_points forward and text embedding, num points is {self.num_points}')

        return enhanced





import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import Tuple, Union, Optional, List

# ==========================================
# 1. 数据处理部分: 生成 Patch 的多点点云
# ==========================================

class DepthToMultiPoint3D:
    """
    从深度图和相机参数生成 Patch 的多点 3D 结构 (Mini Point Cloud)。
    不再是一个 Patch 一个点，而是一个 Patch 包含 KxK 个点 (例如 3x3=9个点)。
    """
    def __init__(self, patch_size: int = 14, fixed_image_size: Tuple[int, int] = (448, 448), grid_n: int = 3):
        """
        Args:
            patch_size: ViT的patch大小 (通常是14)
            fixed_image_size: 固定的图像输入尺寸 (H, W)
            grid_n: 每个 Patch 内部采样的网格大小。
                    grid_n=3 表示每个 patch 采样 3x3=9 个点。
        """
        self.patch_size = patch_size
        self.fixed_image_size = fixed_image_size
        self.grid_n = grid_n
        
        # 计算 Patch 网格数量
        self.grid_h = fixed_image_size[0] // patch_size
        self.grid_w = fixed_image_size[1] // patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.points_per_patch = grid_n * grid_n
        
        # 预计算像素采样网格 (uv coordinates)
        self._init_subpixel_grid()
        
        print(f"[DepthToMultiPoint] Initialized: {fixed_image_size} -> {self.grid_h}x{self.grid_w} patches.")
        print(f"[DepthToMultiPoint] Sampling {grid_n}x{grid_n} points per patch.")

    def _init_subpixel_grid(self):
        """
        预计算每个 Patch 内部 KxK 个点的像素坐标。
        结果保存在 self.uu_flat, self.vv_flat，形状为 (num_patches, points_per_patch)
        """
        # 1. 计算 Patch 内部的采样偏移量 (相对于 Patch 左上角)
        # 例如 patch_size=14, grid_n=3, step=4.66, offsets=[2.33, 7.0, 11.66]
        step = self.patch_size / self.grid_n
        offsets = np.arange(self.grid_n) * step + step / 2.0
        
        # 2. 生成所有 Patch 的左上角坐标
        patch_y = np.arange(self.grid_h) * self.patch_size
        patch_x = np.arange(self.grid_w) * self.patch_size
        
        # 3. 利用 Meshgrid 和 Broadcasting 生成全图所有采样点
        # grid_indices: (grid_h, grid_w)
        gy, gx = np.meshgrid(patch_y, patch_x, indexing='ij')
        
        # sub_offsets: (grid_n, grid_n)
        soy, sox = np.meshgrid(offsets, offsets, indexing='ij')
        
        # 核心广播操作:
        # gy: (H, W, 1, 1) + soy: (1, 1, n, n) -> (H, W, n, n)
        vv_grid = gy[:, :, None, None] + soy[None, None, :, :]
        uu_grid = gx[:, :, None, None] + sox[None, None, :, :]
        
        # 4. 展平为 (num_patches, points_per_patch)
        # (H, W, n, n) -> (H*W, n*n)
        self.uu_flat = uu_grid.reshape(self.num_patches, -1)
        self.vv_flat = vv_grid.reshape(self.num_patches, -1)

    def load_depth_from_png(self, path_or_arr: Union[str, np.ndarray], scale_factor: float = 1000.0) -> np.ndarray:
        """读取深度图并转为米"""
        if isinstance(path_or_arr, str):
            img = Image.open(path_or_arr)
            depth_raw = np.array(img)
        else:
            depth_raw = path_or_arr
            
        # 转换为 float32 并除以比例因子 (毫米 -> 米)
        depth_m = depth_raw.astype(np.float32) / scale_factor
        return depth_m

    def resize_data(self, depth: np.ndarray, rgb: Optional[np.ndarray] = None):
        """Resize 数据并返回缩放比例"""
        orig_h, orig_w = depth.shape
        target_h, target_w = self.fixed_image_size
        
        scale_w = target_w / orig_w
        scale_h = target_h / orig_h
        
        # Depth: 使用 NEAREST 保持物理意义
        depth_pil = Image.fromarray(depth)
        depth_resized = np.array(depth_pil.resize((target_w, target_h), resample=Image.NEAREST))
        
        # RGB: 使用 Bilinear
        rgb_resized = None
        if rgb is not None:
            if isinstance(rgb, np.ndarray):
                rgb_pil = Image.fromarray(rgb)
            else:
                rgb_pil = rgb
            rgb_resized = rgb_pil.resize((target_w, target_h), resample=Image.BILINEAR)
            
        return depth_resized, rgb_resized, scale_w, scale_h

    def get_subsampled_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        根据预计算的 grid 坐标，从深度图中采样深度值。
        包含去噪逻辑：如果某个子点无效，用该 Patch 的中位数填充。
        """
        h, w = depth.shape
        # 将 float 坐标转为整数索引，并限制在图像范围内
        u_idx = np.clip(np.round(self.uu_flat).astype(int), 0, w - 1)
        v_idx = np.clip(np.round(self.vv_flat).astype(int), 0, h - 1)
        
        # 索引深度: (num_patches, points_per_patch)
        z_samples = depth[v_idx, u_idx]
        
        # --- 鲁棒性处理 ---
        # 计算每个 Patch 的中位数 (忽略 NaN 和 0)
        # 注意：如果整个 patch 都是 0，nanmedian 会报 warning 或返回 nan，需要处理
        with np.errstate(all='ignore'):
            # 将 0 视为 NaN 以便计算 median
            z_for_median = z_samples.copy()
            z_for_median[z_for_median <= 0] = np.nan
            patch_medians = np.nanmedian(z_for_median, axis=1, keepdims=True)
        
        # 将计算出的 NaN median (即整个 patch 无效) 替换为 0
        patch_medians = np.nan_to_num(patch_medians, nan=0.0)
        
        # 找出无效的采样点 (<=0 或 NaN)
        invalid_mask = (z_samples <= 0) | np.isnan(z_samples)
        
        # 用该 Patch 的中位数替换无效点
        # 这样即使某个像素是黑洞，只要 Patch 里有点，就能保持几何平面不崩塌
        z_final = np.where(invalid_mask, patch_medians, z_samples)
        
        return z_final

    def robust_dynamic_normalize(self, coords: np.ndarray) -> np.ndarray:
        """
        对 (N, K*K, 3) 的点云数据进行鲁棒归一化。
        保持 Patch 内部的相对几何形状。
        """
        # 展平以便统计全局分布: (N * K * K, 3)
        flat_coords = coords.reshape(-1, 3)
        
        # 1. 过滤无效点
        valid_mask = np.isfinite(flat_coords).all(axis=1)
        # 还要过滤掉全 0 点 (原点通常无意义)
        non_zero = np.abs(flat_coords).sum(axis=1) > 1e-6
        valid_mask = valid_mask & non_zero
        
        if valid_mask.sum() == 0:
            return np.zeros_like(coords)

        valid_data = flat_coords[valid_mask]

        # 2. 计算统计边界 (2% - 98%) 抗噪
        p_min = np.percentile(valid_data, 2, axis=0)
        p_max = np.percentile(valid_data, 98, axis=0)
        
        # 3. 计算中心
        center = (p_min + p_max) / 2.0
        
        # 4. 计算缩放 (各向同性，保持长宽比)
        max_range = (p_max - p_min).max()
        scale = max_range / 2.0
        
        if scale < 1e-6: scale = 1.0

        # 5. 应用归一化 (利用 Broadcasting)
        # coords: (N, M, 3) - center: (3,)
        coords_norm = (coords - center) / scale
        
        # 6. 截断 (Clip)，防止极值影响 Transformer
        coords_norm = np.clip(coords_norm, -1.2, 1.2)
        
        return coords_norm

    def process_single_view(
        self,
        depth_input: Union[np.ndarray, str],
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        rgb_input: Optional[Union[np.ndarray, str]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        处理单张视图
        Returns:
            coords_norm: (num_patches, points_per_patch, 3)
            rgb_res: Resized RGB image
        """
        # 0. 加载
        if isinstance(depth_input, str):
            depth_m = self.load_depth_from_png(depth_input)
        else:
            depth_m = depth_input
            
        rgb_arr = None
        if rgb_input is not None:
            if isinstance(rgb_input, str):
                rgb_arr = np.array(Image.open(rgb_input))
            else:
                rgb_arr = rgb_input

        # 1. Resize
        depth_res, rgb_res, sw, sh = self.resize_data(depth_m, rgb_arr)
        
        # 2. 修正内参
        fx = intrinsic[0, 0] * sw
        fy = intrinsic[1, 1] * sh
        cx = intrinsic[0, 2] * sw
        cy = intrinsic[1, 2] * sh
        
        # 3. 获取多点深度 (N, K*K)
        z_cam = self.get_subsampled_depth(depth_res)
        
        # 4. 反投影 (Back-projection) -> Camera Coords
        # (N, K*K)
        x_cam = (self.uu_flat - cx) * z_cam / fx
        y_cam = (self.vv_flat - cy) * z_cam / fy
        
        # 组合齐次坐标 Stack: (N, K*K, 4)
        ones = np.ones_like(z_cam)
        points_cam_homo = np.stack([x_cam, y_cam, z_cam, ones], axis=-1)
        
        # 5. 转换到世界坐标
        # Extrinsic (C2W): (4, 4)
        # 为了矩阵乘法方便，先 reshape 成 (Total_Points, 4)
        N_patches = points_cam_homo.shape[0]
        K_points = points_cam_homo.shape[1]
        
        flat_cam = points_cam_homo.reshape(-1, 4) # (N*K*K, 4)
        
        # Formula: P_world = T_c2w * P_cam
        # (N*K*K, 4) @ (4, 4).T -> (N*K*K, 4)
        flat_world = flat_cam @ extrinsic.T
        
        # 取前3维并 Reshape 回 (N, K*K, 3)
        points_world = flat_world[:, :3].reshape(N_patches, K_points, 3)
        
        # 6. 归一化
        coords_norm = self.robust_dynamic_normalize(points_world)
        
        return coords_norm, rgb_res

    def process_multi_view(
        self,
        depth_list: List[Union[str, np.ndarray]],
        intrinsic_list: List[np.ndarray],
        extrinsic_list: List[np.ndarray],
        rgb_list: Optional[List[Union[str, np.ndarray]]] = None
    ) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
        """
        处理 Batch 数据
        Returns:
            coords_batch: (B, num_patches, points_per_patch, 3)
        """
        batch_coords = []
        batch_rgbs = [] if rgb_list else None
        
        for i in range(len(depth_list)):
            d_in = depth_list[i]
            K = intrinsic_list[i]
            T = extrinsic_list[i]
            r_in = rgb_list[i] if rgb_list else None
            
            c_norm, r_res = self.process_single_view(d_in, K, T, r_in)
            
            batch_coords.append(c_norm)
            if batch_rgbs is not None:
                batch_rgbs.append(r_res)
        
        # Stack to (B, N, K*K, 3)
        coords_tensor = np.stack(batch_coords, axis=0)
        
        return coords_tensor, batch_rgbs


# ==========================================
# 2. 模型编码部分: 几何感知编码器
# ==========================================

class GeometryAwareEncoding(nn.Module):
    """
    几何感知 3D 编码器
    输入: (B, N, num_points, 3) - 每个 Patch 包含多个点的点云
    输出: (B, N, hidden_size)
    
    原理: 
    将点云解耦为 [质心 Centroid] + [相对坐标 Relative Coords]
    1. 质心使用正弦编码 (Sinusoidal PE)，表示 Patch 在世界的绝对位置。
    2. 相对坐标直接通过 MLP 编码，表示 Patch 的局部几何形状 (平面、边缘、倾斜)。
    """
    def __init__(
        self, 
        hidden_size: int = 3584, # Qwen2.5-VL-7B 尺寸
        num_points: int = 9,     # 对应 3x3 grid
        num_3d_freqs: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_points = num_points
        self.num_3d_freqs = num_3d_freqs
        
        # --- 分支 1: 质心位置编码 (Global Position) ---
        # xyz * 2 (sin/cos) * freqs
        self.centroid_dim = 3 * 2 * num_3d_freqs 
        self.centroid_mlp = nn.Sequential(
            nn.Linear(self.centroid_dim, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU()
        )
        
        # --- 分支 2: 局部几何编码 (Local Geometry) ---
        # 输入是 Flatten 后的相对坐标 (num_points * 3)
        # 这里使用 MLP 来捕捉各点之间的相对关系 (类似于 PointNet)
        self.geometry_input_dim = num_points * 3 
        self.geometry_mlp = nn.Sequential(
            nn.Linear(self.geometry_input_dim, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU()
        )
        
        # --- 融合层 ---
        # 将位置信息和几何信息融合
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout)
        )
        
        # 可学习的融合系数 (初始化为较小值，避免训练初期破坏预训练特征)
        self.fusion_alpha = nn.Parameter(torch.tensor([0.1]))
        
        print(f"[GeometryEncoding] Initialized with {num_points} points/patch.")

    def get_sinusoidal_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """
        生成正弦位置编码
        Args: coords (B, N, 3)
        Returns: (B, N, 3 * 2 * num_freqs)
        """
        device = coords.device
        # 频率: 2^0, 2^1, ... 或者 log space
        freqs = torch.pow(
            10000.0,
            -torch.arange(0, self.num_3d_freqs, device=device, dtype=torch.float32) / self.num_3d_freqs
        )
        
        encoded = []
        for dim in range(3):
            c = coords[:, :, dim:dim+1] # (B, N, 1)
            angles = c * freqs.view(1, 1, -1) # (B, N, F)
            encoded.extend([torch.sin(angles), torch.cos(angles)])
            
        return torch.cat(encoded, dim=-1)

    def forward(
        self,
        vision_tokens: torch.Tensor,
        points_3d: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            vision_tokens: (B, N, hidden_size)
            points_3d: (B, N, num_points, 3) - 从 DepthToMultiPoint3D 获得的坐标
            attention_mask: (B, N)
        """
        if points_3d is None:
            return vision_tokens
            
        B, N, K, _ = points_3d.shape
        assert K == self.num_points, f"Model expects {self.num_points} points, got {K}"
        
        # 1. 计算质心 (Centroid) -> 表示 "绝对位置"
        # (B, N, 3)
        centroid = torch.mean(points_3d, dim=2)
        
        # 2. 计算相对坐标 (Relative Coordinates) -> 表示 "几何形状"
        # 减去质心，消除绝对位置影响
        # (B, N, K, 3)
        relative_coords = points_3d - centroid.unsqueeze(2)
        
        # 3. 编码质心 (Branch 1)
        # 使用 Sinusoidal 编码处理坐标数值范围大的问题
        centroid_enc = self.get_sinusoidal_encoding(centroid) # (B, N, centroid_dim)
        global_feat = self.centroid_mlp(centroid_enc)         # (B, N, h/2)
        
        # 4. 编码几何 (Branch 2)
        # 将相对坐标展平: (B, N, K*3)
        # 因为我们是网格化采样，点的顺序是固定的 (左上 -> 右下)，
        # 所以全连接层可以学习到空间结构 (例如 point_0 和 point_2 的 Z 差值代表坡度)
        local_flat = relative_coords.view(B, N, -1) 
        local_feat = self.geometry_mlp(local_flat)            # (B, N, h/2)
        
        # 5. 拼接与融合
        combined = torch.cat([global_feat, local_feat], dim=-1) # (B, N, h)
        pos_features = self.fusion_mlp(combined)
        
        # 6. 残差连接注入到 Vision Tokens
        enhanced = vision_tokens + self.fusion_alpha * pos_features
        
        # 7. Masking
        if attention_mask is not None:
            enhanced = enhanced * attention_mask.unsqueeze(-1)
            
        return enhanced

# ==========================================
# 3. 测试与示例代码
# ==========================================

# if __name__ == "__main__":
#     # ---------------------------
#     # 步骤 1: 数据准备 (模拟)
#     # ---------------------------
#     print("\n--- Testing Data Processing ---")
    
#     # 初始化处理器 (使用 3x3=9 个点)
#     processor = DepthToMultiPoint3D(patch_size=14, fixed_image_size=(448, 448), grid_n=3)
    
#     # 模拟输入数据
#     # 深度图: 渐变深度
#     dummy_depth = np.linspace(1, 10, 448*448).reshape(448, 448) * 1000 # 毫米
#     # 内参 (fx, fy, cx, cy)
#     dummy_K = np.array([[500, 0, 224], [0, 500, 224], [0, 0, 1]])
#     # 外参 (Identity, 位于原点)
#     dummy_T = np.eye(4)
    
#     # 处理单张
#     coords_batch, _ = processor.process_multi_view(
#         [dummy_depth, dummy_depth], # Batch size = 2
#         [dummy_K, dummy_K],
#         [dummy_T, dummy_T]
#     )
    
#     print(f"Output Coords Shape: {coords_batch.shape}") 
#     # 预期: (2, 1024, 9, 3)  (因为 448/14 = 32, 32*32=1024 patches)
    
#     # ---------------------------
#     # 步骤 2: 模型前向传播
#     # ---------------------------
#     print("\n--- Testing Model Encoding ---")
    
#     B, N, K, _ = coords_batch.shape
#     hidden_dim = 128 # 测试用小维度
    
#     # 初始化模型
#     model = GeometryAwareEncoding(hidden_size=hidden_dim, num_points=K)
    
#     # 模拟 Vision Tokens
#     vision_tokens = torch.randn(B, N, hidden_dim)
    
#     # 转换坐标为 Tensor
#     coords_tensor = torch.from_numpy(coords_batch).float()
    
#     # Forward
#     output = model(vision_tokens, coords_tensor)
    
#     print(f"Input Tokens: {vision_tokens.shape}")
#     print(f"Output Tokens: {output.shape}")
#     print("Success! Dimensions match.")



