import torch
import torch.nn as nn


from qwen_3d.model.coords3dcontext import Coords3DContext


# 2. 修改 Wrapper，不再持有 parent_model，而是持有 context
class MergerWith3DWrapper(nn.Module):
    def __init__(self, original_merger, position_3d_encoder, context: Coords3DContext):
        super().__init__()
        self.position_3d_encoder = position_3d_encoder
        self.original_merger = original_merger
        self.context = context  # 这是一个普通对象，model.to() 不会去遍历它，解除了循环引用

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        # hidden_states: [Total_Valid_Tokens, 1280] (扁平化的)
        # 1. 监控数据统计量
        
        # 1. 获取坐标和 Mask
        coords_3d = self.context.coords_3d       # [Batch, Max_N, 3]
        coords_mask = self.context.mask          # [Batch, Max_N] (0/1)
        text_embeddings = self.context.text_embeddings
        # print(f"coords_3d shape is : {coords_3d.shape}")
        # print(f"coords_3d mask shape is : {coords_mask.shape}")
        # print(f"text_embeddings shape is : {text_embeddings.shape}")
        # exit(0)

        # 2. 注入逻辑
        if coords_3d is not None and coords_mask is not None:
            # === 核心逻辑：对齐 ===
            # hidden_states 是去除了 padding 的扁平化序列
            # coords_3d 是包含 padding 的 Batch 序列
            # 我们利用 mask 直接取出有效坐标，自动完成 "Flatten" 和 "Remove Padding"
            
            # 确保 mask 是布尔类型
            mask_bool = coords_mask.bool()
            
            # 取出有效坐标: [Batch, Max, 3] -> [Total_Valid_Tokens, 3]
            # valid_coords = coords_3d[mask_bool]
            # print(f"Dtype check - Feat: {hidden_states.dtype}, Coords: {valid_coords.dtype}")

            valid_coords = coords_3d[mask_bool].to(dtype=hidden_states.dtype) 
            # print(f"Dtype check - Feat: {hidden_states.dtype}, Coords: {valid_coords.dtype}")
            # exit(0)
            # print(f"valid coords_3d shape is : {valid_coords.shape}")

            
            # 3. 长度校验 (防止意外)
            if valid_coords.shape[0] != hidden_states.shape[0]:
                # 如果长度不一致（极少见，可能是特殊 token 差异），做截断或保护
                print(f"Warning: Dim mismatch. Feat: {hidden_states.shape[0]}, Coords: {valid_coords.shape[0]}")
                exit(0)
                min_len = min(valid_coords.shape[0], hidden_states.shape[0])
                valid_coords = valid_coords[:min_len]
                # 对应的特征也暂时切片用于计算（最后再加回去）
                feat_slice = hidden_states[:min_len]
            else:
                feat_slice = hidden_states

            # 4. 计算 3D 编码
            # encoder 期望输入 [Batch, Seq, 3]，我们这里视为 Batch=1, Seq=Total
            valid_coords_unsqueezed = valid_coords.unsqueeze(0) # [1, Total, 3]
            feat_slice_unsqueezed = feat_slice.unsqueeze(0)     # [1, Total, 1280]
            
            # 假设 mask 全为 1 (因为已经筛选过 valid 了)
            dummy_mask = torch.ones(1, valid_coords.shape[0], device=valid_coords.device)

            enhanced_features = self.position_3d_encoder(
                vision_tokens=feat_slice_unsqueezed, 
                coords_3d=valid_coords_unsqueezed,
                text_embeddings=text_embeddings,
                attention_mask=None
            )
            
            # [1, Total, 1280] -> [Total, 1280]
            enhanced_features = enhanced_features.squeeze(0)
            
            # 5. 直接相加 (残差连接)
            # 如果之前有长度不一致，只加匹配的部分
            if valid_coords.shape[0] != hidden_states.shape[0]:
                print("长度不一致，只加匹配的部分")
                hidden_states = hidden_states
                exit(-1)
            else:
                enhanced_features = enhanced_features
        else :
            print("coords_3d is None or coords_mask is None")
            exit(-1)
        # 6. 进入原有的 Merger 层 (执行降采样和投影)
        # 输入: [Total, 1280] -> 输出: [Total/4, 3584]
        return self.original_merger(enhanced_features, *args, **kwargs)
# ==================== 3D坐标生成工具 ====================
