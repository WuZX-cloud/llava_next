import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Optional, List, Tuple
import math


class CoordinateEncoding(nn.Module):
    """
    为视觉patch token添加2D坐标编码
    """
    def __init__(self, hidden_dim: int, max_positions: int = 1024):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_positions = max_positions
        
        # 为x和y坐标创建可学习的embedding
        self.x_embedding = nn.Embedding(max_positions, hidden_dim // 2)
        self.y_embedding = nn.Embedding(max_positions, hidden_dim // 2)
        
    def forward(self, patch_tokens: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            patch_tokens: shape [batch_size, num_patches, hidden_dim]
            grid_size: (height, width) 网格大小
        Returns:
            添加了坐标编码的patch tokens
        """
        batch_size, num_patches, _ = patch_tokens.shape
        h, w = grid_size
        
        # 生成坐标网格
        y_coords = torch.arange(h, device=patch_tokens.device).unsqueeze(1).expand(h, w)
        x_coords = torch.arange(w, device=patch_tokens.device).unsqueeze(0).expand(h, w)
        
        # 展平坐标
        y_coords = y_coords.reshape(-1)[:num_patches]  # [num_patches]
        x_coords = x_coords.reshape(-1)[:num_patches]  # [num_patches]
        
        # 获取坐标编码
        x_enc = self.x_embedding(x_coords)  # [num_patches, hidden_dim//2]
        y_enc = self.y_embedding(y_coords)  # [num_patches, hidden_dim//2]
        
        # 拼接x和y的编码
        coord_enc = torch.cat([x_enc, y_enc], dim=-1)  # [num_patches, hidden_dim]
        
        # 扩展batch维度并添加到patch tokens
        coord_enc = coord_enc.unsqueeze(0).expand(batch_size, -1, -1)
        
        return patch_tokens + coord_enc


class SinusoidalCoordinateEncoding(nn.Module):
    """
    使用正弦位置编码的坐标编码（类似Transformer位置编码）
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
    def forward(self, patch_tokens: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            patch_tokens: shape [batch_size, num_patches, hidden_dim]
            grid_size: (height, width) 网格大小
        Returns:
            添加了坐标编码的patch tokens
        """
        batch_size, num_patches, hidden_dim = patch_tokens.shape
        h, w = grid_size
        device = patch_tokens.device
        
        # 生成坐标网格 [0, 1]范围
        y_coords = torch.linspace(0, 1, h, device=device).unsqueeze(1).expand(h, w)
        x_coords = torch.linspace(0, 1, w, device=device).unsqueeze(0).expand(h, w)
        
        # 展平
        y_coords = y_coords.reshape(-1)[:num_patches]
        x_coords = x_coords.reshape(-1)[:num_patches]
        
        # 生成正弦编码
        coord_enc = torch.zeros(num_patches, hidden_dim, device=device)
        
        # 频率
        div_term = torch.exp(torch.arange(0, hidden_dim // 2, device=device).float() * 
                            (-math.log(10000.0) / (hidden_dim // 2)))
        
        # x坐标编码 (使用前半部分维度)
        coord_enc[:, 0::4] = torch.sin(x_coords.unsqueeze(1) * div_term[::2])
        coord_enc[:, 1::4] = torch.cos(x_coords.unsqueeze(1) * div_term[::2])
        
        # y坐标编码 (使用后半部分维度)
        coord_enc[:, 2::4] = torch.sin(y_coords.unsqueeze(1) * div_term[::2])
        coord_enc[:, 3::4] = torch.cos(y_coords.unsqueeze(1) * div_term[::2])
        
        # 扩展batch维度
        coord_enc = coord_enc.unsqueeze(0).expand(batch_size, -1, -1)
        
        return patch_tokens + coord_enc


class Qwen2VLWithCoordEncoding(nn.Module):
    """
    在Qwen2.5-VL中集成坐标编码的模型
    """
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        coord_encoding_type: str = "learnable",  # "learnable" or "sinusoidal"
        freeze_base_model: bool = False
    ):
        super().__init__()
        
        # 加载预训练模型
        self.base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # 获取hidden dimension
        self.hidden_dim = self.base_model.config.hidden_size
        
        # 初始化坐标编码
        if coord_encoding_type == "learnable":
            self.coord_encoding = CoordinateEncoding(self.hidden_dim)
        elif coord_encoding_type == "sinusoidal":
            self.coord_encoding = SinusoidalCoordinateEncoding(self.hidden_dim)
        else:
            raise ValueError(f"Unknown coord_encoding_type: {coord_encoding_type}")
        
        # 是否冻结基础模型
        if freeze_base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        前向传播，在视觉特征中注入坐标编码
        """
        # 如果有图像输入,先提取视觉特征并添加坐标编码
        if pixel_values is not None:
            # 获取视觉特征
            vision_outputs = self.base_model.visual(
                pixel_values=pixel_values,
                grid_thw=image_grid_thw
            )
            
            # 添加坐标编码
            if image_grid_thw is not None:
                batch_size = pixel_values.shape[0]
                enhanced_features = []
                
                for i in range(batch_size):
                    # 获取当前图像的grid size
                    t, h, w = image_grid_thw[i]
                    grid_size = (h.item(), w.item())
                    
                    # 提取当前图像的特征
                    start_idx = sum([image_grid_thw[j][0] * image_grid_thw[j][1] * image_grid_thw[j][2] 
                                    for j in range(i)])
                    end_idx = start_idx + t * h * w
                    img_features = vision_outputs[start_idx:end_idx].unsqueeze(0)
                    
                    # 添加坐标编码
                    img_features_with_coord = self.coord_encoding(img_features, grid_size)
                    enhanced_features.append(img_features_with_coord.squeeze(0))
                
                vision_outputs = torch.cat(enhanced_features, dim=0)
            
            # 使用增强后的视觉特征
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                vision_hidden_states=vision_outputs,
                **kwargs
            )
        else:
            # 纯文本输入
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
        
        return outputs
    
    def generate(self, **kwargs):
        """生成文本"""
        return self.base_model.generate(**kwargs)


# 使用示例
def example_usage():
    """
    使用示例代码
    """
    # 1. 初始化模型
    model = Qwen2VLWithCoordEncoding(
        model_name="Qwen/Qwen2-VL-7B-Instruct",
        coord_encoding_type="learnable",  # 或 "sinusoidal"
        freeze_base_model=False  # 是否冻结基础模型
    )
    
    # 2. 加载processor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    
    # 3. 准备输入数据
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "path/to/image1.jpg"},
                {"type": "image", "image": "path/to/image2.jpg"},
                {"type": "text", "text": "请描述这两张图片的内容和它们之间的关系。"}
            ]
        }
    ]
    
    # 4. 处理输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[msg["image"] for msg in messages for item in msg["content"] 
                if item["type"] == "image"],
        padding=True,
        return_tensors="pt"
    )
    
    # 5. 训练模式
    model.train()
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    
    # 6. 推理模式
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7
        )
    
    generated_text = processor.batch_decode(
        generated_ids, 
        skip_special_tokens=True
    )[0]
    
    print(f"Generated text: {generated_text}")
    
    return model, processor


if __name__ == "__main__":
    # 运行示例
    model, processor = example_usage()