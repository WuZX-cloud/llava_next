import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,

)
from typing import List, Dict, Optional, Tuple

from qwen_3d.model.coords3dcontext import Coords3DContext
from qwen_3d.model.threeD import ThreeDPositionEncoding, TextGuided3DPositionEncoding
from qwen_3d.model.merger import MergerWith3DWrapper

from typing import Union, Optional
from typing_extensions import Unpack  # 推荐使用 typing_extensions 以兼容旧版 Python
from transformers.cache_utils import Cache
# from transformers.utils.generic import TransformersKwargs




# ==================== 修改后的Qwen模型 ====================

class Qwen2_5_VLWith3D(Qwen2_5_VLForConditionalGeneration):
    """
    继承Qwen2.5-VL,在视觉编码器后添加3D Position Encoding
    """
    def __init__(self, config, args):
        super().__init__(config)
        self.enable_3d = False
        if args.enable_3d == "True":
            self.enable_3d = True

        if hasattr(args, 'lambda_sparse'):
            self.lambda_sparse = args.lambda_sparse
        else:
            self.lambda_sparse = 0   # 或者给个默认值
        
        # 【关键】初始化 Context 容器
        self.coords_context = Coords3DContext()

        # 【新增】标志位：是否正在进行生成任务
        self.is_generating = False 

        self.merge_type = args.merge_type

        if self.enable_3d:
            position_3d_encoder = ThreeDPositionEncoding(
                hidden_size=config.vision_config.hidden_size,
                text_dim=config.hidden_size,
                num_3d_freqs=args.num_3d_freqs,
                dropout=0.1,
                type=args.norm_type,
                merge_type=self.merge_type,
                grid_n=args.grid_n,
                type_3d=args.type_3d
            )
            # 【关键一步】把 context 挂载给 encoder 实例！
            # 这样 encoder 内部就能访问 context 了
            position_3d_encoder.shared_context = self.coords_context
            # 执行替换
            if hasattr(self.visual, 'merger'):
                print("✓ [Qwen+3D] 正在执行 visual.merger 模块替换...")
                original_merger = self.visual.merger
                # self.visual.merger = original_merger
                # 传入 context 而不是 self
                self.visual.merger = MergerWith3DWrapper(
                    original_merger, 
                    position_3d_encoder, 
                    self.coords_context 
                )
                print("✓ [Qwen+3D]  visual.merger 模块替换完成")
            else:
                raise AttributeError("未找到 self.visual.merger")
    
    def get_text_embedding(self, input_ids):
        text_features = None
        if self.merge_type == 'text_guide' or self.merge_type == "vision_text_guide" or self.merge_type == "film_text_guide" or self.merge_type == 'concat_text2':
            with torch.no_grad():
                # self.model.embed_tokens 就是你要找的层
                # 输出: [Batch, Seq_Len, 3584]
                text_features = self.model.embed_tokens(input_ids)
        if self.merge_type == 'deep_text_guide' or self.merge_type == 'concat_text' or self.merge_type == "text_cross_attention" or self.merge_type=="text_cross_scale":
            with torch.no_grad():
                # 拿到所有层的特征
                outputs = self.model(input_ids=input_ids, output_hidden_states=True)
                # 推荐取倒数第 2 层，兼顾深度与通用语义
                text_features = outputs.hidden_states[-2] 
        if text_features is None:
            print('text embedding get error!')
            exit(-1)

        return text_features

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        question_mask: Optional[torch.Tensor] = None,
        question_input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        coords_3d: Optional[torch.Tensor] = None,
        coords_3d_mask: Optional[torch.Tensor]= None,
        return_dict: bool = True,
        # num_items_in_batch= None,
        # **kwargs,
    ) :
        # print(f"self.enable_3d is : {self.enable_3d}")
        # --- DEBUG 区域 ---
        # if not hasattr(self, "_debug_printed"):
        #     print("\n=== DEBUG FORWARD INPUTS ===")
        #     print(f"input_ids shape: {input_ids.shape if input_ids is not None else 'None'}")
        #     print(f"labels shape: {labels.shape if labels is not None else 'None'}")
        #     print(f"image_grid_thw: {image_grid_thw if image_grid_thw is not None else 'None'}")
            
        #     # 看看输入是不是全是 0 或者全是 padding
        #     if input_ids is not None:
        #         print(f"input_ids sample: {input_ids[0, :10]}")
            
        #     self._debug_printed = True
        # # ------------------
        # exit(0)
        kwargs = {"return_dict" : return_dict}
        if "num_items_in_batch" in kwargs:
            a = kwargs["num_items_in_batch"]
            print(f"num_items_in_batch is {a}")
            kwargs.pop("num_items_in_batch")

        # 标记当前次 forward 是否负责清理
        should_cleanup = False
        # 1. 更新 Context (将 grid_thw 也存进去)
        if self.enable_3d:
            # 优先使用 image_grid_thw，如果是视频则可能需要 video_grid_thw
            # Qwen2.5-VL 通常将两者合并处理，这里假设主要是图片
            # 如果 pixel_values_videos 存在，逻辑可能需要根据实际情况调整
            # 如果传入了新数据，就更新 Context
            if coords_3d is not None:
                # 1. 获取 Text Embeddings (不计算梯度，只做特征提取)
                # input_ids: [Batch, Seq_Len]
                text_embeddings = None
                if 'text' in self.merge_type:
                    
                    text_embeddings = self.get_text_embedding(input_ids=question_input_ids)

                self.coords_context.update(coords_3d, coords_3d_mask, text_embeddings, question_mask)

                # 【关键逻辑】
                # 只有当 “我不是在 generate 过程中” 时，我才负责跑完后清理现场
                # 如果 is_generating 为 True，说明外层的 generate 函数会负责清理，我就别管了
                if not self.is_generating:
                    should_cleanup = True  
        try:
            # 1. 在开始前，先清空旧的 loss（为了安全）
            self.coords_context.pop_aux_losses() # 如果你实现了单独的 clear

            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                rope_deltas=rope_deltas,
                cache_position=cache_position,
                second_per_grid_ts=second_per_grid_ts,
                **kwargs,
            )
            # 3. 如果在训练模式，且有 loss
            if not self.is_generating and outputs.loss is not None:
                # 从 Context 中提取刚才深层网络塞进去的 loss
                aux_loss_sum = self.coords_context.pop_aux_losses()
                
                # 定义稀疏 loss 的权重系数 (lambda)
                lambda_sparse = self.lambda_sparse 
                
                # 合并 Loss
                total_loss = outputs.loss + lambda_sparse * aux_loss_sum
                
                # 替换原本的 loss
                outputs.loss = total_loss
                # (可选) 打印一下看看数值比例，防止 sparsity_loss 过大主导了训练
                # print(f"Main Loss: {outputs.loss.item()}, Aux Loss: {aux_loss_sum.item()}")

            return outputs

        finally:
            # 只有训练模式(或单次推理)下，跑完立刻清理，防止显存泄漏
            if self.enable_3d and should_cleanup:
                self.coords_context.clear()


    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        question_mask: Optional[torch.Tensor] = None,
        question_input_ids: Optional[torch.LongTensor] = None,
        position_ids: torch.LongTensor = None,
        past_key_values: List[torch.FloatTensor] = None,
        inputs_embeds: torch.FloatTensor = None,
        labels: torch.LongTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        pixel_values: torch.Tensor = None,
        pixel_values_videos: torch.FloatTensor = None,
        image_grid_thw: torch.LongTensor = None,  # <--- 重点关注这个参数
        video_grid_thw: torch.LongTensor = None,
        rope_deltas: torch.LongTensor = None,
        coords_3d: torch.Tensor = None,
        coords_3d_mask: torch.Tensor = None,
        **kwargs,
    ):

        # 1. 开启生成模式保护
        self.is_generating = True
        # 1. 更新 Context (将 grid_thw 也存进去)
        if self.enable_3d:
            text_embeddings = None
            if 'text' in self.merge_type :
            # 优先使用 image_grid_thw，如果是视频则可能需要 video_grid_thw
            # Qwen2.5-VL 通常将两者合并处理，这里假设主要是图片
            # 如果 pixel_values_videos 存在，逻辑可能需要根据实际情况调整
            # with torch.no_grad():
            #     # self.model.embed_tokens 就是你要找的层
            #     # 输出: [Batch, Seq_Len, 3584]
            #     text_embeddings = self.model.embed_tokens(input_ids)
                text_embeddings = self.get_text_embedding(input_ids=question_input_ids)
            
            self.coords_context.update(coords_3d, coords_3d_mask, text_embeddings, None)
 
        try:
            # 3. 构造参数
            # 【核心修复】：把所有显式定义的参数，如果不是 None，全部加回去
            gen_kwargs = kwargs.copy()
            
            # === 必须加回去的标准参数 ===
            if input_ids is not None:           gen_kwargs['input_ids'] = input_ids
            if attention_mask is not None:      gen_kwargs['attention_mask'] = attention_mask  # 关键修复！
            if position_ids is not None:        gen_kwargs['position_ids'] = position_ids
            if past_key_values is not None:     gen_kwargs['past_key_values'] = past_key_values
            
            # === 必须加回去的 Qwen2.5-VL 视觉参数 ===
            if pixel_values is not None:        gen_kwargs['pixel_values'] = pixel_values      # 关键修复！
            if pixel_values_videos is not None: gen_kwargs['pixel_values_videos'] = pixel_values_videos
            if image_grid_thw is not None:      gen_kwargs['image_grid_thw'] = image_grid_thw
            if video_grid_thw is not None:      gen_kwargs['video_grid_thw'] = video_grid_thw
            if rope_deltas is not None:         gen_kwargs['rope_deltas'] = rope_deltas

            # 注意：inputs_embeds 通常设为 None 即可，不要强制加回去，除非你真的传了 Embedding
            if inputs_embeds is not None:       gen_kwargs['inputs_embeds'] = inputs_embeds

            # 4. 调用父类 generate
            return super().generate(**gen_kwargs)
        finally:
            # 5. 生成结束，关闭开关并清理
            self.is_generating = False

            if self.enable_3d:
                self.coords_context.clear()    
