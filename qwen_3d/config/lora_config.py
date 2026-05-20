from peft import LoraConfig, TaskType

# 说明：
# - 本文件仅维护一个字典 `LORA_CONFIG`
# - train_3d.py 通过 `lora_config = LORA_CONFIG[args.lora_type]` 获取配置
# - 两个键：
#   1) visual_save_all_lora: 视觉塔解冻（通过 modules_to_save 保存/训练 visual），语言模型 LoRA
#   2) merger_save_all_lora: 视觉塔仅 merger 解冻（保存/训练 visual.merger），视觉其余部分 + 语言模型 LoRA

LORA_CONFIG = {
    # 视觉解冻 + 语言 LoRA
    "visual_save_all_lora": LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=[
                # 这几个只存在于 LLM，视觉塔没有，直接写没问题
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                # 这三个视觉塔也有，必须加路径前缀限定只匹配 LLM
                "model\\.layers\\.\\d+\\.mlp\\.gate_proj",
                "model\\.layers\\.\\d+\\.mlp\\.up_proj",
                "model\\.layers\\.\\d+\\.mlp\\.down_proj",
            ],
            modules_to_save=["visual"],
            bias="none",
            inference_mode=False,
        ),
    # 仅 merger 解冻 + 视觉其余部分&语言 LoRA
    "merger_save_all_lora": LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        # 同时覆盖语言与视觉（视觉 blocks 内也会命中 gate/up/down 等线性层）
        target_modules=[
            # --- 1. 语言模型部分 (LLM) ---
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            # --- 2. 视觉模型 attention 的线性层 ---
            "qkv",
            "attn.proj",
        ],
        # 仅保存/训练 merger（其余视觉部分通过 LoRA 训练）
        modules_to_save=["visual.merger"],
        bias="none",
        inference_mode=False,
    ),
}