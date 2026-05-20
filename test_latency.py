# 标准 Python3 脚本：在容器里用 python 直接跑即可（python 应指向 Python3）
import argparse
import gc
import os
import time
from types import SimpleNamespace
from typing import Dict, Optional, Tuple


import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info
from qwen_3d.model.model import Qwen2_5_VLWith3D


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ===============================
# 1) 计算 Params
# ===============================
def compute_params(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ===============================
# 2) 估算 FLOPs（尽力而为）
# ===============================
def compute_flops_with_torch_profiler(
    model: torch.nn.Module,
    forward_kwargs: Dict[str, torch.Tensor],
    device: torch.device,
) -> Optional[float]:
    """
    返回 FLOPs（单位：FLOPs），如果当前 PyTorch/设备不支持则返回 None。
    注：这里只统计一次 forward（不含 generate 循环）。
    """
    try:
        with torch.no_grad():
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU
                ]
                + ([torch.profiler.ProfilerActivity.CUDA] if device.type == "cuda" else []),
                record_shapes=False,
                profile_memory=False,
                with_flops=True,
            ) as prof:
                _ = model(**forward_kwargs)
        flops_total = 0.0
        for evt in prof.key_averages():
            if evt.flops is not None:
                flops_total += float(evt.flops)
        return flops_total if flops_total > 0 else None
    except Exception:
        return None


# ===============================
# 3) 测试 Latency
# ===============================
def measure_latency_forward(
    model: torch.nn.Module,
    forward_kwargs: Dict[str, torch.Tensor],
    device: torch.device,
    warmup: int = 20,
    runs: int = 50,
) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(**forward_kwargs)
        _sync_if_cuda(device)
        t0 = time.time()
        for _ in range(runs):
            _ = model(**forward_kwargs)
        _sync_if_cuda(device)
        t1 = time.time()
    return (t1 - t0) / runs


def measure_latency_generate(
    model: torch.nn.Module,
    generate_kwargs: Dict[str, torch.Tensor],
    device: torch.device,
    warmup: int = 10,
    runs: int = 20,
    max_new_tokens: int = 1,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model.generate(
                **generate_kwargs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
        _sync_if_cuda(device)
        t0 = time.time()
        for _ in range(runs):
            _ = model.generate(
                **generate_kwargs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
        _sync_if_cuda(device)
        t1 = time.time()
    return (t1 - t0) / runs


def build_dummy_inputs(
    processor,
    device,
    image_size=448,
    question="Please describe the image briefly.",
    num_images=1,
    enable_3d=False,
    coords_len=1024,
):
    """
    构造一条最小可跑的 Qwen2.5-VL 图文输入。
    返回 (forward_kwargs, generate_kwargs)。
    """
    images = []
    for _ in range(num_images):
        raw = os.urandom(image_size * image_size * 3)  # RGB
        images.append(Image.frombytes("RGB", (image_size, image_size), raw))

    # 官方 messages 格式（和你 eval_3d.py 一致）
    content = []
    for _ in range(num_images):
        content.append({"type": "image", "image": images.pop(0)})
    content.append({"type": "text", "text": "\n" + question})

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    )

    # 3D 相关（只给 Qwen2_5_VLWith3D 用；不启用则不传）
    if enable_3d:
        coords_3d = torch.randn(1, coords_len, 3, dtype=torch.float32)
        coords_3d_mask = torch.ones(1, coords_len, dtype=torch.float32)
        q_enc = processor.tokenizer(
            [question],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
            add_special_tokens=False,
        )
        inputs["coords_3d"] = coords_3d
        inputs["coords_3d_mask"] = coords_3d_mask
        inputs["question_input_ids"] = q_enc["input_ids"]

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # forward: 不要 labels，避免走 loss 分支
    forward_kwargs = {
        k: v
        for k, v in inputs.items()
        if k
        in {
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "rope_deltas",
            "coords_3d",
            "coords_3d_mask",
            "question_input_ids",
        }
    }
    generate_kwargs = forward_kwargs.copy()
    return forward_kwargs, generate_kwargs


def load_qwen25_vl_base(
    model_name_or_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    return model.to(device).eval()


def load_qwen25_vl_with_3d(
    base_model_name_or_path: str,
    device: torch.device,
    dtype: torch.dtype,
    enable_3d: bool = True,
    merge_type: str = "text_cross_attention",
    num_3d_freqs: int = 4,
    norm_type: str = "none",
    grid_n: int = 5,
    type_3d: str = "nf",
) -> torch.nn.Module:
    """
    按你 train_3d.py 的方式：先实例化结构，再把 base 权重灌进来（含 merger key rename）。
    """
    config = AutoConfig.from_pretrained(base_model_name_or_path, trust_remote_code=True)
    args = SimpleNamespace(
        enable_3d="True" if enable_3d else "False",
        lambda_sparse=0.0,
        merge_type=merge_type,
        num_3d_freqs=num_3d_freqs,
        norm_type=norm_type,
        grid_n=grid_n,
        type_3d=type_3d,
        fixed_image_size=(224, 224),
        is_w2c="True",
        record="False",
    )
    model = Qwen2_5_VLWith3D(config, args=args).to(dtype)

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    base_state = base_model.state_dict()
    if enable_3d:
        new_state = {}
        for k, v in base_state.items():
            if k.startswith("visual.merger."):
                new_state[k.replace("visual.merger.", "visual.merger.original_merger.")] = v
            else:
                new_state[k] = v
    else:
        new_state = base_state

    model.load_state_dict(new_state, strict=False)
    del base_model
    return model.to(device).eval()


def benchmark_one(
    name: str,
    model: torch.nn.Module,
    processor,
    device: torch.device,
    enable_3d_inputs: bool,
    image_size: int,
    coords_len: int,
    warmup: int,
    runs: int,
    gen_warmup: int,
    gen_runs: int,
    max_new_tokens: int,
) -> None:
    forward_kwargs, generate_kwargs = build_dummy_inputs(
        processor=processor,
        device=device,
        image_size=image_size,
        num_images=2,
        enable_3d=enable_3d_inputs,
        coords_len=coords_len,
    )

    total_params, trainable_params = compute_params(model)
    flops = compute_flops_with_torch_profiler(model, forward_kwargs, device=device)
    lat_fwd = measure_latency_forward(model, forward_kwargs, device=device, warmup=warmup, runs=runs)
    lat_gen = measure_latency_generate(
        model,
        generate_kwargs,
        device=device,
        warmup=gen_warmup,
        runs=gen_runs,
        max_new_tokens=max_new_tokens,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

    print(f"\n===== {name} =====")
    print(f"Params(total): {total_params/1e6:.2f} M")
    print(f"Params(trainable): {trainable_params/1e6:.2f} M")
    if flops is None:
        print("FLOPs(1x forward): N/A (torch profiler 不支持或统计为 0)")
    else:
        print(f"FLOPs(1x forward): {flops/1e9:.2f} G")
    print(f"Latency(forward): {lat_fwd*1000:.3f} ms")
    print(f"Latency(generate, max_new_tokens={max_new_tokens}): {lat_gen*1000:.3f} ms")


def free_model(model: Optional[torch.nn.Module], device: torch.device) -> None:
    if model is None:
        return
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="基础模型（原版Qwen2.5-VL）名称或本地路径",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--coords_len", type=int, default=512, help="3D coords token 数，默认 32*32=1024")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--gen_warmup", type=int, default=10)
    parser.add_argument("--gen_runs", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--enable_3d", action="store_true", help="对 Qwen2_5_VLWith3D 启用 3D 分支并传 coords")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)

    base = load_qwen25_vl_base(args.base_model, device=device, dtype=dtype)
    benchmark_one(
        name="Qwen2_5_VLForConditionalGeneration (base)",
        model=base,
        processor=processor,
        device=device,
        enable_3d_inputs=False,
        image_size=args.image_size,
        coords_len=args.coords_len,
        warmup=args.warmup,
        runs=args.runs,
        gen_warmup=args.gen_warmup,
        gen_runs=args.gen_runs,
        max_new_tokens=args.max_new_tokens,
    )
    free_model(base, device=device)
    base = None

    with3d = load_qwen25_vl_with_3d(
        args.base_model,
        device=device,
        dtype=dtype,
        enable_3d=args.enable_3d,
    )
    benchmark_one(
        name=f"Qwen2_5_VLWith3D (enable_3d={args.enable_3d})",
        model=with3d,
        processor=processor,
        device=device,
        enable_3d_inputs=args.enable_3d,
        image_size=args.image_size,
        coords_len=args.coords_len,
        warmup=args.warmup,
        runs=args.runs,
        gen_warmup=args.gen_warmup,
        gen_runs=args.gen_runs,
        max_new_tokens=args.max_new_tokens,
    )
    free_model(with3d, device=device)
    with3d = None


if __name__ == "__main__":
    main()