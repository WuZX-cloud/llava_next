from huggingface_hub import snapshot_download

model_path = snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
    local_dir="qwen_vl_finetune/model/Qwen2.5-VL-7B-Instruct",
    local_dir_use_symlinks=False,  # 真实文件，不用软链接
    resume_download=False,         # 不用断点续传（避免坏文件）
    force_download=True,
)

print("Model downloaded to:", model_path)
