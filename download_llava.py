from huggingface_hub import snapshot_download
import torchvision.transforms as transforms

# 下载 ControlNet 模型
controlnet_path = snapshot_download(
    repo_id="lmms-lab/llava-next-interleave-qwen-0.5b",
    local_dir="./models/llava-next",
    force_download = True
)

