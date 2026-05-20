from huggingface_hub import snapshot_download
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    RevisionNotFoundError,
    HFValidationError,
    EntryNotFoundError,
)
import time
import logging


# 可选：显示下载日志
logging.basicConfig(level=logging.INFO)

def download_with_retry(
    repo_id: str,
    repo_type: str = "dataset",
    local_dir: str = "data",
    max_retries: int = 5,
    retry_delay: float = 5.0,
):
    """
    支持断点续传 + 网络中断自动重试的下载函数
    """
    for attempt in range(1, max_retries + 1):
        try:
            print(f"🔄 尝试第 {attempt}/{max_retries} 次下载...")
            local_path = snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                local_dir=local_dir,
                allow_patterns=[
                    "*.bin",
                    "*.safetensors",
                    "*.pt",
                    "model*"
                ],
                            resume_download=False,          # ✅ 启用断点续传
                local_dir_use_symlinks=False,  # 按你需求
                force_download=True
                # 注意：不要设置 force_download=True！
            )
            print(f"✅ 下载成功！数据保存在: {local_path}")
            return local_path

        except (ConnectionError, TimeoutError, OSError) as e:
            # 网络或连接类错误，可以重试
            print(f"⚠️ 第 {attempt} 次下载失败（网络问题）: {e}")
            if attempt < max_retries:
                print(f"⏳ {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                print("❌ 所有重试均失败，退出。")
                raise

        except (RepositoryNotFoundError, RevisionNotFoundError, HFValidationError, EntryNotFoundError) as e:
            # 仓库或文件结构错误，不可恢复，直接报错
            print(f"❌ 仓库或路径错误，无法重试: {e}")
            raise

        except Exception as e:
            # 其他未知错误
            print(f"❌ 未知错误: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                raise

# 调用函数
if __name__ == "__main__":
    download_with_retry(
        repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
        repo_type="model",
        local_dir="models/Qwen2.5-VL-7B-Instruct",
        max_retries=20,
        retry_delay=10.0
    )
