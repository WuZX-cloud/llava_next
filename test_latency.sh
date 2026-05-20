

export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=2 python test_latency.py \
    --base_model "models/Qwen2.5-VL-7B-Instruct" --device 'cuda'  --enable_3d

