for GPU_ID in 0 1 2 3; do
    used=$(nvidia-smi -i $GPU_ID \
        --query-gpu=memory.used \
        --format=csv,noheader,nounits)
    echo "GPU $GPU_ID: ${used}MB used"
done