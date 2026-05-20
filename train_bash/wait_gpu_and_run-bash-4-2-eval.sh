#!/bin/bash

# ================= 配置区域 =================
# GPU ID
GPUS=(5)
# 显存阈值 (MB)
MEM_THRESHOLD=1000
# 要求的持续空闲时间 (分钟)
WAIT_DURATION_MIN=5
# 运行命令
CMD="bash train_bash/qwen_3d/train_3d_bash4-2-eval.sh"

# --- 动态心跳配置 ---
# 当 GPU 忙碌时，多久检测一次 (秒) -> 慢节奏
INTERVAL_BUSY=120
# 当 GPU 疑似空闲正在计时验证时，多久检测一次 (秒) -> 快节奏
# 设置短一点(如5-10秒)可以更敏锐地捕捉到中间的短暂占用
INTERVAL_IDLE_CHECK=10
# ===========================================

REQUIRED_DURATION_SEC=$((WAIT_DURATION_MIN * 60))
IDLE_START_TIME=0
CURRENT_SLEEP=$INTERVAL_BUSY

echo "----------------------------------------------------"
echo "开始监控 GPU: ${GPUS[*]}"
echo "策略: 发现空闲后，需连续空闲 ${WAIT_DURATION_MIN} 分钟才执行。"
echo "检测频率: 忙碌时每 ${INTERVAL_BUSY}s，验证空闲时每 ${INTERVAL_IDLE_CHECK}s。"
echo "----------------------------------------------------"

while true; do
    ALL_IDLE=true
    
    # 1. 检测显存
    for GPU_ID in "${GPUS[@]}"; do
        used=$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits)
        if [ -z "$used" ]; then used=99999; fi
        
        if [ "$used" -ge "$MEM_THRESHOLD" ]; then
            ALL_IDLE=false
            # 为了日志简洁，只有在之前是空闲状态突然变忙时，或者处于长休眠模式下才输出
            if [ "$IDLE_START_TIME" -ne 0 ] || [ "$CURRENT_SLEEP" -eq "$INTERVAL_BUSY" ]; then
                 echo "[$(date '+%H:%M:%S')] GPU $GPU_ID 忙碌 (使用: ${used}MB)"
            fi
            break
        fi
    done

    CURRENT_TIME=$(date +%s)

    # 2. 状态机逻辑
    if [ "$ALL_IDLE" = true ]; then
        # === 状态：空闲 ===
        
        # 切换到“快节奏”检测模式
        CURRENT_SLEEP=$INTERVAL_IDLE_CHECK

        if [ "$IDLE_START_TIME" -eq 0 ]; then
            # A. 刚进入空闲状态
            IDLE_START_TIME=$CURRENT_TIME
            echo "[$(date '+%H:%M:%S')] 发现空闲！开始 ${WAIT_DURATION_MIN} 分钟倒计时验证..."
        else
            # B. 已经在倒计时中
            ELAPSED=$((CURRENT_TIME - IDLE_START_TIME))
            REMAINING=$((REQUIRED_DURATION_SEC - ELAPSED))
            
            if [ "$ELAPSED" -ge "$REQUIRED_DURATION_SEC" ]; then
                # C. 验证通过
                echo ""
                echo "SUCCESS: GPU 已连续空闲超过 ${WAIT_DURATION_MIN} 分钟。"
                echo "EXEC: $CMD"
                echo "----------------------------------------------------"
                
                eval "$CMD"
                
                echo "----------------------------------------------------"
                echo "任务结束，脚本退出。"
                break
            else
                # D. 继续验证中（打印进度，使用 \r 可以在同一行刷新，显得更整洁，或者直接echo）
                # 这里使用普通echo，如果觉得刷屏太多，可以注释掉下面这行
                echo "[$(date '+%H:%M:%S')] 验证中... 已空闲 ${ELAPSED}s / ${REQUIRED_DURATION_SEC}s"
            fi
        fi
    else
        # === 状态：忙碌 ===
        
        if [ "$IDLE_START_TIME" -ne 0 ]; then
            echo "[$(date '+%H:%M:%S')] 警告：验证期间检测到 GPU 活动！计时器重置。"
        fi
        
        # 重置计时器
        IDLE_START_TIME=0
        # 切换回“慢节奏”检测模式
        CURRENT_SLEEP=$INTERVAL_BUSY
        echo "GPU 忙碌中，等待 ${CURRENT_SLEEP}s 后再次检测..."
    fi

    sleep "$CURRENT_SLEEP"
done



