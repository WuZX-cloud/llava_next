#!/bin/bash

# ================= 配置区域 =================
# GPU ID
GPUS=(0 1 2 3)
# 显存阈值 (MB)
MEM_THRESHOLD=2500
# 要求的持续空闲时间 (分钟)
WAIT_DURATION_MIN=5

# 运行命令列表 (数组格式，按顺序执行)
# 你可以在这里添加任意多个命令，用引号包裹，换行分隔
CMDS=(
    "bash qwen_vl_finetune/scripts/train_qwen_vl_only_q_and_a.sh"
    "bash qwen_vl_finetune/scripts/train_qwen_vl_no_vision.sh"
)

# --- 动态心跳配置 ---
# 当 GPU 忙碌时，多久检测一次 (秒) -> 慢节奏
INTERVAL_BUSY=120
# 当 GPU 疑似空闲正在计时验证时，多久检测一次 (秒) -> 快节奏
INTERVAL_IDLE_CHECK=10
# ===========================================

REQUIRED_DURATION_SEC=$((WAIT_DURATION_MIN * 60))

echo "----------------------------------------------------"
echo "开始监控 GPU: ${GPUS[*]}"
echo "策略: 发现空闲后，需连续空闲 ${WAIT_DURATION_MIN} 分钟才执行。"
echo "共加载了 ${#CMDS[@]} 个任务等待执行。"
echo "检测频率: 忙碌时每 ${INTERVAL_BUSY}s，验证空闲时每 ${INTERVAL_IDLE_CHECK}s。"
echo "----------------------------------------------------"

# 遍历任务列表
for i in "${!CMDS[@]}"; do
    CMD="${CMDS[$i]}"
    TASK_NUM=$((i + 1))
    
    echo ""
    echo "===================================================="
    echo "准备执行任务 [$TASK_NUM/${#CMDS[@]}]: $CMD"
    echo "正在等待 GPU 满足空闲条件..."
    echo "===================================================="

    # 每次进入新任务前，重置计时器和睡眠状态
    IDLE_START_TIME=0
    CURRENT_SLEEP=$INTERVAL_BUSY

    # 针对当前任务的 GPU 等待循环
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
                    
                    # 执行命令 (此处是阻塞运行，直到该任务跑完才会继续往下走)
                    eval "$CMD"
                    
                    echo "----------------------------------------------------"
                    echo "任务 [$TASK_NUM/${#CMDS[@]}] 结束！"
                    
                    # 跳出当前的 while 监控循环，让外层的 for 循环进入下一个任务
                    break
                else
                    # D. 继续验证中
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
    
    # 在上一个任务刚结束，准备进入下一个任务的监控前，给系统 5 秒钟的缓冲时间
    # 避免上一个进程刚释放显存，下一个任务立刻误判
    sleep 5 
done

echo ""
echo "===================================================="
echo "🎉 所有 ${#CMDS[@]} 个任务已全部执行完毕，脚本安全退出。"
echo "===================================================="