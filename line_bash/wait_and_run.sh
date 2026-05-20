#!/bin/bash

# ================= 配置区域 =================
GPUS=(0 1 2 3)
MEM_THRESHOLD=2500
WAIT_DURATION_MIN=5

# 任务队列文件路径
QUEUE_FILE="line_bash/queue.txt"

INTERVAL_BUSY=120
INTERVAL_IDLE_CHECK=10
# ===========================================

REQUIRED_DURATION_SEC=$((WAIT_DURATION_MIN * 60))

# 如果文件不存在，自动创建一个空的
if [ ! -f "$QUEUE_FILE" ]; then
    touch "$QUEUE_FILE"
fi

echo "----------------------------------------------------"
echo "开始监控 GPU: ${GPUS[*]}"
echo "队列面板: $QUEUE_FILE"
echo "说明: 脚本不会删除任务，而是会在任务末尾打上状态标签。"
echo "状态包含: [WAITING], [RUNNING], [SUCCEED], [ERROR]"
echo "----------------------------------------------------"

# 辅助函数：安全地更新文件中指定行的状态
update_status() {
    local target_line=$1
    local raw_cmd=$2
    local status_tag=$3
    # 使用 awk 安全替换整行内容，避免 sed 遇到特殊字符时报错
    awk -v n="$target_line" -v s="${raw_cmd} ${status_tag}" 'NR==n {$0=s} 1' "$QUEUE_FILE" > "${QUEUE_FILE}.tmp" && mv "${QUEUE_FILE}.tmp" "$QUEUE_FILE"
}

# 守护进程无限循环
while true; do
    
    LINE_NUM=0
    CMD=""
    
    # 1. 扫描文件，寻找第一个未完成的任务
    while IFS= read -r line; do
        ((LINE_NUM++))
        
        # 跳过空行
        if [ -z "$(echo "$line" | tr -d '[:space:]')" ]; then continue; fi
        
        # 跳过已经明确结束的任务
        if [[ "$line" == *"[SUCCEED]" ]] || [[ "$line" == *"[ERROR]" ]]; then continue; fi
        
        # 找到了待办任务！
        # 清理掉可能残留的旧状态标签 (比如上次脚本意外中断留下的 [WAITING] 或 [RUNNING])
        CMD=$(echo "$line" | sed 's/ \[[A-Z_]*\]$//')
        break
    done < "$QUEUE_FILE"

    # 如果没有找到任何待办任务，休眠一会再重新扫描文件
    if [ -z "$CMD" ]; then
        sleep 10
        continue
    fi

    echo ""
    echo "===================================================="
    echo "锁定任务 (第 $LINE_NUM 行): $CMD"
    echo "正在等待 GPU 满足空闲条件..."
    echo "===================================================="

    # 标记状态为等待中
    update_status "$LINE_NUM" "$CMD" "[WAITING]"

    IDLE_START_TIME=0
    CURRENT_SLEEP=$INTERVAL_BUSY

    # 2. GPU 监控循环
    while true; do
        ALL_IDLE=true
        
        for GPU_ID in "${GPUS[@]}"; do
            used=$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits)
            if [ -z "$used" ]; then used=99999; fi
            
            if [ "$used" -ge "$MEM_THRESHOLD" ]; then
                ALL_IDLE=false
                if [ "$IDLE_START_TIME" -ne 0 ] || [ "$CURRENT_SLEEP" -eq "$INTERVAL_BUSY" ]; then
                     echo "[$(date '+%H:%M:%S')] GPU $GPU_ID 忙碌 (使用: ${used}MB)"
                fi
                break
            fi
        done

        CURRENT_TIME=$(date +%s)

        if [ "$ALL_IDLE" = true ]; then
            CURRENT_SLEEP=$INTERVAL_IDLE_CHECK

            if [ "$IDLE_START_TIME" -eq 0 ]; then
                IDLE_START_TIME=$CURRENT_TIME
                echo "[$(date '+%H:%M:%S')] 发现空闲！开始 ${WAIT_DURATION_MIN} 分钟倒计时验证..."
            else
                ELAPSED=$((CURRENT_TIME - IDLE_START_TIME))
                
                if [ "$ELAPSED" -ge "$REQUIRED_DURATION_SEC" ]; then
                    echo ""
                    echo "SUCCESS: 验证通过，准备执行！"
                    echo "EXEC: $CMD"
                    echo "----------------------------------------------------"
                    
                    # === 核心状态流转 ===
                    # 1. 标记状态为正在运行
                    update_status "$LINE_NUM" "$CMD" "[RUNNING]"
                    
                    # 2. 执行命令
                    eval "$CMD"
                    EXIT_CODE=$? # 获取命令的退出状态码 (0表示成功，非0表示报错)
                    
                    # 3. 根据执行结果更新最终状态
                    if [ $EXIT_CODE -eq 0 ]; then
                        echo ">> 任务执行成功结束！"
                        update_status "$LINE_NUM" "$CMD" "[SUCCEED]"
                    else
                        echo ">> 任务执行失败退出 (Exit Code: $EXIT_CODE)！"
                        update_status "$LINE_NUM" "$CMD" "[ERROR]"
                    fi
                    
                    echo "----------------------------------------------------"
                    break
                else
                    echo "[$(date '+%H:%M:%S')] 验证中... 已空闲 ${ELAPSED}s / ${REQUIRED_DURATION_SEC}s"
                fi
            fi
        else
            if [ "$IDLE_START_TIME" -ne 0 ]; then
                echo "[$(date '+%H:%M:%S')] 警告：验证期间检测到 GPU 活动！计时器重置。"
            fi
            IDLE_START_TIME=0
            CURRENT_SLEEP=$INTERVAL_BUSY
        fi

        sleep "$CURRENT_SLEEP"
    done
    
    # 任务间缓冲
    sleep 5 
done