#!/bin/bash

# 定义所有 lora_type 的值
lora_types=('merger_save_all_lora' "visual_save_all_lora" "all_lora")

# 日志目录
log_dir="./train_bash/qwen/new_label-training_logs"
mkdir -p ${log_dir}

# 获取当前时间戳
timestamp=$(date +"%Y%m%d_%H%M%S")

# 创建总日志文件
summary_log="${log_dir}/summary_${timestamp}.log"

# 定义日志函数,同时输出到屏幕和文件
log_message() {
    echo "$1"
    echo "$1" >> ${summary_log}
}

# 记录开始信息
log_message "========================================="
log_message "批量训练开始"
log_message "时间: $(date)"
log_message "总共需要训练 ${#lora_types[@]} 个模型"
log_message "========================================="
log_message ""

# 循环遍历每个 lora_type
for lora_type in "${lora_types[@]}"; do
    log_message "========================================="
    log_message "开始训练: lora_type=${lora_type}"
    log_message "时间: $(date)"
    log_message "========================================="
    
    # 定义每个训练的详细日志文件
    detail_log="${log_dir}/train_${lora_type}_${timestamp}.log"
    
    # 运行训练脚本并保存日志
    bash qwen_vl_finetune/scripts/train_qwen_vl.sh -l ${lora_type} 2>&1 | tee ${detail_log}
    
    # 检查上一个命令是否成功
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log_message "❌ 错误: lora_type=${lora_type} 训练失败!"
        log_message "查看详细日志: ${detail_log}"
        log_message ""
        # 选择是继续还是退出
        # exit 1  # 如果失败就退出
        continue  # 如果失败就继续下一个
    fi
    
    log_message "✅ 完成训练: lora_type=${lora_type}"
    log_message "详细日志已保存: ${detail_log}"
    log_message ""

    sleep 60
done

log_message "========================================="
log_message "所有训练任务完成!"
log_message "总共训练了 ${#lora_types[@]} 个模型"
log_message "结束时间: $(date)"
log_message "========================================="
log_message ""
log_message "查看汇总日志: ${summary_log}"
log_message "查看详细日志: ${log_dir}/"



