#!/bin/bash
# ==============================================
# 多 GPU 并行推理脚本 (0-3 四张显卡)
# 每张显卡分配 3-4 个 scenario_type_id
# 使用示例: bash train_script/inference/run_all_infer_multi_gpu.sh
# ==============================================

# 定义 GPU 数量与任务总数
NUM_GPUS=4
TOTAL_SCENARIOS=14

# 脚本路径
INFER_SCRIPT="train_script/inference/asyncdriver_infer.sh"

# 日志输出目录
LOG_DIR="logs/infer"
mkdir -p $LOG_DIR

# 任务分配逻辑
echo "Launching inference for all $TOTAL_SCENARIOS scenarios using $NUM_GPUS GPUs..."

for ((i=0; i<TOTAL_SCENARIOS; i++)); do
    GPU_ID=$((i % NUM_GPUS))  # 轮流分配GPU 0-3
    echo "→ Launching scenario_type_id=$i on GPU $GPU_ID"
    
    # 后台运行并重定向日志
    bash $INFER_SCRIPT $GPU_ID $i > $LOG_DIR/scenario_${i}_gpu${GPU_ID}.log 2>&1 &
    
    # 为避免瞬间创建太多进程，可稍微延迟
    sleep 1
done

# 等待所有任务结束
wait

echo "✅ All 14 scenario inferences completed!"
echo "Logs saved in: $LOG_DIR/"
