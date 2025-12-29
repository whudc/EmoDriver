#!/bin/bash
# ==================================================
# 单 GPU 顺序推理脚本（最稳定版本）
# 所有 scenario 顺序执行，固定使用一张 GPU
# 使用示例:
#   bash train_script/inference/run_all_infer_single_gpu_sequential.sh
# ==================================================

# 固定使用的 GPU
GPU_ID=0

# scenario 总数
TOTAL_SCENARIOS=14

# 推理脚本路径
INFER_SCRIPT="train_script/inference/asyncdriver_true_async.sh"

# 日志目录
LOG_DIR="logs/infer"
mkdir -p "$LOG_DIR"

echo "������ Start sequential inference on GPU $GPU_ID"
echo "Total scenarios: $TOTAL_SCENARIOS"
echo "---------------------------------------------"

for ((i=0; i<TOTAL_SCENARIOS; i++)); do
    LOG_FILE="$LOG_DIR/scenario_${i}_gpu${GPU_ID}.log"

    echo "→ Running scenario_type_id=$i on GPU $GPU_ID"
    echo "  Log file: $LOG_FILE"

    # 严格顺序执行（不使用 &）
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    bash $INFER_SCRIPT $GPU_ID $i 0.5 > "$LOG_FILE" 2>&1

    # 失败即退出（非常重要）
    if [ $? -ne 0 ]; then
        echo "❌ Scenario $i failed!"
        echo "   Please check log: $LOG_FILE"
        exit 1
    fi

    echo "✅ Scenario $i finished successfully"
    echo "---------------------------------------------"
done

echo "All $TOTAL_SCENARIOS scenarios completed successfully!"
echo "Logs saved in: $LOG_DIR/"
