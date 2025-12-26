#   bash train_script/inference/asyncdriver_true_async.sh 0 0 1.5

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "Project root: $PROJECT_ROOT"
echo "PYTHONPATH: $PYTHONPATH"

GPU_ID=$1
SCENARIO_TYPE=$2
LLM_INTERVAL=${3:-0}  # Default 0 (event-driven mode) if not specified

echo "========================================="
echo "True Async AsyncDriver Inference"
echo "GPU: $GPU_ID"
echo "Scenario Type: $SCENARIO_TYPE"
if [ "$LLM_INTERVAL" == "0" ] || [ "$LLM_INTERVAL" == "-1" ]; then
    echo "LLM Mode: Event-Driven (immediate use after inference)"
    SAVE_DIR_NAME="asyncdriver_event_driven"
else
    echo "LLM Update Interval: ${LLM_INTERVAL}s (fixed interval mode)"
    SAVE_DIR_NAME="asyncdriver_true_async_${LLM_INTERVAL}s"
fi
echo "Save Directory: $SAVE_DIR_NAME"
echo "========================================="

CUDA_VISIBLE_DEVICES=$GPU_ID python train_script/inference/simulator_llama4drive_async.py \
--planner llama2/output/llm_load_pretrain_lora_gameformer/checkpoint-2100 \
--base_model llama2/llama-2-13b-chat-hf \
--planner_type llama4drive_async \
--save_dir $SAVE_DIR_NAME \
--ins_wo_stop \
--short_ins 30 \
--lora_r 8 \
--type $SCENARIO_TYPE \
--llm_interval $LLM_INTERVAL \
--async_mode \
--simulation_root_path simulation_save_root_path \
