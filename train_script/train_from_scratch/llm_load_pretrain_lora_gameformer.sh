
root_dir=./qwen
task_name=$(basename "$0" .sh)
PYTHONPATH=.
MASTER_PORT=29520

deepspeed --include=localhost:$1 --master_port $MASTER_PORT  qwen/sft_llama4drive_trainer.py \
--model_name_or_path qwen/qwen/Qwen3-8B \
--train_files  data/stage1_train_180k_processed_fixed.json \
--validation_files data/stage1_val_20k_processed_fixed.json \
--feature_len 80 \
--use_all_tokens False \
--adapter_fusion True \
--enable_lora True \
--ins_wo_stop True \
--lora_r 8 \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 1 \
--do_train \
--use_fast_tokenizer False \
--output_dir ${root_dir}/output/${task_name} \
--evaluation_strategy steps \
--learning_rate 1e-4 \
--gradient_accumulation_steps 1 \
--num_train_epochs 3 \
--warmup_steps 150 \
--load_in_bits 4 \
--logging_dir ${root_dir}/tensorboard/${task_name} \
--logging_steps 10 \
--logging_strategy steps \
--save_strategy steps \
--preprocessing_num_workers 10 \
--save_total_limit 10 \
--save_steps 100 \
--eval_steps 100000000000 \
--seed 7 \
--disable_tqdm False \
--ddp_find_unused_parameters False \
--block_size 2048 \
--report_to tensorboard \
--add_special_tokens "<map>,</map>" \
--resize_token_embeddings True \
--map_input_size 256 \
--deepspeed ${root_dir}/ds_config.json \
--ddp_timeout 18000000 \
--gameformer_ckpt training_log/model_epoch_48_valADE_1.1187.pth \
--lora_ckpt llama2/output/mix_driveqa_decision/adapter_model.bin \
--overwrite_output_dir \
# # debug
# --max_train_samples 200 \
# --max_eval_samples 50

# 2>&1 | tee -a ${root_dir}/log/train_${task_name}.log > /dev/null &
# --layers_to_transform 39 \
# 2>&1 | tee -a ${root_dir}/log/train_${task_name}.log > /dev/null &
