root_dir=./qwen
task_name=drive_qa
MASTER_PORT=29501

deepspeed --include=localhost:$1 --master_port $MASTER_PORT qwen/sft_qwen_trainer.py \
--model_name_or_path qwen/qwen/Qwen3-8B \
--train_files data/drive_qa_mini40k.json \
--validation_files data/drive_qa_mini2k.json \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 1 \
--do_train \
--do_eval \
--use_fast_tokenizer False \
--output_dir ${root_dir}/output/${task_name} \
--eval_strategy steps \
--learning_rate 5e-4 \
--gradient_accumulation_steps 10 \
--freeze_map_adapter True \
--num_train_epochs 1 \
--warmup_steps 25 \
--load_in_bits 4 \
--lora_r 8 \
--lora_alpha 32 \
--target_modules q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
--logging_dir ${root_dir}/tensorboard/${task_name} \
--logging_steps 10 \
--logging_strategy steps \
--save_strategy steps \
--preprocessing_num_workers 10 \
--save_total_limit 1 \
--save_steps 1000 \
--eval_steps 1000 \
--seed 7 \
--disable_tqdm False \
--ddp_find_unused_parameters False \
--block_size 3072 \
--report_to tensorboard \
--overwrite_output_dir \
--resize_token_embeddings True \
--map_input_size 256 \
--deepspeed ${root_dir}/ds_config.json \
--bf16 \
--bf16_full_eval \
--ddp_timeout 18000000 \
--add_special_tokens "<map>,</map>"
# --resume_from_checkpoint ${root_dir}/checkpoint-20400 \
# --layers_to_transform 39 \
