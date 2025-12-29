CUDA_VISIBLE_DEVICES=$1 python train_script/inference/simulator_qwen4drive.py \
--planner qwen/output/llm_load_pretrain_lora_gameformer_qwen/checkpoint-16875 \
--base_model qwen/qwen/Qwen3-8B \
--planner_type qwen4drive_lora_ins_wo_stop \
--save_dir asyncdriver_self_2 \
--ins_wo_stop \
--short_ins 30 \
--lora_r 8 \
--type $2 \
--simulation_root_path simulation_save_root_path \

# --planner llama2/output/llm_load_pretrain_lora_gameformer/ckpt \