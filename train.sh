#!/bin/bash
export NUPLAN_MAPS_ROOT=/data/DC/dataset/nuplan/dataset/maps
export NUPLAN_DATA_ROOT=/data/DC/dataset/nuplan/dataset
export NUPLAN_EXP_ROOT=/data/DC/dataset/nuplan/exp
export CUDA_HOME=/data/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/data/DC/AsyncDriver:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# 顺序执行训练脚本

echo "=== 开始执行第一个训练脚本 ==="
python train_script/train_gameformer.py --name qwen --train_set ./data/stage1_train_180k_processed_fixed.json --valid_set ./data/stage1_val_20k_processed_fixed.json


echo "=== 第一个脚本执行完成，开始执行第二个训练脚本 ==="
bash train_script/train_qa/train_driveqa_qwen.sh 0,1,2,3


echo "=== 第二个脚本执行完成，开始执行第三个训练脚本 ==="
bash train_script/train_qa/train_mixed_desion_qa_qwen.sh 0,1,2,3

echo "=== 第三个脚本执行完成，开始执行第4个训练脚本 ==="
bash train_script/train_from_scratch/llm_load_pretrain_lora_gameformer_qwen.sh 0,1,2,3

bash train_script/inference/run_all.sh

echo "=== 所有训练脚本执行完成 ==="