#!/bin/bash

# 修改~/.bashrc文件
BASHRC="$HOME/.bashrc"
if [ -f "$BASHRC" ]; then
    # 使用sed命令替换内容
    sed -i 's|source /data/envs/anaconda3/bin/activate|source /data/miniconda3/bin/activate|' $BASHRC
    sed -i 's|export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda/lib64|export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/data/cuda-11.8/lib64|' $BASHRC
    sed -i 's|export PATH=\$PATH:/usr/local/cuda/bin|export PATH=\$PATH:/data/cuda-11.8/bin|' $BASHRC
    sed -i 's|export CUDA_HOME=/usr/local/cuda|export CUDA_HOME=/data/cuda-11.8|' $BASHRC
else
    echo "$BASHRC not found!"
    exit 1
fi

# 修改/etc/resolv.conf文件
RESOLV_CONF="/etc/resolv.conf"
if [ -f "$RESOLV_CONF" ]; then
    # 添加nameserver配置
    echo "nameserver 8.8.8.8" | sudo tee -a $RESOLV_CONF > /dev/null
    echo "nameserver 1.1.1.1" | sudo tee -a $RESOLV_CONF > /dev/null
else
    echo "$RESOLV_CONF not found!"
    exit 1
fi

echo "修改完成！"
