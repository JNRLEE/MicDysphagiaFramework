#!/bin/bash
echo "Creating conda environment for MicDysphagiaFramework..."

# 創建並啟用conda環境
conda create -n micdys python=3.9 -y
conda activate micdys

# 安裝PyTorch (根據CUDA版本選擇適當的安裝命令)
# 如果有CUDA支援:
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
# 如果沒有CUDA:
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# 安裝requirements.txt中的依賴
pip install -r requirements.txt

echo "Environment setup completed!"
echo "To activate the environment, use: conda activate micdys" 