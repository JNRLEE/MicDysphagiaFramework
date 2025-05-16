# Linux 環境設定指南

此文件提供在 Linux 環境下設置 MicDysphagiaFramework 專案的步驟。

## 環境需求

- Anaconda 或 Miniconda
- Python 3.9+
- 支援 PyTorch 的硬體設備

## 安裝步驟

### 1. 設置 Conda 環境

提供了兩種方式設置環境：

#### 方法 A：使用提供的腳本 (推薦)

```bash
# 給予腳本執行權限
chmod +x setup_conda_env.sh

# 執行設定腳本
./setup_conda_env.sh
```

#### 方法 B：手動設置環境

```bash
# 創建新的環境
conda create -n micdys python=3.9 -y
conda activate micdys

# 如果系統有CUDA支援，安裝GPU版本的PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
# 如果沒有CUDA，則安裝CPU版本
# conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# 安裝其他依賴
pip install -r requirements.txt
```

### 2. 資料準備

確保資料和Excel檔案都已經正確放置在指定位置：

```
/home/sbplab/JNRLEE/MicDysphagiaFramework/data/Processed(Cut)
/home/sbplab/JNRLEE/MicDysphagiaFramework/data/吞嚥聲音名單(共同編輯).xlsx
```

如果資料位置不同，請修改 `config/custom_audio_fcnn_classification.yaml` 檔案中的路徑設定。

### 3. 測試安裝

執行以下命令測試環境是否正確設置：

```bash
cd /home/sbplab/JNRLEE
conda activate micdys
python MicDysphagiaFramework/scripts/run_experiments.py --config MicDysphagiaFramework/config/custom_audio_fcnn_classification.yaml
```

## 常見問題

### 找不到模組錯誤

如果遇到 "ModuleNotFoundError" 例如 "No module named 'yaml'"，請確認所有的依賴都已正確安裝：

```bash
conda activate micdys
pip install -r MicDysphagiaFramework/requirements.txt
```

### CUDA 相關錯誤

如果遇到 CUDA 相關錯誤，請確認系統上的 CUDA 版本，並安裝對應版本的 PyTorch：

```bash
# 查看CUDA版本
nvidia-smi

# 根據CUDA版本安裝對應的PyTorch版本
# 訪問 https://pytorch.org/get-started/locally/ 獲得最新的安裝命令
```

### 資料路徑錯誤

如果程式找不到資料檔案，請檢查並修改配置檔案中的路徑：
```
MicDysphagiaFramework/config/custom_audio_fcnn_classification.yaml
``` 