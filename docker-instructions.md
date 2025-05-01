# MicDysphagiaFramework Docker 使用指南

本文檔提供如何構建和使用 MicDysphagiaFramework Docker 映像的說明。

## 構建 Docker 映像

在專案根目錄執行以下命令來構建 Docker 映像：

```bash
docker build -t micdysphagia:latest .
```

## 使用 Docker 映像

### 基本用法

顯示幫助信息：

```bash
docker run micdysphagia:latest
```

### 運行具體訓練任務

使用配置文件運行訓練任務：

```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs micdysphagia:latest --config config/your_config.yaml
```

### 使用 GPU

如果要使用 GPU 進行訓練，請確保主機已安裝 NVIDIA Docker 支持，然後使用以下命令：

```bash
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs micdysphagia:latest --config config/your_config.yaml
```

### 數據掛載說明

- `-v $(pwd)/data:/app/data`: 將本地數據目錄掛載到容器中
- `-v $(pwd)/outputs:/app/outputs`: 將容器的輸出結果保存到本地

## 常見命令

### 訓練模型

```bash
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs micdysphagia:latest --config config/train_config.yaml
```

### 僅評估模型

```bash
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs micdysphagia:latest --config config/eval_config.yaml --eval_only --checkpoint outputs/model_best.pth
```

### 調試模式

```bash
docker run -it --gpus all -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs micdysphagia:latest --config config/debug_config.yaml --debug
```

## 開發環境

若需要在容器內進行開發，可以使用以下命令啟動一個交互式 shell：

```bash
docker run -it --gpus all -v $(pwd):/app micdysphagia:latest /bin/bash
```

這將啟動一個 bash 會話，允許您在容器內執行命令。

## 自定義入口點

如果需要使用其他腳本（如 main.py 或前處理腳本），可以使用 `--entrypoint` 參數：

```bash
docker run --entrypoint python -v $(pwd)/data:/app/data micdysphagia:latest main.py --config config/your_config.yaml
```

或執行數據預處理：

```bash
docker run --entrypoint python -v $(pwd)/data:/app/data micdysphagia:latest scripts/preprocess_data.py
``` 