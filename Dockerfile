# MicDysphagiaFramework 專案的 Dockerfile
# 基於 PyTorch 建立用於音頻分析與機器學習的環境
# 最後更新: 2025-04-15

# 使用最新穩定版 PyTorch 與 CUDA 支援
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 設置工作目錄
WORKDIR /app

# 防止 Python 創建 .pyc 文件和 __pycache__ 目錄
ENV PYTHONDONTWRITEBYTECODE=1
# 確保 Python 輸出立即刷新到終端
ENV PYTHONUNBUFFERED=1
# 將 /app 加入 Python 路徑
ENV PYTHONPATH=/app

# 安裝系統依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    libsndfile1 \
    ffmpeg \
    sox \
    libsox-dev \
    vim \
    htop \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 複製並安裝 Python 依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    # 安裝額外的 PyTorch 依賴（按需取消註釋）
    # pip install --no-cache-dir pytorch-lightning && \
    # 清理 pip 快取
    rm -rf /root/.cache/pip

# 建立必要的目錄結構
RUN mkdir -p /app/data /app/outputs /app/results /app/checkpoints /app/logs

# 複製專案文件（.dockerignore 會排除不必要的文件）
COPY . .

# 確保腳本可執行
RUN chmod +x scripts/*.py

# 暴露 TensorBoard 端口（可選）
EXPOSE 6006

# 設置健康檢查（可選）
HEALTHCHECK --interval=5m --timeout=3s \
  CMD python -c "import torch; print(torch.__version__)" || exit 1

# 設定工作目錄的工作權限
RUN chmod -R 755 /app

# 設定預設的入口點
ENTRYPOINT ["python", "scripts/run_experiments.py"]

# 設定默認的指令
CMD ["--help"]

# 使用方法:
# 基本用法: docker run micdysphagia:latest
# 使用 GPU: docker run --gpus all micdysphagia:latest --config config/your_config.yaml
# 掛載數據: docker run -v /path/to/data:/app/data -v /path/to/outputs:/app/outputs micdysphagia:latest
# 啟動 TensorBoard: docker run -d -p 6006:6006 micdysphagia:latest tensorboard --logdir=/app/outputs --host=0.0.0.0 