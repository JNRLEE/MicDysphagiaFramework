# Dockerfile用於構建MicDysphagiaFramework容器
# 此Dockerfile設置了一個支持PyTorch GPU訓練環境的容器

# 使用支援CUDA的PyTorch基礎映像
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# 設置工作目錄
WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libsndfile1 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 複製requirements.txt
COPY requirements.txt .

# 安裝Python依賴
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式碼
COPY . .

# 排除.gitignore中指定的檔案和目錄
# 不複製資料集、虛擬環境、輸出結果等
RUN mkdir -p data outputs results

# 設定環境變數
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# 設定默認的入口點
ENTRYPOINT ["python", "scripts/run_experiments.py"]

# 設定默認的指令
CMD ["--help"] 