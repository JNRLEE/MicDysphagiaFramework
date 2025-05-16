# MicDysphagiaFramework 數據結構說明

本文件詳細說明 MicDysphagiaFramework 產生的實驗結果數據結構，以幫助開發數據讀取與分析工具。

## 目錄結構

每個實驗會在 `results/` 下創建一個子目錄，命名格式為 `{實驗名稱}_{時間戳}`。例如：

```
results/
└── audio_swin_regression_20250417_142912/
    ├── config.json                 # 實驗配置文件副本
    ├── model_structure.json        # 模型結構信息
    ├── training_history.json       # 訓練歷史記錄
    ├── models/                     # 模型權重保存目錄
    │   ├── best_model.pth          # 最佳模型權重
    │   ├── checkpoint_epoch_0.pth  # 第0輪模型權重檢查點
    │   └── checkpoint_epoch_1.pth  # 第1輪模型權重檢查點
    ├── hooks/                      # 模型鉤子數據
    │   ├── training_summary.pt     # 整體訓練摘要
    │   ├── evaluation_results_test.pt  # 驗證集評估結果
    │   ├── epoch_0_validation_predictions.pt  # 第0輪驗證集預測結果
    │   ├── epoch_1_validation_predictions.pt  # 第1輪驗證集預測結果
    │   ├── test_set_activations_feature_extractor.pt # 測試集特徵提取器激活值
    │   ├── epoch_0/                # 第0輪數據
    │   │   ├── epoch_summary.pt    # 輪次摘要
    │   │   ├── batch_0_data.pt     # 第0批次數據
    │   │   ├── head_activation_batch_0.pt  # 頭部層激活值
    │   │   ├── head_gradient_batch_0.pt    # 頭部層梯度張量
    │   │   ├── head_gradient_batch_0_stats.json # 頭部層梯度統計量（含分位數）
    │   │   ├── head_gradient_batch_0_hist.pt # 頭部層梯度直方圖
    │   │   └── gns_stats_epoch_0.json      # GNS統計量
    │   └── epoch_1/                # 第1輪數據
    │       └── ...                 # 同上
    ├── results/                    # 實驗結果
    │   └── results.json            # 最終結果摘要
    ├── tensorboard_logs/           # TensorBoard日誌
    ├── test_predictions.pt         # 測試集評估結果
    └── logs/                       # 訓練日誌
```

## 文件格式與內容

### 1. 配置文件 (`config.json`)

實驗配置的JSON副本，包含完整的實驗參數，包括：

- 模型類型與參數
- 數據集設置
- 訓練參數（學習率、批次大小等）
- 評估與回調設置

```json
{
  "experiment_name": "audio_swin_regression",
  "model": {
    "name": "swin_transformer",
    "params": {
      "input_channels": 1,
      "patch_size": 4,
      "embed_dim": 96,
      "depths": [2, 2, 6, 2],
      "num_heads": [3, 6, 12, 24],
      "window_size": 7
    }
  },
  "trainer": {
    "epochs": 100,
    "batch_size": 32,
    "optimizer": {
      "name": "adam",
      "params": {
        "lr": 0.001
      }
    }
  }
}
```

### 2. 模型權重 (`models/`)

#### 最佳模型 (`best_model.pth`)

根據驗證集表現保存的最佳模型權重，適用於推論或遷移學習。通過 `torch.load()` 讀取：

```python
import torch

model = YourModelClass()  # 先創建模型實例
model.load_state_dict(torch.load('results/experiment_name/models/best_model.pth'))
model.eval()  # 設置為評估模式
```

#### 檢查點 (`checkpoint_epoch_N.pth`)

各輪次的完整檢查點，包含模型權重、優化器狀態和訓練信息，適用於恢復訓練：

```python
import torch

checkpoint = torch.load('results/experiment_name/models/checkpoint_epoch_5.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

### 3. 模型結構 (`model_structure.json`)

包含模型架構的詳細信息，包括層結構、參數數量和形狀：

```json
{
  "model_summary": "SwinTransformer(\n  (patch_embed): PatchEmbed(...)",
  "layer_info": [
    {
      "name": "patch_embed",
      "type": "PatchEmbed",
      "parameters": 27744,
      "shape": [96, 56, 56]
    },
    {
      "name": "layers.0",
      "type": "BasicLayer",
      "parameters": 553248,
      "shape": [96, 56, 56]
    }
  ],
  "total_parameters": 28427856
}
```

### 4. 訓練歷史 (`training_history.json`)

包含每輪的訓練和驗證指標，用於分析訓練過程：

```json
{
  "loss": [0.568, 0.342, 0.245],
  "val_loss": [0.495, 0.301, 0.256],
  "mae": [0.245, 0.156, 0.098],
  "val_mae": [0.225, 0.145, 0.102],
  "lr": [0.001, 0.001, 0.0005]
}
```

讀取方式：

```python
import json

with open('results/experiment_name/training_history.json', 'r') as f:
    history = json.load(f)
  
# 繪製損失曲線
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

### 5. 鉤子數據 (`hooks/`)

#### 訓練摘要 (`training_summary.pt`)

包含整體訓練過程的摘要信息：

```python
import torch

training_summary = torch.load('results/experiment_name/hooks/training_summary.pt')
print(f"總訓練時間: {training_summary['total_training_time']} 秒")
print(f"每批次平均時間: {training_summary['avg_batch_time']} 秒")
```

#### 輪次摘要 (`epoch_N/epoch_summary.pt`)

包含特定輪次的訓練摘要：

```python
epoch_summary = torch.load('results/experiment_name/hooks/epoch_0/epoch_summary.pt')
print(f"輪次耗時: {epoch_summary['epoch_time']} 秒")
print(f"輪次損失: {epoch_summary['loss']}")
```

#### 批次數據 (`epoch_N/batch_M_data.pt`)

包含特定批次的完整信息，包括輸入、輸出、目標和損失：

```python
batch_data = torch.load('results/experiment_name/hooks/epoch_0/batch_0_data.pt')
inputs = batch_data['inputs']      # 輸入張量
outputs = batch_data['outputs']    # 模型輸出
targets = batch_data['targets']    # 目標張量
loss = batch_data['loss']          # 損失值
```

#### 激活值 (`epoch_N/layer_name_activation_batch_M.pt`)

包含特定層在特定批次的激活值：

```python
activation = torch.load('results/experiment_name/hooks/epoch_0/head_activation_batch_0.pt')
act_values = activation['activation']   # 激活值張量
layer_name = activation['layer_name']   # 層名稱
```

#### 梯度 (`epoch_N/param_name_gradient_batch_M.pt`)

- **內容說明**: 特定參數 (如 `layer.weight`, `layer.bias`) 在特定批次或 epoch (`_all`) 的原始梯度張量。
- **檔案格式**: PyTorch 張量 (`.pt`)。
- **讀取方式**: `grad_tensor = torch.load('path/to/gradient.pt')`
- **用途**: 可用於細粒度的梯度檢查、可視化或進一步分析。

#### 梯度統計量 (`epoch_N/param_name_gradient_batch_M_stats.json`)

- **內容說明**: 特定參數梯度的詳細統計信息。
- **檔案格式**: JSON，包含如下欄位：
  ```json
  {
    "mean": 0.001,
    "std": 0.01,
    "min": -0.05,
    "max": 0.06,
    "norm": 1.23,
    "quantile_25": -0.005,   // 25% 分位數
    "quantile_50": 0.001,    // 中位數
    "quantile_75": 0.008,    // 75% 分位數
    "timestamp": "2024-04-27T10:00:00Z",
    "epoch": 0,
    "batch": 0               // 若為epoch結尾保存則為null
  }
  ```
- **讀取方式**: 標準 JSON 解析。
- **用途**: 快速了解梯度的基本分布特性。

#### 梯度直方圖 (`epoch_N/param_name_gradient_batch_M_hist.pt`)

- **內容說明**: 特定參數梯度的直方圖數據，包含計數 (`hist`) 和箱體邊界 (`bin_edges`)。
- **檔案格式**: PyTorch Dictionary (`.pt`)。
  ```python
  hist_data = {
      'hist': tensor([...]),      # 計數張量
      'bin_edges': tensor([...]), # 箱體邊界張量
      'timestamp': '...', 
      'epoch': 0,
      'batch': 0
  }
  ```
- **讀取方式**: `hist_data = torch.load('path/to/gradient_hist.pt')`
- **用途**: 用於繪製梯度分布直方圖，視覺化分析梯度分布情況。

#### 驗證集預測 (`epoch_N_validation_predictions.pt`)

- **內容說明**: 每個訓練輪次結束時的驗證集預測結果，包含原始輸出、預測類別和真實標籤。
- **檔案格式**: PyTorch Dictionary (`.pt`)。
  ```python
  validation_data = {
      'outputs': tensor([...]),    # 模型原始輸出
      'targets': tensor([...]),    # 真實標籤
      'predictions': tensor([...]) # 預測類別（對於分類任務）
  }
  ```
- **讀取方式**: `validation_data = torch.load('path/to/epoch_N_validation_predictions.pt')`
- **用途**: 用於分析模型在每個訓練輪次結束時的驗證集表現，追蹤模型訓練過程中的預測變化。

#### 評估結果 (`evaluation_results_dataset.pt`)

包含在特定數據集上的評估結果：

```python
evaluation = torch.load('results/experiment_name/hooks/evaluation_results_test.pt')
metrics = evaluation['metrics']           # 評估指標
predictions = evaluation['predictions']   # 模型預測
targets = evaluation['targets']           # 真實標籤
```

如果是分類任務，還會包括類別概率：

```python
probabilities = evaluation['probabilities']  # 類別概率（形狀: [樣本數, 類別數]）
```

#### 層激活值捕獲 (`dataset_set_activations_layer_name.pt`)

- **內容說明**: 特定數據集上特定層的激活值，用於餘弦相似度分析和其他高級模型分析。
- **檔案格式**: PyTorch Dictionary (`.pt`)。
  ```python
  activation_data = {
      'layer_name': 'feature_extractor',
      'activations': tensor([...]),  # 形狀: [樣本數, 特徵維度]
      'targets': tensor([...]),      # 真實標籤
      'sample_ids': [...],           # 樣本ID (如果提供)
      'timestamp': '2024-05-02T12:34:56.789Z'
  }
  ```
- **讀取方式**: `activation_data = torch.load('path/to/dataset_set_activations_layer_name.pt')`
- **用途**: 用於分析模型內部特徵空間，如餘弦相似度圖、t-SNE可視化、特徵聚類等。

#### 評估結果 (`hooks/evaluation_results_test.pt`)

包含在測試集上的評估結果：

```python
evaluation = torch.load('results/experiment_name/hooks/evaluation_results_test.pt')
metrics = evaluation['metrics']           # 評估指標
predictions = evaluation['predictions']   # 模型預測
targets = evaluation['targets']           # 真實標籤
```

#### GNS 統計量 (`epoch_N/gns_stats_epoch_N.json`)

- **內容說明**：每個 epoch 計算一次 GNS (Gradient Noise Scale) 統計量，記錄於 hooks/epoch_N/gns_stats_epoch_N.json。
- **檔案格式**：JSON，包含如下欄位：
  ```json
  {
    "gns": 0.123,                // GNS數值
    "total_var": 1.234,          // 所有batch梯度總變異數
    "mean_norm_sq": 0.567,       // 所有batch梯度均值的平方範數
    "epoch": 0,                  // epoch編號
    "timestamp": "2024-04-26T21:54:47.351Z", // 記錄時間
    "reference": "https://arxiv.org/abs/2006.08536"
  }
  ```
- **讀取方式**：
  ```python
  import json
  with open('results/experiment_name/hooks/epoch_0/gns_stats_epoch_0.json') as f:
      gns_stats = json.load(f)
  print(f"GNS: {gns_stats['gns']}")
  ```
- **用途**：可用於分析訓練過程中的梯度雜訊規模，協助調整 batch size 或學習率。

### 6. 實驗結果 (`results/results.json`)

包含實驗的最終結果摘要：

```json
{
  "test_metrics": {
    "loss": 0.235,
    "mae": 0.089,
    "r2_score": 0.856
  },
  "best_val_metrics": {
    "loss": 0.256,
    "mae": 0.102,
    "epoch": 87
  },
  "training_time": 3650.25,
  "model_size": 106.8
}
```

## 注意事項

1. 所有 `.pt` 文件都是 PyTorch 對象，可以使用 `torch.load()` 讀取
2. 所有 `.json` 文件可以使用標準 JSON 方法解析
3. TensorBoard 日誌需要使用 TensorBoard 工具進行可視化:
   ```
   tensorboard --logdir=results/experiment_name/tensorboard_logs
   ```
4. 如果實驗提前終止，某些文件可能不存在或不完整
5. 每個 epoch 的 GNS 統計量皆儲存於 hooks/epoch_N/gns_stats_epoch_N.json，可直接用於自動化分析與可視化
6. 每個 epoch 的驗證集預測結果都會保存在 hooks/epoch_N_validation_predictions.pt，可用於分析模型訓練過程中的預測變化
7. 對於分類任務，評估結果文件將包含類別概率，可用於繪製 ROC 曲線和計算其他進階指標
