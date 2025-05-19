# MicDysphagiaFramework 數據結構說明

本文件詳細說明 MicDysphagiaFramework 產生的實驗結果數據結構，以幫助開發數據讀取與分析工具。

## 目錄結構

每個實驗會在 `results/` 下創建一個子目錄，命名格式為 `{實驗名稱}_{時間戳}`。例如：

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
    ├── feature_vectors/            # 特徵向量分析數據
    │   ├── feature_analysis.json   # 特徵分析摘要
    │   ├── epoch_1/               # 特定epoch的特徵向量
    │   │   ├── layer_backbone_7_features.pt    # 特定層的特徵向量
    │   │   ├── layer_backbone_7_cosine_similarity.pt # 特徵間餘弦相似度矩陣
    │   │   └── layer_backbone_7_tsne.pt        # t-SNE降維結果
    │   └── epoch_2/               # 另一個epoch的特徵向量
    │       └── ...                 # 同上
    ├── datasets/                   # 數據集信息
    │   ├── train_dataset_info.pt   # 訓練集資訊
    │   ├── val_dataset_info.pt     # 驗證集資訊
    │   ├── test_dataset_info.pt    # 測試集資訊
    │   └── dataset_statistics.json # 綜合統計資訊
    ├── results/                    # 實驗結果
    │   └── results.json            # 最終結果摘要
    ├── tensorboard_logs/           # TensorBoard日誌
    ├── test_predictions.pt         # 測試集評估結果
    └── logs/                       # 訓練日誌

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

### 6. 特徵向量分析數據 (`feature_vectors/`)

此目錄包含特定模型層的特徵向量及其分析結果，支持餘弦相似度分析和t-SNE可視化等高級分析。

#### 特徵向量摘要 (`feature_analysis.json`)

包含各層特徵向量的摘要統計信息，適用於比較不同epoch間特徵變化：

```json
{
  "backbone.7": {
    "10": {
      "epoch": 10,
      "num_samples": 256,
      "feature_dim": 512,
      "timestamp": "2025-05-15T12:34:56.789Z",
      "feature_norm_mean": 16.234,
      "feature_std": 0.765,
      "num_classes": 4,
      "class_distribution": {
        "0": 64,
        "1": 68,
        "2": 62,
        "3": 62
      }
    },
    "20": {
      "epoch": 20,
      "num_samples": 256,
      "feature_dim": 512,
      "timestamp": "2025-05-15T13:45:56.789Z",
      "feature_norm_mean": 16.753,
      "feature_std": 0.653,
      "num_classes": 4,
      "class_distribution": {
        "0": 64,
        "1": 68,
        "2": 62,
        "3": 62
      }
    }
  }
}
```

讀取方式：

```python
import json

with open('results/experiment_name/feature_vectors/feature_analysis.json', 'r') as f:
    analysis = json.load(f)

# 查看特定層在不同epoch的特徵變化
import matplotlib.pyplot as plt
epochs = sorted([int(e) for e in analysis['backbone.7'].keys()])
norms = [analysis['backbone.7'][str(e)]['feature_norm_mean'] for e in epochs]
plt.figure(figsize=(10, 6))
plt.plot(epochs, norms, marker='o')
plt.title('特徵向量範數隨訓練變化')
plt.xlabel('Epoch')
plt.ylabel('平均特徵範數')
plt.grid(True)
plt.show()
```

#### 特徵向量文件 (`epoch_N/layer_X_features.pt`)

包含特定epoch下特定層的特徵向量：

```python
import torch

features_data = torch.load('results/experiment_name/feature_vectors/epoch_10/layer_backbone_7_features.pt')
print(f"特徵向量形狀: {features_data['activations'].shape}")  # [樣本數, 特徵維度]
print(f"目標標籤: {features_data['targets'].shape}")  # [樣本數]
```

特徵向量文件結構：

```python
{
    'layer_name': 'backbone.7',        # 層名稱
    'activations': tensor([...]),      # 特徵向量張量，形狀: [樣本數, 特徵維度]
    'targets': tensor([...]),          # 目標標籤（如果有），形狀: [樣本數]
    'timestamp': '2025-05-15T12:34:56.789Z',  # 保存時間
    'epoch': 10                        # 訓練輪次
}
```

**目前框架處理說明:**
-   **`'activations'` (強制性):** 此欄位必須存在。
    -   期望形狀: `[樣本數, 特徵維度]` (2D張量)。
    -   實際處理: `HookDataLoader` 會確保此欄位存在且為張量。後續的 `IntermediateDataAnalyzer` 在進行 t-SNE 或餘弦相似度計算時，若 `activations` 張量維度大於2 (例如 `[樣本數, C, H, W]`)，會自動將其展平 (reshape) 為 `[樣本數, 總特徵數]`。
-   **`'targets'` (可選):** 此欄位為可選。
    -   期望形狀: `[樣本數]`。
    -   實際處理: 如果此欄位不存在，`IntermediateDataAnalyzer` 在執行 `analyze_feature_vectors` 時，若 `generate_random_labels` 參數設置為 `True`，則會為每個樣本生成隨機標籤。如果為 `False` 且無標籤，則部分依賴標籤的分析 (如 t-SNE 著色) 可能受影響。
-   **`'layer_name'` (期望存在，可自動推斷):**
    -   實際處理: 如果檔案內不存在此欄位，`HookDataLoader` 會嘗試從檔案名稱中解析 (例如，從 `layer_backbone_7_features.pt` 推斷出 `layer_backbone_7`)。
-   **`'epoch'` (期望存在，可自動推斷):**
    -   實際處理: 如果檔案內不存在此欄位，`HookDataLoader` 會嘗試從其所在的目錄名稱中解析 (例如，從 `epoch_10/` 目錄推斷出 `10`)。
-   **`'timestamp'` (期望存在):**
    -   實際處理: 目前的載入器不強制檢查此欄位，也不會自動推斷。建議在生成檔案時包含此資訊。

#### 餘弦相似度矩陣 (`epoch_N/layer_X_cosine_similarity.pt`)

包含特徵向量間的餘弦相似度矩陣和類別間相似度統計：

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

similarity_data = torch.load('results/experiment_name/feature_vectors/epoch_10/layer_backbone_7_cosine_similarity.pt')

# 繪製相似度熱圖
plt.figure(figsize=(12, 10))
sns.heatmap(similarity_data['similarity_matrix'].numpy(), cmap='viridis')
plt.title('特徵向量餘弦相似度熱圖')
plt.show()

# 查看類別內和類別間相似度
print(f"類別內平均相似度: {similarity_data['intra_class_avg_similarity']}")
print(f"類別間平均相似度: {similarity_data['inter_class_avg_similarity']}")
```

餘弦相似度文件結構：

```python
{
    'similarity_matrix': tensor([...]),   # 餘弦相似度矩陣，形狀: [樣本數, 樣本數]
    'timestamp': '2025-05-15T12:34:56.789Z',
    'num_samples': 256,                  # 樣本數量
    'classes': [0, 1, 2, 3],             # 類別列表
    'intra_class_avg_similarity': 0.85,  # 類別內平均相似度
    'inter_class_avg_similarity': 0.32,  # 類別間平均相似度
    'sample_to_centroid_avg_similarity': 0.79,  # 樣本到類別質心平均相似度
    'intra_class_similarities_by_class': {  # 各類別內相似度
        "0": 0.87,
        "1": 0.83,
        "2": 0.86,
        "3": 0.84
    }
}
```

#### t-SNE可視化文件 (`epoch_N/layer_X_tsne.pt`)

包含特徵向量的二維t-SNE嵌入，用於可視化：

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

tsne_data = torch.load('results/experiment_name/feature_vectors/epoch_10/layer_backbone_7_tsne.pt')

# 繪製t-SNE散點圖
plt.figure(figsize=(12, 10))
coords = tsne_data['tsne_coordinates'].numpy()
labels = tsne_data['targets'].numpy()

# 為每個類別分配不同顏色
unique_labels = np.unique(labels)
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

for i, label in enumerate(unique_labels):
    mask = labels == label
    plt.scatter(coords[mask, 0], coords[mask, 1], c=[colors[i]], label=f'Class {label}')

plt.title('t-SNE Visualization')
plt.legend()
plt.show()
```

t-SNE文件結構：

```python
  {
       'tsne_coordinates': tsne_coords,
       'targets': tensor([...]), # 目標標籤（如果有）
       'label_names': original_labels,  # 原始標籤文本
       'label_mapping':   # 映射字典, 根據yaml而定，例如`score`, `DrLee_Evaluation`, `DrTai_Evaluation`, `selection`, e. g= {0: "正常", 1: "輕度", 2: "中度"},
       'label_field': "DrLee_Evaluation",  # 標籤欄位名稱, 根據yaml而定，例如`score`, `DrLee_Evaluation`, `DrTai_Evaluation`, `selection`
       'num_samples': len(tsne_coords),  # 樣本數量
       'original_dim': embeddings.shape[1:],
       'timestamp': datetime.now().isoformat()
   }
```

### 7. 數據集信息 (`datasets/`)

包含關於訓練、驗證和測試數據集的詳細信息：

#### 數據集信息文件 (`XXX_dataset_info.pt`)

包含特定數據集的詳細信息：

```python
import torch

dataset_info = torch.load('results/experiment_name/datasets/train_dataset_info.pt')
print(f"資料集大小: {dataset_info['size']}")
print(f"特徵維度: {dataset_info['feature_dims']}")
print(f"類別分佈: {dataset_info['class_distribution']}")
```

數據集信息文件結構：

```python
{
    'dataset_type': 'train',          # 數據集類型
    'size': 1024,                     # 樣本數量
    'feature_dims': [224, 224, 3],    # 特徵維度
    'class_distribution': {           # 類別分佈
        "0": 250,
        "1": 274,
        "2": 248,
        "3": 252
    },
    'preprocessing_params': {         # 預處理參數
        'normalization': 'standard',
        'augmentation': ['rotation', 'flip']
    },
    'file_paths': [                   # 數據文件路徑
        "data/patient001.wav",
        "data/patient002.wav",
        ...
    ],
    'split_method': 'stratified'      # 數據集分割方法
}
```

#### 數據集統計信息 (`dataset_statistics.json`)

包含所有數據集的統計摘要：

```json
{
  "dataset_sizes": {
    "train": 1024,
    "val": 256,
    "test": 512
  },
  "class_distribution": {
    "train": {"0": 250, "1": 274, "2": 248, "3": 252},
    "val": {"0": 64, "1": 68, "2": 62, "3": 62},
    "test": {"0": 125, "1": 137, "2": 124, "3": 126}
  },
  "feature_statistics": {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  },
  "timestamp": "2025-05-15T12:00:00.000Z"
}
```

### 8. 實驗結果 (`results/results.json`)

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
8. 特徵向量分析數據存儲在 feature_vectors 目錄，支持不同epoch間的特徵變化分析
9. 在模型配置中可以指定要捕獲的特定層和epoch，以減少存儲開銷並專注於重要訓練階段
10. 餘弦相似度分析提供了類別內和類別間相似度統計，可用於評估特徵空間中的分類能力
