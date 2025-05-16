# MicDysphagiaFramework 與 SBP_analyzer 集成指南

## 概述

本指南介紹如何將 MicDysphagiaFramework 與 SBP_analyzer 工具集成，實現深度學習模型訓練過程的高級分析和可視化。MicDysphagiaFramework 的 Hook 系統提供了標準化的數據收集機制，而 SBP_analyzer 則提供了進階的分析和可視化功能。

## 功能對應關係

| MicDysphagiaFramework Hook 功能 | SBP_analyzer 分析功能 |
|--------------------------------|---------------------|
| 評估結果捕獲 (EvaluationResultsHook) | 混淆矩陣分析、ROC曲線、類別精確率/召回率 |
| 激活值捕獲 (ActivationCaptureHook) | 餘弦相似度分析、t-SNE 降維可視化、特徵聚類 |
| 驗證預測保存 | 預測變化追蹤、模型收斂性分析 |
| 梯度捕獲 | 梯度分布分析、GNS 統計 |

## 集成步驟

### 1. 設置 MicDysphagiaFramework 以收集數據

在 MicDysphagiaFramework 配置文件中啟用相關 Hook：

```yaml
hooks:
  # 模型分析鉤子：收集層激活值和梯度
  model_analytics:
    enabled: true
    monitored_layers: ['feature_extractor.0', 'feature_extractor.3', 'feature_extractor.6']
    monitored_params: ['head.weight']
    save_frequency: 1
    save_validation_predictions: true
  
  # 評估結果捕獲鉤子
  evaluation_capture:
    enabled: true
    datasets: ['test']
    save_probabilities: true
  
  # 激活值捕獲鉤子
  activation_capture:
    enabled: true
    target_layers: ['feature_extractor']
    datasets: ['test']
    include_sample_ids: true
```

### 2. 運行 MicDysphagiaFramework 實驗

執行實驗以生成所需數據：

```bash
python scripts/run_experiments.py --config config/your_config.yaml
```

實驗完成後，結果將保存在 `results/{實驗名稱}_{時間戳}/` 目錄下。

### 3. 使用 SBP_analyzer 分析結果

安裝 SBP_analyzer（如果尚未安裝）：

```bash
pip install sbp-analyzer
```

運行分析腳本：

```bash
# 基本分析（生成所有標準圖表）
sbp-analyzer analyze --experiment_dir results/your_experiment_timestamp

# 特定類別的分析
sbp-analyzer analyze --experiment_dir results/your_experiment_timestamp --analysis_type confusion_matrix
sbp-analyzer analyze --experiment_dir results/your_experiment_timestamp --analysis_type cosine_similarity
sbp-analyzer analyze --experiment_dir results/your_experiment_timestamp --analysis_type gradient_distribution
sbp-analyzer analyze --experiment_dir results/your_experiment_timestamp --analysis_type prediction_tracking
```

分析結果將保存在 `analysis_results/{實驗名稱}_{時間戳}/` 目錄下。

## 主要分析功能

### 1. 混淆矩陣分析

使用 `evaluation_results_test.pt` 中的預測標籤和真實標籤生成混淆矩陣：

```bash
sbp-analyzer analyze --experiment_dir results/your_experiment_timestamp --analysis_type confusion_matrix
```

生成的混淆矩陣圖表將保存在 `analysis_results/{實驗名稱}_{時間戳}/confusion_matrix.png`。

### 2. 餘弦相似度分析

使用 `test_set_activations_feature_extractor.pt` 中的激活值生成餘弦相似度熱圖：

```bash
sbp-analyzer analyze --experiment_dir results/your_experiment_timestamp --analysis_type cosine_similarity --layer feature_extractor
```

生成的餘弦相似度圖表將保存在 `analysis_results/{實驗名稱}_{時間戳}/cosine_similarity_feature_extractor.png`。

### 3. 預測變化追蹤

使用各輪次的驗證集預測結果 `epoch_N_validation_predictions.pt` 追蹤模型預測變化：

```bash
sbp-analyzer analyze --experiment_dir results/your_experiment_timestamp --analysis_type prediction_tracking
```

生成的預測變化追蹤圖表將保存在 `analysis_results/{實驗名稱}_{時間戳}/prediction_tracking.png`。

### 4. ROC 曲線分析

使用 `evaluation_results_test.pt` 中的類別概率生成 ROC 曲線：

```bash
sbp-analyzer analyze --experiment_dir results/your_experiment_timestamp --analysis_type roc_curve
```

生成的 ROC 曲線圖表將保存在 `analysis_results/{實驗名稱}_{時間戳}/roc_curve.png`。

## 自定義分析

對於高級用戶，可以直接讀取保存的數據進行自定義分析：

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

# 載入評估結果
eval_results = torch.load('results/your_experiment_timestamp/hooks/evaluation_results_test.pt')

# 載入激活值
activations = torch.load('results/your_experiment_timestamp/hooks/test_set_activations_feature_extractor.pt')

# 載入驗證集預測
val_predictions = [
    torch.load(f'results/your_experiment_timestamp/hooks/epoch_{i}_validation_predictions.pt')
    for i in range(10)  # 載入前10輪
]

# 自定義分析代碼
```

## 注意事項

1. 確保 MicDysphagiaFramework 和 SBP_analyzer 的版本兼容。
2. 大型實驗可能會生成大量數據，請確保有足夠的磁盤空間。
3. 一些高級分析可能需要較長時間運行，特別是對於大型數據集。