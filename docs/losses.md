# 損失函數設計與使用

本文檔詳細說明了MicDysphagiaFramework支持的損失函數、設計理念、配置方法和使用示例。

## 設計理念

損失函數在模型訓練中起著關鍵作用，尤其在吞嚥障礙評估任務中，既需要預測準確的EAT-10評分(回歸任務)，又需要正確排序不同病例的嚴重程度(排序任務)。因此，我們設計了多種損失函數，並支持靈活組合：

1. **標準損失函數**：支持PyTorch標準損失函數，適用於回歸和分類任務
2. **排序損失函數**：支持Pairwise和Listwise兩類排序損失函數，專注於保持數據間的相對順序
3. **組合損失函數**：允許多個損失函數加權組合，同時優化多個目標

損失函數系統採用工廠模式設計，支持通過配置文件動態選擇和組合損失函數，簡化了損失函數的配置和使用。

## 損失函數類型

### 標準損失函數

標準損失函數主要用於回歸和分類任務，框架支持所有PyTorch內置的損失函數：

| 損失函數 | 用途 | 適用場景 |
|--------|------|---------|
| MSELoss | 均方誤差損失 | 回歸任務，如EAT-10評分預測 |
| L1Loss | 平均絕對誤差 | 回歸任務，對異常值不敏感 |
| SmoothL1Loss | 平滑L1損失 | 回歸任務，結合MSE和L1的優點 |
| HuberLoss | Huber損失 | 回歸任務，對異常值更穩健 |
| CrossEntropyLoss | 交叉熵損失 | 多類別分類任務 |
| BCELoss | 二元交叉熵損失 | 二分類任務 |
| BCEWithLogitsLoss | 帶Logits的BCE | 二分類任務，數值更穩定 |

### 排序損失函數

排序損失函數專注於優化樣本間的相對順序關係，特別適合需要正確排序吞嚥障礙嚴重程度的場景：

#### 1. PairwiseRankingLoss (成對排序損失)

對樣本對進行比較，確保預測排序與真實排序一致。

```python
# 損失函數配置示例
loss_config = {
    "type": "PairwiseRankingLoss",
    "parameters": {
        "margin": 0.3,              # 排序間隔
        "sampling_ratio": 0.5,       # 採樣比例
        "sampling_strategy": "score_diff",  # 採樣策略
        "use_exp": False            # 是否使用指數加權
    }
}
```

**採樣策略選項**：
- `score_diff`: 基於真實分數差異採樣，優先選擇差異大的對
- `random`: 隨機採樣
- `hard_negative`: 優先選擇預測錯誤的"困難"對

#### 2. ListwiseRankingLoss (列表排序損失)

將整個批次作為排序列表處理，優化整體排序結果。

```python
# 損失函數配置示例
loss_config = {
    "type": "ListwiseRankingLoss",
    "parameters": {
        "method": "listnet",        # 方法：listnet, listmle, approxndcg
        "temperature": 1.0,         # softmax溫度參數
        "k": 10,                   # top-k評估參數
        "group_size": 0,           # 分組大小，0表示不分組
        "stochastic": True         # 是否添加隨機擾動
    }
}
```

**方法選項**：
- `listnet`: 使用交叉熵比較真實和預測的排序概率分布
- `listmle`: 最大似然排序，優化排序概率
- `approxndcg`: 近似優化NDCG評估指標

#### 3. LambdaRankLoss

高級排序損失函數，結合了成對和列表方法的優點，並考慮NDCG等評估指標。

```python
# 損失函數配置示例
loss_config = {
    "type": "LambdaRankLoss",
    "parameters": {
        "sigma": 1.0,              # sigmoid尺度參數
        "k": 10,                   # NDCG@k評估
        "sampling_ratio": 0.3      # 採樣比例
    }
}
```

### 組合損失函數

組合損失函數允許多個損失函數加權組合，同時優化多個目標：

```python
# 組合損失函數配置示例
loss_config = {
    "combined": {
        "mse": {
            "type": "MSELoss",
            "weight": 0.6,
            "parameters": {
                "reduction": "mean"
            }
        },
        "ranking": {
            "type": "PairwiseRankingLoss",
            "weight": 0.4,
            "parameters": {
                "margin": 0.3,
                "sampling_strategy": "score_diff"
            }
        }
    }
}
```

**特殊功能**：
- 支持不同損失函數的加權組合
- 可選自適應權重調整，根據訓練過程動態調整權重
- 提供詳細的單個損失值，便於監控訓練過程

## 配置與使用

### YAML配置示例

在配置文件中，可以在`training`部分配置損失函數：

**標準損失函數**:
```yaml
training:
  loss:
    type: "MSELoss"
    parameters:
      reduction: "mean"
```

**排序損失函數**:
```yaml
training:
  loss:
    type: "PairwiseRankingLoss"
    parameters:
      margin: 0.3
      sampling_ratio: 0.5
      sampling_strategy: "score_diff"
```

**組合損失函數**:
```yaml
training:
  loss:
    combined:
      mse:
        type: "MSELoss"
        weight: 0.6
      pairwise:
        type: "PairwiseRankingLoss"
        weight: 0.4
        parameters:
          margin: 0.3
          sampling_strategy: "score_diff"
```

**高級組合損失函數(帶自適應權重)**:
```yaml
training:
  loss:
    combined:
      mse:
        type: "MSELoss"
        weight: 0.6
      listwise:
        type: "ListwiseRankingLoss"
        weight: 0.4
        parameters:
          method: "listnet"
          temperature: 0.5
    adaptive_weights: true
    weight_update_freq: 100
    weight_update_ratio: 0.1
```

### 程式碼使用示例

如果需要在代碼中直接使用損失函數，可以這樣做：

```python
from losses import LossFactory

# 創建損失函數
loss_config = {
    "type": "PairwiseRankingLoss",
    "parameters": {
        "margin": 0.3,
        "sampling_strategy": "score_diff"
    }
}
loss_fn = LossFactory.get_loss(loss_config)

# 直接從配置文件創建
from utils.config_loader import ConfigLoader
config = ConfigLoader.load("config/my_config.yaml")
loss_fn = LossFactory.create_from_config(config["training"]["loss"])

# 使用損失函數
outputs = model(inputs)
loss = loss_fn(outputs, targets)
```

## 推薦配置

根據不同任務，我們推薦使用以下損失函數配置：

### EAT-10回歸任務

適合準確預測EAT-10分數的損失函數配置：

```yaml
training:
  loss:
    combined:
      mse:
        type: "MSELoss"
        weight: 0.7
      pairwise:
        type: "PairwiseRankingLoss"
        weight: 0.3
        parameters:
          margin: 0.3
          sampling_strategy: "score_diff"
```

### 吞嚥障礙嚴重程度分類任務

適合分類吞嚥障礙嚴重程度的損失函數配置：

```yaml
training:
  loss:
    type: "CrossEntropyLoss"
    parameters:
      reduction: "mean"
```

### 混合任務(同時預測分數和排序)

適合同時優化分數預測和排序的損失函數配置：

```yaml
training:
  loss:
    combined:
      mse:
        type: "MSELoss"
        weight: 0.5
      l1:
        type: "L1Loss"
        weight: 0.1
      lambda_rank:
        type: "LambdaRankLoss"
        weight: 0.4
        parameters:
          sigma: 1.0
          k: 5
```

## 損失函數選擇與調優指南

1. **回歸任務**：首選MSELoss或HuberLoss，根據數據分布可選擇適合的損失函數
2. **關注順序**：當患者間的相對評分順序重要時，添加PairwiseRankingLoss或ListwiseRankingLoss
3. **批次大小影響**：排序損失函數對批次大小敏感，建議較大批次(32+)以確保足夠的對比樣本
4. **採樣策略選擇**：初期使用`score_diff`策略，進入後期可嘗試`hard_negative`策略
5. **權重調整**：初始可設置均等權重，根據訓練過程調整或啟用自適應權重
6. **損失監控**：通過監控各個損失函數的值，分析對模型訓練的影響並相應調整配置

通過合理選擇和組合損失函數，可以提高模型在吞嚥障礙評估任務上的性能，同時保證評分的準確性和排序的合理性。 