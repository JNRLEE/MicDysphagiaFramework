# 排序損失函數配置指南

本文件提供在吞嚥障礙評估框架中配置和使用排序損失函數的詳細說明。

## 目錄

- [基本配置](#基本配置)
- [成對排序損失](#成對排序損失)
- [列表排序損失](#列表排序損失)
- [組合損失函數](#組合損失函數)
- [配置範例](#配置範例)
- [自訂損失函數](#自訂損失函數)

## 基本配置

在模型配置文件中，您可以通過以下方式指定要使用的排序損失函數：

```yaml
model:
  name: dysphagia_ranking_model
  type: ranking

training:
  loss:
    type: ranking_loss
    name: margin_ranking_loss  # 可選: margin_ranking_loss, ranknet, fidelity, listmle, approx_ndcg, lambda_rank
    params:
      margin: 1.0  # 僅適用於 margin_ranking_loss
      sigma: 1.0   # 僅適用於 ranknet, lambda_rank
```

## 成對排序損失

### 邊際排序損失 (Margin Ranking Loss)

適用於成對比較，確保正確排序的樣本間有足夠的評分差距。

```yaml
loss:
  type: ranking_loss
  name: margin_ranking_loss
  params:
    margin: 1.0    # 期望維持的最小評分差距
    reduction: mean  # 可為 'none', 'mean', 'sum'
```

### RankNet 損失

基於概率方法的成對排序損失，使用交叉熵計算樣本對的排序損失。

```yaml
loss:
  type: ranking_loss
  name: ranknet
  params:
    sigma: 1.0     # sigmoid函數的縮放因子
    reduction: mean  # 可為 'none', 'mean', 'sum'
```

### 保真度損失 (Fidelity Loss)

基於量子力學中保真度概念的排序損失函數，適合處理複雜的排序關係。

```yaml
loss:
  type: ranking_loss
  name: fidelity
  params:
    reduction: mean  # 可為 'none', 'mean', 'sum'
```

## 列表排序損失

### ListMLE 損失

基於最大似然估計的列表排序損失，適合直接優化整個排序列表。

```yaml
loss:
  type: ranking_loss
  name: listmle
  params:
    eps: 1e-10  # 用於數值穩定性的小常數
```

### 近似 NDCG 損失

直接優化 NDCG 評估指標的可微分近似版本。

```yaml
loss:
  type: ranking_loss
  name: approx_ndcg
  params:
    temperature: 1.0  # softmax 的溫度參數
    eps: 1e-10        # 用於數值穩定性的小常數
```

### LambdaRank 損失

通過重新加權成對比較來間接優化 NDCG 等排序評估指標。

```yaml
loss:
  type: ranking_loss
  name: lambda_rank
  params:
    sigma: 1.0  # sigmoid 函數的縮放因子
    k: 10       # 截斷的排名位置，即 NDCG@k
```

## 組合損失函數

您可以組合多個損失函數，並設定自適應權重調整策略：

```yaml
loss:
  type: ranking_loss
  name: adaptive_weighted
  components:
    - name: ranknet
      weight: 0.4
      params:
        sigma: 1.0
    - name: listmle
      weight: 0.3
      params:
        eps: 1e-10
    - name: lambda_rank
      weight: 0.3
      params:
        sigma: 1.0
        k: 10
  adaptation:
    method: loss_ratio  # 可為 'grad_norm', 'loss_ratio', 'uncertainty'
    frequency: 10       # 權重調整頻率
```

## 配置範例

### 吞嚥障礙評估的成對排序配置

```yaml
model:
  name: dysphagia_severity_ranker
  type: ranking
  input_dim: 20
  hidden_dims: [64, 32, 16]

training:
  epochs: 50
  batch_size: 32
  optimizer:
    name: adam
    lr: 0.001
  loss:
    type: ranking_loss
    name: ranknet
    params:
      sigma: 1.0

evaluation:
  metrics:
    - ndcg@5
    - ndcg@10
    - spearman_correlation
    - kendall_tau
```

### 使用組合損失函數的配置

```yaml
model:
  name: dysphagia_clinical_ranker
  type: ranking
  input_dim: 128
  hidden_dims: [256, 128, 64]

training:
  epochs: 100
  batch_size: 16
  optimizer:
    name: adam
    lr: 0.0005
    weight_decay: 0.0001
  loss:
    type: ranking_loss
    name: adaptive_weighted
    components:
      - name: ranknet
        weight: 0.4
        params:
          sigma: 1.0
      - name: approx_ndcg
        weight: 0.3
        params:
          temperature: 0.5
      - name: fidelity
        weight: 0.3
    adaptation:
      method: grad_norm
      frequency: 5

evaluation:
  metrics:
    - ndcg@5
    - ndcg@10
    - precision@5
    - mean_average_precision
    - clinical_agreement_score
```

## 自訂損失函數

若要自訂損失函數，您需要：

1. 在 `losses/ranking_losses.py` 中實作損失函數類別
2. 將損失函數註冊到損失函數註冊表中
3. 在配置文件中引用您的自訂損失函數名稱

示例：

```python
# 在 losses/ranking_losses.py 中添加您的自訂損失函數

class MyCustomRankingLoss(nn.Module):
    def __init__(self, param1=1.0, param2=0.5):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        
    def forward(self, predictions, targets):
        # 實作您的損失計算邏輯
        loss = ...
        return loss

# 註冊您的損失函數
RANKING_LOSS_REGISTRY.register("my_custom_loss", MyCustomRankingLoss)
```

然後在配置文件中：

```yaml
loss:
  type: ranking_loss
  name: my_custom_loss
  params:
    param1: 2.0
    param2: 0.75
```

透過以上配置，您可以根據吞嚥障礙評估的具體需求，選擇合適的排序損失函數或其組合，以達到最佳的評估效果。 