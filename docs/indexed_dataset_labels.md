# 使用不同標籤欄位的指南

MicDysphagiaFramework支援使用多種標籤欄位進行模型訓練，這些標籤欄位存儲在`data/metadata/data_index.csv`文件中。本文檔將指導您如何選擇和使用不同的標籤欄位。

## 可用的標籤欄位

在`data_index.csv`中，有以下主要標籤欄位可供選擇：

| 標籤欄位 | 類型 | 類別數量 | 說明 |
|---------|------|---------|------|
| `score` | 回歸 | 1 (輸出) | EAT-10問卷得分，範圍0-40的整數值 |
| `DrLee_Evaluation` | 分類 | 3 | Dr. Lee對每筆資料的分類：{"聽起來正常", "輕度異常", "重度異常"} |
| `DrTai_Evaluation` | 分類 | 4 | Dr. Tai對每筆資料的分類：{"正常", "無OR輕微吞嚥障礙", "重度吞嚥障礙", "吞嚥障礙"} |
| `selection` | 分類 | 9+ | 錄音時的動作類型：{"乾吞嚥1口", "乾吞嚥2口", "乾吞嚥3口", "吞果凍", "吞水10ml", "吞水20ml", "無動作", "餅乾1塊", "餅乾2塊"} |

## 在配置文件中設置標籤欄位

要使用特定的標籤欄位，您需要在配置文件的`data`部分設置`label_field`參數：

```yaml
data:
  # 其他配置...
  use_index: true
  index_path: 'data/metadata/data_index.csv'
  label_field: 'DrLee_Evaluation'  # 選擇標籤欄位
```

## 自動類別數量設置

從v1.2.0版本開始，框架支援自動設置模型輸出類別數量。當使用`scripts/run_experiments.py`運行實驗時，系統會自動從數據集中獲取類別數量，並更新模型配置。

這意味著您可以更改標籤欄位而不需要手動調整模型的`num_classes`參數。不過，為了清晰起見，建議在配置文件中設置正確的初始值：

```yaml
model:
  # 其他配置...
  parameters:
    # 其他參數...
    num_classes: 3  # 會根據選擇的標籤欄位自動更新
    is_classification: true  # 分類任務設為true，回歸任務設為false
```

## 分類任務與回歸任務

根據選擇的標籤欄位，您需要配置模型進行分類或回歸任務：

### 分類任務 (DrLee_Evaluation, DrTai_Evaluation, selection)

分類任務需要設置：
- `model.parameters.is_classification: true`
- `model.parameters.num_classes: N` (N為類別數量)
- `training.loss.type: 'CrossEntropyLoss'`
- 評估指標：accuracy, precision, recall, f1

### 回歸任務 (score)

回歸任務需要設置：
- `model.parameters.is_classification: false`
- `model.parameters.num_classes: 1`
- `training.loss.type: 'MSELoss'` 或 `'L1Loss'`
- 評估指標：mse, rmse, mae, r2

## 示例配置文件

框架提供了多個示例配置文件，展示如何使用不同的標籤欄位：

- `config/example_classification_indexed.yaml` - 使用DrLee_Evaluation (3類)
- `config/example_classification_drtai.yaml` - 使用DrTai_Evaluation (4類)
- `config/example_classification_selection.yaml` - 使用selection (9類+)
- `config/example_regression_score.yaml` - 使用score進行回歸任務

## 注意事項

1. **缺失值處理**：某些記錄可能缺少特定標籤欄位的值。框架會自動過濾掉這些記錄，或者在配置中設置`filter_criteria`進行更精確的篩選。

2. **類別不平衡**：不同標籤欄位的類別分布可能不均衡。考慮使用加權損失函數或數據增強技術來處理這個問題。

3. **標籤質量**：不同標籤欄位的質量和一致性可能有所不同。例如，專家評估可能比問卷得分更主觀。

4. **多標籤學習**：目前框架僅支持單一標籤欄位訓練。未來版本將支持多標籤學習。 