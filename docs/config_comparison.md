# MicDysphagiaFramework 配置文件比較指南

本文檔提供了 MicDysphagiaFramework 不同配置文件的詳細比較，幫助使用者選擇最適合其需求的配置方案。

## 配置文件類型概述

MicDysphagiaFramework 提供了多種配置文件，主要分為以下幾類：

1. **基於索引 CSV 的配置** - 使用 `data/metadata/data_index.csv` 作為數據源
   - `example_classification_indexed.yaml` - 用於分類任務
   - `example_eat10_regression_indexed.yaml` - 用於 EAT-10 回歸任務

2. **傳統直接模式配置** - 直接從文件系統加載數據
   - `example_classification.yaml` - 用於分類任務
   - `example_eat10_regression.yaml` - 用於 EAT-10 回歸任務

3. **特定模型架構配置** - 針對特定模型架構優化的配置
   - `custom_audio_cnn_classification.yaml` - 使用 CNN 進行音頻分類
   - `custom_audio_fcnn_classification.yaml` - 使用全連接神經網絡進行音頻分類

## 詳細配置比較

### 索引模式 vs. 傳統模式

#### `example_classification_indexed.yaml` vs. `example_classification.yaml`

| 特性 | 索引模式 | 傳統模式 |
|------|----------|----------|
| **數據源** | 從 data_index.csv 加載 | 直接從文件系統加載 |
| **標籤選擇** | 支持多種標籤欄位 (score, DrTai_Evaluation, DrLee_Evaluation 等) | 僅支持固定標籤 |
| **數據篩選** | 可按患者 ID、動作類型、處理狀態等篩選 | 無內建篩選功能 |
| **退化機制** | 支持索引不可用時退化到傳統模式 | 不適用 |
| **適用場景** | 大型數據集、需要靈活標籤選擇、需要數據篩選 | 小型數據集、標籤固定、無需複雜篩選 |
| **優點** | 更靈活、支持複雜查詢、方便管理大型數據集 | 設置簡單、依賴少、直觀 |
| **缺點** | 需要維護索引CSV、設置稍複雜 | 缺乏靈活性、難以篩選數據 |

#### `example_eat10_regression_indexed.yaml` vs. `example_eat10_regression.yaml`

與上方分類任務的比較類似，主要差異在於任務類型為回歸而非分類。回歸任務針對 EAT-10 評分（連續值）進行預測，而非離散類別。

### 不同模型架構配置比較

#### `custom_audio_cnn_classification.yaml` vs. `custom_audio_fcnn_classification.yaml`

| 特性 | CNN 模型 | FCNN 模型 |
|------|----------|----------|
| **模型架構** | 卷積神經網絡 | 全連接神經網絡 |
| **參數量** | 中等 | 較大 (取決於隱藏層配置) |
| **適用數據類型** | 音頻、頻譜圖 (最適合) | 特徵向量、處理後的音頻 |
| **特點** | 能捕捉局部特徵和空間關係 | 適合處理已提取的特徵 |
| **訓練速度** | 中等 | 較快 (較簡單的架構) |
| **性能表現** | 適合識別音頻中的空間模式 | 適合基於已提取特徵的分類 |
| **適用場景** | 直接從原始音頻或頻譜圖學習特徵 | 使用預先提取的特徵進行分類 |

### 索引模式特定配置選項

以下是僅在基於索引的配置文件中可用的關鍵配置選項：

```yaml
data:
  source:
    use_index: true                         # 啟用索引CSV功能
    index_path: "data/metadata/data_index.csv"  # 索引CSV路徑
    label_column: "DrTai_Evaluation"        # 使用的標籤欄位
    label_mapping:                          # 標籤映射（文字標籤轉數字）
      "正常": 0
      "無OR 輕微吞嚥障礙": 1 
      "重度吞嚥障礙": 2
      "吞嚥障礙": 2
    filters:                                # 數據篩選條件
      selection: ["乾吞嚥1口", "乾吞嚥2口", "乾吞嚥3口"]  # 只使用特定動作類型
      status: "processed"                   # 只使用處理完成的數據
```

### 模型特定配置選項

#### CNN 配置示例

```yaml
model:
  type: "cnn"
  parameters:
    input_channels: 1
    num_classes: 3
    kernel_size: 3
    conv_layers: [16, 32, 64, 128]  # 卷積層通道數
    fc_layers: [256, 128]           # 全連接層大小
    dropout: 0.5
    pooling_type: "max"
    activation: "relu"
```

#### FCNN 配置示例

```yaml
model:
  type: "fcnn"
  parameters:
    input_size: 256  # 輸入特徵維度
    hidden_layers: [128, 64, 32]  # 隱藏層大小
    num_classes: 3
    dropout: 0.3
    activation: "relu"
    batch_norm: true
```

## 配置選擇指南

### 何時使用索引模式

1. **數據集較大或結構複雜**：索引模式提供更好的組織和查詢能力
2. **需要靈活的標籤選擇**：可以輕鬆切換不同的標籤欄位（EAT-10 分數、醫生評估等）
3. **需要複雜的數據篩選**：可根據各種條件（患者ID、動作類型、處理狀態等）篩選數據
4. **進行多標籤或多任務學習**：索引模式更容易支持多標籤學習

### 何時使用傳統模式

1. **數據集較小且結構簡單**：傳統模式設置更簡單直接
2. **標籤固定且一致**：不需要在不同標籤之間切換
3. **無需數據篩選**：使用所有可用數據
4. **環境限制**：在無法輕易建立或維護索引CSV的環境中

### 模型選擇建議

1. **直接處理原始音頻**：
   - 首選 CNN 或 Swin Transformer 架構
   - 使用 `custom_audio_cnn_classification.yaml` 或類似配置

2. **使用預先提取的特徵**：
   - 首選 FCNN 架構
   - 使用 `custom_audio_fcnn_classification.yaml` 或類似配置

3. **頻譜圖分析**：
   - 首選 CNN 或 Swin Transformer 架構
   - 調整模型參數以適應頻譜圖輸入

4. **計算資源受限情況**：
   - 使用較輕量的模型配置
   - 減少卷積層或隱藏層數量
   - 考慮使用 `example_classification.yaml` 的簡化版本

## 性能考量

不同配置在性能方面也存在差異：

1. **數據加載速度**：
   - 索引模式通常比傳統模式更快，特別是對於大型數據集
   - 索引模式支持更高效的批處理和預加載

2. **內存使用**：
   - CNN 模型通常比 FCNN 模型使用更少的內存
   - 索引模式可能需要額外內存來維護索引結構

3. **訓練時間**：
   - FCNN 模型通常訓練速度較快
   - CNN 和 Swin Transformer 適合 GPU 加速

## 結論

選擇合適的配置文件是基於您的具體需求、數據特性和可用計算資源的權衡。我們建議：

1. 對於研究性質的工作，使用索引模式配置，獲得最大的靈活性
2. 對於生產部署，可能需要根據環境限制選擇更簡化的傳統模式
3. 模型架構選擇應基於您的數據類型和任務需求
4. 可以從 `tests/benchmark_results.json` 中獲取不同配置的性能比較，幫助做出決策

## 參考配置範例

完整的配置範例可在 `config/` 目錄下找到，建議先查看以下文件作為起點：

- `config/example_classification_indexed.yaml` - 索引模式分類的標準配置
- `config/example_eat10_regression_indexed.yaml` - 索引模式回歸的標準配置
- `config/example_classification.yaml` - 傳統模式分類的標準配置
- `config/example_eat10_regression.yaml` - 傳統模式回歸的標準配置 