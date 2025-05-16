# 索引數據集使用指南

本文檔詳細說明如何使用MicDysphagiaFramework中的索引數據集功能，包括配置、數據準備、標籤選擇和常見用例。

## 1. 索引數據集概述

索引數據集功能允許使用者通過一個中央CSV檔案（data_index.csv）來管理和加載數據，而不是直接從檔案系統讀取。這帶來以下好處：

- **靈活的標籤選擇**：可以輕鬆切換不同的標籤欄位（如EAT-10分數、醫生評估等）
- **數據篩選**：可以根據各種條件（如動作類型、處理狀態等）篩選數據
- **按患者分割數據**：可以確保同一患者的數據不會同時出現在訓練集和測試集中
- **退化機制**：當索引CSV不可用時，系統會自動退化到原始的直接加載模式

## 2. 數據索引CSV格式

索引CSV文件（通常命名為`data_index.csv`）應包含以下欄位：

| 欄位名稱 | 描述 | 必要性 |
|---------|------|-------|
| file_path | 音頻文件的絕對路徑 | 必要 |
| score | EAT-10問卷得分 (0-40分) | 必要 |
| patient_id | 患者識別碼 | 可選 (用於按患者拆分) |
| feature_path | 特徵檔案路徑 | 可選 (用於FeatureDataset) |
| codes_path | 編碼檔案路徑 | 可選 (用於FeatureDataset) |
| DrLee_Evaluation | Dr Lee的分類評估 | 可選 (用於分類任務) |
| DrTai_Evaluation | Dr Tai的分類評估 | 可選 (用於分類任務) |
| selection | 錄音時的動作類型 | 可選 (用於篩選或分類) |
| status | 數據處理狀態 | 可選 (用於篩選) |

### 範例CSV內容

```csv
file_path,score,patient_id,DrLee_Evaluation,DrTai_Evaluation,selection,status,feature_path,codes_path
/path/to/file1.wav,15,p001,聽起來正常,正常,乾吞嚥1口,processed,/path/to/file1_features.npy,/path/to/file1_codes.npy
/path/to/file2.wav,25,p002,輕度異常,無OR 輕微吞嚥障礙,吞水10ml,processed,/path/to/file2_features.npy,/path/to/file2_codes.npy
/path/to/file3.wav,5,p001,重度異常,重度吞嚥障礙,餅乾1塊,raw,/path/to/file3_features.npy,/path/to/file3_codes.npy
```

## 3. 配置索引數據集

在配置文件中，可以通過以下方式啟用和配置索引數據集功能：

### 3.1 基本配置

```yaml
data:
  type: 'feature'  # 可以是 'audio', 'spectrogram', 或 'feature'
  index_path: 'data/data_index.csv'  # 索引CSV文件路徑
  label_field: 'score'  # 使用哪個欄位作為標籤
  filter_criteria:  # 可選的篩選條件
    status: 'processed'
    selection: '乾吞嚥1口'
```

### 3.2 回歸任務配置範例

```yaml
data:
  type: 'feature'
  index_path: 'data/data_index.csv'
  label_field: 'score'  # 使用EAT-10分數作為回歸目標
  filter_criteria:
    status: 'processed'
  batch_size: 32
  num_workers: 4
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
```

### 3.3 分類任務配置範例

```yaml
data:
  type: 'feature'
  index_path: 'data/data_index.csv'
  label_field: 'DrLee_Evaluation'  # 使用醫生評估作為分類目標
  filter_criteria:
    status: 'processed'
  batch_size: 32
  num_workers: 4
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
```

## 4. 使用索引數據集的方法

### 4.1 通過配置文件使用

最簡單的方式是在配置文件中指定`index_path`和其他相關設置，然後使用標準的數據集創建流程：

```python
from utils.config_loader import load_config
from data.dataset_factory import create_dataset

# 加載配置
config = load_config('config/example_classification_indexed.yaml')

# 創建數據集
train_dataset, val_dataset, test_dataset = create_dataset(config)
```

### 4.2 直接在代碼中使用

也可以直接在代碼中創建和使用索引數據集：

```python
from data.feature_dataset import FeatureDataset

# 創建特徵數據集，使用索引CSV
dataset = FeatureDataset(
    index_path='data/data_index.csv',
    label_field='DrLee_Evaluation',  # 使用醫生評估作為分類目標
    filter_criteria={'status': 'processed', 'selection': '乾吞嚥1口'},
    transform=None  # 可選的數據轉換
)

# 檢查數據集信息
print(f"數據集大小: {len(dataset)}")
print(f"類別數量: {dataset.num_classes}")
print(f"標籤映射: {dataset.get_label_map()}")

# 獲取第一個樣本
data, label = dataset[0]
```

### 4.3 按患者ID拆分數據

使用索引數據集的一個重要優勢是能夠按患者ID拆分數據，確保同一患者的數據不會同時出現在訓練集和測試集中：

```python
from data.feature_dataset import FeatureDataset
from torch.utils.data import Subset

# 創建數據集
dataset = FeatureDataset(
    index_path='data/data_index.csv',
    label_field='score'
)

# 按患者ID拆分
train_indices, val_indices, test_indices = dataset.split_by_patient(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42  # 設置隨機種子以確保結果可重現
)

# 創建子集
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)
```

## 5. 支持的標籤類型

索引數據集支持以下類型的標籤：

### 5.1 回歸標籤

- **score**: EAT-10問卷得分 (0-40分)，用於回歸任務

### 5.2 分類標籤

- **DrLee_Evaluation**: Dr Lee的分類評估，包含以下類別：
  - "聽起來正常"
  - "輕度異常"
  - "重度異常"

- **DrTai_Evaluation**: Dr Tai的分類評估，包含以下類別：
  - "正常"
  - "無OR 輕微吞嚥障礙"
  - "重度吞嚥障礙"
  - "吞嚥障礙"

- **selection**: 錄音時的動作類型，包含以下類別：
  - "乾吞嚥1口"
  - "乾吞嚥2口"
  - "乾吞嚥3口"
  - "吞果凍"
  - "吞水10ml"
  - "吞水20ml"
  - "無動作"
  - "餅乾1塊"
  - "餅乾2塊"

## 6. 退化機制

當索引CSV不可用或篩選條件沒有匹配到任何記錄時，系統會自動退化到原始的直接加載模式：

```python
# 使用不存在的索引路徑，但啟用退化機制
dataset = FeatureDataset(
    index_path='non_existent_path.csv',
    fallback_to_direct=True,  # 啟用退化機制
    data_dir='data/features',  # 直接模式下的數據目錄
    label_file='data/labels.csv'  # 直接模式下的標籤文件
)

# 系統會自動退化到直接從data_dir加載數據
```

可以通過設置`fallback_to_direct=False`來禁用退化機制，這樣當索引CSV不可用時會直接拋出異常。

## 7. 常見問題與解決方案

### 7.1 索引CSV找不到

**問題**: `FileNotFoundError: 數據索引文件不存在: /path/to/data_index.csv`

**解決方案**:
- 確認索引CSV文件路徑是否正確
- 確認文件是否存在於指定位置
- 如果需要，啟用退化機制: `fallback_to_direct=True`

### 7.2 標籤欄位不存在

**問題**: `ValueError: 數據索引中不存在標籤欄位: some_field`

**解決方案**:
- 確認索引CSV文件中是否包含指定的標籤欄位
- 檢查欄位名稱是否正確（注意大小寫）
- 使用可用的標籤欄位，如: `score`, `DrLee_Evaluation`, `DrTai_Evaluation`, `selection`

### 7.3 篩選後沒有記錄

**問題**: 篩選條件太嚴格，導致沒有匹配的記錄

**解決方案**:
- 放寬篩選條件
- 檢查索引CSV中的值是否與篩選條件匹配
- 啟用退化機制: `fallback_to_direct=True`

### 7.4 按患者ID拆分失敗

**問題**: 無法按患者ID拆分數據

**解決方案**:
- 確認索引CSV中包含`patient_id`欄位
- 確認`patient_id`欄位有有效值
- 如果無法按患者ID拆分，系統會自動退化到隨機拆分

## 8. 最佳實踐

1. **保持索引CSV的一致性**：確保索引CSV中的路徑和標籤信息是最新的
2. **使用相對路徑**：在索引CSV中使用相對路徑，以便在不同環境中使用
3. **提供充分的篩選條件**：使用篩選條件來選擇特定類型的數據
4. **按患者ID拆分**：在醫療應用中，按患者ID拆分數據通常是最佳實踐
5. **設置隨機種子**：設置隨機種子以確保結果可重現
6. **啟用退化機制**：在生產環境中啟用退化機制，以增強系統的魯棒性

## 9. 進階用法

### 9.1 自定義標籤映射

如果需要自定義標籤映射，可以通過以下方式實現：

```python
from utils.data_index_loader import DataIndexLoader

# 加載索引
loader = DataIndexLoader('data/data_index.csv')

# 獲取標籤映射
mapping = loader.get_mapping_dict('DrLee_Evaluation')
print(mapping)  # 例如: {'聽起來正常': 0, '輕度異常': 1, '重度異常': 2}

# 自定義映射
custom_mapping = {'聽起來正常': 'normal', '輕度異常': 'mild', '重度異常': 'severe'}
```

### 9.2 複雜篩選條件

可以使用複雜的篩選條件來選擇特定的數據：

```python
# 複合篩選條件
filter_criteria = {
    'status': 'processed',
    'selection': '乾吞嚥1口',
    'patient_id': 'p001'
}

# 使用篩選條件創建數據集
dataset = FeatureDataset(
    index_path='data/data_index.csv',
    label_field='score',
    filter_criteria=filter_criteria
)
```

## 10. 總結

索引數據集功能提供了一種靈活、強大的方式來管理和加載數據，特別適合處理複雜的醫療數據。通過使用中央CSV索引文件，可以輕鬆切換不同的標籤、篩選特定的數據，並確保數據拆分的科學性。同時，退化機制確保了系統在索引不可用時仍能正常運行。 