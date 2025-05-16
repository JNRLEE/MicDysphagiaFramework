# MicDysphagiaFramework 資料讀取增強計劃

## 1. 系統分析

根據需求和現有架構，我們需要對以下模組進行修改：

### 1.1 需要更動的核心模組：

1. **data/dataset_factory.py** - ✅ 完成 - 加入對索引CSV的支援，可根據配置選擇使用索引CSV或原始方式創建數據集
2. **data/audio_dataset.py** - ✅ 完成 - 已支援從索引CSV讀取，同時保持向後兼容
3. **data/spectrogram_dataset.py** - ✅ 完成 - 已支援從索引CSV讀取，同時保持向後兼容
4. **data/feature_dataset.py** - ✅ 完成 - 已支援從索引CSV讀取，同時保持向後兼容
5. **utils/config_loader.py** - ✅ 完成 - 已更新配置架構以支援新的數據源選項

### 1.2 需要新增的模組：

1. **data/indexed_dataset.py** - ✅ 完成 - 已建立基於索引CSV的數據集基類，提供統一的數據加載介面
2. **utils/data_index_loader.py** - ✅ 完成 - 已實現讀取與解析data_index.csv的功能，包含數據篩選和標籤映射

### 1.3 數據格式說明

#### data_index.csv

核心數據索引文件，位於 `data/data_index.csv`，包含以下主要欄位：

- `file_path`: 音頻文件的絕對路徑 (主索引鍵)
- `score`: EAT-10 問卷得分(0-40分, 整數)
- `patient_id`: 患者識別碼
- `recording_date`: 錄音日期
- `features_shape`, `codes_shape`: 資料維度
- `status`: 數據處理狀態（'raw', 'processed', 'failed'）
- `selection`: 錄音時的動作類型{"乾吞嚥1口", "乾吞嚥2口", "乾吞嚥3口", "吞果凍", "吞水10ml", "吞水20ml", "無動作", "餅乾1塊", "餅乾2塊"}
- `DrTai_Evaluation`: Dr. Tai對每一筆資料的分類{"正常", "無OR 輕微吞嚥障礙", "重度吞嚥障礙", "吞嚥障礙"}
- `DrLee_Evaluation`: Dr Lee對每一筆資料的分類{"聽起來正常", "輕度異常", "重度異常"}

#### 單一數據源的多重表示

每個音頻數據在系統中以三種形式表示：

1. 原始WAV文件 - 用於直接處理和聽覺分析
2. 特徵向量(features.npy) - 中間特徵表示
3. 編碼向量(codes.npy) - 用於機器學習的抽象表示

這三種表示通過file_path關聯，確保數據一致性和可追蹤性。

#### Processed(Cut)資料架構

```
Processed(Cut)/
├── {資料檔名（可能因為不同時期的資料而有不同的命名邏輯）} # 對應`data/processed/data_index.csv`的file_path
│   ├──  {資料檔名}_codes.npy  # 對應`data/processed/data_index.csv`的codes_path
│   ├──  {資料檔名}_features.npy # 對應`data/processed/data_index.csv`的feature_path
│   ├── {patient_id}_info.json # 對應`data/processed/data_index.csv`的wav_path
│   ├── Probe0_RX_IN_TDM4CH0.wav
```

## 2. 詳細開發計劃

### 階段一：設計與準備 - ✅ 已完成

1. **建立索引加載器** - ✅ 完成

   - ✅ 在`utils`目錄下建立`data_index_loader.py`
   - ✅ 實現讀取與解析`data_index.csv`的功能
   - ✅ 加入數據驗證與清理功能
   - ✅ 提供方法根據需求篩選數據
2. **建立標籤選擇機制** - ✅ 完成

   - ✅ 允許根據配置選擇不同的標籤欄位
   - ✅ 支援標籤數據轉換（例如從文字標籤到數字類別）
   - ✅ 設計一個彈性的標籤映射系統

### 階段二：建立基於索引的數據集 - ✅ 已完成

1. **建立基礎索引數據集類** - ✅ 完成

   - ✅ 在`data`目錄下建立`indexed_dataset.py`
   - ✅ 實現基於索引CSV的數據加載邏輯
   - ✅ 提供統一的數據檢索和驗證機制
   - ✅ 設計彈性處理各種數據格式的方法
2. **整合索引數據集與現有數據集** - ✅ 完成

   - ✅ 為現有的AudioDataset、SpectrogramDataset與FeatureDataset增加索引支援
   - ✅ 設計一個退化機制，當索引中沒有找到項目時回退到原始加載方式

### 階段三：更新工廠與配置系統 - ✅ 已完成

1. **更新DatasetFactory** - ✅ 完成

   - ✅ 在`dataset_factory.py`中加入對索引數據集的支援
   - ✅ 實現依據配置動態選擇數據源的機制
2. **更新配置系統** - ✅ 完成

   - ✅ 在`config_loader.py`中添加新的配置選項
   - ✅ 為數據加載添加標籤選擇機制的配置項
   - ✅ 為數據源選擇添加優先級配置

### 階段四：測試與整合 - ✅ 已完成

1. **建立單元測試** - ✅ 完成

   - ✅ 模組內簡單測試代碼
   - ✅ feature\_dataset.py 測試問題已解決
   - ✅ 建立獨立的測試文件
     - ✅ 創建 `tests/test_data_index_loader.py` 文件
     - ✅ 創建 `tests/test_indexed_dataset.py` 文件
   - ✅ 添加測試覆蓋率檢查
     - ✅ 創建 `tests/run_indexed_dataset_tests.py` 用於運行測試並生成覆蓋率報告
2. **建立整合測試** - ✅ 完成

   - ✅ 基本數據加載流程測試
   - ✅ 完整訓練流程測試
     - ✅ 創建 `tests/test_indexed_dataset_training.py` 用於測試索引數據集與訓練流程的整合
     - ✅ 測試回歸任務和分類任務

### 階段五：文檔與錯誤處理 - ⚠️ 進行中

1. **文檔更新** - ⚠️ 未完成

   - ⚠️ 創建 `docs/indexed_dataset_guide.md`，詳細說明如何使用索引CSV功能
   - ⚠️ 更新主 `README.md`，添加索引CSV功能的簡要說明
   - ⚠️ 在配置示例中添加註釋，說明每個配置項的作用
2. **邊緣情況處理** - ✅ 完成

   - ✅ 基本錯誤處理邏輯
   - ✅ 全面測試與處理邊緣情況
     - ✅ 索引CSV格式錯誤時的錯誤信息
     - ✅ 文件路徑無效時的退化機制測試
     - ✅ 標籤欄位不存在時的錯誤處理

## 3. 具體實現步驟

所有實現步驟皆已完成，並已整合至系統中。核心修改包括：

1. ✅ 建立了`utils/data_index_loader.py`：實現讀取CSV索引檔案、篩選資料和標籤映射功能。
2. ✅ 建立了`data/indexed_dataset.py`：創建基於索引CSV的資料集基類，提供統一的資料加載介面和退化機制。
3. ✅ 更新了`data/audio_dataset.py`：使其繼承自IndexedDatasetBase，同時支援索引模式和原始直接模式。
4. ✅ 更新了`data/spectrogram_dataset.py`：使其繼承自IndexedDatasetBase，支援從索引CSV讀取頻譜圖資料。
5. ✅ 更新了`data/feature_dataset.py`：使其繼承自IndexedDatasetBase，支援從索引CSV讀取特徵資料。
6. ✅ 更新了`data/dataset_factory.py`：加入對索引模式的支援，可根據配置選擇使用索引CSV或原始方式創建資料集。
7. ✅ 更新了`utils/config_loader.py`：添加對索引CSV相關配置的驗證。
8. ✅ 新增了兩個示例配置檔案：
   - `config/example_classification_indexed.yaml`：用於使用索引CSV進行分類任務
   - `config/example_eat10_regression_indexed.yaml`：用於使用索引CSV進行回歸任務
9. ✅ 擴展了`config/config_schema.yaml`：添加索引CSV相關的配置選項。

## 4. 測試計劃

1. **測試數據索引加載器** - ✅ 完成

   - ✅ 基本功能測試（在模組內部）
   - ✅ 專門的測試文件 `tests/test_data_index_loader.py`
   - ✅ 測試覆蓋率檢查
2. **測試索引數據集** - ✅ 完成

   - ✅ 基本功能測試（在模組內部）
   - ✅ 專門的測試文件 `tests/test_indexed_dataset.py`
   - ✅ 測試覆蓋率檢查
3. **測試整合功能** - ✅ 完成

   - ✅ 基本加載測試
   - ✅ 完整訓練流程測試 `tests/test_indexed_dataset_training.py`

## 5. 部署計劃

1. **測試環境部署** - ✅ 完成
2. **生產環境部署** - ⚠️ 進行中
   - ✅ 代碼部署
   - ⚠️ 文檔更新 (未完成)
   - ⚠️ 使用說明 (未完成)

## 6. 未來擴展計劃

1. **數據處理增強**

   - 添加更多數據預處理選項
   - 支援更複雜的篩選條件
2. **多標籤學習支援**

   - 支援同時使用多個標籤欄位進行訓練
   - 實現多任務學習功能
3. **自動數據更新**

   - 添加自動更新索引CSV的功能
   - 實現增量學習支援

## 7. 總結

我們已經成功實現了從索引CSV加載數據的功能，這使得系統能夠更靈活地處理數據，特別是：

- 可以簡單切換不同標籤欄位（EAT-10分數、醫生評估等）
- 可以輕鬆進行數據篩選（如只使用特定動作類型的數據）
- 在索引CSV不可用時，系統能優雅地回退到原始的數據加載模式

所有核心功能代碼和測試都已完成，只剩下文檔工作需要補完：

1. **待完成的文檔任務**:
   - 創建 `docs/indexed_dataset_guide.md` 使用指南文檔
   - 更新主要 README.md 添加功能簡介
   - 為配置示例添加詳細註釋
