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

核心數據索引文件，位於 `data/metadata/data_index.csv`，包含以下主要欄位：

- `file_path`: 音頻資料夾的絕對路徑 (主索引鍵)
- `wav_path`, `features_path`, `codes_path`:音頻資料夾內的wav, feature.npy與codes.npy的絕對路徑
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

#### Processed資料架構

```
Processed(Cut) /
│  └── {資料檔名}          # 對應`data/metadata/data_index.csv`的file_path
│     ├── {資料檔名}_codes.npy      # 對應`data/metadata/data_index.csv`的codes_path
│     └── {資料檔名}_features.npy   # 對應`data/metadata/data_index.csv`的feature_path
│     └── {資料檔名}_features.npy   # 對應`data/metadata/data_index.csv`的feature_path
│     └── Probe0_RX_IN_TDM4CH0.wav # 對應`data/metadata/data_index.csv`的wav_path
│     └── patient_id_info.json     # 功能已被data/metadata/data_index.csv取代
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

## 5.1 部署計劃

1. **測試環境部署** - ✅ 完成

   - ✅ 單元測試全部通過
   - ✅ 資料讀取和索引功能正常
   - ✅ 與訓練流程整合無誤
2. **資料重組** - ✅ 完成

   - ✅ 建立標準化的資料目錄結構
     - data/metadata/ - 存放索引文件和元數據
     - data/raw/ - 存放原始音頻文件
     - data/processed/ - 存放處理後的特徵和頻譜圖
   - ✅ 移動 `data_index.csv` 至 `data/metadata/` 目錄
   - ✅ 更新索引CSV檔案中的檔案路徑
   - ✅ 創建並完成 `scripts/reorganize_data.py` 工具腳本，用於自動化數據遷移過程
3. **生產環境部署** - ✅ 完成

   - ✅ 代碼部署
   - ✅ 目錄結構重組
   - ✅ 配置文件更新
   - ✅ 文檔更新
   - ✅ 使用說明
   - ✅ 新結構的 .gitignore 更新
4. **試跑模型測試** - ⚠️ 進行中（epoch 以2-10為主，不要測太花時間的情形）

   - ✅ 創建測試腳本
   - ⚠️ 執行以下測試命令，並收集結果:
     ```bash
     # 使用分類配置執行測試
     python scripts/run_experiments.py --config config/example_classification_indexed.yaml

     # 使用回歸配置執行測試
     python scripts/run_experiments.py --config config/example_eat10_regression_indexed.yaml

     # 使用自定義CNN分類配置執行測試
     python scripts/run_experiments.py --config config/custom_audio_cnn_classification.yaml

     # 使用自定義FCNN分類配置執行測試
     python scripts/run_experiments.py --config config/custom_audio_fcnn_classification.yaml
     ```
   - ⚠️ 測試不同模型架構的執行效率與準確性
   - ⚠️ 測試不同資料篩選條件對模型效能的影響
   - ⚠️ 比較使用與不使用索引CSV的性能差異
5. **部署驗證** - ⚠️ 規劃中

   - ⚠️ 確認模型訓練結果與之前版本一致
   - ⚠️ 確認所有訓練參數和指標正確記錄
   - ⚠️ 確認存檔的中間模型參數完整
   - ⚠️ 驗證不同標籤欄位的分類效果
   - ⚠️ 建立自動驗證測試腳本 - `scripts/validate_deployment.py`
   - ⚠️ 測試TensorBoard日誌記錄是否正確
6. **配置文件差異分析** - ⚠️ 待完成

   - ⚠️ 比較 `config/example_classification_indexed.yaml` 與 `config/custom_audio_cnn_classification.yaml` 的功能差異
   - ⚠️ 比較 `config/example_eat10_regression_indexed.yaml` 與 `config/custom_audio_fcnn_classification.yaml` 的功能差異
   - ⚠️ 記錄各配置檔案的最佳使用場景和優缺點
   - ⚠️ 建立配置文件選擇指南 - `docs/config_guide.md`
   - ⚠️ 創建配置參考表 - `docs/config_comparison.md`

## 5.2 成功整合指標

成功整合的關鍵指標與驗證步驟如下：（epoch 以2-10為主，不要測太花時間的情形）

1. **功能完整性** - ⚠️ 進行中

   - ⚠️ 確保 `config/example_classification_indexed.yaml` 能夠成功運行完整的訓練流程
   - ⚠️ 確保 `config/example_eat10_regression_indexed.yaml` 能夠成功運行完整的訓練流程
   - ⚠️ 確認模型能夠正確處理索引化的數據集
   - ⚠️ 驗證標籤欄位選擇功能正常工作
   - ⚠️ 測試篩選條件功能（例如根據 selection 或 status 欄位篩選）
   - ⚠️ 驗證不同數據類型（音頻、頻譜圖、特徵向量）的加載正確性
2. **性能指標** - ⚠️ 待評估

   - ⚠️ 使用索引數據集訓練的模型準確率不應低於使用傳統方式訓練的模型
   - ⚠️ 索引方式的數據加載時間應接近或優於傳統方式
   - ⚠️ 測量並記錄訓練時間、推理時間和內存使用情況
   - ⚠️ 比較不同配置文件下的模型性能差異
   - ⚠️ 建立基準性能測試報告 - `tests/benchmark_results.md`
   - ⚠️ 使用自動化腳本收集關鍵指標 - `scripts/collect_performance_metrics.py`
3. **資料保存完整性** - ⚠️ 待評估

   - ⚠️ 確認訓練過程中的指標正確記錄到 `experiments.log`
   - ⚠️ 確認模型結構和參數正確保存
   - ⚠️ 確認輸出目錄結構符合預期
   - ⚠️ 使用標準化命名格式命名輸出圖像
   - ⚠️ 驗證TensorBoard日誌的完整性和正確性
   - ⚠️ 驗證配置文件的正確保存
4. **穩定性測試** - ⚠️ 規劃中

   - ⚠️ 在不同大小的數據集上測試索引功能
   - ⚠️ 測試退化機制（當索引CSV不可用時）
   - ⚠️ 測試從錯誤和異常中恢復的能力
   - ⚠️ 驗證多次運行的結果一致性
   - ⚠️ 進行邊界條件測試（如極少數據、不平衡數據等）
   - ⚠️ 創建穩定性測試報告模板 - `tests/stability_test_template.md`
5. **最終驗收** - ⚠️ 待完成

   - ⚠️ 整合測試報告，包括所有功能點的測試結果
   - ⚠️ 性能對比報告，比較各種配置的性能差異
   - ⚠️ 完善使用文檔，確保使用者能夠理解和應用新功能
   - ⚠️ 最終代碼審查和質量檢查
   - ⚠️ 創建最終驗收報告 - `docs/indexed_dataset_acceptance.md`
   - ⚠️ 建立部署後的維護計劃

## 6. 未來擴展計劃

### 6.1 數據處理增強 - 🔍 規劃中

1. **進階數據預處理選項**

   - 設計更多音頻增強技術，如頻譜增強、時間拉伸、音調偏移等
   - 實現自動音頻質量評估，過濾低質量樣本
   - 支援自定義預處理流程配置
2. **複雜篩選條件支援**

   - 擴展DataIndexLoader，支援複合邏輯條件（AND, OR, NOT）
   - 實現數值範圍篩選（如篩選特定EAT-10分數範圍）
   - 添加基於時間的篩選（如特定日期範圍的數據）
3. **數據平衡與採樣**

   - 實現類別平衡採樣機制，處理不平衡數據集
   - 支援基於患者ID的分層採樣
   - 添加加權採樣功能，根據標籤重要性調整採樣概率

### 6.2 多標籤學習支援 - 🧩 概念階段

1. **多任務學習框架**

   - 設計支援同時使用多個標籤欄位的數據集類
   - 實現多任務損失函數（如同時預測EAT-10分數和醫生評估）
   - 創建多輸出模型架構
2. **標籤關係建模**

   - 分析並利用不同標籤之間的關係（如EAT-10分數與醫生評估的關聯）
   - 實現標籤依賴性建模，提高預測準確性
   - 支援半監督學習，利用未標記數據
3. **交叉驗證方法**

   - 實現適用於多標籤學習的交叉驗證策略
   - 支援多目標評估指標
   - 添加多標籤學習的可視化工具

### 6.3 自動數據更新與增量學習 - 🔄 概念階段

1. **數據索引自動更新**

   - 創建監視目錄變化的工具，自動更新索引CSV
   - 實現數據一致性檢查，確保索引與文件系統同步
   - 建立數據版本控制機制
2. **增量學習支援**

   - 設計增量更新模型的訓練流程
   - 實現模型參數的連續微調
   - 支援新類別的動態添加
3. **在線學習能力**

   - 建立流式數據處理機制
   - 實現漸進式特徵提取和模型更新
   - 支援概念漂移檢測和模型調整

### 6.4 性能優化 - ⚡ 規劃中

1. **數據加載優化**

   - 實現數據預加載和緩存機制
   - 優化內存使用，支援大規模數據集
   - 添加多進程數據加載選項
2. **分佈式訓練支援**

   - 實現分佈式數據並行處理
   - 支援模型並行訓練
   - 添加檢查點保存和恢復機制
3. **硬件加速**

   - 優化GPU利用率
   - 支援混合精度訓練
   - 實現模型量化，提高推理速度

## 7. 總結

我們已經成功實現了基於索引CSV的數據加載功能，大幅提升了MicDysphagiaFramework的資料處理靈活性。此次增強的核心價值包括：

### 7.1 主要成果

- **靈活的標籤選擇機制**：通過簡單的配置變更，使用者可以輕鬆切換不同的標籤欄位（EAT-10分數、醫生評估等），無需修改代碼
- **強大的數據篩選能力**：支援多種條件篩選數據，如特定動作類型、處理狀態等，幫助研究人員聚焦於特定研究問題
- **清晰的資料組織結構**：重新組織了資料目錄結構，提高了代碼的可維護性和數據的可追蹤性
- **優雅的退化機制**：在索引CSV不可用時，系統能自動回退到傳統的直接加載模式，確保系統的魯棒性

### 7.2 關鍵進展

所有核心功能代碼已實現並通過測試，包括：

1. **核心組件**：

   - 完成數據索引加載器（DataIndexLoader）
   - 完成索引數據集基類（IndexedDatasetBase）
   - 完成各專用數據集的索引支援整合
2. **完善的測試**：

   - 單元測試確保各模組功能正確
   - 整合測試驗證整個系統協同工作
   - 覆蓋率測試確保代碼質量
3. **資料目錄重組**：

   - 標準化資料存儲結構
   - 建立元數據與實際數據的分離
   - 提供資料遷移工具

### 7.3 後續行動項目

我們仍需完成以下文檔工作，確保使用者能夠充分利用新功能：

1. **使用指南**：

   - 完善 `docs/indexed_dataset_guide.md` 使用指南文檔
   - 更新主 README.md，添加索引CSV功能簡介
   - 為配置示例添加詳細註釋
2. **實際部署測試**：

   - 完成試跑模型測試計劃
   - 收集性能對比數據
   - 驗證不同配置下的穩定性

### 7.4 願景展望

此次功能增強為未來發展奠定了基礎。我們期望MicDysphagiaFramework能夠:

- 支援更複雜的學習任務，如多標籤學習和半監督學習
- 處理更大規模的數據集，通過優化的數據加載和處理機制
- 提供更豐富的數據分析工具，幫助研究人員深入理解吞嚥障礙數據
- 成為臨床研究與AI技術結合的典範，推動吞嚥障礙診斷與治療的進步

通過不斷迭代和優化，我們的框架將持續為吞嚥障礙研究提供強大而靈活的技術支持。

#### 資料夾結構

```
data/
├── metadata/                     # 元數據資訊
│   ├── data_index.csv            # 核心數據索引文件
│   └── 吞嚥聲音名單(共同編輯).xlsx  # 原始資料名單
│
├── raw/                          # 原始資料
│   └── <患者ID>/                 # 以患者ID命名的子資料夾
│       └── <音頻文件>.wav         # 原始WAV音頻文件
│
├── /Processed(Cut)               # 處理後的資料
│
├── audio_dataset.py              # 音頻資料集實現
├── feature_dataset.py            # 特徵資料集實現
├── indexed_dataset.py            # 索引資料集基類
├── spectrogram_dataset.py        # 頻譜圖資料集實現
└── dataset_factory.py            # 資料集工廠
```

#### Processed(Cut) 資料架構

```
Processed(Cut) /
│  └── {資料檔名}          # 對應`data/metadata/data_index.csv`的file_path
│     ├── {資料檔名}_codes.npy      # 對應`data/metadata/data_index.csv`的codes_path
│     └── {資料檔名}_features.npy   # 對應`data/metadata/data_index.csv`的feature_path
│     └── {資料檔名}_features.npy   # 對應`data/metadata/data_index.csv`的feature_path
│     └── Probe0_RX_IN_TDM4CH0.wav # 對應`data/metadata/data_index.csv`的wav_path
│     └── patient_id_info.json     # 功能已被data/metadata/data_index.csv取代
```
