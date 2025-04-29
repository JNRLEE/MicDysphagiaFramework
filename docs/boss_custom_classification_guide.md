# 老闆自定義分類功能使用指南

本文檔介紹如何使用自定義分類功能，根據老闆提供的Excel檔案自動分類患者數據。

## 功能概述

原始系統使用EAT-10分數和選擇類型對患者進行分類。自定義分類功能允許您從Excel檔案讀取分類信息，覆蓋原始分類邏輯，提供更靈活的分類方案。

主要特點：
- 從Excel檔案讀取患者ID和分類
- 只處理在Excel中存在的患者資料
- 自動調整模型輸出層匹配Excel中的分類數量
- 同時支持回歸和分類任務

## 配置說明

要使用自定義分類功能，請在配置檔案中添加以下配置：

```yaml
data:
  filtering:
    # 自定義分類設定
    custom_classification:
      enabled: true # 啟用自定義分類
      excel_path: '/path/to/your/classification.xlsx' # Excel檔案路徑
      patient_id_column: 'A' # 患者ID所在欄位（可以是欄位名稱或Excel欄位代號如A, B, C...）
      class_column: 'P' # 分類所在欄位（可以是欄位名稱或Excel欄位代號如A, B, C...）
```

## 使用方法

1. **準備Excel檔案**：
   - 確保Excel檔案中至少包含患者ID和分類兩個欄位
   - 患者ID應與數據集中的ID格式保持一致
   - 分類欄位應包含文字分類標籤

2. **配置YAML檔案**：
   - 複製 `config/boss_custom_classification.yaml` 或修改現有配置
   - 設置 `custom_classification.enabled` 為 `true`
   - 設置 `excel_path` 為您的Excel檔案路徑
   - 設置 `patient_id_column` 為患者ID所在欄位
   - 設置 `class_column` 為分類所在欄位

3. **執行訓練**：
   ```bash
   python scripts/run_experiments.py --config config/your_config.yaml
   ```

## 工作原理

1. 系統會讀取Excel檔案，創建患者ID到分類的映射
2. 創建數據集時，系統會過濾掉不在Excel中的患者
3. 模型創建時，輸出層數量會自動調整為Excel中的分類數
4. 訓練和評估與常規分類相同

## 注意事項

1. **相容性**：
   - 啟用自定義分類後，原始的分類設定（如score_thresholds、class_config等）將被忽略
   - 自定義分類僅考慮患者ID，不考慮選擇類型等其他因素

2. **錯誤處理**：
   - 如果Excel檔案路徑不正確，系統會顯示錯誤並禁用自定義分類
   - 如果患者不在Excel中，系統會跳過該患者的數據

3. **日誌輸出**：
   - 系統會記錄自定義分類統計信息，如類別數、類別名稱等
   - 數據過濾統計也會顯示因自定義分類被過濾的樣本數

## 測試自定義分類

您可以使用提供的測試腳本驗證自定義分類功能是否正常工作：

```bash
python scripts/test_boss_classification.py
```

測試結果將保存在 `tests/output/boss_classification_test/test_results.json` 中。

## 故障排除

1. **找不到Excel檔案**：
   - 確保Excel檔案路徑正確，最好使用絕對路徑
   - 檢查檔案格式是否為.xlsx而非.xls

2. **無法讀取分類數據**：
   - 檢查欄位名稱或欄位代號是否正確
   - 確保Excel檔案中包含有效的患者ID和分類

3. **數據集為空**：
   - 檢查數據集中的患者ID是否與Excel中的格式一致
   - 確保至少有一個患者同時存在於數據集和Excel中

## 示例配置

```yaml
data:
  type: 'audio'
  source:
    wav_dir: '/path/to/audio/files'
  filtering:
    custom_classification:
      enabled: true
      excel_path: '/path/to/分類名單.xlsx'
      patient_id_column: 'A'  # 或使用實際的欄位名稱
      class_column: 'P'  # 或使用實際的欄位名稱
    task_type: 'classification'

model:
  type: 'swin_transformer'
  parameters:
    # 注意：當自定義分類啟用時，num_classes會被自動設置
    num_classes: 10  # 此值會被覆蓋
    is_classification: true
```

## 開發人員信息

若有任何問題或需要更多協助，請聯繫系統開發人員。 