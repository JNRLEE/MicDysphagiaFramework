# MicDysphagiaFramework 開發指南

## 目前急需開發的功能

為了支援 SBP_analyzer 中的進階模型分析功能（混淆矩陣和餘弦相似度圖），需要在 MicDysphagiaFramework 中增加以下數據收集和保存功能：

### 1. 標準化測試集結果保存 (用於混淆矩陣)

**目標**: 在測試/評估階段結束後，將所有測試樣本的真實標籤和模型預測標籤保存到標準位置。

**實作任務**:

1. **修改 Trainer 類的 evaluate/test 方法**:
   - 在 `trainers/base_trainer.py` 或特定 trainer 實現中，確保 `evaluate` 或 `test` 方法能收集完整的預測結果。
   - 收集過程中累積所有批次的 `targets` 和 `predictions`。

2. **創建標準化的 Evaluation Hook**:
   - 在 `models/hook_bridge.py` 中增加或修改 Hook，使其在評估完成時被觸發。
   - Hook 應收集完整的 `targets` (真實標籤) 和 `predictions` (模型預測)。
   - 使用標準化的鍵名（`'targets'` 和 `'predictions'`）保存數據。

3. **保存格式與位置**:
   ```python
   # 保存格式示例 (使用 torch.save)
   results = {
       'targets': all_targets,       # 形狀: [num_samples]
       'predictions': all_preds,     # 形狀: [num_samples]
       'probabilities': all_probs,   # 形狀: [num_samples, num_classes] (可選)
       'metrics': metrics_dict,      # 如準確率、F1分數等
       'timestamp': timestamp        # 保存時間
   }
   torch.save(results, os.path.join(hooks_dir, 'evaluation_results_test.pt'))
   ```

4. **配置系統整合**:
   - 確保此 Hook 可通過 `config.yaml` 中的配置來啟用/停用。
   - 在 `utils/save_manager.py` 中確保創建了正確的 hooks 目錄結構。

### 2. 保存測試集目標層激活值 (用於餘弦相似度圖)

**目標**: 在測試集評估過程中，捕獲並保存特定目標層的激活值。

**實作任務**:

1. **實現 Activation Hook**:
   - 在 `models/hook_bridge.py` 中創建一個新的 Hook 類，用於捕獲指定層的激活值。
   - 使用 PyTorch 的 `register_forward_hook` 將 Hook 附加到模型的目標層。
   - 設計 Hook 以收集整個測試集的激活值，形成一個大的張量。

2. **配置目標層**:
   - 在 `config.yaml` 中添加新的配置選項，允許用戶指定要捕獲激活值的層：
   ```yaml
   hooks:
     activation_capture:
       enabled: true
       target_layers: ['encoder.final_layer', 'head.pre_logits']  # 目標層名稱列表
       datasets: ['test']  # 在哪些數據集上捕獲
   ```

3. **保存格式與位置**:
   ```python
   # 單一層的全部測試集激活值 (理想格式)
   activations = {
       'layer_name': layer_name,              # 層名稱
       'activations': all_samples_activations, # 形狀: [num_samples, embedding_dim]
       'sample_ids': sample_ids,              # 樣本ID (可選)
       'targets': targets,                    # 對應的真實標籤 (用於後續分析)
       'timestamp': timestamp                 # 保存時間
   }
   torch.save(activations, os.path.join(hooks_dir, f'test_set_activations_{layer_name}.pt'))
   ```

4. **擴展現有 Hook 註冊機制**:
   - 修改 `models/hook_bridge.py` 中的 `get_analyzer_callbacks_from_config` 函數，使其能夠識別並創建激活值捕獲 Hook。
   - 確保 Hook 僅在指定的數據集（如測試集）上被觸發。

### 3. 文件和文檔更新

1. **更新數據結構文檔**:
   - 修改 `framework_data_structure.md`，添加新增文件的格式說明。
   - 明確記錄 `hooks/evaluation_results_test.pt` 的用途和標準格式。
   - 明確記錄 `hooks/test_set_activations_{layer_name}.pt` 的用途和標準格式。

2. **更新 README.md**:
   - 在特性列表中添加新的分析能力說明。
   - 更新使用示例，展示如何配置這些新的 Hook。

3. **添加使用示例**:
   - 創建示例配置文件，展示如何啟用和配置這些新的數據捕獲功能。
   - 在 `docs/` 目錄下添加進階分析指南，說明如何使用這些新數據。

### 4. 測試計劃

1. **單元測試**:
   - 為新增的 Hook 和功能編寫單元測試。
   - 測試不同模型架構下的激活值捕獲。
   - 測試不同分類任務的評估結果保存。

2. **整合測試**:
   - 執行完整的訓練-評估流程，確保新的 Hook 不影響原有功能。
   - 驗證生成的文件格式是否符合預期，並且可以被 SBP_analyzer 正確讀取。

### 開發優先順序

1. 首先實現標準化測試集結果保存 (混淆矩陣所需)
2. 然後實現目標層激活值捕獲 (餘弦相似度圖所需)
3. 最後更新文檔和增加測試

完成後，SBP_analyzer 將能夠使用這些數據生成更深入的模型分析。
