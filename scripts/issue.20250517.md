# 特徵向量保存問題分析與解決方案

測試時暫時先使用
'index_path: 'data/metadata/data_index_text.csv' '，並確保我們設定想要儲存的epoch跟結果相符合

## 問題描述

1. **配置分散問題**

   - 目前 `save_every` 和 `save_frequency` 是分開的兩個配置項
   - `save_every` 在 `training` 部分定義
   - `save_frequency` 在 `hooks.activation_capture` 部分定義
   - 這種分散的配置容易造成混淆和不一致
2. **保存頻率不一致問題**

   - 雖然設置了 `save_frequency: 1`，但特徵向量只保存了 epoch 2
   - 這表明 `save_frequency` 的設置沒有正確生效
   - 需要檢查 `ActivationCaptureHook` 的實現邏輯
3. **Hook 與特徵向量保存不同步**

   - `hooks` 目錄和 `feature_vectors` 目錄的保存頻率不一致
   - 需要統一使用 `training.save_every` 作為保存頻率

## 目錄結構比較

### 預期結構 (根據 framework_data_structure.md)

```
results/
└── audio_feature_vectors_YYYYMMDD_HHMMSS/
    ├── ...
    ├── hooks/
    │   ├── epoch_0/               # 第0輪數據
    │   │   ├── epoch_summary.pt    # 輪次摘要
    │   │   └── ...
    │   ├── epoch_1/                # 第1輪數據
    │   │   └── ...
    │   ├── epoch_2/                # 第2輪數據
    │   │   └── ...
    │   ├── epoch_0_validation_predictions.pt # 第0輪驗證集預測結果
    │   ├── epoch_1_validation_predictions.pt # 第1輪驗證集預測結果
    │   ├── epoch_2_validation_predictions.pt # 第2輪驗證集預測結果
    │   └── ...
    ├── feature_vectors/
    │   ├── feature_analysis.json   # 特徵分析摘要
    │   ├── epoch_0/               # 第0輪特徵向量
    │   │   ├── layer_backbone_7_features.pt    # 特定層的特徵向量
    │   │   ├── layer_backbone_7_cosine_similarity.pt # 特徵間餘弦相似度矩陣
    │   │   └── layer_backbone_7_tsne.pt        # t-SNE降維結果
    │   ├── epoch_1/               # 第1輪特徵向量
    │   │   └── ...  
    │   └── epoch_2/               # 第2輪特徵向量
    │       └── ...
    └── ...
```

### 實際結構 (目前)

```
results/audio_feature_vectors_20250516_132051/
├── feature_vectors/  
  ├── 
└── hooks/
    ├── evaluation_results_test.pt
    ├── epoch_2/
    ├── epoch_1/
    └── epoch_0/
```

### 主要差異

1. **feature_vectors 目錄**:

   - 預期: 應包含 epoch_0、epoch_1 三個資料夾
   - 實際: 只有epoch_1
   - 差異: 所有特徵向量都未被保存
2. **hooks 目錄**:

   - 預期: 包含 epoch_0、epoch_1兩個資料夾和相應的預測結果
   - 實際: 包含 epoch_0、epoch_1兩個資料夾，但沒有看到驗證預測結果文件
   - 差異: hooks 目錄結構基本正確，但可能缺少驗證預測結果文件

## 根本原因分析

經過代碼詳細檢查，發現了兩個關鍵的根本問題：

1. **配置參數分散與覆蓋**：
   - `training.save_every` 用於控制模型檢查點保存
   - `hooks.activation_capture.save_frequency` 用於控制特徵向量保存
   - 雖然我們在 `run_experiments.py` 中將 `training.save_every` 傳遞給 `hooks.activation_capture.save_frequency`，但這並不能完全解決問題

2. **ActivationCaptureHook 實現中的邏輯問題**：
   - 分析發現 `ActivationCaptureHook` 類在每次評估結束時會被重複調用，存在邏輯問題
   - 存在一個關鍵缺陷：沒有追蹤已經保存過的 epoch，導致後面的 epoch 會覆蓋前面的
   - 每次保存特徵向量時，沒有區分不同 epoch，導致只有最後一個 epoch 被保存

3. **hooks 和 feature_vectors 目錄的行為不一致**：
   - `hooks` 目錄的創建有單獨的邏輯，如 `save_manager.save_hook_data()`
   - `feature_vectors` 目錄的創建直接在 `ActivationCaptureHook.on_evaluation_end()` 中

## 詳細修復方案

我們實施了以下修復方案：

1. **在 models/hook_bridge.py 中**：

   A. 在 `ActivationCaptureHook` 類中添加 `captured_epochs` 集合以跟踪已處理的 epochs：
   ```python
   def __init__(self, ...):
       # 原有初始化代碼...
       self.captured_epochs = set()  # 新增：記錄已經捕獲的epochs
   ```

   B. 抽取公共邏輯到 `_should_process_epoch` 方法：
   ```python
   def _should_process_epoch(self, epoch: int) -> bool:
       # 如果已經處理過這個epoch，直接返回False避免重複處理
       if epoch in self.captured_epochs:
           logger.info(f"跳過epoch {epoch}的激活值處理，已經處理過")
           return False
           
       # 檢查是否符合target_epochs或save_frequency條件...
       return should_process
   ```

   C. 修改 `on_evaluation_begin` 和 `on_evaluation_end` 使用新的方法：
   ```python
   def on_evaluation_begin(self, model: nn.Module, logs: Dict[str, Any] = None) -> None:
       # 獲取當前epoch...
       # 修改：使用抽取出的方法判斷是否應該捕獲
       should_capture = self._should_process_epoch(self.current_epoch)
       # 其餘邏輯...
   
   def on_evaluation_end(self, model: nn.Module, results: Dict[str, Any] = None, logs: Dict[str, Any] = None) -> None:
       # 修改：使用抽取出的方法判斷是否應該保存
       should_save = self._should_process_epoch(self.current_epoch)
       # 其餘邏輯...
       
       # 保存完成後標記此epoch已處理
       self.captured_epochs.add(self.current_epoch)
   ```

   D. 改進回調參數處理邏輯：
   ```python
   def get_analyzer_callbacks_from_config(config: Dict[str, Any]) -> List[Any]:
       # 從training部分獲取save_every作為默認保存頻率
       default_save_frequency = config.get('training', {}).get('save_every', 1)
       
       # 修改：優先使用training.save_every作為保存頻率
       save_frequency = activation_config.get('save_frequency', None)
       if save_frequency is None:
           save_frequency = default_save_frequency
       elif default_save_frequency != save_frequency:
           logger.warning(f"發現衝突的保存頻率設定...")
           save_frequency = default_save_frequency
   ```

2. **在 run_experiments.py 中**：

   A. 獲取並覆蓋 `save_frequency` 設置：
   ```python
   # 獲取並添加分析器回調
   # 修改：統一使用 training.save_every 作為保存頻率
   save_every = config.get('training', {}).get('save_every', 0)
   
   # 如果hooks.activation_capture配置中有save_frequency，使用training.save_every覆蓋它
   if 'hooks' in config and 'activation_capture' in config['hooks'] and 'save_frequency' in config['hooks']['activation_capture']:
       logger.info(f"覆蓋hooks.activation_capture.save_frequency={config['hooks']['activation_capture']['save_frequency']}為training.save_every={save_every}")
       config['hooks']['activation_capture']['save_frequency'] = save_every
   ```

## 修復後預期行為

使用以上修復方案後，系統應該能正確保存每個符合條件的 epoch 的特徵向量：

1. 當設置 `training.save_every = 1` 時，所有 epoch (0, 1, 2, ...) 的特徵向量都會被保存
2. 如果設置 `training.save_every = 2`，則 epoch 0, 2, 4, ... 的特徵向量會被保存（假設 `save_first_last` 為 true）
3. `feature_vectors` 和 `hooks` 目錄將會有一致的 epoch 保存行為

## 配置建議

在YAML配置文件中，建議只設置 `training.save_every` 即可控制所有保存行為：

```yaml
training:
  epochs: 50
  learning_rate: 0.001
  save_every: 1  # 每個epoch都保存模型檢查點和特徵向量

hooks:
  activation_capture:
    enabled: true  
    target_layers: ["backbone.7", "fc"]  # 要捕獲的層
    # 不需要再設置save_frequency，會自動使用training.save_every的值
    compute_similarity: true
    compute_tsne: true
```

## 額外建議

1. **檢查回調實現**：
   確保 `ActivationCaptureHook` 和其他回調正確解釋 `save_frequency` 參數，並在適當的 epoch 上保存數據。

2. **目錄結構一致性**：
   檢查 `SaveManager` 是否在所有地方都使用相同的目錄結構，特別是 `hooks` 和 `feature_vectors` 目錄。

3. **監控日誌**：
   運行實驗時查看日誌輸出，確保保存頻率設置被正確應用到所有回調。

此修復應該能讓使用者只需配置 `training.save_every` 就能統一控制所有保存行為，避免配置分散帶來的不一致問題。

# 特徵向量捕獲只保存最後一個Epoch的問題

## 問題描述

在YAML配置中設定了 `save_frequency: 1`，理論上應該保存每個epoch的特徵向量，但實際上系統只保存了最後一個epoch的特徵向量。

## 問題原因分析

經過代碼審查，發現問題主要出在 `ActivationCaptureHook` 類的實現中:

1. 在評估階段，系統可能多次調用 `on_evaluation_begin` 和 `on_evaluation_end` 方法，但沒有正確跟踪哪些epoch已經處理過。

2. 雖然類中有一個 `captured_epochs` 集合用於跟踪已經處理過的epoch，但該集合沒有在判斷是否處理特定epoch時被充分使用。

3. 在 `_compute_tsne` 方法中，PCA降維組件數設置為固定的50，當樣本數少於50時會導致錯誤。

## 解決方案

### 1. 改進 `_should_process_epoch` 方法

確保該方法首先檢查epoch是否已經處理過，防止重複處理。

```python
def _should_process_epoch(self, epoch: int) -> bool:
    # 如果已經處理過這個epoch，直接返回False避免重複處理
    if epoch in self.captured_epochs:
        logger.info(f"跳過epoch {epoch}的激活值處理，已經處理過")
        return False
        
    # 優先使用target_epochs
    if self.target_epochs is not None:
        should_process = epoch in self.target_epochs
        # ...剩餘邏輯保持不變
```

### 2. 在評估開始和評估結束時檢查epoch是否已處理

在 `on_evaluation_begin` 和 `on_evaluation_end` 方法的開頭添加額外檢查：

```python
# 檢查當前epoch是否已被處理過
if self.current_epoch in self.captured_epochs:
    logger.info(f"評估開始：epoch {self.current_epoch} 已被處理過，跳過此次處理")
    return
```

### 3. 修復 t-SNE 計算中的PCA組件數問題

改進 `_compute_tsne` 方法，使PCA組件數適應樣本數：

```python
# 樣本數和特徵數
n_samples, n_features = features_np.shape

# 修改：根據樣本數調整PCA組件數
# 選擇 min(樣本數-1, 特徵數, 50) 作為組件數
n_components = min(n_samples - 1, n_features, 50)
if n_components < 2:
    n_components = 2  # 確保至少有兩個組件
```

## 改進後的結果

修復後，系統應該能夠根據配置文件中的 `save_frequency` 或 `target_epochs` 設置正確保存每個指定epoch的特徵向量，並且防止重複處理相同的epoch。

這些更改還能解決在處理小批量數據時的t-SNE錯誤，使系統更穩健。

## 實現的技術要點

1. **適當的狀態跟踪**：使用 `captured_epochs` 集合跟踪已處理的epochs，確保不重複處理
2. **重複檢查**：在關鍵處理點（評估開始和結束）都檢查epoch狀態
3. **動態參數調整**：根據實際數據特性動態調整算法參數（如PCA組件數）
4. **詳細日誌**：添加更多日誌輸出，幫助診斷問題
