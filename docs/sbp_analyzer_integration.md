# SBP_analyzer 整合指南

本文檔說明如何在 MicDysphagiaFramework 中整合和使用 SBP_analyzer 工具包，以實現模型訓練過程中的深度分析。

## 目錄

1. [安裝指南](#安裝指南)
2. [基本概念](#基本概念)
3. [配置指南](#配置指南)
4. [分析回調](#分析回調)
5. [模型鉤子](#模型鉤子)
6. [分析結果查看](#分析結果查看)
7. [進階使用](#進階使用)
8. [故障排除](#故障排除)

## 安裝指南

SBP_analyzer 是一個獨立的工具包，需要單獨安裝。建議使用開發模式安裝，以便於修改和更新。

```bash
# 先切換到 SBP_analyzer 目錄
cd /path/to/SBP_analyzer

# 安裝開發版本
pip install -e .
```

## 基本概念

SBP_analyzer 提供兩種主要功能：

1. **回調 (Callbacks)**：在模型訓練的不同階段執行分析和監控功能
2. **模型鉤子 (Model Hooks)**：無侵入式獲取模型內部狀態（激活值、梯度、權重等）

這些功能與 MicDysphagiaFramework 整合，無需修改現有訓練代碼即可獲得深入的模型和數據分析。

## 配置指南

在 YAML 配置文件中添加 `analysis` 部分以啟用 SBP_analyzer 功能。以下是一個完整的配置示例：

```yaml
# ... 其他配置 ...

analysis:
  enabled: true                           # 是否啟用分析功能
  
  # 模型分析配置
  model_analytics:
    enabled: true                         # 是否啟用模型分析
    output_dir: 'analysis_results'        # 分析結果保存目錄
    save_frequency: 1                     # 分析結果保存頻率(每多少epoch)
    monitored_layers:                     # 要監控的層名稱，如果為null則監控所有層
      - 'encoder.layer1'
      - 'encoder.layer2'
      - 'decoder'
    monitored_params:                     # 要監控的參數名稱
      - 'encoder.layer1.weight'
      - 'decoder.weight'
  
  # 數據監控配置
  data_monitor:
    enabled: true
    # ... 數據監控相關配置 ...

  # 特徵監控配置
  feature_monitor:
    enabled: true
    # ... 特徵監控相關配置 ...
```

## 分析回調

SBP_analyzer 提供多種分析回調，每種回調針對模型訓練的不同方面：

### ModelAnalyticsCallback

模型內部狀態分析回調，監控層激活值、參數梯度和權重分布。

```python
from sbp_analyzer import ModelAnalyticsCallback

callback = ModelAnalyticsCallback(
    output_dir='analysis_results',
    monitored_layers=['encoder.layer1', 'decoder'],
    save_frequency=1
)

# 添加到訓練器
trainer.add_callback(callback)
```

### DataMonitorCallback

監控數據分布和統計特性，檢測數據漂移和異常值。

```python
from sbp_analyzer import DataMonitorCallback

callback = DataMonitorCallback()
trainer.add_callback(callback)
```

### FeatureMonitorCallback

監控和分析特徵提取和表示學習效果。

```python
from sbp_analyzer import FeatureMonitorCallback

callback = FeatureMonitorCallback()
trainer.add_callback(callback)
```

## 模型鉤子

模型鉤子提供深入了解模型內部狀態的能力，主要包含以下鉤子：

- **ActivationHook**：獲取模型中間層激活值
- **GradientHook**：獲取模型參數梯度
- **ModelHookManager**：統一管理多種鉤子

這些鉤子已經整合到 SBP_analyzer 的回調系統中，無需直接使用。如需手動使用，可參考以下範例：

```python
from sbp_analyzer import ModelHookManager

# 創建鉤子管理器
hook_manager = ModelHookManager(model)

# 執行前向和反向傳播
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()

# 獲取分析數據
activations = hook_manager.get_activations()
gradients = hook_manager.get_gradients()
weights = hook_manager.get_weights()

# 獲取統計信息
activation_stats = hook_manager.get_layer_output_statistics()
gradient_stats = hook_manager.get_gradient_statistics()
weight_stats = hook_manager.get_weight_statistics()

# 清理鉤子（重要！）
hook_manager.remove_hooks()
```

## 分析結果查看

SBP_analyzer 會將分析結果保存在指定的輸出目錄中，通常包含：

1. **JSON 格式的詳細分析數據**：位於 `{output_dir}/model_analytics_{timestamp}/epoch_{n}/analysis.json`
2. **模型結構信息**：位於 `{output_dir}/model_analytics_{timestamp}/model_structure.json`
3. **TensorBoard 記錄**：可以通過 TensorBoard 查看各種統計指標的變化趨勢

```bash
# 啟動 TensorBoard 查看結果
tensorboard --logdir results/
```

## 進階使用

### 自定義回調

您可以通過繼承 BaseCallback 類來創建自定義回調：

```python
from sbp_analyzer import BaseCallback

class MyCustomCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        # 初始化自定義參數

    def on_epoch_end(self, epoch, model, train_logs, val_logs, logs=None):
        # 自定義邏輯
        pass
```

### 模型層識別

要監控特定層，需要知道模型中各層的名稱。可以使用以下代碼列出模型中所有層：

```python
for name, module in model.named_modules():
    print(f"層名稱: {name}, 類型: {type(module).__name__}")
```

## 故障排除

### 常見問題：

1. **記憶體不足**：監控所有層可能導致記憶體使用量增加，建議只監控關鍵層
2. **性能下降**：過於頻繁的分析可能影響訓練速度，適當調高保存頻率
3. **分析報錯**：確保監控的層名稱正確存在

### 解決方案：

- 減少監控的層數量
- 降低分析頻率
- 檢查模型結構，確認層名稱
- 如需更深入的調試，啟用日誌：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
``` 