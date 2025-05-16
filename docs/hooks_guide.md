# MicDysphagiaFramework Hook 系統指南

## 概述

MicDysphagiaFramework 的 Hook 系統允許用戶在模型訓練和評估的關鍵點上插入自定義的處理邏輯，以收集額外的數據或執行特定操作。本文檔將詳細介紹實現的 Hook 功能：

1. **評估結果捕獲 Hook**：收集測試集的真實標籤和模型預測，用於生成混淆矩陣等分析圖表
2. **激活值捕獲 Hook**：捕獲模型特定層的激活值，用於餘弦相似度分析等進階模型分析
3. **驗證預測保存**：在每個訓練輪次結束時保存驗證集的預測結果，用於追蹤模型訓練過程

## 配置 Hook

要使用 Hook 功能，需要在配置文件中添加相應的設置。以下是一個完整的配置示例：

```yaml
hooks:
  # 模型分析鉤子：收集層激活值和梯度
  model_analytics:
    enabled: true
    monitored_layers: ['encoder.final_layer', 'head.pre_logits']  # 要監控的層
    monitored_params: ['encoder.layer1.weight', 'head.weight']  # 要監控的參數
    save_frequency: 1  # 保存頻率（每 N 個 epoch 保存一次）
    save_validation_predictions: true  # 是否在每個 epoch 保存驗證集預測結果
  
  # 評估結果捕獲鉤子：收集測試集的真實標籤和模型預測（用於混淆矩陣）
  evaluation_capture:
    enabled: true
    datasets: ['test', 'val']  # 要捕獲結果的數據集
    save_probabilities: true  # 是否保存類別概率（用於 ROC 曲線）
  
  # 激活值捕獲鉤子：收集特定層的激活值（用於餘弦相似度圖）
  activation_capture:
    enabled: true
    target_layers: ['encoder.final_layer', 'head.pre_logits']  # 要捕獲激活值的層名稱列表
    datasets: ['test']  # 在哪些數據集上捕獲
    include_sample_ids: true  # 是否包含樣本ID
```

## 功能詳解

### 1. 評估結果捕獲 (EvaluationResultsHook)

此 Hook 會在模型評估階段結束時，保存所有測試樣本的真實標籤和模型預測，用於後續分析。

#### 使用方法：

1. 在配置文件中啟用此功能：
   ```yaml
   hooks:
     evaluation_capture:
       enabled: true
       datasets: ['test']  # 可添加多個數據集，如 ['test', 'val']
       save_probabilities: true  # 分類任務保存類別概率
   ```

2. 執行實驗後，會在 `results/{實驗名稱}_{時間戳}/hooks/` 目錄下生成以下文件：
   - `evaluation_results_test.pt`: 測試集評估結果
   - `evaluation_results_val.pt`: 驗證集評估結果 (如果設置了 'val')

3. 讀取保存的結果：
   ```python
   import torch
   
   # 載入測試集評估結果
   eval_results = torch.load('results/experiment/hooks/evaluation_results_test.pt')
   
   # 獲取預測和真實標籤
   predictions = eval_results['predictions']
   targets = eval_results['targets']
   
   # 如果保存了概率（分類任務）
   if 'probabilities' in eval_results:
       probabilities = eval_results['probabilities']
   ```

### 2. 激活值捕獲 (ActivationCaptureHook)

此 Hook 會在模型評估過程中捕獲指定層的激活值，用於分析模型內部特徵空間。

#### 使用方法：

1. 在配置文件中啟用此功能：
   ```yaml
   hooks:
     activation_capture:
       enabled: true
       target_layers: ['feature_extractor', 'head']  # 要捕獲的層名稱
       datasets: ['test']  # 可添加多個數據集
       include_sample_ids: true  # 是否包含樣本ID
   ```

2. 執行實驗後，會在 `results/{實驗名稱}_{時間戳}/hooks/` 目錄下生成以下文件：
   - `test_set_activations_feature_extractor.pt`: 測試集特徵提取器層的激活值
   - `test_set_activations_head.pt`: 測試集頭部層的激活值

3. 讀取保存的激活值：
   ```python
   import torch
   
   # 載入測試集特徵提取器層的激活值
   activation_data = torch.load('results/experiment/hooks/test_set_activations_feature_extractor.pt')
   
   # 獲取激活值和標籤
   activations = activation_data['activations']
   targets = activation_data['targets']
   
   # 如果包含了樣本ID
   if 'sample_ids' in activation_data:
       sample_ids = activation_data['sample_ids']
   ```

### 3. 驗證預測保存

此功能會在每個訓練輪次結束時保存驗證集的預測結果，用於追蹤模型訓練過程中的預測變化。

#### 使用方法：

1. 在配置文件中啟用此功能：
   ```yaml
   hooks:
     model_analytics:
       enabled: true
       save_validation_predictions: true  # 啟用驗證集預測保存
       # 其他參數...
   ```

2. 執行實驗後，會在 `results/{實驗名稱}_{時間戳}/hooks/` 目錄下生成以下文件：
   - `epoch_0_validation_predictions.pt`: 第0輪驗證集預測結果
   - `epoch_1_validation_predictions.pt`: 第1輪驗證集預測結果
   - ... 每個輪次都會生成一個文件

3. 讀取保存的預測結果：
   ```python
   import torch
   
   # 載入第0輪驗證集預測結果
   val_predictions = torch.load('results/experiment/hooks/epoch_0_validation_predictions.pt')
   
   # 獲取模型輸出、預測和真實標籤
   outputs = val_predictions['outputs']
   targets = val_predictions['targets']
   
   # 如果是分類任務，還會包含預測類別
   if 'predictions' in val_predictions:
       predictions = val_predictions['predictions']
   ```

## 後續分析

保存的數據可用於 SBP_analyzer 進行進階分析，生成以下可視化圖表：

1. **混淆矩陣**：使用 `evaluation_results_test.pt` 中的預測和真實標籤
2. **餘弦相似度圖**：使用 `test_set_activations_layer_name.pt` 中的激活值
3. **預測變化追蹤**：使用各個 epoch 的 `epoch_N_validation_predictions.pt` 文件
4. **ROC 曲線**：使用 `evaluation_results_test.pt` 中的類別概率（分類任務）

## 注意事項

1. 捕獲層激活值時，請確保指定的層名稱正確存在於模型中。
2. 保存大量數據可能會佔用較多磁盤空間，建議在配置中合理選擇需要監控的層和數據集。
3. 激活值捕獲僅在評估階段進行，不會影響訓練過程的性能。