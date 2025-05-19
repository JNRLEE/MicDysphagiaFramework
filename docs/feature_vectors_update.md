# 特徵向量格式更新說明

## 背景

為了使特徵向量分析功能能夠順利運行，我們對特徵向量的儲存方式進行了以下重要修改：

1. **特徵向量結構優化**:
   - 將高維特徵向量展平為2D格式 `[batch_size, feature_dim]`
   - 保持特徵向量文件結構一致性，確保包含必要欄位

2. **標籤保存**:
   - 為所有特徵向量添加類別標籤 (`targets` 字段)
   - 在無法獲取真實標籤時，提供隨機標籤生成功能

3. **目錄結構規範**:
   - 遵循 `feature_vectors/epoch_X/layer_X_features.pt` 的目錄結構
   - 更新特徵向量分析摘要，添加類別分佈信息

4. **t-SNE 可視化支援**:
   - 確保 t-SNE 文件包含標籤信息，便於可視化分析
   - 加強特徵向量降維前的預處理

## 修改的文件

1. `models/hook_bridge.py`: 更新了 ActivationCaptureHook 類，添加以下特性：
   - 展平高維特徵向量為2D格式
   - 添加隨機標籤生成功能
   - 優化 t-SNE 計算過程

2. `config/example_feature_vectors.yaml`: 更新配置文件，添加新選項：
   - `generate_random_labels`: 無標籤時是否生成隨機標籤

## 新增的工具

我們提供了一個修復工具腳本 `scripts/fix_feature_vectors.py`，用於處理現有的特徵向量文件：

```bash
# 檢查需要修改的文件（不實際修改）
python scripts/fix_feature_vectors.py --experiment_dir results/your_experiment_dir --dry_run

# 修復特徵向量文件
python scripts/fix_feature_vectors.py --experiment_dir results/your_experiment_dir
```

此腳本會：
1. 將高維特徵向量展平為2D格式
2. 為沒有標籤的特徵向量添加隨機標籤
3. 更新 t-SNE 文件，添加標籤
4. 更新特徵分析摘要文件，添加類別分佈信息

## 兼容性說明

這些更改完全向後兼容，不會影響已有的模型訓練和評估過程。但請注意以下事項：

1. 使用舊版格式產生的特徵向量文件需要使用修復腳本處理後才能用於新的特徵向量分析工具
2. 如果需要精確的類別相似度分析，請確保使用真實標籤，而不是隨機生成的標籤

## 使用建議

1. 對於新的實驗，建議在配置文件中設置 `generate_random_labels: true`
2. 對於已完成的實驗，使用修復腳本處理特徵向量文件
3. 在分析特徵向量時，注意區分真實標籤和隨機生成的標籤

## 進一步改進計劃

1. 添加從數據集中獲取真實標籤的功能
2. 改進特徵向量可視化工具，支援更多降維和聚類算法
3. 添加特徵向量質量評估指標，如類內/類間距離比 