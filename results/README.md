# MicDysphagiaFramework 實驗結果

本目錄包含 MicDysphagiaFramework 生成的實驗結果。每個實驗都存儲在獨立的子目錄中，以實驗名稱和時間戳命名。

## 目錄結構

```
results/
└── {實驗名稱}_{時間戳}/               # 例：audio_swin_regression_20250417_142912/
    ├── config.json                 # 實驗配置文件
    ├── model_structure.json        # 模型結構信息
    ├── training_history.json       # 訓練歷史記錄
    ├── models/                     # 模型權重保存目錄
    │   ├── best_model.pth          # 最佳模型權重
    │   └── checkpoint_epoch_N.pth  # 各輪次的檢查點
    ├── hooks/                      # 模型鉤子數據
    │   ├── training_summary.pt     # 整體訓練摘要
    │   ├── evaluation_results_test.pt  # 測試集評估結果
    │   ├── epoch_N_validation_predictions.pt  # 每輪驗證集預測結果
    │   ├── test_set_activations_layer_name.pt # 測試集層激活值
    │   └── epoch_N/                # 各輪次數據
    │       ├── ...                 # 其他數據
    │       ├── *_gradient.pt       # 參數梯度張量
    │       ├── *_gradient_stats.json # 梯度統計量 (含分位數)
    │       ├── *_gradient_hist.pt  # 梯度直方圖數據
    │       └── gns_stats_epoch_N.json  # GNS統計量
    ├── results/                    # 實驗結果
    │   └── results.json            # 最終結果摘要
    ├── tensorboard_logs/           # TensorBoard日誌
    └── logs/                       # 訓練日誌
```

## 實驗分析

使用 SBP_analyzer 工具可以對這些實驗結果進行深入分析，包含訓練過程中的梯度統計量、梯度直方圖數據、GNS 統計量等：

```bash
# 使用run_analysis.py腳本分析實驗
python run_analysis.py --experiment audio_swin_regression_20250417_142912
```

分析結果將存儲在 `analysis_results/{實驗名稱}_{時間戳}/` 目錄下。

## 新增功能說明

### 驗證集預測保存

- **功能**: 每個訓練輪次結束時自動保存驗證集的預測結果
- **文件位置**: `hooks/epoch_N_validation_predictions.pt`
- **用途**: 追蹤模型訓練過程中預測表現的變化，分析模型收斂情況

### 評估結果保存

- **功能**: 在評估階段結束時保存測試集的評估結果，包含真實標籤、模型預測和完整指標
- **文件位置**: `hooks/evaluation_results_test.pt`
- **用途**: 用於生成混淆矩陣、ROC曲線等分析圖表

### 目標層激活值捕獲

- **功能**: 在測試集評估過程中，捕獲並保存特定目標層的激活值
- **文件位置**: `hooks/test_set_activations_layer_name.pt`
- **用途**: 用於餘弦相似度分析、t-SNE可視化等進階模型分析

## GNS 統計量與梯度分析說明

- **GNS**: 每個 epoch 計算一次 GNS 統計量，記錄於 `hooks/epoch_N/gns_stats_epoch_N.json`。
- **梯度統計量**: 每個參數的詳細梯度統計量 (含分位數) 記錄於 `hooks/epoch_N/*_gradient_stats.json`。
- **梯度直方圖**: 每個參數的梯度直方圖數據記錄於 `hooks/epoch_N/*_gradient_hist.pt`。
- 這些數據可用於分析訓練穩定性、梯度雜訊規模、梯度分布等，協助調整超參數。
- 詳細格式與讀取方式請參考 `framework_data_structure.md`。

## 詳細說明

請參考 `framework_data_structure.md` 文件了解更多關於數據結構的詳細信息，包括各個文件的格式、內容和讀取方法。