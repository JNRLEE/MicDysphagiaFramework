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
  ├── epoch_2/
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
