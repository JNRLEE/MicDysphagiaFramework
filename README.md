# MicDysphagiaFramework：吞嚥障礙評估統一框架

MicDysphagiaFramework 是一個統一的吞嚥障礙評估框架，整合了 EAT-10 評分回歸和多類別分類功能，提供了靈活的配置系統和模塊化設計。通過 YAML 配置文件，使用者可以輕鬆地切換不同的數據源、模型架構、訓練策略和評估方法。

## 功能特點

- **統一配置系統**：使用 YAML 配置文件管理所有實驗參數
- **PyTorch 深度學習框架**：基於 PyTorch 構建，提供豐富的模型選擇
- **模塊化設計**：數據處理、模型架構、訓練流程和可視化模塊獨立且可擴展
- **多種數據類型**：支持原始音頻、頻譜圖和特徵向量作為輸入
- **多種模型架構**：支持 Swin Transformer、ResNet、全連接神經網絡和 CNN 等
- **自動數據適配**：在不同數據類型與模型架構間提供橋接功能，自動轉換格式
- **多元損失函數**：
  - 支持標準回歸與分類損失函數
  - 提供豐富的排序學習損失函數:
    - 成對方法(Pairwise): MarginRankingLoss, RankNetLoss
    - 列表方法(Listwise): ListMLELoss, ApproxNDCGLoss, LambdaRankLoss
  - 實現損失函數組合機制，支持權重動態調整和自定義損失函數組合
- **豐富的擴展功能**：
  - 視覺提示 (Visual Prompting)
  - t-SNE 可視化
  - 按患者 ID 拆分數據集
  - 自動調整學習率

## 模型與數據類型兼容性

下表顯示了框架中不同模型與數據類型之間的兼容性:

| 模型 \ 數據類型 | 音頻(Audio) | 頻譜圖(Spectrogram) | 特徵(Features) |
|--------------|-----------|-----------------|------------|
| Swin Transformer | ✅ | ✅ | ✅ |
| FCNN | ✅ | ✅ | ✅ |
| CNN | ✅ | ✅ | ✅ |
| ResNet | ✅ | ✅ | ✅ |

框架提供自動數據適配功能，可以在不同數據類型和模型架構之間進行自動轉換，確保輸入格式正確。

## 安裝指南

### 環境需求

```
Python 3.8+
PyTorch 2.0+
```

### 安裝步驟

1. 克隆存儲庫：

```bash
git clone https://github.com/your-username/MicDysphagiaFramework.git
cd MicDysphagiaFramework
```

2. 創建虛擬環境：

```bash
# 使用 conda
conda create -n micdys python=3.8
conda activate micdys

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安裝依賴：

```bash
# 安裝基本依賴
pip install -r requirements.txt

# 安裝 PyTorch (建議)
conda install pytorch torchvision torchaudio -c pytorch
```

## 目錄結構

```
MicDysphagiaFramework/
├── config/                      # 配置文件目錄
│   ├── config_schema.yaml       # 配置模式定義
│   ├── example_eat10_regression.yaml  # EAT-10 回歸示例配置
│   └── example_classification.yaml    # 分類示例配置
│
├── data/                        # 數據處理模塊
│   ├── dataset_factory.py       # 數據集工廠
│   ├── audio_dataset.py         # 音頻數據集
│   ├── spectrogram_dataset.py   # 頻譜圖數據集
│   └── feature_dataset.py       # 特徵數據集
│
├── models/                      # 模型定義
│   ├── model_factory.py         # 模型工廠
│   ├── swin_transformer.py      # Swin Transformer 模型
│   ├── fcnn.py                  # 全連接神經網絡
│   ├── cnn_model.py             # CNN 模型
│   └── resnet_model.py          # ResNet 模型
│
├── trainers/                    # 訓練模塊
│   ├── trainer_factory.py       # 訓練器工廠
│   └── pytorch_trainer.py       # PyTorch 訓練器
│
│
├── tests/                       # 功能測試
│   ├── test_model_data_bridging.py  # 模型數據橋接測試
│   ├── model_data_bridging_report.json # 兼容性測試報告
│   └── loss_tests/              # 損失函數測試
│
├── losses/                      # 損失函數
│   ├── loss_factory.py          # 損失函數工廠
│   ├── ranking_losses.py        # 排序損失函數
│   ├── combined_losses.py       # 組合損失函數
│   └── __init__.py              # 包初始化文件
│
├── utils/                       # 工具函數
│   ├── config_loader.py         # 配置加載器
│   ├── data_adapter.py          # 數據適配器
│   ├── logging_utils.py         # 日誌工具
│   └── metrics.py               # 評估指標
│
├── visualization/               # 可視化模塊
│   ├── visualize_results.py     # 結果可視化
│   ├── tsne_visualizer.py       # t-SNE 可視化
│   └── confusion_matrix.py      # 混淆矩陣可視化
│
├── docs/                        # 文檔
│   ├── model_data_compatibility.md  # 模型與數據兼容性文檔
│   ├── losses.md                # 損失函數設計與使用文檔
│   └── ranking_losses.md        # 排序損失函數數學原理與應用場景
│
├── scripts/                     # 腳本
│   ├── prepare_spectrograms.py  # 頻譜圖預處理
│   └── extract_features.py      # 特徵提取
│
├── main.py                      # 主程序
├── requirements.txt             # 依賴列表
└── README.md                    # 說明文件
```

## 使用指南

### 基本用法

1. **創建或修改配置文件**：

   根據您的需求修改 `config/` 目錄下的配置文件，或創建新的配置文件。

2. **運行訓練**：

   ```bash
   python main.py --config config/example_eat10_regression.yaml
   ```

3. **使用默認配置**：

   ```bash
   python main.py --config config/my_config.yaml --default_config config/config_schema.yaml
   ```

4. **僅評估模型**：

   ```bash
   python main.py --config config/example_eat10_regression.yaml --eval_only --checkpoint path/to/checkpoint.pth
   ```

5. **指定輸出目錄和設備**：

   ```bash
   python main.py --config config/example_eat10_regression.yaml --output_dir runs/experiment1 --device cuda:0
   ```

### 配置文件結構

配置文件分為幾個主要部分：

1. **global**: 全局配置
2. **data**: 數據配置
3. **model**: 模型配置
4. **training**: 訓練配置
5. **evaluation**: 評估配置
6. **visualization**: 可視化配置

詳細的配置選項請參考 `config/config_schema.yaml`。

### 損失函數配置

在配置文件的 `training` 部分可以設置損失函數：

```yaml
training:
  loss:
    type: "MSELoss"  # 使用均方誤差損失
    parameters:
      reduction: "mean"
```

排序損失函數配置範例：

#### 成對(Pairwise)損失函數

```yaml
training:
  loss:
    type: "MarginRankingLoss"  # 邊界排序損失
    parameters:
      margin: 0.3
      sampling_ratio: 0.5
      sampling_strategy: "score_diff"
```

```yaml
training:
  loss:
    type: "RankNetLoss"  # RankNet損失
    parameters:
      sigma: 1.0
      sampling_ratio: 0.4
```

#### 列表(Listwise)損失函數

```yaml
training:
  loss:
    type: "ListMLELoss"  # 列表最大似然估計損失
    parameters:
      batch_size_per_group: 16
      temperature: 1.0
```

```yaml
training:
  loss:
    type: "ApproxNDCGLoss"  # 近似NDCG損失
    parameters:
      temperature: 0.1
      group_size: 8
```

```yaml
training:
  loss:
    type: "LambdaRankLoss"  # LambdaRank損失
    parameters:
      sigma: 1.0
      group_size: 10
```

#### 組合多個損失函數

靜態權重組合：

```yaml
training:
  loss:
    combined:
      mse:
        type: "MSELoss"
        weight: 0.6
      ranking:
        type: "MarginRankingLoss"
        weight: 0.4
        parameters:
          margin: 0.3
          sampling_strategy: "score_diff"
```

動態權重組合：

```yaml
training:
  loss:
    combined:
      mse:
        type: "MSELoss"
        weight:
          start: 0.8
          end: 0.3
          schedule: "linear"  # 線性減少MSE的權重
      listmle:
        type: "ListMLELoss"
        weight:
          start: 0.2
          end: 0.7
          schedule: "linear"  # 線性增加ListMLE的權重
        parameters:
          batch_size_per_group: 16
```

更多損失函數配置示例和原理說明，請參考 `docs/losses.md` 和 `docs/ranking_losses.md`。

### 示例：EAT-10 回歸配置

```yaml
global:
  experiment_name: 'eat10_regression_swin'
  seed: 42
  debug: false
  device: 'auto'
  output_dir: 'runs/eat10_regression'

data:
  type: 'audio'
  source:
    wav_dir: '/path/to/wav/data'
  preprocessing:
    audio:
      sr: 16000
      duration: 5
      normalize: true
    spectrogram:
      method: 'mel'
      n_mels: 128
  # ... 更多配置

model:
  type: 'swin_transformer'
  parameters:
    model_name: 'swin_tiny_patch4_window7_224'
    pretrained: true
    num_classes: 1
    input_channels: 3
    is_classification: false
  # ... 更多配置
```

完整示例請參考 `config/example_eat10_regression.yaml` 和 `config/example_classification.yaml`。

## 擴展框架

### 添加新的數據集

1. 在 `data/` 目錄下創建新的數據集類
2. 在 `data/dataset_factory.py` 中註冊新的數據集

### 添加新的模型

1. 在 `models/` 目錄下創建新的模型類
2. 在 `models/model_factory.py` 中註冊新的模型

### 添加新的損失函數

1. 在 `losses/` 目錄下實現新的損失函數
2. 在 `losses/loss_factory.py` 中註冊新的損失函數

## 貢獻指南

1. Fork 存儲庫
2. 創建您的功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打開 Pull Request

## 許可證

本項目使用 MIT 許可證 - 詳情請參閱 [LICENSE](LICENSE) 文件。

## 致謝

- 本框架整合了 EAT10Regression 和 CombinationMonitor 兩個項目的功能
- 特別感謝所有貢獻者和使用者 