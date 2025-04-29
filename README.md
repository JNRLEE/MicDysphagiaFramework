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
- **訓練過程監控**：
  - 整合 TensorBoard 實時監控訓練過程
  - 支持損失、指標、學習率等訓練參數的視覺化
  - 自動記錄實驗數據和配置
- **回調接口與模型鉤子**：
  - 提供標準化的回調接口，支持在訓練過程關鍵點插入監控和分析功能
  - 模型鉤子系統可無侵入式獲取神經網絡內部狀態（激活值、梯度、權重）
  - 支持與SBP_analyzer集成，實現訓練過程的深度分析和可視化
- **豐富的擴展功能**：
  - 視覺提示 (Visual Prompting)
  - t-SNE 可視化
  - 按患者 ID 拆分數據集
  - 自動調整學習率

## Minimal Executable Unit (MEU) 測試說明

本框架所有主要模組（如 `utils/config_loader.py`、`utils/constants.py`、`utils/data_adapter.py`、`utils/save_manager.py` 等）皆已內建 Minimal Executable Unit (MEU) 測試區塊。

**MEU 特色與用途：**
- 每個模組檔案底部皆有一段自動化測試程式，會檢查該模組的關鍵功能是否能正確運作，並測試常見錯誤情境。
- 測試內容包含：正常情境（正確功能）、異常情境（預期報錯），並會明確顯示「測試成功」或「遇到錯誤（預期行為）」。
- 所有 MEU 測試程式皆有完整中文註解與標準 docstring，方便理解與維護。

**如何執行 MEU 測試：**
1. 進入專案根目錄。
2. 依序執行下列指令（以 `config_loader.py` 為例）：
   ```bash
   python -m utils.config_loader
   python -m utils.constants
   python -m utils.data_adapter
   python -m utils.save_manager
   ```
3. 每個檔案都會自動執行其 MEU 區塊，並在終端機輸出測試結果。

如需將這些測試自動化、整合進 CI/CD，或有任何測試失敗訊息需要 debug，請參考原始碼底部的 MEU 範例或聯絡專案維護者。

## 模型與數據類型兼容性

下表顯示了框架中不同模型與數據類型之間的兼容性:


| 模型 \ 數據類型  | 音頻(Audio) | 頻譜圖(Spectrogram) | 特徵(Features) |
| ---------------- | ----------- | ------------------- | -------------- |
| Swin Transformer | ✅          | ✅                  | ✅             |
| FCNN             | ✅          | ✅                  | ✅             |
| CNN              | ✅          | ✅                  | ✅             |
| ResNet           | ✅          | ✅                  | ✅             |

框架提供自動數據適配功能，可以在不同數據類型和模型架構之間進行自動轉換，確保輸入格式正確。

## 環境設置

### 方法 1: 使用 Docker（推薦）

Docker 提供了一個獨立且一致的環境，確保在任何平台上都能順利運行此框架：

```bash
# 構建 Docker 映像
docker build -t micdysphagia:latest .

# 運行容器（基本用法）
docker run micdysphagia:latest

# 運行訓練任務（使用 GPU）
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs micdysphagia:latest --config config/your_config.yaml
```

更多 Docker 使用說明，請參考 `docker-instructions.md` 文件。

### 方法 2: 本地環境設置

1. **創建虛擬環境**：
```bash
# 使用 conda
conda create -n micdys python=3.9
conda activate micdys

# 或使用 venv
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

2. **安裝依賴**：
```bash
pip install -r requirements.txt
```

3. **特定平台設置**：
   - **macOS (Apple Silicon)**：
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
     ```
   - **CUDA 支援環境**：
     ```bash
     # 使用 conda
     conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
     # 或使用 pip
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```

## 開發指南

### 代碼風格

- 每個函式必須有標準化的 docstring，其中包含 Args、Returns、Description、References
- 使用 [Google Python 風格指南](https://google.github.io/styleguide/pyguide.html) 進行編碼

### 實驗管理

- 所有實驗執行程式需將 metadata 存入 experiments.log
- 使用標準命名格式命名輸出圖像
- 每個訓練腳本都必須呼叫 torch.manual_seed 並在 requirements.txt 中記錄使用的所有套件版本

### 新增套件

- 新增的套件必須立即更新至環境設定檔，例如 requirements.txt 或 Dockerfile

## 安裝指南

### 方法一：使用 Conda (推薦)

1. 安裝 [Anaconda](https://www.anaconda.com/download/) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. 克隆存儲庫：
```bash
git clone https://github.com/your-username/MicDysphagiaFramework.git
cd MicDysphagiaFramework
```

3. 創建並激活 conda 環境：
```bash
# 創建環境
conda create -n micdys python=3.9
# 激活環境
conda activate micdys
```

4. 安裝依賴：
```bash
# 安裝基本依賴
pip install -r requirements.txt
```

5. 安裝 PyTorch (根據您的系統選擇合適的命令)：
   - **macOS 用戶 (Apple Silicon)**:
     ```bash
     pip install torch torchvision torchaudio
     ```
   - **Windows/Linux 使用 NVIDIA GPU**:
     ```bash
     conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
     ```
   - **僅 CPU**:
     ```bash
     conda install pytorch torchvision torchaudio cpuonly -c pytorch
     ```

### 方法二：使用標準 venv

1. 克隆存儲庫：
```bash
git clone https://github.com/your-username/MicDysphagiaFramework.git
cd MicDysphagiaFramework
```

2. 創建虛擬環境：
```bash
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

3. 安裝依賴：
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. 安裝 PyTorch (根據您的系統選擇合適的命令，參考 [PyTorch 官網](https://pytorch.org/get-started/locally/))

## 快速入門

### 運行實驗

1. **執行標準實驗**：
```bash
python scripts/run_experiments.py --config config/audio_swin_regression.yaml
```

2. **使用特定設備**：
```bash
python scripts/run_experiments.py --config config/audio_swin_regression.yaml --device cuda:0
```

3. **指定輸出目錄**：
```bash
python scripts/run_experiments.py --config config/audio_swin_regression.yaml --output_dir results/my_experiment
```

### 監控訓練過程

本框架整合了 TensorBoard 用於訓練監控。在訓練過程中，所有實驗指標、損失和學習率變化都會自動記錄到 TensorBoard 日誌中。

1. **啟動 TensorBoard**：
```bash
tensorboard --logdir outputs/
```

2. **查看訓練指標**：
   - 打開瀏覽器訪問 http://localhost:6006
   - 查看損失曲線、評估指標、學習率變化等

3. **同時查看多個實驗**：
```bash
tensorboard --logdir_spec=exp1:outputs/experiment1,exp2:outputs/experiment2
```

## 目錄結構

```
MicDysphagiaFramework/
├── config/                      # 配置文件目錄
│   ├── config_schema.yaml       # 配置模式定義
│   ├── audio_swin_regression.yaml  # 音頻 Swin 回歸模型配置
│   └── audio_swin_classification.yaml  # 音頻 Swin 分類模型配置
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
│   ├── resnet_model.py          # ResNet 模型
│   ├── model_structure.py       # 模型結構信息提取工具
│   └── hook_bridge.py           # 模型鉤子橋接模塊
│
├── trainers/                    # 訓練模塊
│   ├── trainer_factory.py       # 訓練器工廠
│   └── pytorch_trainer.py       # PyTorch 訓練器
│
├── tests/                       # 功能測試
│   ├── test_model_data_bridging.py  # 模型數據橋接測試
│   ├── model_data_bridging_report.json # 兼容性測試報告
│   ├── test_callback_interface.py   # 回調接口測試
│   ├── dataloader_test/         # 數據加載器測試數據
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
│   ├── metrics.py               # 評估指標
│   ├── save_manager.py          # 存檔管理器
│   └── callback_interface.py    # 回調接口定義
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
│   ├── run_experiments.py       # 運行實驗腳本
│   ├── prepare_spectrograms.py  # 頻譜圖預處理
│   └── extract_features.py      # 特徵提取
│
├── results/                     # 實驗結果目錄
│   ├── experiment1_timestamp/   # 實驗結果子目錄
│   ├── experiment2_timestamp/   # 實驗結果子目錄
│   └── frame_data_structure.md        # 數據結構說明文檔
│
├── main.py                      # 主程序
├── requirements.txt             # 依賴列表
└── README.md                    # 說明文件
```

## 實驗結果數據結構

每個實驗在 `results/` 目錄下都有一個以實驗名稱和時間戳命名的子目錄，例如 `audio_swin_classification_20250417_143624`，包含以下內容：

- **config.json**：實驗配置的完整副本
- **models/**：
  - **best_model.pth**：根據驗證集表現最佳的模型
  - **checkpoint_epoch_X.pth**：每個訓練週期的模型檢查點
- **tensorboard_logs/**：TensorBoard 日誌文件，用於可視化訓練過程
- **hooks/**：模型鉤子捕獲的中間層激活值和梯度數據
- **logs/**：訓練和評估過程的詳細日誌

詳細的數據結構說明請參考 `results/data_structure.md`。

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

完整示例請參考 `config/audio_swin_regression.yaml` 和 `config/audio_swin_classification.yaml`。

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

### 使用回調接口和模型鉤子

回調接口系統允許在模型訓練過程中的關鍵點插入自定義代碼，實現深度分析、可視化和監控。

#### 基本使用方法

1. **創建自定義回調**:

   ```python
   from utils.callback_interface import CallbackInterface
   
   class MyCallback(CallbackInterface):
       def on_train_begin(self, model: nn.Module, logs: Dict[str, Any] = None) -> None:
           print("訓練開始")
           
       def on_epoch_end(self, epoch: int, model: nn.Module, train_logs: Dict[str, Any], 
                      val_logs: Dict[str, Any], logs: Dict[str, Any] = None) -> None:
           print(f"Epoch {epoch} 結束，訓練損失: {train_logs['loss']}, 驗證損失: {val_logs['loss']}")
   ```

2. **註冊回調到訓練器**:

   ```python
   trainer = PyTorchTrainer(config, model)
   trainer.add_callback(MyCallback())
   ```

3. **使用模型鉤子監控內部狀態**:

   ```python
   from models.hook_bridge import SimpleModelHookManager
   
   class ActivationMonitorCallback(CallbackInterface):
       def __init__(self):
           self.hook_manager = None
           
       def on_train_begin(self, model: nn.Module, logs: Dict[str, Any] = None) -> None:
           # 創建鉤子管理器
           self.hook_manager = SimpleModelHookManager(
               model,
               monitored_layers=['model.layers.0', 'head'],
               monitored_params=['model.layers.0.blocks.0.mlp.fc1.weight']
           )
           
       def on_batch_end(self, batch: int, model: nn.Module, inputs: torch.Tensor = None, 
                      targets: torch.Tensor = None, outputs: torch.Tensor = None, 
                      loss: torch.Tensor = None, logs: Dict[str, Any] = None) -> None:
           # 更新批次數據
           self.hook_manager.update_batch_data(inputs, outputs, targets, loss, batch_idx=batch)
           # 保存當前數據
           self.hook_manager.save_current_data()
           
       def on_train_end(self, model: nn.Module, history: Dict[str, List] = None, 
                      logs: Dict[str, Any] = None) -> None:
           # 清理鉤子
           if self.hook_manager:
               self.hook_manager.remove_hooks()
   ```

#### 可用的回調方法

- `on_train_begin`: 訓練開始時調用
- `on_epoch_begin`: 每個epoch開始時調用
- `on_batch_begin`: 每個批次開始時調用
- `on_batch_end`: 每個批次結束時調用
- `on_epoch_end`: 每個epoch結束時調用
- `on_train_end`: 訓練結束時調用
- `on_evaluation_begin`: 評估開始時調用
- `on_evaluation_end`: 評估結束時調用

#### 使用現有的分析回調

框架提供了內置的 `SimpleModelAnalyticsCallback` 分析回調，可以自動收集模型的激活值和梯度信息：

```python
from models.hook_bridge import create_analyzer_callback

# 創建分析器回調
analyzer_callback = create_analyzer_callback(
    output_dir='results/my_experiment',
    monitored_layers=['model.layers.0', 'model.layers.3', 'head'],
    monitored_params=['model.layers.0.blocks.0.mlp.fc1.weight', 'head.weight'],
    save_frequency=1
)

# 添加到訓練器
trainer.add_callback(analyzer_callback)
```

#### 與 SBP_analyzer 集成

回調接口支持與 SBP_analyzer（吞嚥聲音處理分析器）集成，當 SBP_analyzer 可用時，框架會自動使用其提供的高級功能：

```python
from models.hook_bridge import get_analyzer_callbacks_from_config

# 從配置中獲取分析器回調
analyzer_callbacks = get_analyzer_callbacks_from_config(config)

# 添加到訓練器
for callback in analyzer_callbacks:
    trainer.add_callback(callback)
```

## 貢獻指南

1. Fork 存儲庫
2. 創建您的功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打開 Pull Request

## 許可證

本項目使用 MIT 許可證 - 詳情請參閱 [LICENSE](LICENSE) 文件。

## 致謝

- 本研究由台北榮民總醫院耳鼻喉頭頸醫學部與國立陽明交通大學共同參與