"""
此腳本展示如何在實際訓練流程中使用損失函數。
模擬一個完整的訓練循環，使用不同的損失函數進行模型優化。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import logging
import matplotlib.pyplot as plt
from pathlib import Path

# 引入損失函數
from losses import LossFactory

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("losses_example")

# 設定隨機種子
def set_seed(seed: int = 42):
    """設定隨機種子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 創建一個簡單的回歸模型
class SimpleModel(nn.Module):
    """簡單的回歸模型，用於預測EAT-10分數"""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 64):
        super(SimpleModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# 生成模擬數據
def generate_data(num_samples: int = 100, input_size: int = 10, noise_level: float = 0.5):
    """生成模擬數據
    
    Args:
        num_samples: 樣本數量
        input_size: 輸入特徵維度
        noise_level: 噪聲水平
        
    Returns:
        特徵張量、分數張量和患者ID列表
    """
    # 特徵
    X = torch.randn(num_samples, input_size)
    
    # 生成真實權重
    true_weights = torch.randn(input_size)
    
    # 計算分數 (線性組合 + 噪聲)
    scores = torch.matmul(X, true_weights) + noise_level * torch.randn(num_samples)
    
    # 保證分數在合理範圍內 (0-40)
    scores = torch.clamp(scores, min=0, max=40)
    
    # 患者ID (每個患者有多個樣本)
    num_patients = num_samples // 3  # 平均每個患者有3個樣本
    patient_ids = []
    for i in range(num_patients):
        samples_per_patient = random.randint(1, 5)  # 每個患者1-5個樣本
        patient_id = f"P{i:03d}"
        patient_ids.extend([patient_id] * samples_per_patient)
    
    # 裁剪到正確數量
    patient_ids = patient_ids[:num_samples]
    
    return X, scores, patient_ids

# 訓練模型
def train_model(
    model: nn.Module, 
    X: torch.Tensor, 
    scores: torch.Tensor, 
    loss_config: dict, 
    num_epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 0.001
):
    """使用指定損失函數訓練模型
    
    Args:
        model: 要訓練的模型
        X: 特徵張量
        scores: 目標分數張量
        loss_config: 損失函數配置
        num_epochs: 訓練輪數
        batch_size: 批次大小
        learning_rate: 學習率
        
    Returns:
        訓練損失歷史和測試損失
    """
    # 拆分訓練集和測試集
    num_samples = X.size(0)
    indices = torch.randperm(num_samples)
    
    train_size = int(num_samples * 0.8)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    X_train, scores_train = X[train_indices], scores[train_indices]
    X_test, scores_test = X[test_indices], scores[test_indices]
    
    # 創建優化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 創建損失函數
    loss_fn = LossFactory.create_from_config(loss_config)
    
    # 記錄訓練損失
    train_losses = []
    
    # 訓練循環
    model.train()
    for epoch in range(num_epochs):
        # 批次處理
        epoch_loss = 0.0
        num_batches = 0
        
        for i in range(0, train_size, batch_size):
            # 獲取批次數據
            batch_indices = train_indices[i:i+batch_size]
            X_batch = X[batch_indices]
            scores_batch = scores[batch_indices]
            
            # 前向傳播
            predictions = model(X_batch)
            
            # 計算損失
            loss = loss_fn(predictions, scores_batch.unsqueeze(1))
            
            # 反向傳播和優化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 累計損失
            epoch_loss += loss.item()
            num_batches += 1
        
        # 計算平均損失
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        
        # 每20輪打印一次
        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # 測試模型
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test)
        test_loss = nn.MSELoss()(test_predictions, scores_test.unsqueeze(1)).item()
        
        # 計算MAE
        test_mae = nn.L1Loss()(test_predictions, scores_test.unsqueeze(1)).item()
    
    logger.info(f"測試集 MSE: {test_loss:.4f}, MAE: {test_mae:.4f}")
    
    return train_losses, test_loss

# 運行不同損失函數的訓練並比較結果
def run_loss_comparison():
    """使用不同損失函數訓練模型並比較結果"""
    
    # 設定隨機種子
    set_seed(42)
    
    # 生成模擬數據
    X, scores, patient_ids = generate_data(num_samples=200, input_size=10, noise_level=2.0)
    
    # 定義要測試的損失函數
    loss_configs = {
        "MSE": {
            "type": "MSELoss",
            "parameters": {}
        },
        "L1": {
            "type": "L1Loss",
            "parameters": {}
        },
        "SmoothL1": {
            "type": "SmoothL1Loss",
            "parameters": {}
        },
        "Huber": {
            "type": "HuberLoss",
            "parameters": {"delta": 1.0}
        },
        "Pairwise": {
            "type": "PairwiseRankingLoss",
            "parameters": {
                "margin": 1.0,
                "sampling_ratio": 0.3,
                "sampling_strategy": "score_diff"
            }
        },
        "Combined": {
            "combined": {
                "mse": {
                    "type": "MSELoss",
                    "parameters": {},
                    "weight": 1.0
                },
                "ranking": {
                    "type": "PairwiseRankingLoss",
                    "parameters": {
                        "margin": 1.0,
                        "sampling_ratio": 0.3
                    },
                    "weight": 0.5
                }
            }
        }
    }
    
    # 存儲訓練結果
    all_train_losses = {}
    all_test_losses = {}
    
    # 為每個損失函數訓練一個模型
    for loss_name, loss_config in loss_configs.items():
        logger.info(f"使用 {loss_name} 損失函數訓練模型...")
        
        # 創建模型
        model = SimpleModel(input_size=X.size(1))
        
        # 訓練模型
        train_losses, test_loss = train_model(
            model=model,
            X=X,
            scores=scores,
            loss_config=loss_config,
            num_epochs=200,
            batch_size=32,
            learning_rate=0.001
        )
        
        # 保存結果
        all_train_losses[loss_name] = train_losses
        all_test_losses[loss_name] = test_loss
    
    # 繪製訓練損失曲線
    plt.figure(figsize=(10, 6))
    for loss_name, losses in all_train_losses.items():
        plt.plot(losses, label=f"{loss_name} (Test MSE: {all_test_losses[loss_name]:.4f})")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("損失函數比較")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 保存圖像
    plt.savefig("loss_comparison.png")
    logger.info("圖表已保存為 loss_comparison.png")
    
    # 打印最終測試損失
    logger.info("最終測試MSE比較:")
    for loss_name, test_loss in sorted(all_test_losses.items(), key=lambda x: x[1]):
        logger.info(f"{loss_name}: {test_loss:.4f}")

if __name__ == "__main__":
    run_loss_comparison() 