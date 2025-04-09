"""
排序損失函數使用示例
此示例展示如何在吞嚥障礙評估模型中應用排序損失函數
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# 引入排序損失函數
from model.losses.ranking_losses import (
    MarginRankingLoss, 
    RankNetLoss, 
    ListMLELoss, 
    ApproxNDCGLoss,
    LambdaRankLoss,
    CombinedRankingLoss
)

# 設定隨機種子以確保結果可重現
torch.manual_seed(42)
np.random.seed(42)


class SimpleRankingModel(nn.Module):
    """
    簡單的排序模型
    用於示範如何使用排序損失函數進行訓練
    """
    
    def __init__(self, input_dim, hidden_dim=64):
        """
        初始化簡單排序模型
        
        參數:
            input_dim (int): 輸入特徵維度
            hidden_dim (int): 隱藏層維度
        """
        super(SimpleRankingModel, self).__init__()
        
        # 使用簡單的前饋神經網絡
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        """
        前向傳播
        
        參數:
            x (torch.Tensor): 輸入特徵，形狀為[batch_size, list_size, input_dim]
            
        返回:
            torch.Tensor: 預測分數，形狀為[batch_size, list_size]
        """
        batch_size, list_size, input_dim = x.shape
        
        # 重塑以便處理每個項目
        x_flat = x.view(-1, input_dim)
        
        # 得到每個項目的分數
        scores_flat = self.network(x_flat).squeeze(-1)
        
        # 重塑回原始批次和列表維度
        scores = scores_flat.view(batch_size, list_size)
        
        return scores


def generate_synthetic_data(n_samples=100, list_size=10, input_dim=20):
    """
    生成合成數據用於排序任務
    
    參數:
        n_samples (int): 樣本數量
        list_size (int): 每個樣本中的項目數量
        input_dim (int): 每個項目的特徵維度
        
    返回:
        tuple: (特徵, 相關性分數)
    """
    # 隨機生成特徵
    features = np.random.normal(0, 1, (n_samples, list_size, input_dim))
    
    # 生成線性加權向量
    weights = np.random.normal(0, 1, input_dim)
    
    # 計算基礎分數（線性組合）
    base_scores = np.sum(features * weights, axis=2)
    
    # 添加一些隨機性以模擬噪聲
    relevance = base_scores + np.random.normal(0, 0.5, (n_samples, list_size))
    
    # 標準化相關性分數到[0, 4]範圍（模擬吞嚥障礙嚴重程度評分）
    relevance = (relevance - np.min(relevance, axis=1, keepdims=True))
    relevance = 4 * relevance / np.max(relevance, axis=1, keepdims=True)
    
    return torch.FloatTensor(features), torch.FloatTensor(relevance)


def train_and_evaluate(model, dataloader, loss_fn, optimizer, epochs=50, device="cpu"):
    """
    訓練模型並評估性能
    
    參數:
        model (nn.Module): 模型
        dataloader (DataLoader): 數據加載器
        loss_fn (nn.Module): 損失函數
        optimizer (Optimizer): 優化器
        epochs (int): 訓練輪數
        device (str): 訓練設備
        
    返回:
        list: 訓練過程中的損失值
    """
    model.to(device)
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        # 訓練
        model.train()
        for features, relevance in dataloader:
            features, relevance = features.to(device), relevance.to(device)
            
            # 前向傳播
            scores = model(features)
            
            # 計算損失
            if isinstance(loss_fn, (MarginRankingLoss, RankNetLoss)):
                # 對於成對損失函數，我們需要生成成對數據
                scores_i, scores_j, y_pairs = generate_pairs(scores, relevance)
                loss = loss_fn(scores_i, scores_j, y_pairs)
            else:
                # 對於列表損失函數，直接使用批次數據
                loss = loss_fn(scores, relevance)
            
            # 反向傳播和優化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
        # 計算平均損失
        avg_loss = epoch_loss / batch_count
        loss_history.append(avg_loss)
        
        # 打印訓練進度
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return loss_history


def generate_pairs(scores, relevance):
    """
    從批次數據生成成對比較
    
    參數:
        scores (torch.Tensor): 預測分數，形狀為[batch_size, list_size]
        relevance (torch.Tensor): 真實相關性，形狀為[batch_size, list_size]
        
    返回:
        tuple: (scores_i, scores_j, y_pairs) 用於成對損失函數
    """
    batch_size, list_size = scores.size()
    device = scores.device
    
    # 創建所有可能的項目對索引
    i_idx, j_idx = torch.triu_indices(list_size, list_size, offset=1, device=device)
    num_pairs = i_idx.size(0)
    
    scores_i_list = []
    scores_j_list = []
    y_pairs_list = []
    
    for b in range(batch_size):
        # 獲取當前批次的所有項目對
        s_i = scores[b, i_idx]
        s_j = scores[b, j_idx]
        r_i = relevance[b, i_idx]
        r_j = relevance[b, j_idx]
        
        # 計算標籤差異並轉換為{-1,0,1}
        y_diff = r_i - r_j
        y_pairs = torch.sign(y_diff)
        
        scores_i_list.append(s_i)
        scores_j_list.append(s_j)
        y_pairs_list.append(y_pairs)
    
    # 合併所有批次的數據
    scores_i = torch.cat(scores_i_list)
    scores_j = torch.cat(scores_j_list)
    y_pairs = torch.cat(y_pairs_list)
    
    # 移除標籤為0的對（表示相關性相同）
    valid_mask = y_pairs != 0
    scores_i = scores_i[valid_mask]
    scores_j = scores_j[valid_mask]
    y_pairs = y_pairs[valid_mask]
    
    return scores_i, scores_j, y_pairs


def evaluate_ndcg(model, dataloader, k=5, device="cpu"):
    """
    計算模型在數據集上的NDCG@k
    
    參數:
        model (nn.Module): 模型
        dataloader (DataLoader): 數據加載器
        k (int): 只考慮前k個結果
        device (str): 評估設備
        
    返回:
        float: 平均NDCG@k
    """
    model.eval()
    ndcg_scores = []
    
    with torch.no_grad():
        for features, relevance in dataloader:
            features, relevance = features.to(device), relevance.to(device)
            
            # 前向傳播
            scores = model(features)
            
            # 計算每個批次的NDCG@k
            batch_ndcg = []
            for i in range(scores.size(0)):
                # 獲取預測排序
                _, pred_indices = torch.sort(scores[i], descending=True)
                sorted_relevance = relevance[i][pred_indices]
                
                # 計算DCG@k
                gains = torch.pow(2.0, sorted_relevance[:k]) - 1.0
                discounts = torch.log2(torch.arange(1, k + 1, device=device) + 1.0)
                dcg = torch.sum(gains / discounts)
                
                # 計算理想DCG@k
                _, ideal_indices = torch.sort(relevance[i], descending=True)
                ideal_relevance = relevance[i][ideal_indices]
                ideal_gains = torch.pow(2.0, ideal_relevance[:k]) - 1.0
                idcg = torch.sum(ideal_gains / discounts)
                
                # 避免除零錯誤
                if idcg > 0:
                    ndcg = dcg / idcg
                else:
                    ndcg = torch.tensor(1.0, device=device)
                
                batch_ndcg.append(ndcg.item())
            
            ndcg_scores.extend(batch_ndcg)
    
    return np.mean(ndcg_scores)


def compare_loss_functions():
    """
    比較不同排序損失函數的性能
    """
    # 生成合成數據
    features, relevance = generate_synthetic_data(n_samples=500, list_size=10, input_dim=20)
    
    # 劃分訓練集和測試集
    train_size = int(0.8 * len(features))
    train_features, test_features = features[:train_size], features[train_size:]
    train_relevance, test_relevance = relevance[:train_size], relevance[train_size:]
    
    # 創建數據加載器
    train_dataset = TensorDataset(train_features, train_relevance)
    test_dataset = TensorDataset(test_features, test_relevance)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 定義不同的損失函數
    loss_functions = {
        "Margin Ranking Loss": MarginRankingLoss(margin=0.5),
        "RankNet Loss": RankNetLoss(sigma=1.0),
        "ListMLE Loss": ListMLELoss(),
        "ApproxNDCG Loss": ApproxNDCGLoss(temperature=1.0, k=5),
        "LambdaRank Loss": LambdaRankLoss(sigma=1.0, k=5),
        "Combined Loss": CombinedRankingLoss(
            loss_modules=[
                RankNetLoss(sigma=1.0),
                ApproxNDCGLoss(temperature=1.0, k=5)
            ],
            weights=[0.5, 0.5],
            adaptive_weight=True
        )
    }
    
    # 儲存各損失函數的性能指標
    results = {}
    
    for loss_name, loss_fn in loss_functions.items():
        print(f"\n訓練使用 {loss_name}:")
        
        # 初始化模型和優化器
        model = SimpleRankingModel(input_dim=20)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 訓練模型
        loss_history = train_and_evaluate(
            model, train_loader, loss_fn, optimizer, epochs=30, device=device
        )
        
        # 評估在測試集上的NDCG
        ndcg = evaluate_ndcg(model, test_loader, k=5, device=device)
        
        # 儲存結果
        results[loss_name] = {
            "loss_history": loss_history,
            "ndcg": ndcg
        }
        
        print(f"{loss_name} - 測試集 NDCG@5: {ndcg:.4f}")
    
    return results


def plot_results(results):
    """
    繪製實驗結果
    
    參數:
        results (dict): 實驗結果
    """
    # 繪製損失曲線
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for loss_name, result in results.items():
        plt.plot(result["loss_history"], label=loss_name)
    
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Loss Function Comparison")
    plt.legend()
    plt.grid(True)
    
    # 繪製NDCG比較
    plt.subplot(1, 2, 2)
    loss_names = list(results.keys())
    ndcg_values = [result["ndcg"] for result in results.values()]
    
    bar_positions = np.arange(len(loss_names))
    plt.bar(bar_positions, ndcg_values)
    plt.xticks(bar_positions, loss_names, rotation=45, ha="right")
    plt.xlabel("Loss Function")
    plt.ylabel("NDCG@5")
    plt.title("Ranking Performance Comparison")
    plt.grid(True, axis="y")
    plt.tight_layout()
    
    plt.savefig("ranking_loss_comparison.png")
    plt.show()


def main():
    """
    主函數
    """
    print("排序損失函數比較實驗")
    print("="*50)
    
    # 比較不同損失函數
    results = compare_loss_functions()
    
    # 繪製結果
    plot_results(results)
    
    # 打印實驗結論
    print("\n實驗結論:")
    print("-"*50)
    best_loss = max(results.items(), key=lambda x: x[1]["ndcg"])
    print(f"1. 在該合成數據集上，{best_loss[0]}取得了最佳NDCG@5: {best_loss[1]['ndcg']:.4f}")
    print("2. 列表損失函數通常比成對損失函數產生更好的排序質量")
    print("3. 組合多種損失函數可以融合不同優化目標，提高模型的整體性能")
    print("4. 對於吞嚥障礙嚴重程度評估等排序任務，直接優化NDCG的損失函數更加有效")


if __name__ == "__main__":
    main() 