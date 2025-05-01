# -*- coding: utf-8 -*-
"""
# 排序損失函數實現範例

此文件提供了排序學習中常用損失函數的實現範例及其在吞嚥障礙評估中的應用示例。
這些範例展示了如何在PyTorch框架中實現和使用各種排序損失函數，並包含詳細的中文註釋。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable


# ===== 成對排序損失函數 =====

class MarginRankingLoss(nn.Module):
    """
    邊際排序損失函數
    
    計算成對樣本間的排序損失，確保正確排序的樣本間有足夠的評分差距
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        """
        初始化
        
        Args:
            margin: 期望維持的最小評分差距
            reduction: 損失計算方式，可為'none', 'mean', 'sum'
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向計算
        
        Args:
            predictions: 模型預測分數，形狀為 [batch_size]
            targets: 目標分數/標籤，形狀為 [batch_size]
            
        Returns:
            計算得到的損失值
        """
        # 生成所有可能的對比樣本對
        batch_size = predictions.size(0)
        pred_i = predictions.unsqueeze(1).expand(batch_size, batch_size)
        pred_j = predictions.unsqueeze(0).expand(batch_size, batch_size)
        
        target_i = targets.unsqueeze(1).expand(batch_size, batch_size)
        target_j = targets.unsqueeze(0).expand(batch_size, batch_size)
        
        # 計算標籤差異
        target_diff = target_i - target_j
        
        # 僅考慮標籤有差異的樣本對
        mask = torch.eye(batch_size, device=predictions.device) == 0
        mask = mask & (target_diff != 0)
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        # 標籤差異的符號作為排序方向
        sign = torch.sign(target_diff[mask])
        
        # 預測差異
        pred_diff = pred_i[mask] - pred_j[mask]
        
        # 計算邊際排序損失
        losses = torch.relu(self.margin - sign * pred_diff)
        
        # 根據指定方式處理損失
        if self.reduction == 'none':
            return losses
        elif self.reduction == 'mean':
            return losses.mean()
        else:  # sum
            return losses.sum()


class RankNetLoss(nn.Module):
    """
    RankNet損失函數
    
    使用交叉熵計算樣本對之間的排序損失，適合直接優化排序結果
    """
    
    def __init__(self, sigma: float = 1.0, reduction: str = 'mean'):
        """
        初始化
        
        Args:
            sigma: sigmoid函數的縮放因子
            reduction: 損失計算方式，可為'none', 'mean', 'sum'
        """
        super().__init__()
        self.sigma = sigma
        self.reduction = reduction
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向計算
        
        Args:
            predictions: 模型預測分數，形狀為 [batch_size]
            targets: 目標分數/標籤，形狀為 [batch_size]
            
        Returns:
            計算得到的損失值
        """
        # 生成所有可能的對比樣本對
        batch_size = predictions.size(0)
        pred_i = predictions.unsqueeze(1).expand(batch_size, batch_size)
        pred_j = predictions.unsqueeze(0).expand(batch_size, batch_size)
        
        target_i = targets.unsqueeze(1).expand(batch_size, batch_size)
        target_j = targets.unsqueeze(0).expand(batch_size, batch_size)
        
        # 計算標籤差異
        target_diff = target_i - target_j
        
        # 僅考慮標籤有差異的樣本對
        mask = torch.eye(batch_size, device=predictions.device) == 0
        mask = mask & (target_diff != 0)
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        # 標籤差異的符號，轉換為目標概率
        # 如果 target_i > target_j, 則 P_ij = 1
        # 如果 target_i < target_j, 則 P_ij = 0
        target_prob = (torch.sign(target_diff[mask]) + 1) / 2
        
        # 預測差異
        pred_diff = self.sigma * (pred_i[mask] - pred_j[mask])
        
        # 使用sigmoid函數計算預測概率
        pred_prob = torch.sigmoid(pred_diff)
        
        # 計算交叉熵損失
        losses = F.binary_cross_entropy(pred_prob, target_prob)
        
        # 根據指定方式處理損失
        if self.reduction == 'none':
            return losses
        elif self.reduction == 'mean':
            return losses.mean()
        else:  # sum
            return losses.sum()


class FidelityLoss(nn.Module):
    """
    保真度(Fidelity)損失函數
    
    基於量子力學中保真度概念，計算排序樣本對之間的損失
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        初始化
        
        Args:
            reduction: 損失計算方式，可為'none', 'mean', 'sum'
        """
        super().__init__()
        self.reduction = reduction
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向計算
        
        Args:
            predictions: 模型預測分數，形狀為 [batch_size]
            targets: 目標分數/標籤，形狀為 [batch_size]
            
        Returns:
            計算得到的損失值
        """
        # 生成所有可能的對比樣本對
        batch_size = predictions.size(0)
        pred_i = predictions.unsqueeze(1).expand(batch_size, batch_size)
        pred_j = predictions.unsqueeze(0).expand(batch_size, batch_size)
        
        target_i = targets.unsqueeze(1).expand(batch_size, batch_size)
        target_j = targets.unsqueeze(0).expand(batch_size, batch_size)
        
        # 計算標籤差異
        target_diff = target_i - target_j
        
        # 僅考慮標籤有差異的樣本對
        mask = torch.eye(batch_size, device=predictions.device) == 0
        mask = mask & (target_diff != 0)
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        # 標籤差異的符號
        sign = torch.sign(target_diff[mask])
        
        # 預測差異
        pred_diff = pred_i[mask] - pred_j[mask]
        
        # 正規化差異到[-1, 1]區間
        normalized_diff = torch.tanh(pred_diff)
        
        # 當sign為正時，我們希望normalized_diff接近1；為負時，接近-1
        # 使用保真度損失: 1 - cos^2(pi/4 - pi/4 * pred_diff * sign)
        angle = torch.tensor(np.pi/4, device=predictions.device) - torch.tensor(np.pi/4, device=predictions.device) * normalized_diff * sign
        losses = 1 - torch.cos(angle)**2
        
        # 根據指定方式處理損失
        if self.reduction == 'none':
            return losses
        elif self.reduction == 'mean':
            return losses.mean()
        else:  # sum
            return losses.sum()


# ===== 列表排序損失函數 =====

class ListMLELoss(nn.Module):
    """
    ListMLE(Maximum Likelihood Estimation)損失函數
    
    基於Plackett-Luce模型，優化排序列表的最大似然估計
    """
    
    def __init__(self, eps: float = 1e-10):
        """
        初始化
        
        Args:
            eps: 用於數值穩定性的小常數
        """
        super().__init__()
        self.eps = eps
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向計算
        
        Args:
            predictions: 模型預測分數，形狀為 [batch_size, list_size]
            targets: 目標分數/標籤，形狀為 [batch_size, list_size]
            
        Returns:
            計算得到的損失值
        """
        # 獲取每個樣本的排序索引
        # 對targets降序排列，獲取原始索引作為理想排序
        _, indices = targets.sort(descending=True, dim=1)
        
        batch_size, list_size = predictions.size()
        
        # 根據indices重新排列predictions，方便後續計算
        batch_indices = torch.arange(batch_size, device=predictions.device).unsqueeze(1).expand(-1, list_size)
        gather_indices = torch.stack([batch_indices, indices], dim=2)
        
        # 選取按理想排序後的預測值
        sorted_preds = torch.gather(predictions, 1, indices)
        
        # 計算ListMLE損失
        loss = torch.zeros(batch_size, device=predictions.device)
        
        for i in range(list_size):
            # 計算從位置i到末尾的所有預測值
            subset_preds = sorted_preds[:, i:]
            
            # 對當前位置i的預測值進行softmax處理
            # 公式: exp(s_i) / sum(exp(s_j)) for j >= i
            numerator = torch.exp(sorted_preds[:, i])
            denominator = torch.sum(torch.exp(subset_preds), dim=1)
            
            # 計算概率並累加對數似然
            P_i = numerator / (denominator + self.eps)
            loss -= torch.log(P_i + self.eps)
        
        return loss.mean()


class ApproxNDCGLoss(nn.Module):
    """
    近似NDCG(Normalized Discounted Cumulative Gain)損失函數
    
    直接優化NDCG評估指標，使用可微分近似
    """
    
    def __init__(self, temperature: float = 1.0, eps: float = 1e-10):
        """
        初始化
        
        Args:
            temperature: softmax的溫度參數，控制近似排序的平滑程度
            eps: 用於數值穩定性的小常數
        """
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向計算
        
        Args:
            predictions: 模型預測分數，形狀為 [batch_size, list_size]
            targets: 目標分數/標籤，形狀為 [batch_size, list_size]
            
        Returns:
            計算得到的損失值
        """
        # 計算理想DCG
        batch_size, list_size = predictions.size()
        
        # 對每個位置計算折扣係數 1/log2(i+2)
        positions = torch.arange(1, list_size + 1, device=predictions.device).float()
        discounts = 1.0 / torch.log2(positions + 1)
        
        # 計算理想的DCG：按targets降序排列後與折扣相乘
        # 假設targets為相關性分數，通常進行如下轉換：2^rel - 1
        gains = (2 ** targets) - 1.0
        _, indices = targets.sort(descending=True, dim=1)
        batch_indices = torch.arange(batch_size, device=predictions.device).unsqueeze(1).expand(-1, list_size)
        ordered_gains = torch.gather(gains, 1, indices)
        ideal_dcg = torch.sum(ordered_gains * discounts, dim=1)
        
        # 計算預測的近似DCG
        # 使用softmax近似排序，獲得每個項目的軟排名
        pred_scaled = predictions / self.temperature
        
        # 計算每個項目的軟排名
        pred_i = pred_scaled.unsqueeze(2).expand(-1, -1, list_size)
        pred_j = pred_scaled.unsqueeze(1).expand(-1, list_size, -1)
        
        # 對於每個項目i，計算比它分數低的項目j的數量
        # 使用sigmoid函數進行平滑近似
        soft_comparison = torch.sigmoid(pred_i - pred_j)
        # 移除自身比較
        mask = ~torch.eye(list_size, dtype=torch.bool, device=predictions.device).unsqueeze(0)
        soft_comparison = soft_comparison * mask.float()
        
        # 軟排名 = 1 + 比自己低分的數量
        soft_ranks = 1.0 + torch.sum(soft_comparison, dim=2)
        
        # 計算折扣：1/log2(soft_rank + 1)
        soft_discounts = 1.0 / torch.log2(soft_ranks + 1.0)
        
        # 計算近似DCG
        approx_dcg = torch.sum(gains * soft_discounts, dim=1)
        
        # 計算近似NDCG
        approx_ndcg = approx_dcg / (ideal_dcg + self.eps)
        
        # 損失 = 1 - NDCG (使其最小化)
        loss = 1.0 - approx_ndcg
        
        return loss.mean()


class LambdaRankLoss(nn.Module):
    """
    LambdaRank損失函數
    
    通過重新加權成對比較來間接優化NDCG等排序評估指標
    """
    
    def __init__(self, sigma: float = 1.0, k: Optional[int] = None):
        """
        初始化
        
        Args:
            sigma: sigmoid函數的縮放因子
            k: 截斷的排名位置，即NDCG@k，None表示使用全部列表
        """
        super().__init__()
        self.sigma = sigma
        self.k = k
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向計算
        
        Args:
            predictions: 模型預測分數，形狀為 [batch_size, list_size]
            targets: 目標分數/標籤，形狀為 [batch_size, list_size]
            
        Returns:
            計算得到的損失值
        """
        batch_size, list_size = predictions.size()
        k = self.k if self.k is not None else list_size
        k = min(k, list_size)
        
        # 計算每一對項目交換后的NDCG變化
        # 首先計算折扣係數
        positions = torch.arange(1, k + 1, device=predictions.device).float()
        discounts = 1.0 / torch.log2(positions + 1)
        
        # 計算增益(2^rel - 1)
        gains = (2 ** targets) - 1.0
        
        # 對於每一個batch
        total_loss = torch.tensor(0.0, device=predictions.device)
        
        for b in range(batch_size):
            # 計算當前排序下的DCG
            current_pred = predictions[b]
            current_gains = gains[b]
            
            # 對預測進行排序，得到原始的排序索引
            _, indices = current_pred.sort(descending=True)
            
            # 計算當前的DCG
            sorted_gains = current_gains[indices[:k]]
            current_dcg = torch.sum(sorted_gains * discounts[:sorted_gains.size(0)])
            
            # 計算理想的DCG
            ideal_gains, _ = torch.sort(current_gains, descending=True)
            ideal_gains = ideal_gains[:k]
            ideal_dcg = torch.sum(ideal_gains * discounts[:ideal_gains.size(0)])
            
            # 如果理想DCG為0，跳過當前batch
            if ideal_dcg == 0:
                continue
                
            # 計算成對交換的NDCG變化
            lambdas = torch.zeros_like(current_pred)
            
            for i in range(list_size):
                for j in range(list_size):
                    if i == j or current_gains[i] == current_gains[j]:
                        continue
                    
                    # 僅當兩項的相關性分數不同時，才計算NDCG變化
                    gain_diff = current_gains[i] - current_gains[j]
                    
                    # 如果預測順序與理想順序不一致
                    if (gain_diff > 0 and current_pred[i] < current_pred[j]) or \
                       (gain_diff < 0 and current_pred[i] > current_pred[j]):
                       
                        # 複製一份當前索引，並模擬i和j的交換
                        new_indices = indices.clone()
                        i_pos = torch.where(indices == i)[0]
                        j_pos = torch.where(indices == j)[0]
                        
                        # 僅當至少有一個在top-k內時才計算NDCG變化
                        if i_pos < k or j_pos < k:
                            # 交換位置
                            i_pos_val = i_pos.clone()
                            new_indices[i_pos] = j
                            new_indices[j_pos] = i
                            
                            # 計算新的DCG
                            new_sorted_gains = current_gains[new_indices[:k]]
                            new_dcg = torch.sum(new_sorted_gains * discounts[:new_sorted_gains.size(0)])
                            
                            # 計算NDCG變化的絕對值
                            delta_ndcg = torch.abs(new_dcg / ideal_dcg - current_dcg / ideal_dcg)
                            
                            # RankNet梯度，乘以NDCG變化作為權重
                            sig = torch.sigmoid(self.sigma * (current_pred[i] - current_pred[j]))
                            rho = -self.sigma * delta_ndcg
                            
                            if gain_diff > 0:
                                lambdas[i] += rho * (1 - sig)
                                lambdas[j] += rho * sig
                            else:
                                lambdas[i] += rho * sig
                                lambdas[j] += rho * (1 - sig)
            
            # 這裡我們通過直接使用lambdas計算損失
            # 實際上LambdaRank沒有顯式損失函數，僅使用梯度來更新模型
            # 為了與PyTorch整合，我們構造一個虛擬損失，確保其梯度等於lambdas
            pseudo_loss = torch.sum(current_pred * lambdas)
            total_loss += pseudo_loss
            
        return total_loss / batch_size


# ===== 組合損失函數 =====

class AdaptiveWeightedLoss(nn.Module):
    """
    自適應權重的組合損失函數
    
    結合多個損失函數，並根據訓練過程自動調整權重
    """
    
    def __init__(self, losses: List[nn.Module], initial_weights: Optional[List[float]] = None,
                 method: str = 'grad_norm', adapt_freq: int = 10):
        """
        初始化
        
        Args:
            losses: 損失函數列表
            initial_weights: 初始權重列表，None表示均等權重
            method: 權重調整方法，可為'grad_norm', 'loss_ratio', 'uncertainty'
            adapt_freq: 權重調整頻率(每隔多少個batch調整一次)
        """
        super().__init__()
        self.losses = nn.ModuleList(losses)
        n_losses = len(losses)
        
        # 初始化權重
        if initial_weights is None:
            self.weights = torch.ones(n_losses) / n_losses
        else:
            self.weights = torch.tensor(initial_weights)
            self.weights = self.weights / self.weights.sum()  # 標準化
            
        self.method = method
        self.adapt_freq = adapt_freq
        self.iteration = 0
        
        # 用於儲存初始損失值和梯度範數
        self.initial_losses = None
        self.last_grads = None
        
        # 如果使用uncertainty方法，需要參數化權重的對數
        if method == 'uncertainty':
            self.log_weights = nn.Parameter(torch.log(self.weights))
            
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向計算
        
        Args:
            predictions: 模型預測分數
            targets: 目標分數/標籤
            
        Returns:
            組合損失值
        """
        # 計算每個損失函數的損失值
        individual_losses = [loss(predictions, targets) for loss in self.losses]
        
        # 更新迭代計數器
        self.iteration += 1
        
        # 儲存初始損失值
        if self.initial_losses is None:
            self.initial_losses = torch.tensor(individual_losses)
            
        # 根據選定的方法調整權重
        if self.iteration % self.adapt_freq == 0:
            if self.method == 'grad_norm':
                # 基於梯度範數的調整需要手動實現反向傳播以獲取梯度
                # 這裡簡化實現，實際應用中需要在訓練迴圈中獲取梯度
                if hasattr(predictions, 'grad') and predictions.grad is not None:
                    grads = []
                    for loss in individual_losses:
                        predictions.grad.zero_()
                        loss.backward(retain_graph=True)
                        grads.append(predictions.grad.norm().item())
                    
                    # 梯度範數越大，權重越小
                    grads = torch.tensor(grads)
                    if self.last_grads is None:
                        self.last_grads = grads
                    else:
                        # 平滑更新
                        alpha = 0.9
                        grads = alpha * self.last_grads + (1 - alpha) * grads
                        self.last_grads = grads
                    
                    # 避免除零錯誤
                    grads = grads + 1e-10
                    
                    # 更新權重: w_i ∝ 1/||∇L_i||
                    inv_grads = 1.0 / grads
                    self.weights = inv_grads / inv_grads.sum()
            
            elif self.method == 'loss_ratio':
                # 根據損失值相對於初始值的比例調整權重
                # 損失下降較少的，權重增大
                current_losses = torch.tensor(individual_losses)
                loss_ratios = current_losses / self.initial_losses
                
                # 避免除零錯誤
                loss_ratios = loss_ratios + 1e-10
                
                # 更新權重: w_i ∝ L_i(t)/L_i(0)
                self.weights = loss_ratios / loss_ratios.sum()
                
            # uncertainty方法無需顯式調整，因為權重作為模型參數會通過梯度下降自動更新
                
        # 根據當前權重計算加權損失
        if self.method == 'uncertainty':
            # 不確定性方法使用可訓練的權重
            weights = F.softmax(self.log_weights, dim=0)
            # 損失為加權和加上權重正則化項
            combined_loss = sum(w * l for w, l in zip(weights, individual_losses))
            # 添加一個正則化項，避免某個任務權重過度主導
            combined_loss += 0.5 * sum(w for w in weights)
        else:
            # 其他方法直接使用當前權重
            combined_loss = sum(w * l for w, l in zip(self.weights, individual_losses))
            
        return combined_loss


# ===== 排序模型示例 =====

class RankingModel(nn.Module):
    """
    排序模型示例
    
    用於吞嚥障礙評估的排序模型，輸入為患者特徵，輸出為嚴重程度評分
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32, 16]):
        """
        初始化
        
        Args:
            input_dim: 輸入特徵維度
            hidden_dims: 隱藏層維度列表
        """
        super().__init__()
        
        # 構建多層感知機
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
            
        # 最後的輸出層，輸出嚴重程度評分
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向計算
        
        Args:
            x: 輸入特徵，形狀為 [batch_size, input_dim]
            
        Returns:
            預測的嚴重程度評分，形狀為 [batch_size, 1]
        """
        return self.model(x).squeeze(-1)


# ===== 實際應用示例 =====

def dysphagia_ranking_example():
    """
    吞嚥障礙排序評估示例
    
    展示如何訓練一個排序模型用於吞嚥障礙評估
    """
    # 假設我們有吞嚥障礙患者的特徵和對應的嚴重程度標籤
    # 特徵可能包括：年齡、性別、病史、症狀評分、語音特徵等
    batch_size = 32
    feature_dim = 20
    
    # 生成模擬數據
    np.random.seed(42)
    X_train = torch.rand(100, feature_dim)
    y_train = torch.randint(0, 5, (100,)).float()  # 嚴重程度從0到4
    
    # 創建數據加載器
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 創建排序模型
    model = RankingModel(input_dim=feature_dim)
    
    # 選擇損失函數
    # 在這個例子中，我們使用組合損失函數，結合成對和列表方法
    pairwise_loss = RankNetLoss()
    listwise_loss = ListMLELoss()
    lambda_loss = LambdaRankLoss(k=10)  # 優化NDCG@10
    
    # 創建自適應權重的組合損失
    combined_loss = AdaptiveWeightedLoss(
        losses=[pairwise_loss, listwise_loss, lambda_loss],
        initial_weights=[0.4, 0.3, 0.3],
        method='loss_ratio'  # 根據損失下降比例自動調整權重
    )
    
    # 創建優化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 訓練模型
    epochs = 5
    print("開始訓練排序模型...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            
            # 前向傳播
            predictions = model(batch_X)
            
            # 計算損失
            loss = combined_loss(predictions, batch_y)
            
            # 反向傳播
            loss.backward()
            
            # 更新參數
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # 打印訓練進度
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    print("訓練完成！")
    
    # 模型評估和結果分析
    model.eval()
    
    # 生成測試數據
    X_test = torch.rand(20, feature_dim)
    y_test = torch.randint(0, 5, (20,)).float()
    
    # 預測嚴重程度
    with torch.no_grad():
        predictions = model(X_test)
    
    # 計算評估指標
    # 這裡我們計算NDCG@5和NDCG@10
    def compute_ndcg(preds, labels, k):
        """計算NDCG@k"""
        batch_size = preds.size(0)
        
        # 對每個位置計算折扣係數
        positions = torch.arange(1, k + 1, device=preds.device).float()
        discounts = 1.0 / torch.log2(positions + 1)
        
        ndcg_sum = 0.0
        
        for i in range(batch_size):
            # 獲取當前樣本的預測和標籤
            pred = preds[i]
            label = labels[i]
            
            # 按預測值排序的索引
            _, pred_indices = pred.sort(descending=True)
            pred_indices = pred_indices[:k]
            
            # 按標籤值排序的索引
            _, label_indices = label.sort(descending=True)
            label_indices = label_indices[:k]
            
            # 計算DCG
            pred_labels = label[pred_indices]
            gains = (2 ** pred_labels) - 1
            dcg = torch.sum(gains * discounts[:len(gains)])
            
            # 計算理想DCG
            ideal_labels = label[label_indices]
            ideal_gains = (2 ** ideal_labels) - 1
            idcg = torch.sum(ideal_gains * discounts[:len(ideal_gains)])
            
            # 防止除零錯誤
            if idcg > 0:
                ndcg = dcg / idcg
            else:
                ndcg = 0.0
                
            ndcg_sum += ndcg
            
        return ndcg_sum / batch_size
    
    # 使用原始的y_test是單個標籤，為了計算NDCG，我們生成多個項目
    # 在實際應用中，我們會有多個項目和對應的標籤
    expanded_y_test = y_test.unsqueeze(1).expand(-1, 10)
    expanded_preds = predictions.unsqueeze(1).expand(-1, 10)
    
    # 添加一些隨機性以模擬多個項目
    noise = torch.rand_like(expanded_y_test) * 0.5 - 0.25
    expanded_y_test = expanded_y_test + noise
    expanded_y_test = torch.clamp(expanded_y_test, 0, 4)  # 確保標籤在合理範圍內
    
    # 再次添加預測的隨機性
    pred_noise = torch.rand_like(expanded_preds) * 0.3 - 0.15
    expanded_preds = expanded_preds + pred_noise
    
    # 計算NDCG@5和NDCG@10
    ndcg_5 = compute_ndcg(expanded_preds, expanded_y_test, 5)
    ndcg_10 = compute_ndcg(expanded_preds, expanded_y_test, 10)
    
    print(f"NDCG@5: {ndcg_5:.4f}")
    print(f"NDCG@10: {ndcg_10:.4f}")
    
    # 繪製前5個樣本的真實值與預測值比較
    plt.figure(figsize=(10, 6))
    x = np.arange(5)
    width = 0.35
    
    plt.bar(x - width/2, y_test[:5].numpy(), width, label='真實嚴重程度')
    plt.bar(x + width/2, predictions[:5].numpy(), width, label='預測嚴重程度')
    
    plt.xlabel('患者編號')
    plt.ylabel('嚴重程度評分')
    plt.title('吞嚥障礙評估模型預測結果')
    plt.xticks(x)
    plt.legend()
    plt.tight_layout()
    
    # 在實際應用中，可以保存圖表
    # plt.savefig('dysphagia_results.png')
    # plt.show()
    
    return model


if __name__ == "__main__":
    print("排序損失函數實際應用示例")
    model = dysphagia_ranking_example() 