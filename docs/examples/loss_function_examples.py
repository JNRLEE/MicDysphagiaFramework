"""
這個文件提供了MicDysphagiaFramework中核心排序損失函數的實現範例。
這些函數展示了如何實現各種排序學習損失函數，以及如何將它們組合使用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PairwiseRankingLoss(nn.Module):
    """
    成對排序損失函數，基於邊際排序損失原理。
    適用於需要學習相對順序的場景，如按嚴重程度排序吞嚥障礙病例。
    """
    def __init__(self, margin=1.0, exp_weight=False, sampling='random', sampling_rate=0.5):
        """
        初始化PairwiseRankingLoss
        
        Args:
            margin (float): 預測值之間的最小邊距
            exp_weight (bool): 是否使用指數權重強調真實差距大的樣本對
            sampling (str): 採樣策略，可選'random', 'score_diff', 'hard_negative'
            sampling_rate (float): 採樣比例，控制使用多少樣本對進行訓練
        """
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin
        self.exp_weight = exp_weight
        self.sampling = sampling
        self.sampling_rate = sampling_rate
    
    def forward(self, predictions, targets):
        """
        計算排序損失
        
        Args:
            predictions (torch.Tensor): 模型預測值，形狀為 [batch_size]
            targets (torch.Tensor): 真實標籤值，形狀為 [batch_size]
            
        Returns:
            torch.Tensor: 排序損失值
        """
        batch_size = predictions.size(0)
        
        # 構建所有可能的樣本對
        i_idx, j_idx = [], []
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    i_idx.append(i)
                    j_idx.append(j)
        
        i_idx = torch.tensor(i_idx, device=predictions.device)
        j_idx = torch.tensor(j_idx, device=predictions.device)
        
        # 採樣策略
        if self.sampling_rate < 1.0:
            num_pairs = len(i_idx)
            sample_size = max(1, int(num_pairs * self.sampling_rate))
            
            if self.sampling == 'random':
                # 隨機採樣
                indices = torch.randperm(num_pairs)[:sample_size]
            
            elif self.sampling == 'score_diff':
                # 按真實分數差異採樣
                target_diff = torch.abs(targets[i_idx] - targets[j_idx])
                _, indices = torch.sort(target_diff, descending=True)
                indices = indices[:sample_size]
            
            elif self.sampling == 'hard_negative':
                # 硬負例採樣（預測難度大的樣本對）
                pred_i, pred_j = predictions[i_idx], predictions[j_idx]
                target_i, target_j = targets[i_idx], targets[j_idx]
                
                # 計算損失函數
                target_diff = target_i - target_j
                pred_diff = pred_i - pred_j
                
                sign = torch.sign(target_diff)
                raw_loss = torch.relu(-sign * pred_diff + self.margin)
                
                _, indices = torch.sort(raw_loss, descending=True)
                indices = indices[:sample_size]
            
            i_idx = i_idx[indices]
            j_idx = j_idx[indices]
        
        # 獲取選定對的預測值和真實值
        pred_i, pred_j = predictions[i_idx], predictions[j_idx]
        target_i, target_j = targets[i_idx], targets[j_idx]
        
        # 計算排序方向
        target_diff = target_i - target_j
        sign = torch.sign(target_diff)
        
        # 計算邊際排序損失
        pred_diff = pred_i - pred_j
        losses = torch.relu(-sign * pred_diff + self.margin)
        
        # 應用指數權重（可選）
        if self.exp_weight:
            weight = torch.exp(torch.abs(target_diff))
            losses = losses * weight
        
        return losses.mean()


class ListMLELoss(nn.Module):
    """
    ListMLE（最大似然估計）損失函數，用於優化整個排序列表。
    """
    def __init__(self, temperature=1.0):
        """
        初始化ListMLELoss
        
        Args:
            temperature (float): 控制概率分布平滑程度的溫度參數
        """
        super(ListMLELoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, predictions, targets):
        """
        計算ListMLE損失
        
        Args:
            predictions (torch.Tensor): 模型預測值，形狀為 [batch_size]
            targets (torch.Tensor): 真實標籤值，形狀為 [batch_size]
            
        Returns:
            torch.Tensor: ListMLE損失值
        """
        # 按真實標籤值排序獲取索引
        _, indices = torch.sort(targets, descending=True)
        
        # 整理預測值按照索引排序
        sorted_preds = predictions[indices]
        batch_size = predictions.size(0)
        
        # 計算ListMLE損失
        loss = 0
        for i in range(batch_size):
            # 當前位置及之後的預測值
            remaining_preds = sorted_preds[i:]
            
            # 對剩餘預測值應用softmax，計算當前位置的概率
            scaled_preds = remaining_preds / self.temperature
            probs = F.softmax(scaled_preds, dim=0)
            
            # 累加負對數似然
            loss -= torch.log(probs[0] + 1e-10)
        
        return loss / batch_size


class ApproxNDCGLoss(nn.Module):
    """
    ApproxNDCG損失函數，直接優化NDCG評估指標。
    通過可微近似實現NDCG的直接優化。
    """
    def __init__(self, temperature=1.0, k=None):
        """
        初始化ApproxNDCGLoss
        
        Args:
            temperature (float): 控制概率分布平滑程度的溫度參數
            k (int, optional): 計算NDCG@k，若為None則使用所有項目
        """
        super(ApproxNDCGLoss, self).__init__()
        self.temperature = temperature
        self.k = k
    
    def dcg(self, scores, ranks):
        """計算DCG (Discounted Cumulative Gain)"""
        gains = torch.pow(2, scores) - 1
        discounts = torch.log2(ranks + 1)
        return (gains / discounts).sum()
    
    def forward(self, predictions, targets):
        """
        計算ApproxNDCG損失
        
        Args:
            predictions (torch.Tensor): 模型預測值，形狀為 [batch_size]
            targets (torch.Tensor): 真實標籤值，形狀為 [batch_size]
            
        Returns:
            torch.Tensor: 1 - ApproxNDCG值，使其作為最小化損失函數
        """
        batch_size = predictions.size(0)
        k = self.k if self.k is not None else batch_size
        k = min(k, batch_size)
        
        # 計算排序概率矩陣 P(i排在位置j)
        scaled_preds = predictions / self.temperature
        P = torch.zeros((batch_size, batch_size), device=predictions.device)
        
        for i in range(batch_size):
            # 計算項目i排在各位置的概率
            s = torch.zeros(batch_size, device=predictions.device)
            s[i] = 1
            P[i] = F.softmax(scaled_preds * s, dim=0)
        
        # 計算期望DCG
        ranks = torch.arange(1, batch_size + 1, device=predictions.device).float()
        discount = 1. / torch.log2(ranks + 1)
        
        # 限制為前k個位置
        if k < batch_size:
            discount = discount[:k]
        
        # 計算所有可能排序的DCG期望
        gains = torch.pow(2, targets) - 1
        approx_dcg = 0
        
        for i in range(batch_size):
            # 計算項目i對DCG的貢獻
            if k < batch_size:
                contribution = (P[i, :k] * discount).sum() * gains[i]
            else:
                contribution = (P[i] * discount).sum() * gains[i]
            approx_dcg += contribution
        
        # 計算理想DCG (IDCG)
        _, indices = torch.sort(targets, descending=True)
        sorted_gains = gains[indices]
        idcg = (sorted_gains[:k] / discount[:k]).sum() if k < batch_size else (sorted_gains / discount).sum()
        
        # 計算NDCG，轉換為損失函數（1-NDCG）
        approx_ndcg = approx_dcg / (idcg + 1e-10)
        return 1 - approx_ndcg


class LambdaRankLoss(nn.Module):
    """
    LambdaRank損失函數，通過重新加權成對損失優化排序評估指標。
    """
    def __init__(self, sigma=1.0, k=10):
        """
        初始化LambdaRankLoss
        
        Args:
            sigma (float): 控制sigmoid函數斜率的參數
            k (int): 計算NDCG@k
        """
        super(LambdaRankLoss, self).__init__()
        self.sigma = sigma
        self.k = k
    
    def compute_delta_ndcg(self, targets, i, j):
        """計算交換項目i和j導致的NDCG變化"""
        batch_size = targets.size(0)
        
        # 創建排序
        sorted_idx = torch.argsort(targets, descending=True)
        ideal_dcg = self.compute_dcg(targets, sorted_idx)
        
        # 計算原始DCG
        rank = torch.zeros(batch_size, dtype=torch.long, device=targets.device)
        rank[sorted_idx] = torch.arange(batch_size, device=targets.device)
        
        # 交換i和j的位置
        swapped_rank = rank.clone()
        swapped_rank[i], swapped_rank[j] = rank[j], rank[i]
        
        # 計算交換後的DCG
        swapped_sorted_idx = torch.zeros(batch_size, dtype=torch.long, device=targets.device)
        swapped_sorted_idx[swapped_rank] = torch.arange(batch_size, device=targets.device)
        swapped_dcg = self.compute_dcg(targets, swapped_sorted_idx)
        
        # 計算NDCG變化
        delta = torch.abs((swapped_dcg - ideal_dcg) / ideal_dcg)
        return delta
    
    def compute_dcg(self, targets, indices):
        """計算DCG值"""
        k = min(self.k, targets.size(0))
        topk_idx = indices[:k]
        gains = (2 ** targets[topk_idx]) - 1
        discounts = torch.log2(torch.arange(1, k + 1, dtype=torch.float, device=targets.device) + 1)
        return (gains / discounts).sum()
    
    def forward(self, predictions, targets):
        """
        計算LambdaRank損失
        
        Args:
            predictions (torch.Tensor): 模型預測值，形狀為 [batch_size]
            targets (torch.Tensor): 真實標籤值，形狀為 [batch_size]
            
        Returns:
            torch.Tensor: LambdaRank損失值
        """
        batch_size = predictions.size(0)
        
        # 構建所有可能的樣本對
        i_idx, j_idx = [], []
        for i in range(batch_size):
            for j in range(batch_size):
                if targets[i] > targets[j]:  # 只關注不同標籤的樣本對
                    i_idx.append(i)
                    j_idx.append(j)
        
        if not i_idx:  # 沒有合適的樣本對
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        i_idx = torch.tensor(i_idx, device=predictions.device)
        j_idx = torch.tensor(j_idx, device=predictions.device)
        
        # 計算每對的NDCG增益
        delta_ndcg = torch.zeros(len(i_idx), device=predictions.device)
        for idx in range(len(i_idx)):
            i, j = i_idx[idx].item(), j_idx[idx].item()
            delta_ndcg[idx] = self.compute_delta_ndcg(targets, i, j)
        
        # 獲取預測值
        s_i = predictions[i_idx]
        s_j = predictions[j_idx]
        
        # 計算RankNet概率和梯度權重
        pij = 1 / (1 + torch.exp(-self.sigma * (s_i - s_j)))
        lambda_ij = self.sigma * (1 - pij) * delta_ndcg
        
        # 計算損失
        loss = -torch.sum(torch.log(pij + 1e-10) * delta_ndcg)
        return loss / len(i_idx)


class CombinedLoss(nn.Module):
    """
    組合損失函數，支持多個損失函數的加權組合。
    可以實現自適應權重調整。
    """
    def __init__(self, losses, weights=None, adaptive=False, update_rate=0.1):
        """
        初始化CombinedLoss
        
        Args:
            losses (list): 損失函數列表
            weights (list, optional): 損失函數權重列表，必須與losses長度相同且和為1
            adaptive (bool): 是否使用自適應權重調整
            update_rate (float): 權重更新率
        """
        super(CombinedLoss, self).__init__()
        self.losses = nn.ModuleList(losses)
        
        if weights is None:
            n = len(losses)
            self.weights = torch.ones(n) / n
        else:
            assert len(weights) == len(losses), "權重數量必須與損失函數數量相同"
            assert abs(sum(weights) - 1.0) < 1e-6, "權重總和必須為1"
            self.weights = torch.tensor(weights)
        
        self.adaptive = adaptive
        self.update_rate = update_rate
        self.initial_losses = None
    
    def forward(self, predictions, targets):
        """
        計算組合損失
        
        Args:
            predictions (torch.Tensor): 模型預測值
            targets (torch.Tensor): 真實標籤值
            
        Returns:
            torch.Tensor: 組合損失值
        """
        # 計算每個損失函數的損失值
        individual_losses = []
        for loss_fn in self.losses:
            loss_val = loss_fn(predictions, targets)
            individual_losses.append(loss_val)
        
        # 記錄初始損失值，用於自適應權重調整
        if self.adaptive and self.initial_losses is None:
            self.initial_losses = [l.detach() for l in individual_losses]
        
        # 自適應權重調整
        if self.adaptive and self.initial_losses is not None:
            with torch.no_grad():
                # 計算損失比率
                ratios = [l / init_l for l, init_l in zip(individual_losses, self.initial_losses)]
                
                # 計算新權重
                ratio_sum = sum(ratios)
                new_weights = [(r / ratio_sum) for r in ratios]
                
                # 更新權重
                for i in range(len(self.weights)):
                    self.weights[i] = (1 - self.update_rate) * self.weights[i] + self.update_rate * new_weights[i]
        
        # 計算加權組合損失
        combined_loss = 0
        for i, loss in enumerate(individual_losses):
            combined_loss += self.weights[i] * loss
        
        return combined_loss


# 使用示例
def loss_function_example():
    # 創建隨機示例數據
    predictions = torch.randn(10)
    targets = torch.randint(0, 5, (10,)).float()
    
    # PairwiseRanking損失
    pairwise_loss = PairwiseRankingLoss(
        margin=0.5, 
        exp_weight=True,
        sampling='score_diff',
        sampling_rate=0.3
    )
    pairwise_result = pairwise_loss(predictions, targets)
    print(f"PairwiseRanking損失: {pairwise_result.item():.4f}")
    
    # ListMLE損失
    listmle_loss = ListMLELoss(temperature=0.5)
    listmle_result = listmle_loss(predictions, targets)
    print(f"ListMLE損失: {listmle_result.item():.4f}")
    
    # ApproxNDCG損失
    approx_ndcg_loss = ApproxNDCGLoss(temperature=0.1, k=5)
    approx_ndcg_result = approx_ndcg_loss(predictions, targets)
    print(f"ApproxNDCG損失: {approx_ndcg_result.item():.4f}")
    
    # LambdaRank損失
    lambda_rank_loss = LambdaRankLoss(sigma=1.0, k=5)
    lambda_rank_result = lambda_rank_loss(predictions, targets)
    print(f"LambdaRank損失: {lambda_rank_result.item():.4f}")
    
    # 組合損失
    combined_loss = CombinedLoss(
        losses=[pairwise_loss, listmle_loss, approx_ndcg_loss],
        weights=[0.5, 0.3, 0.2],
        adaptive=True,
        update_rate=0.1
    )
    combined_result = combined_loss(predictions, targets)
    print(f"組合損失: {combined_result.item():.4f}")
    print(f"自適應調整後的權重: {combined_loss.weights}")


if __name__ == "__main__":
    loss_function_example() 