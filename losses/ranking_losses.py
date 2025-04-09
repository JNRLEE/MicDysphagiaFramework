"""
此模組實現了用於排序學習的損失函數，包括pairwise和listwise排序損失。
這些損失函數特別適用於有分數指標的數據排序任務。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

from .loss_factory import LossFactory


@LossFactory.register_loss("PairwiseRankingLoss")
class PairwiseRankingLoss(nn.Module):
    """
    成對排序損失函數，對於有分數指標的樣本對執行成對比較。
    支持隨機採樣樣本對以增加訓練多樣性。
    
    主要思想是讓模型學習對兩個樣本進行相對排序，而不是絕對分數。
    """
    
    def __init__(self, 
                 margin: float = 0.0, 
                 sampling_ratio: float = 0.25,
                 sampling_strategy: str = 'score_diff', 
                 use_exp: bool = False):
        """
        初始化PairwiseRankingLoss
        
        Args:
            margin: 排序間隔，預測分數之間的最小差距要求
            sampling_ratio: 從所有可能的對中採樣的比例
            sampling_strategy: 採樣策略，可以是 'random'、'score_diff' 或 'hard_negative'
            use_exp: 是否使用指數加權差異
        """
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin
        self.sampling_ratio = sampling_ratio
        self.sampling_strategy = sampling_strategy
        self.use_exp = use_exp
        
        # 內部排序損失使用MarginRankingLoss
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    
    def sample_pairs(self, 
                     scores: torch.Tensor, 
                     predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        從批次中採樣對比用的樣本對
        
        Args:
            scores: 真實分數，形狀為 [batch_size]
            predictions: 預測分數，形狀為 [batch_size]
            
        Returns:
            pred_i, pred_j, target: 兩個預測和對應的目標（1表示i優於j，-1表示j優於i）
        """
        device = scores.device
        batch_size = scores.size(0)
        
        # 計算所有可能的對
        num_total_pairs = batch_size * (batch_size - 1) // 2
        
        # 根據抽樣率確定實際使用的對數
        num_pairs = max(1, int(num_total_pairs * self.sampling_ratio))
        
        # 創建所有可能的索引對
        pairs = []
        for i in range(batch_size):
            for j in range(i+1, batch_size):
                pairs.append((i, j))
        
        # 根據策略對對進行採樣
        if self.sampling_strategy == 'random':
            # 隨機採樣
            pairs = random.sample(pairs, num_pairs)
        
        elif self.sampling_strategy == 'score_diff':
            # 基於真實分數差異的採樣，優先選擇分數差異較大的對
            diffs = []
            for i, j in pairs:
                diffs.append((abs(scores[i].item() - scores[j].item()), (i, j)))
            
            # 對差異排序並選擇前num_pairs個
            diffs.sort(reverse=True)
            pairs = [p for _, p in diffs[:num_pairs]]
            
            # 再次打亂以增加隨機性
            random.shuffle(pairs)
            
        elif self.sampling_strategy == 'hard_negative':
            # 選擇"困難"對，即預測與真實排序不符的對
            hard_pairs = []
            for i, j in pairs:
                # 檢查預測和真實是否不一致
                if (predictions[i] > predictions[j] and scores[i] < scores[j]) or \
                   (predictions[i] < predictions[j] and scores[i] > scores[j]):
                    hard_pairs.append((i, j))
            
            # 如果困難對不夠，則補充隨機對
            if len(hard_pairs) < num_pairs:
                remaining = set(pairs) - set(hard_pairs)
                hard_pairs.extend(random.sample(list(remaining), 
                                              min(num_pairs - len(hard_pairs), len(remaining))))
            else:
                # 如果困難對太多，則隨機採樣
                hard_pairs = random.sample(hard_pairs, num_pairs)
            
            pairs = hard_pairs
        
        # 創建張量
        pred_i = torch.zeros(len(pairs), device=device)
        pred_j = torch.zeros(len(pairs), device=device)
        target = torch.zeros(len(pairs), device=device)
        
        # 填充張量
        for idx, (i, j) in enumerate(pairs):
            pred_i[idx] = predictions[i]
            pred_j[idx] = predictions[j]
            
            # 設置目標：1表示i應排在j之前，-1表示j應排在i之前
            if scores[i] > scores[j]:
                target[idx] = 1.0
            elif scores[i] < scores[j]:
                target[idx] = -1.0
            else:
                # 如果分數相等，則隨機分配以增加多樣性
                target[idx] = 1.0 if random.random() > 0.5 else -1.0
        
        return pred_i, pred_j, target
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        計算成對排序損失
        
        Args:
            predictions: 模型預測值，形狀為 [batch_size, 1] 或 [batch_size]
            targets: 真實分數，形狀為 [batch_size, 1] 或 [batch_size]
            
        Returns:
            排序損失值
        """
        # 確保輸入是一維的
        if predictions.dim() > 1:
            predictions = predictions.squeeze()
        if targets.dim() > 1:
            targets = targets.squeeze()
        
        # 採樣對
        pred_i, pred_j, target = self.sample_pairs(targets, predictions)
        
        if self.use_exp:
            # 使用指數加權差異可以放大對重要對的懲罰
            return F.relu((pred_j - pred_i) * target + self.margin).mean()
        else:
            # 使用標準MarginRankingLoss
            return self.ranking_loss(pred_i, pred_j, target)


@LossFactory.register_loss("ListwiseRankingLoss")
class ListwiseRankingLoss(nn.Module):
    """
    列表排序損失函數，整個列表一起考慮而不是單獨的對。
    實現了多種列表排序方法，包括ListNet、ListMLE等。
    """
    
    def __init__(self, 
                 method: str = 'listnet', 
                 temperature: float = 1.0, 
                 k: int = 10,
                 group_size: int = 0,
                 stochastic: bool = True):
        """
        初始化ListwiseRankingLoss
        
        Args:
            method: 列表排序方法，可以是 'listnet'、'listmle' 或 'approxndcg'
            temperature: softmax溫度參數，控制對頂部項目的關注度
            k: Top-k排序損失參數，僅考慮前k個項目
            group_size: 如果>0，將batch分成group_size的子批次
            stochastic: 是否應用隨機擾動以增加多樣性
        """
        super(ListwiseRankingLoss, self).__init__()
        self.method = method
        self.temperature = temperature
        self.k = k
        self.group_size = group_size
        self.stochastic = stochastic
    
    def _apply_perturbation(self, 
                           targets: torch.Tensor, 
                           noise_level: float = 0.05) -> torch.Tensor:
        """
        應用小的隨機擾動以增加多樣性
        
        Args:
            targets: 真實分數
            noise_level: 擾動水平
            
        Returns:
            應用擾動後的目標
        """
        if self.stochastic and self.training:
            noise = torch.randn_like(targets) * noise_level * targets.abs()
            return targets + noise
        return targets
        
    def _listnet_loss(self, 
                     predictions: torch.Tensor, 
                     targets: torch.Tensor) -> torch.Tensor:
        """
        ListNet損失: 使用交叉熵比較真實和預測的概率分佈
        
        Args:
            predictions: 預測分數
            targets: 真實分數
            
        Returns:
            ListNet損失值
        """
        # 應用溫度縮放和softmax轉換為概率分佈
        pred_probs = F.softmax(predictions / self.temperature, dim=0)
        target_probs = F.softmax(targets / self.temperature, dim=0)
        
        # 使用KL散度計算分佈之間的差異
        loss = F.kl_div(pred_probs.log(), target_probs, reduction='sum')
        return loss
        
    def _listmle_loss(self, 
                     predictions: torch.Tensor, 
                     targets: torch.Tensor) -> torch.Tensor:
        """
        ListMLE (Maximum Likelihood Estimation) 損失
        
        Args:
            predictions: 預測分數
            targets: 真實分數
            
        Returns:
            ListMLE損失值
        """
        # 根據目標分數對索引進行排序
        _, target_indices = torch.sort(targets, descending=True)
        
        # 獲取預測，按照目標排序
        sorted_predictions = predictions[target_indices]
        
        # 計算損失: -log(P(π*|s))，其中π*是理想排序
        losses = F.log_softmax(sorted_predictions, dim=0)
        loss = -torch.sum(losses)
        
        return loss
        
    def _approxndcg_loss(self, 
                        predictions: torch.Tensor, 
                        targets: torch.Tensor) -> torch.Tensor:
        """
        近似NDCG損失: 學習最大化NDCG
        
        Args:
            predictions: 預測分數
            targets: 真實分數
            
        Returns:
            近似NDCG損失值
        """
        # 計算理想DCG
        _, indices = torch.sort(targets, descending=True)
        ideal_dcg = self._dcg(targets[indices])
        
        # 使用SoftMax近似排序
        approx_ranks = self._approx_ranks(predictions)
        approx_dcg = torch.sum(targets / torch.log2(approx_ranks + 1))
        
        # 損失是1減去NDCG
        loss = 1.0 - (approx_dcg / ideal_dcg)
        return loss
    
    def _dcg(self, 
            relevance: torch.Tensor, 
            k: Optional[int] = None) -> torch.Tensor:
        """
        計算DCG (Discounted Cumulative Gain)
        
        Args:
            relevance: 相關性分數
            k: 只考慮前k個項目
            
        Returns:
            DCG值
        """
        if k is None or k > relevance.size(0):
            k = relevance.size(0)
            
        # 計算位置折扣
        positions = torch.arange(1, k + 1, device=relevance.device, dtype=torch.float)
        discounts = torch.log2(positions + 1)
        
        # 只考慮前k個，並應用折扣
        dcg = torch.sum(relevance[:k] / discounts)
        return dcg
    
    def _approx_ranks(self, 
                     predictions: torch.Tensor) -> torch.Tensor:
        """
        通過softmax計算近似排名
        
        Args:
            predictions: 預測分數
            
        Returns:
            近似排名
        """
        batch_size = predictions.size(0)
        # 計算所有可能的比較結果
        diff = predictions.unsqueeze(0) - predictions.unsqueeze(1)
        # 使用sigmoid函數將差異轉換為概率
        prob = torch.sigmoid(diff)
        # 每個元素的近似排名
        approx_ranks = torch.sum(prob, dim=1) + 0.5
        return approx_ranks
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        計算列表排序損失
        
        Args:
            predictions: 模型預測值，形狀為 [batch_size, 1] 或 [batch_size]
            targets: 真實分數，形狀為 [batch_size, 1] 或 [batch_size]
            
        Returns:
            排序損失值
        """
        # 確保輸入是一維的
        if predictions.dim() > 1:
            predictions = predictions.squeeze()
        if targets.dim() > 1:
            targets = targets.squeeze()
        
        # 擾動目標以增加多樣性
        targets = self._apply_perturbation(targets)
        
        # 確定是否需要分組
        if self.group_size > 0 and targets.size(0) > self.group_size:
            # 將批次分成子批次
            num_groups = (targets.size(0) + self.group_size - 1) // self.group_size
            loss = 0.0
            
            for i in range(num_groups):
                start_idx = i * self.group_size
                end_idx = min(start_idx + self.group_size, targets.size(0))
                
                group_pred = predictions[start_idx:end_idx]
                group_targets = targets[start_idx:end_idx]
                
                # 計算每個子批次的損失
                if self.method == 'listnet':
                    loss += self._listnet_loss(group_pred, group_targets)
                elif self.method == 'listmle':
                    loss += self._listmle_loss(group_pred, group_targets)
                elif self.method == 'approxndcg':
                    loss += self._approxndcg_loss(group_pred, group_targets)
                
            # 計算平均損失
            loss = loss / num_groups
        else:
            # 對整個批次計算損失
            if self.method == 'listnet':
                loss = self._listnet_loss(predictions, targets)
            elif self.method == 'listmle':
                loss = self._listmle_loss(predictions, targets)
            elif self.method == 'approxndcg':
                loss = self._approxndcg_loss(predictions, targets)
            else:
                raise ValueError(f"未知的列表排序方法: {self.method}")
        
        return loss


@LossFactory.register_loss("LambdaRankLoss")
class LambdaRankLoss(nn.Module):
    """
    LambdaRank損失函數，針對排序優化設計。
    結合了pairwise和listwise的優點，是一種更高級的排序損失。
    """
    
    def __init__(self, 
                 sigma: float = 1.0, 
                 k: int = 10, 
                 sampling_ratio: float = 0.3):
        """
        初始化LambdaRankLoss
        
        Args:
            sigma: sigmoid函數的尺度參數
            k: 在NDCG@k評估中使用的k值
            sampling_ratio: 從所有可能的對中採樣的比例
        """
        super(LambdaRankLoss, self).__init__()
        self.sigma = sigma
        self.k = k
        self.sampling_ratio = sampling_ratio
    
    def dcg_gain(self, targets: torch.Tensor, k: int = None) -> torch.Tensor:
        """
        計算DCG增益
        
        Args:
            targets: 目標分數
            k: 僅考慮前k個項目
            
        Returns:
            DCG增益值
        """
        if k is None:
            k = targets.size(0)
        
        _, indices = torch.sort(targets, descending=True)
        gains = torch.pow(2, targets[indices]) - 1
        discounts = torch.log2(torch.arange(1, targets.size(0) + 1, dtype=torch.float, device=targets.device) + 1)
        dcg = torch.sum(gains[:k] / discounts[:k])
        return dcg
    
    def ndcg_delta(self, targets: torch.Tensor, i: int, j: int) -> float:
        """
        計算交換i和j位置後的NDCG差異
        
        Args:
            targets: 目標分數
            i: 第一個位置
            j: 第二個位置
            
        Returns:
            NDCG變化值
        """
        # 創建排序索引
        _, indices = torch.sort(targets, descending=True)
        ideal_dcg = self.dcg_gain(targets)
        
        # 交換i和j的位置
        new_indices = indices.clone()
        new_indices[i], new_indices[j] = new_indices[j], new_indices[i]
        
        # 計算原始DCG和交換後的DCG
        original = self.dcg_gain(targets[indices])
        swapped = self.dcg_gain(targets[new_indices])
        
        # 計算NDCG變化
        delta = torch.abs((original - swapped) / ideal_dcg)
        return delta.item()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        計算LambdaRank損失
        
        Args:
            predictions: 模型預測值，形狀為 [batch_size, 1] 或 [batch_size]
            targets: 真實分數，形狀為 [batch_size, 1] 或 [batch_size]
            
        Returns:
            LambdaRank損失值
        """
        # 確保輸入是一維的
        if predictions.dim() > 1:
            predictions = predictions.squeeze()
        if targets.dim() > 1:
            targets = targets.squeeze()
        
        batch_size = targets.size(0)
        device = targets.device
        
        # 計算所有可能的對
        num_total_pairs = batch_size * (batch_size - 1) // 2
        num_pairs = max(1, int(num_total_pairs * self.sampling_ratio))
        
        # 創建所有可能的索引對
        all_pairs = []
        for i in range(batch_size):
            for j in range(i+1, batch_size):
                all_pairs.append((i, j))
        
        # 隨機採樣對
        sampled_pairs = random.sample(all_pairs, num_pairs)
        
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        for i, j in sampled_pairs:
            # 計算Sij
            if targets[i] > targets[j]:
                Sij = 1
                delta_ndcg = self.ndcg_delta(targets, i, j)
            elif targets[i] < targets[j]:
                Sij = -1
                delta_ndcg = self.ndcg_delta(targets, i, j)
            else:
                continue
            
            # 計算pij
            pred_diff = predictions[i] - predictions[j]
            pij = 1.0 / (1.0 + torch.exp(-self.sigma * pred_diff))
            
            # 計算λij * |Δ NDCG|
            lambda_ij = (1.0 - pij) * Sij * delta_ndcg
            
            # 計算梯度
            loss = loss - lambda_ij * pred_diff
            
        return loss / len(sampled_pairs) if sampled_pairs else loss 