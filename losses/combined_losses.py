"""
此模組實現損失函數組合功能，允許多個損失函數加權組合使用。
適用於需要同時優化多個目標的情況，如同時使用回歸損失和排序損失。
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Union, Any
from collections import OrderedDict

from .loss_factory import LossFactory

logger = logging.getLogger(__name__)

@LossFactory.register_loss("CombinedLoss")
class CombinedLoss(nn.Module):
    """
    組合損失函數，允許多個損失函數加權組合使用。
    可以動態調整各個損失的權重，支持自適應重新加權機制。
    """
    
    def __init__(self, 
                 losses: Dict[str, nn.Module], 
                 weights: Dict[str, float] = None,
                 adaptive_weights: bool = False,
                 weight_update_freq: int = 100,
                 weight_update_ratio: float = 0.1):
        """
        初始化組合損失函數
        
        Args:
            losses: 損失函數字典，鍵為損失名稱，值為損失函數實例
            weights: 損失權重字典，鍵為損失名稱，值為權重
            adaptive_weights: 是否自適應調整權重
            weight_update_freq: 權重更新頻率（訓練步數）
            weight_update_ratio: 權重更新比例
        """
        super(CombinedLoss, self).__init__()
        
        self.losses = nn.ModuleDict(losses)
        
        # 如果沒有指定權重，則平均分配
        if weights is None:
            weights = {name: 1.0 / len(losses) for name in losses.keys()}
        else:
            # 確保所有損失都有權重
            for name in losses.keys():
                if name not in weights:
                    weights[name] = 1.0
                    logger.warning(f"損失函數 {name} 未指定權重，設為默認值 1.0")
        
        # 將權重註冊為緩衝區，這樣它們會被保存和加載，但不是模型參數
        self.register_buffer('_weights', torch.tensor(list(weights.values()), dtype=torch.float))
        
        # 保存損失名稱以維持順序
        self.loss_names = list(losses.keys())
        
        # 自適應權重設置
        self.adaptive_weights = adaptive_weights
        self.weight_update_freq = weight_update_freq
        self.weight_update_ratio = weight_update_ratio
        self.steps = 0
        
        # 用於追蹤每個損失函數的歷史值
        self.register_buffer('_loss_history', torch.zeros(len(losses), dtype=torch.float))
        self.history_size = 0
    
    @property
    def weights(self) -> Dict[str, float]:
        """
        獲取當前損失權重
        
        Returns:
            損失權重字典
        """
        return {name: self._weights[i].item() for i, name in enumerate(self.loss_names)}
    
    def update_weights(self, weights: Dict[str, float]):
        """
        更新損失權重
        
        Args:
            weights: 新的損失權重字典
        """
        for i, name in enumerate(self.loss_names):
            if name in weights:
                self._weights[i] = weights[name]
        
        logger.info(f"更新損失權重: {self.weights}")
    
    def _update_adaptive_weights(self, current_losses: Dict[str, float]):
        """
        根據當前損失值自適應更新權重
        
        Args:
            current_losses: 當前各損失函數的值
        """
        if not self.adaptive_weights:
            return
        
        self.steps += 1
        
        # 只有達到更新頻率時才更新
        if self.steps % self.weight_update_freq != 0:
            return
        
        # 更新歷史損失值
        for i, name in enumerate(self.loss_names):
            # 使用指數移動平均更新歷史
            if self.history_size > 0:
                self._loss_history[i] = 0.9 * self._loss_history[i] + 0.1 * current_losses[name]
            else:
                self._loss_history[i] = current_losses[name]
        
        self.history_size += 1
        
        # 只有在有足夠歷史數據時才進行自適應
        if self.history_size < 5:
            return
        
        # 計算新權重：權重與損失值成反比
        loss_values = torch.clamp(self._loss_history, min=1e-8)
        inverse_losses = 1.0 / loss_values
        new_weights = inverse_losses / inverse_losses.sum()
        
        # 平滑更新
        updated_weights = (1 - self.weight_update_ratio) * self._weights + self.weight_update_ratio * new_weights
        
        # 重新歸一化
        updated_weights = updated_weights / updated_weights.sum()
        
        # 更新權重
        self._weights.copy_(updated_weights)
        logger.debug(f"自適應更新損失權重: {self.weights}")
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        計算組合損失
        
        Args:
            predictions: 模型預測值
            targets: 真實目標值
            **kwargs: 其他參數，可以傳遞給特定的損失函數
            
        Returns:
            組合損失值
        """
        individual_losses = OrderedDict()
        total_loss = 0.0
        
        # 計算每個損失函數的值
        for i, (name, loss_fn) in enumerate(self.losses.items()):
            try:
                # 嘗試使用kwargs調用損失函數
                loss_value = loss_fn(predictions, targets, **kwargs)
                
                # 加入到總損失
                individual_losses[name] = loss_value.item()
                total_loss = total_loss + self._weights[i] * loss_value
            except Exception as e:
                logger.error(f"計算損失 {name} 時出錯: {str(e)}")
                # 如果發生錯誤，將這個損失設為0
                individual_losses[name] = 0.0
        
        # 自適應更新權重
        self._update_adaptive_weights(individual_losses)
        
        # 返回組合損失
        return total_loss
    
    def get_individual_losses(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> Dict[str, float]:
        """
        計算並返回各個損失函數的值，不加權
        
        Args:
            predictions: 模型預測值
            targets: 真實目標值
            **kwargs: 其他參數
            
        Returns:
            各個損失函數的值字典
        """
        individual_losses = {}
        
        for name, loss_fn in self.losses.items():
            try:
                loss_value = loss_fn(predictions, targets, **kwargs)
                individual_losses[name] = loss_value.item()
            except Exception as e:
                logger.error(f"計算損失 {name} 時出錯: {str(e)}")
                individual_losses[name] = 0.0
        
        return individual_losses


@LossFactory.register_loss("WeightedMSELoss")
class WeightedMSELoss(nn.Module):
    """
    加權MSE損失，允許為不同樣本指定不同權重。
    適用於某些樣本比其他樣本更重要的情況。
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        初始化加權MSE損失
        
        Args:
            reduction: 損失歸約方式，可以是'none', 'mean', 'sum'
        """
        super(WeightedMSELoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, 
               predictions: torch.Tensor, 
               targets: torch.Tensor, 
               weights: torch.Tensor = None) -> torch.Tensor:
        """
        計算加權MSE損失
        
        Args:
            predictions: 模型預測值，形狀為 [batch_size, ...]
            targets: 真實目標值，形狀為 [batch_size, ...]
            weights: 樣本權重，形狀為 [batch_size]，如果為None則所有樣本權重相等
            
        Returns:
            損失值
        """
        # 計算每個樣本的MSE
        mse = (predictions - targets) ** 2
        
        if weights is not None:
            # 如果提供了權重，則應用權重
            weights = weights.view(*weights.shape, *([1] * (mse.dim() - weights.dim())))
            mse = mse * weights
        
        # 應用歸約
        if self.reduction == 'none':
            return mse
        elif self.reduction == 'sum':
            return mse.sum()
        else:  # mean
            return mse.mean()


@LossFactory.register_loss("FocalLoss")
class FocalLoss(nn.Module):
    """
    焦點損失(Focal Loss)，減少易分類樣本的權重，增加難分類樣本的權重。
    適用於類別不平衡問題。
    """
    
    def __init__(self, 
                alpha: float = 0.25, 
                gamma: float = 2.0, 
                reduction: str = 'mean'):
        """
        初始化焦點損失
        
        Args:
            alpha: 平衡正負樣本的參數
            gamma: 聚焦參數，控制易分類樣本的下降權重
            reduction: 損失歸約方式，可以是'none', 'mean', 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, 
               predictions: torch.Tensor, 
               targets: torch.Tensor) -> torch.Tensor:
        """
        計算焦點損失
        
        Args:
            predictions: 模型預測值，形狀為 [batch_size, num_classes] 的logits
            targets: 真實標籤，形狀為 [batch_size] 的整數類別索引
            
        Returns:
            焦點損失值
        """
        # 使用sigmoid函數將logits轉換為概率
        if predictions.size(-1) == 1:  # 二分類情況
            predictions = torch.sigmoid(predictions)
            
            # 獲取正負樣本的預測概率
            p = predictions.view(-1)
            p_t = torch.empty_like(p)
            p_t[targets.view(-1) == 1] = p[targets.view(-1) == 1]
            p_t[targets.view(-1) == 0] = 1 - p[targets.view(-1) == 0]
            
            # 計算alpha_t
            alpha_t = torch.ones_like(p_t) * self.alpha
            alpha_t[targets.view(-1) == 0] = 1 - self.alpha
            
            # 計算焦點損失
            focal_weight = (1 - p_t) ** self.gamma
            loss = -alpha_t * focal_weight * torch.log(torch.clamp(p_t, min=1e-8))
        else:  # 多分類情況
            # 使用softmax函數將logits轉換為概率
            predictions_softmax = F.softmax(predictions, dim=1)
            
            # 獲取每個類別的真實概率
            batch_size = predictions.size(0)
            # 獲取每個樣本對應的實際類別的預測概率
            p_t = predictions_softmax[
                torch.arange(batch_size, device=predictions.device),
                targets
            ]
            
            # 計算focal weight
            focal_weight = (1 - p_t) ** self.gamma
            
            # 計算交叉熵損失
            ce_loss = F.cross_entropy(predictions, targets, reduction='none')
            
            # 應用focal weight
            loss = focal_weight * ce_loss
        
        # 應用歸約
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # mean
            return loss.mean() 