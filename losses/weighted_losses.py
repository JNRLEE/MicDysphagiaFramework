"""
此模組實現了加權損失函數，特別針對不平衡分類問題設計。
主要功能是根據類別頻率自動調整每個類別的權重，使模型在訓練過程中能更好地學習少數類。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import logging

logger = logging.getLogger(__name__)

class WeightedCrossEntropyLoss(nn.Module):
    """
    加權交叉熵損失，自動根據訓練數據中各類別的頻率設置權重。
    權重計算公式：weight = 1 / (class_frequency * total_samples)
    
    Args:
        weight: 可選的手動權重，如果提供則覆蓋自動計算的權重
        reduction: 損失縮減方式，可選'none', 'mean', 'sum'
        eps: 避免除以零的小數值
        normalize_weights: 是否標準化權重使其和為 num_classes
        
    References:
        1. 類不平衡問題的損失函數：https://arxiv.org/abs/1901.05555
    """
    def __init__(
        self, 
        weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
        eps: float = 1e-8,
        normalize_weights: bool = True
    ):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.manual_weight = weight
        self.reduction = reduction
        self.eps = eps
        self.normalize_weights = normalize_weights
        self.weight = None
        self.class_counts = None
        self.num_classes = None
        
    def update_weight_from_labels(self, labels: torch.Tensor) -> None:
        """
        根據訓練數據的標籤分布更新類別權重
        
        Args:
            labels: 訓練數據的標籤，形狀為 [batch_size]
            
        Returns:
            None
        """
        # 獲取標籤的類別數量
        if self.num_classes is None:
            self.num_classes = int(labels.max().item()) + 1
            
        # 初始化類別計數器
        if self.class_counts is None:
            self.class_counts = torch.zeros(self.num_classes, device=labels.device)
        
        # 計算每個類別的樣本數量
        for i in range(self.num_classes):
            self.class_counts[i] += (labels == i).sum().item()
            
        # 計算類別權重：權重與樣本數量成反比
        total_samples = self.class_counts.sum()
        class_frequencies = self.class_counts / total_samples
        self.weight = 1.0 / (class_frequencies + self.eps)
        
        # 標準化權重使其和為 num_classes
        if self.normalize_weights:
            self.weight = self.weight * self.num_classes / self.weight.sum()
            
        logger.info(f"更新類別權重：{self.weight.cpu().numpy()}")
            
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向傳播，計算加權交叉熵損失
        
        Args:
            inputs: 模型輸出，形狀為 [batch_size, num_classes]
            targets: 目標標籤，形狀為 [batch_size]
            
        Returns:
            計算的損失值
        """
        # 使用手動設置的權重，或計算好的權重
        weight = self.manual_weight if self.manual_weight is not None else self.weight
        
        # 確保權重在與輸入相同的設備上
        if weight is not None and weight.device != inputs.device:
            weight = weight.to(inputs.device)
            
        return F.cross_entropy(inputs, targets, weight=weight, reduction=self.reduction)
        
    def get_class_weights(self) -> Optional[torch.Tensor]:
        """
        獲取當前使用的類別權重
        
        Returns:
            類別權重張量
        """
        if self.manual_weight is not None:
            return self.manual_weight
        return self.weight
        
    def reset_stats(self) -> None:
        """
        重置統計信息
        
        Returns:
            None
        """
        self.class_counts = None
        self.weight = None
        self.num_classes = None

# 中文註解：這是weighted_losses.py的Minimal Executable Unit，測試WeightedCrossEntropyLoss能否正確計算權重
if __name__ == "__main__":
    """
    Description: Minimal Executable Unit for weighted_losses.py，測試WeightedCrossEntropyLoss能否正確計算權重。
    Args: None
    Returns: None
    References: 無
    """
    import torch
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 創建不平衡標籤
    labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 2])  # 類別0:5個樣本, 類別1:2個樣本, 類別2:1個樣本
    
    # 創建加權損失函數
    criterion = WeightedCrossEntropyLoss()
    
    # 更新權重
    criterion.update_weight_from_labels(labels)
    
    # 獲取權重
    weights = criterion.get_class_weights()
    print(f"自動計算的權重：{weights}")
    
    # 模擬模型輸出
    outputs = torch.randn(8, 3)  # 8個樣本，3個類別
    
    # 計算損失
    loss = criterion(outputs, labels)
    print(f"損失值：{loss.item()}")
    
    print("測試成功!") 