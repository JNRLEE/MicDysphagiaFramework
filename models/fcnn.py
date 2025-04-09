"""
全連接神經網絡模型：用於吞嚥障礙評估的特徵向量處理
功能：
1. 提供彈性配置的多層全連接神經網絡
2. 支持分類和回歸任務
3. 支持批標準化和多種激活函數
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class FCNN(nn.Module):
    """全連接神經網絡模型類，用於處理特徵向量的分類或回歸任務
    
    Args:
        input_dim: 輸入特徵維度
        hidden_dims: 隱藏層神經元數量列表
        num_classes: 類別數量（分類任務）或輸出維度（回歸任務）
        dropout_rate: Dropout率
        activation: 激活函數，支持 'relu', 'leaky_relu', 'elu', 'gelu'
        is_classification: 是否為分類任務
        batch_norm: 是否使用批標準化
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dims: List[int] = [512, 256],
        num_classes: int = 5,
        dropout_rate: float = 0.2,
        activation: str = 'relu',
        is_classification: bool = True,
        batch_norm: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.is_classification = is_classification
        self.batch_norm = batch_norm
        
        # 創建網絡層
        layers = []
        dims = [input_dim] + hidden_dims
        
        # 獲取激活函數
        act_fn = self._get_activation(activation)
        
        # 建構隱藏層
        for i in range(len(dims) - 1):
            # 全連接層
            layers.append(nn.Linear(dims[i], dims[i+1]))
            
            # 批標準化（如果啟用）
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i+1]))
            
            # 激活函數
            layers.append(act_fn())
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        
        # 特徵提取器
        self.feature_extractor = nn.Sequential(*layers)
        
        # 輸出層
        if is_classification:
            self.head = nn.Linear(dims[-1], num_classes)
        else:
            self.head = nn.Linear(dims[-1], 1)
            
    def _get_activation(self, activation_name: str) -> nn.Module:
        """根據名稱獲取激活函數類
        
        Args:
            activation_name: 激活函數名稱
            
        Returns:
            對應的PyTorch激活函數類
            
        Raises:
            ValueError: 如果提供了不支持的激活函數名稱
        """
        activation_name = activation_name.lower()
        if activation_name == 'relu':
            return nn.ReLU
        elif activation_name == 'leaky_relu':
            return nn.LeakyReLU
        elif activation_name == 'elu':
            return nn.ELU
        elif activation_name == 'gelu':
            return lambda: nn.GELU()
        else:
            raise ValueError(f"不支持的激活函數: {activation_name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播
        
        Args:
            x: 輸入特徵向量，形狀為 [batch_size, input_dim] 或 [batch_size, seq_len, input_dim]
               或 [batch_size, channels, height, width]（來自頻譜圖）
            
        Returns:
            torch.Tensor: 預測結果
                - 分類模式: [batch_size, num_classes] 的logits
                - 回歸模式: [batch_size] 的分數
        """
        # 處理輸入形狀
        input_shape = x.shape
        batch_size = input_shape[0]
        
        # 檢查輸入維度並進行必要的調整
        if len(input_shape) == 4:  # [batch_size, channels, height, width]
            # 輸入是4D（圖像/頻譜圖），展平為2D
            logger.info(f"FCNN接收到4D輸入{input_shape}，展平為2D")
            x = x.reshape(batch_size, -1)
        elif len(input_shape) == 3:  # [batch_size, seq_len, dim]
            # 輸入是3D（序列），對序列維度進行平均
            logger.info(f"FCNN接收到3D輸入{input_shape}，對序列維度取平均")
            x = x.mean(dim=1)
        elif len(input_shape) != 2:  # 非 [batch_size, dim]
            raise ValueError(f"不支持的輸入維度: {input_shape}")
        
        # 檢查特徵維度是否與模型期望的輸入維度一致
        if x.size(1) != self.input_dim:
            logger.warning(f"輸入維度不匹配: 期望 {self.input_dim}, 實際 {x.size(1)}")
            
            # 調整維度（截斷或填充）
            if x.size(1) > self.input_dim:
                logger.info(f"截斷特徵: {x.size(1)} -> {self.input_dim}")
                x = x[:, :self.input_dim]
            else:
                logger.info(f"填充特徵: {x.size(1)} -> {self.input_dim}")
                padding = torch.zeros(batch_size, self.input_dim - x.size(1), device=x.device)
                x = torch.cat([x, padding], dim=1)
        
        # 確保張量形狀正確並且是連續的
        x = x.contiguous()
        
        # 提取特徵
        features = self.feature_extractor(x)
        
        # 輸出預測
        output = self.head(features)
        
        # 如果是回歸模式且輸出維度為1，則壓縮最後一個維度
        if not self.is_classification and output.shape[-1] == 1:
            output = output.squeeze(-1)
            
        return output
    
    def configure_optimizers(
        self, 
        lr: float = 1e-3, 
        weight_decay: float = 1e-5
    ) -> torch.optim.Optimizer:
        """配置優化器
        
        Args:
            lr: 學習率
            weight_decay: 權重衰減
            
        Returns:
            torch.optim.Optimizer: 配置的優化器
        """
        # FCNN通常使用Adam優化器，較大的學習率
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        return optimizer 