"""
CNN模型：用於吞嚥障礙評估的時間序列或頻譜圖處理
功能：
1. 提供彈性配置的卷積神經網絡
2. 支持分類和回歸任務
3. 適用於頻譜圖或時間序列數據
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import math

logger = logging.getLogger(__name__)

class CNNModel(nn.Module):
    """CNN模型類，用於處理時間序列或頻譜圖的分類或回歸任務
    
    Args:
        input_channels: 輸入通道數
        input_size: 輸入尺寸 (高度, 寬度)
        filters: 每層卷積的濾波器數量列表
        kernel_sizes: 每層卷積的核大小列表
        pool_sizes: 每層池化的大小列表
        fc_dims: 全連接層的神經元數量列表
        num_classes: 類別數量（分類任務）或輸出維度（回歸任務）
        dropout_rate: Dropout率
        batch_norm: 是否使用批標準化
        activation: 激活函數，支持 'relu', 'leaky_relu'
        is_classification: 是否為分類任務
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        input_size: Tuple[int, int] = (224, 224),
        filters: List[int] = [32, 64, 128],
        kernel_sizes: List[int] = [3, 3, 3],
        pool_sizes: List[int] = [2, 2, 2],
        fc_dims: List[int] = [512],
        num_classes: int = 5,
        dropout_rate: float = 0.2,
        batch_norm: bool = True,
        activation: str = 'relu',
        is_classification: bool = True
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.input_size = input_size
        self.is_classification = is_classification
        self.num_classes = num_classes
        
        # 確保所有列表長度一致
        assert len(filters) == len(kernel_sizes) == len(pool_sizes), \
            "濾波器、核大小和池化大小列表長度必須相同"
        
        # 創建卷積層
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        
        # 計算每層輸出尺寸，用於計算展平後的特徵維度
        height, width = input_size
        
        # 構建卷積層
        for i, (f, k, p) in enumerate(zip(filters, kernel_sizes, pool_sizes)):
            # 卷積模塊
            conv_block = []
            
            # 卷積層
            conv_block.append(nn.Conv2d(in_channels, f, kernel_size=k, padding=k//2))
            
            # 批標準化
            if batch_norm:
                conv_block.append(nn.BatchNorm2d(f))
            
            # 激活函數
            if activation.lower() == 'relu':
                conv_block.append(nn.ReLU(inplace=True))
            elif activation.lower() == 'leaky_relu':
                conv_block.append(nn.LeakyReLU(0.2, inplace=True))
            else:
                raise ValueError(f"不支持的激活函數: {activation}")
            
            # 池化層
            conv_block.append(nn.MaxPool2d(p))
            
            # 添加到卷積層列表
            self.conv_layers.append(nn.Sequential(*conv_block))
            
            # 更新下一層的輸入通道數
            in_channels = f
            
            # 更新輸出尺寸
            height = math.floor((height - k + 2*(k//2)) / p + 1)
            width = math.floor((width - k + 2*(k//2)) / p + 1)
        
        # 計算展平後的特徵維度
        flattened_dim = filters[-1] * height * width
        logger.info(f"展平後的特徵維度: {flattened_dim}")
        
        # 創建全連接層
        fc_layers = []
        fc_in_dim = flattened_dim
        
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(fc_in_dim, fc_dim))
            if batch_norm:
                fc_layers.append(nn.BatchNorm1d(fc_dim))
            
            # 激活函數
            if activation.lower() == 'relu':
                fc_layers.append(nn.ReLU(inplace=True))
            elif activation.lower() == 'leaky_relu':
                fc_layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            # Dropout
            if dropout_rate > 0:
                fc_layers.append(nn.Dropout(dropout_rate))
            
            fc_in_dim = fc_dim
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
        # 輸出層
        if is_classification:
            self.head = nn.Linear(fc_dims[-1] if fc_dims else flattened_dim, num_classes)
        else:
            self.head = nn.Linear(fc_dims[-1] if fc_dims else flattened_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播
        
        Args:
            x: 輸入數據，形狀為 [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: 預測結果
                - 分類模式: [batch_size, num_classes] 的logits
                - 回歸模式: [batch_size] 的分數
        """
        # 檢查輸入形狀和通道數
        if len(x.shape) != 4:
            raise ValueError(f"CNN模型需要4D輸入 [batch_size, channels, height, width]，但得到了 {len(x.shape)}D 輸入")
        
        batch_size, channels, height, width = x.shape
        
        # 檢查通道數
        if channels != self.input_channels:
            logger.warning(f"輸入通道數不匹配：期望 {self.input_channels}，實際 {channels}")
            # 調整通道數
            if channels < self.input_channels:
                # 少於期望通道，複製現有通道
                factor = (self.input_channels + channels - 1) // channels  # 向上取整
                x = x.repeat(1, factor, 1, 1)[:, :self.input_channels, :, :]
            else:
                # 多於期望通道，只保留前幾個通道
                x = x[:, :self.input_channels, :, :]
        
        # 檢查空間尺寸
        expected_height, expected_width = self.input_size
        if height != expected_height or width != expected_width:
            logger.warning(f"輸入尺寸不匹配：期望 {self.input_size}，實際 ({height}, {width})")
            # 調整尺寸
            x = F.interpolate(x, size=self.input_size, mode='bilinear', align_corners=False)
        
        # 通過卷積層
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 通過全連接層
        x = self.fc_layers(x)
        
        # 輸出預測
        output = self.head(x)
        
        # 如果是回歸模式且輸出維度為1，則壓縮最後一個維度
        if not self.is_classification and output.shape[-1] == 1:
            output = output.squeeze(-1)
            
        return output
    
    def freeze_backbone_layers(self, trainable_layers: int = 1) -> None:
        """凍結卷積主幹網絡的大部分層，只保留最後幾層可訓練
        
        Args:
            trainable_layers: 保持可訓練的最後幾層數量
        """
        logger.info(f"凍結CNN主幹網絡，只保留最後{trainable_layers}層可訓練")
        
        # 計算要凍結的層數
        total_conv_layers = len(self.conv_layers)
        frozen_layers = max(0, total_conv_layers - trainable_layers)
        
        # 凍結指定層
        for i in range(frozen_layers):
            for param in self.conv_layers[i].parameters():
                param.requires_grad = False
            logger.info(f"凍結第 {i+1} 個卷積層")
    
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
        # CNN通常使用Adam優化器
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        return optimizer 