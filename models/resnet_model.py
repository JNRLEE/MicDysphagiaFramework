"""
ResNet模型：用於吞嚥障礙評估的深度殘差網絡
功能：
1. 提供預訓練的ResNet模型
2. 支持分類和回歸任務
3. 提供骨幹網絡凍結功能
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class ResNetModel(nn.Module):
    """ResNet模型類，用於吞嚥障礙評估的分類或回歸任務
    
    Args:
        model_name: 選擇的ResNet模型，支持'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
        pretrained: 是否使用預訓練權重
        num_classes: 類別數量（分類任務）或輸出維度（回歸任務）
        in_channels: 輸入通道數
        dropout_rate: Dropout率
        is_classification: 是否為分類任務
    """
    
    def __init__(
        self,
        model_name: str = 'resnet50',
        pretrained: bool = True,
        num_classes: int = 5,
        in_channels: int = 3,
        dropout_rate: float = 0.2,
        is_classification: bool = True
    ):
        super().__init__()
        
        self.model_name = model_name.lower()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.is_classification = is_classification
        
        # 檢查模型名稱
        valid_models = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        if self.model_name not in valid_models:
            raise ValueError(f"不支持的模型名稱: {model_name}，支持的模型為: {valid_models}")
        
        # 加載ResNet模型
        if pretrained:
            logger.info(f"使用預訓練的{model_name}模型")
            if self.model_name == 'resnet18':
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                self.backbone = models.resnet18(weights=weights)
            elif self.model_name == 'resnet34':
                weights = models.ResNet34_Weights.IMAGENET1K_V1
                self.backbone = models.resnet34(weights=weights)
            elif self.model_name == 'resnet50':
                weights = models.ResNet50_Weights.IMAGENET1K_V1
                self.backbone = models.resnet50(weights=weights)
            elif self.model_name == 'resnet101':
                weights = models.ResNet101_Weights.IMAGENET1K_V1
                self.backbone = models.resnet101(weights=weights)
            elif self.model_name == 'resnet152':
                weights = models.ResNet152_Weights.IMAGENET1K_V1
                self.backbone = models.resnet152(weights=weights)
        else:
            logger.info(f"使用隨機初始化的{model_name}模型")
            if self.model_name == 'resnet18':
                self.backbone = models.resnet18(weights=None)
            elif self.model_name == 'resnet34':
                self.backbone = models.resnet34(weights=None)
            elif self.model_name == 'resnet50':
                self.backbone = models.resnet50(weights=None)
            elif self.model_name == 'resnet101':
                self.backbone = models.resnet101(weights=None)
            elif self.model_name == 'resnet152':
                self.backbone = models.resnet152(weights=None)
        
        # 如果輸入通道數不是3，需要修改第一層卷積
        if in_channels != 3:
            logger.info(f"修改第一層卷積，將輸入通道從3改為{in_channels}")
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 
                self.backbone.conv1.out_channels, 
                kernel_size=self.backbone.conv1.kernel_size, 
                stride=self.backbone.conv1.stride, 
                padding=self.backbone.conv1.padding, 
                bias=False
            )
        
        # 獲取特徵維度
        if self.model_name in ['resnet18', 'resnet34']:
            feature_dim = 512
        else:
            feature_dim = 2048
        
        # 移除原始的全連接層
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # 添加新的分類頭或回歸頭
        head_layers = []
        head_layers.append(nn.Flatten())
        
        if dropout_rate > 0:
            head_layers.append(nn.Dropout(dropout_rate))
        
        if is_classification:
            # 分類頭
            head_layers.append(nn.Linear(feature_dim, num_classes))
        else:
            # 回歸頭
            head_layers.append(nn.Linear(feature_dim, 1))
        
        self.head = nn.Sequential(*head_layers)
    
    def freeze_backbone_layers(self, trainable_layers: int = 1) -> None:
        """凍結ResNet主幹網絡的大部分層，只保留最後幾層可訓練
        
        Args:
            trainable_layers: 保持可訓練的最後幾層數量
        """
        logger.info(f"凍結ResNet主幹網絡，只保留最後{trainable_layers}層可訓練")
        
        # 獲取骨幹網絡的所有層
        backbone_layers = list(self.backbone.children())
        
        # 計算要凍結的層數
        total_layers = len(backbone_layers)
        frozen_layers = total_layers - trainable_layers
        
        # 凍結指定層
        for i in range(min(frozen_layers, total_layers)):
            for param in backbone_layers[i].parameters():
                param.requires_grad = False
            logger.info(f"已凍結第 {i+1} 層")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播
        
        Args:
            x: 輸入數據，形狀為 [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: 預測結果
                - 分類模式: [batch_size, num_classes] 的logits
                - 回歸模式: [batch_size] 的分數
        """
        # 通過主幹網絡
        features = self.backbone(x)
        
        # 通過頭部
        output = self.head(features)
        
        # 如果是回歸模式且輸出維度為1，則壓縮最後一個維度
        if not self.is_classification and output.shape[-1] == 1:
            output = output.squeeze(-1)
            
        return output
    
    def configure_optimizers(
        self, 
        lr: float = 1e-4, 
        weight_decay: float = 1e-5,
        backbone_lr: Optional[float] = None
    ) -> torch.optim.Optimizer:
        """配置優化器
        
        Args:
            lr: 學習率
            weight_decay: 權重衰減
            backbone_lr: 主幹網絡的學習率（如果為None，則使用lr的1/10）
            
        Returns:
            torch.optim.Optimizer: 配置的優化器
        """
        if backbone_lr is None:
            backbone_lr = lr / 10  # 默認主幹網絡學習率為基本學習率的1/10
        
        # 分別配置參數組
        backbone_params = []
        head_params = []
        
        # 分配參數
        for name, param in self.named_parameters():
            if name.startswith('backbone'):
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        # 參數分組
        param_groups = [
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr, "weight_decay": weight_decay}
        ]
        
        # 創建Adam優化器
        optimizer = torch.optim.Adam(param_groups)
        
        return optimizer 