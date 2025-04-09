"""
Swin Transformer 模型：針對吞嚥障礙評估的影像分類/回歸
功能：
1. 提供可配置的Swin Transformer模型
2. 支持視覺提示機制
3. 支持分類和回歸任務
4. 提供骨幹網絡凍結功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from einops import rearrange, repeat
import math
import torchvision.models as models

logger = logging.getLogger(__name__)

class VisualPromptTemplate(nn.Module):
    """視覺提示模板類，用於向輸入圖像添加可學習的提示
    
    Args:
        prompt_size: 提示的高度和寬度
        prompt_channels: 提示的通道數
        image_size: 圖像的高度和寬度
        image_channels: 圖像的通道數
        template_dropout: 提示模板的隨機丟棄概率
        init_range: 提示模板的初始化範圍
    """
    
    def __init__(
        self,
        prompt_size: Tuple[int, int] = (16, 16),
        prompt_channels: int = 3,
        image_size: Tuple[int, int] = (224, 224),
        image_channels: int = 3,
        template_dropout: float = 0.1,
        init_range: float = 0.1
    ):
        super().__init__()
        
        self.prompt_size = prompt_size
        self.prompt_channels = prompt_channels
        self.image_size = image_size
        self.image_channels = image_channels
        self.template_dropout = template_dropout
        self.init_range = init_range
        
        # 創建可學習的提示模板
        self.prompt_template = nn.Parameter(
            torch.randn(1, prompt_channels, prompt_size[0], prompt_size[1]) * init_range
        )
        
        # 提示位置：預設在圖像左上角
        # 提示區域的起始和結束位置
        self.prompt_pos = (0, 0, prompt_size[0], prompt_size[1])
        
        # 丟棄層
        self.dropout = nn.Dropout(template_dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播
        
        Args:
            x: 輸入圖像，形狀為 [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: 添加了視覺提示的圖像
        """
        batch_size = x.size(0)
        
        # 創建提示模板的批次
        prompt_template = self.prompt_template.expand(batch_size, -1, -1, -1)
        
        # 創建遮罩
        mask = torch.zeros(batch_size, 1, self.image_size[0], self.image_size[1], device=x.device)
        mask[:, :, self.prompt_pos[0]:self.prompt_pos[2], self.prompt_pos[1]:self.prompt_pos[3]] = 1
        
        # 將提示模板調整為與輸入圖像相同的大小
        prompt_template = F.interpolate(
            prompt_template,
            size=(self.image_size[0], self.image_size[1]),
            mode='bilinear',
            align_corners=False
        )
        
        # 應用提示模板
        x = x * (1 - mask) + prompt_template * mask
        
        # 應用丟棄
        x = self.dropout(x)
        
        return x

class SwinTransformerModel(nn.Module):
    """Swin Transformer模型類，用於吞嚥障礙評估的分類或回歸任務
    
    Args:
        model_name: 選擇的Swin Transformer模型
        pretrained: 是否使用預訓練權重
        num_classes: 類別數量
        in_channels: 輸入通道數
        input_size: 輸入圖像大小
        use_visual_prompting: 是否使用視覺提示
        prompt_size: 視覺提示的大小
        prompt_dropout: 視覺提示的dropout率
        prompt_init_range: 視覺提示的初始化範圍
        dropout_rate: dropout率
        is_classification: 是否為分類模式
    """
    
    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        num_classes: int = 5,
        in_channels: int = 3,
        input_size: Tuple[int, int] = (224, 224),
        use_visual_prompting: bool = False,
        prompt_size: Tuple[int, int] = (16, 16),
        prompt_dropout: float = 0.1,
        prompt_init_range: float = 0.1,
        dropout_rate: float = 0.1,
        is_classification: bool = True
    ):
        super().__init__()
        
        # 保存設置
        self.input_size = input_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_visual_prompting = use_visual_prompting
        self.is_classification = is_classification
        
        # 加載基礎Swin Transformer模型
        if pretrained:
            if model_name == "swin_tiny_patch4_window7_224":
                weights = models.Swin_T_Weights.IMAGENET1K_V1
                logger.info("使用預訓練權重: Swin_T_Weights.IMAGENET1K_V1")
                self.backbone = models.swin_t(weights=weights)
            elif model_name == "swin_small_patch4_window7_224":
                weights = models.Swin_S_Weights.IMAGENET1K_V1
                logger.info("使用預訓練權重: Swin_S_Weights.IMAGENET1K_V1")
                self.backbone = models.swin_s(weights=weights)
            elif model_name == "swin_base_patch4_window7_224":
                weights = models.Swin_B_Weights.IMAGENET1K_V1
                logger.info("使用預訓練權重: Swin_B_Weights.IMAGENET1K_V1")
                self.backbone = models.swin_b(weights=weights)
            else:
                raise ValueError(f"未知的模型名稱: {model_name}")
        else:
            # 不使用預訓練權重
            if model_name == "swin_tiny_patch4_window7_224":
                self.backbone = models.swin_t(weights=None)
            elif model_name == "swin_small_patch4_window7_224":
                self.backbone = models.swin_s(weights=None)
            elif model_name == "swin_base_patch4_window7_224":
                self.backbone = models.swin_b(weights=None)
            else:
                raise ValueError(f"未知的模型名稱: {model_name}")
            
        # 移除原始的分類頭
        self.backbone.head = nn.Identity()
        
        # 獲取特徵維度
        if "tiny" in model_name:
            feature_dim = 768
        elif "small" in model_name:
            feature_dim = 768
        elif "base" in model_name:
            feature_dim = 1024
        else:
            feature_dim = 768  # 默認
        
        # 創建視覺提示模塊（如果啟用）
        if use_visual_prompting:
            self.visual_prompt = VisualPromptTemplate(
                prompt_size=prompt_size,
                prompt_channels=in_channels,
                image_size=input_size,
                image_channels=in_channels,
                template_dropout=prompt_dropout,
                init_range=prompt_init_range
            )
        
        # 創建輸出頭
        if is_classification:
            # 分類頭
            self.head = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, num_classes)
            )
        else:
            # 回歸頭
            self.head = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, 1)
            )
        
        # 殘差路徑（輕量級分支）
        self.skip_connection = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes if is_classification else 1)
        )
        
        # 初始化權重
        self._initialize_weights()
        
    def freeze_backbone_layers(self, trainable_layers: int = 1) -> None:
        """凍結Swin Transformer主幹網絡的大部分層，只保留最後幾層可訓練
        
        Args:
            trainable_layers: 保持可訓練的最後幾層數量
        """
        logger.info(f"凍結Swin Transformer主幹網絡，只保留最後{trainable_layers}層可訓練")
        
        # 首先凍結所有主幹網絡參數
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 獲取主幹網絡特徵提取器的所有層
        stages = [module for module in self.backbone.features]
        
        # 如果trainable_layers > 0，解凍最後幾層
        if trainable_layers > 0:
            # 計算需要解凍的層的索引
            unfrozen_indices = list(range(max(0, len(stages) - trainable_layers), len(stages)))
            
            # 解凍這些層
            for idx in unfrozen_indices:
                for param in stages[idx].parameters():
                    param.requires_grad = True
                logger.info(f"  保持第 {idx} 層可訓練")
                
    def _initialize_weights(self) -> None:
        """初始化模型權重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播
        
        Args:
            x: 輸入頻譜圖或圖像，形狀為 [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: 預測結果
                - 分類模式: [batch_size, num_classes] 的logits
                - 回歸模式: [batch_size] 的分數
        """
        # 應用視覺提示（如果啟用）
        if self.use_visual_prompting:
            x = self.visual_prompt(x)
        
        # 通過主幹網絡
        features = self.backbone(x)
        
        # 主路徑
        main_output = self.head(features)
        
        # 殘差路徑
        skip_output = self.skip_connection(features)
        
        # 組合輸出
        output = main_output + skip_output
        
        # 如果是回歸模式，且輸出維度為1，則壓縮最後一個維度
        if not self.is_classification and self.num_classes == 1:
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
        prompt_params = []
        backbone_params = []
        head_params = []
        
        # 分配參數
        if self.use_visual_prompting:
            prompt_params.extend(self.visual_prompt.parameters())
            
        backbone_params.extend(self.backbone.parameters())
        head_params.extend(self.head.parameters())
        head_params.extend(self.skip_connection.parameters())
        
        # 參數分組
        param_groups = [
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr, "weight_decay": weight_decay}
        ]
        
        if self.use_visual_prompting:
            param_groups.append({"params": prompt_params, "lr": lr, "weight_decay": 0.0})
        
        # 創建Adam優化器
        optimizer = torch.optim.Adam(param_groups)
        
        return optimizer 