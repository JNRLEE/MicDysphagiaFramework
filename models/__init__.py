"""
模型模組
提供各種深度學習模型架構和相關工具：

1. 模型實現:
   - SwinTransformerModel: Swin Transformer架構，適用於圖像和頻譜圖
   - FCNN: 全連接神經網絡，適用於特徵向量和展平的信號
   - CNNModel: 卷積神經網絡，適用於音頻和圖像
   - ResNetModel: ResNet架構，適用於圖像和頻譜圖

2. 模型工廠:
   - 統一模型創建接口

3. 模型鉤子系統:
   - 監控神經網絡內部狀態
   - 捕獲激活值和梯度
   - 支持模型分析

4. 模型結構分析:
   - 獲取模型架構信息
   - 支持模型層的可視化
"""

# 模型鉤子系統
from .hook_bridge import (
    SimpleModelHookManager as ModelHookManager,
    SimpleActivationHook as ActivationHook,
    SimpleGradientHook as GradientHook
)

# 模型工廠
from .model_factory import create_model

# 模型架構
from .swin_transformer import SwinTransformerModel
from .fcnn import FCNN
from .cnn_model import CNNModel
from .resnet_model import ResNetModel

# 模型結構分析
from .model_structure import ModelStructureInfo, get_registered_model_structure

__all__ = [
    # 模型工廠
    'create_model',
    
    # 模型架構
    'SwinTransformerModel',
    'FCNN',
    'CNNModel',
    'ResNetModel',
    
    # 模型鉤子系統
    'ModelHookManager',
    'ActivationHook',
    'GradientHook',
    
    # 模型結構分析
    'ModelStructureInfo',
    'get_registered_model_structure'
] 