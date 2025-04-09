"""
模型工廠模組：根據配置動態創建模型
功能：
1. 支持PyTorch框架
2. 根據配置選擇適當的模型架構
3. 自動處理模型參數設置
4. 整合視覺提示等擴展功能
"""

import logging
from typing import Dict, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class ModelFactory:
    """模型工廠類，根據配置創建模型"""
    
    @staticmethod
    def create_model(config: Dict[str, Any]) -> Any:
        """創建模型
        
        Args:
            config: 配置字典，包含 'model' 部分
            
        Returns:
            Any: 創建的模型實例
            
        Raises:
            ValueError: 如果配置中指定了不支持的模型類型
        """
        model_config = config['model']
        model_type = model_config.get('type', 'swin_transformer').lower()
        
        # 提取模型參數
        model_params = model_config.get('parameters', {})
        visual_prompting_config = model_config.get('visual_prompting', {})
        backbone_config = model_config.get('backbone', {})
        
        # 根據模型類型創建對應的模型
        if model_type == 'swin_transformer':
            return ModelFactory._create_swin_transformer(config)
        elif model_type == 'fcnn':
            return ModelFactory._create_pytorch_fcnn(config)
        elif model_type == 'resnet':
            return ModelFactory._create_pytorch_resnet(config)
        elif model_type == 'cnn':
            return ModelFactory._create_pytorch_cnn(config)
        else:
            raise ValueError(f"不支持的模型類型: {model_type}")
    
    @staticmethod
    def _create_swin_transformer(config: Dict[str, Any]) -> Any:
        """創建Swin Transformer模型
        
        Args:
            config: 配置字典
            
        Returns:
            Any: 創建的Swin Transformer模型實例
        """
        from .swin_transformer import SwinTransformerModel
        
        model_config = config['model']
        model_params = model_config.get('parameters', {})
        visual_prompting_config = model_config.get('visual_prompting', {})
        backbone_config = model_config.get('backbone', {})
        
        # 提取參數
        model_name = model_params.get('model_name', 'swin_tiny_patch4_window7_224')
        pretrained = model_params.get('pretrained', True)
        input_channels = model_params.get('input_channels', 3)
        input_size = model_params.get('input_size', (224, 224))
        num_classes = model_params.get('num_classes', 5)
        dropout_rate = model_params.get('dropout_rate', 0.1)
        is_classification = model_params.get('is_classification', True)
        
        # 視覺提示參數
        use_visual_prompting = visual_prompting_config.get('enabled', False)
        prompt_size = visual_prompting_config.get('prompt_size', (16, 16))
        prompt_dropout = visual_prompting_config.get('prompt_dropout', 0.1)
        
        # 創建模型
        model = SwinTransformerModel(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_channels=input_channels,
            input_size=input_size,
            use_visual_prompting=use_visual_prompting,
            prompt_size=prompt_size,
            prompt_dropout=prompt_dropout,
            dropout_rate=dropout_rate,
            is_classification=is_classification
        )
        
        # 凍結骨幹網絡
        freeze_backbone = backbone_config.get('freeze', False)
        if freeze_backbone:
            unfreeze_layers = backbone_config.get('unfreeze_layers', 1)
            model.freeze_backbone_layers(trainable_layers=unfreeze_layers)
            logger.info(f"已凍結Swin Transformer骨幹網絡，解凍最後 {unfreeze_layers} 層")
        
        return model
    
    @staticmethod
    def _create_pytorch_fcnn(config: Dict[str, Any]) -> Any:
        """創建PyTorch全連接神經網絡模型
        
        Args:
            config: 配置字典
            
        Returns:
            Any: 創建的全連接神經網絡模型實例
        """
        from .fcnn import FCNN
        
        model_config = config['model']
        model_params = model_config.get('parameters', {})
        
        # 提取參數
        input_dim = model_params.get('input_dim', 1024)
        hidden_dims = model_params.get('hidden_layers', [512, 256])
        num_classes = model_params.get('num_classes', 5)
        dropout_rate = model_params.get('dropout_rate', 0.2)
        activation = model_params.get('activation', 'relu')
        is_classification = model_params.get('is_classification', True)
        batch_norm = model_params.get('batch_norm', True)
        
        # 創建模型
        model = FCNN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            activation=activation,
            is_classification=is_classification,
            batch_norm=batch_norm
        )
        
        return model
    
    @staticmethod
    def _create_pytorch_resnet(config: Dict[str, Any]) -> Any:
        """創建PyTorch ResNet模型
        
        Args:
            config: 配置字典
            
        Returns:
            Any: 創建的ResNet模型實例
        """
        from .resnet_model import ResNetModel
        
        model_config = config['model']
        model_params = model_config.get('parameters', {})
        backbone_config = model_config.get('backbone', {})
        
        # 提取參數
        model_name = model_params.get('model_name', 'resnet50')
        pretrained = model_params.get('pretrained', True)
        input_channels = model_params.get('input_channels', 3)
        num_classes = model_params.get('num_classes', 5)
        dropout_rate = model_params.get('dropout_rate', 0.2)
        is_classification = model_params.get('is_classification', True)
        
        # 創建模型
        model = ResNetModel(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_channels=input_channels,
            dropout_rate=dropout_rate,
            is_classification=is_classification
        )
        
        # 凍結骨幹網絡
        freeze_backbone = backbone_config.get('freeze', False)
        if freeze_backbone:
            unfreeze_layers = backbone_config.get('unfreeze_layers', 1)
            model.freeze_backbone_layers(trainable_layers=unfreeze_layers)
            logger.info(f"已凍結ResNet骨幹網絡，解凍最後 {unfreeze_layers} 層")
        
        return model
    
    @staticmethod
    def _create_pytorch_cnn(config: Dict[str, Any]) -> Any:
        """創建PyTorch CNN模型
        
        Args:
            config: 配置字典
            
        Returns:
            Any: 創建的CNN模型實例
        """
        from .cnn_model import CNNModel
        
        model_config = config['model']
        model_params = model_config.get('parameters', {})
        
        # 提取參數
        input_channels = model_params.get('input_channels', 3)
        input_size = model_params.get('input_size', (224, 224))
        filters = model_params.get('filters', [32, 64, 128])
        kernel_sizes = model_params.get('kernel_sizes', [3, 3, 3])
        pool_sizes = model_params.get('pool_sizes', [2, 2, 2])
        fc_dims = model_params.get('fc_dims', [512])
        num_classes = model_params.get('num_classes', 5)
        dropout_rate = model_params.get('dropout_rate', 0.2)
        batch_norm = model_params.get('batch_norm', True)
        activation = model_params.get('activation', 'relu')
        is_classification = model_params.get('is_classification', True)
        
        # 創建模型
        model = CNNModel(
            input_channels=input_channels,
            input_size=input_size,
            filters=filters,
            kernel_sizes=kernel_sizes,
            pool_sizes=pool_sizes,
            fc_dims=fc_dims,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            activation=activation,
            is_classification=is_classification
        )
        
        return model


def create_model(config: Dict[str, Any]) -> Any:
    """便捷函數，創建模型
    
    Args:
        config: 配置字典
        
    Returns:
        Any: 創建的模型實例
    """
    return ModelFactory.create_model(config) 