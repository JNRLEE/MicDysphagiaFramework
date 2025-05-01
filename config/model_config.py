"""
模型配置模組：提供默認模型配置
功能：
1. 為各種模型類型提供默認配置
2. 方便在配置文件中引用基本配置
3. 支持 CNN、ResNet、Swin Transformer 和 FCNN 模型
"""

from typing import Dict, Any, List, Optional, Union, Tuple

def get_default_model_config(model_type: str) -> Dict[str, Any]:
    """獲取默認模型配置
    
    Args:
        model_type: 模型類型，支持 'cnn', 'resnet', 'swin_transformer', 'fcnn'
        
    Returns:
        Dict[str, Any]: 默認模型配置
        
    Raises:
        ValueError: 如果指定了不支持的模型類型
    """
    
    # CNN 默認配置
    if model_type == 'cnn':
        return {
            'type': 'cnn',
            'parameters': {
                'input_channels': 3,
                'input_size': (224, 224),
                'filters': [32, 64, 128, 256],
                'kernel_sizes': [3, 3, 3, 3],
                'pool_sizes': [2, 2, 2, 2],
                'fc_dims': [512, 256],
                'num_classes': 5,
                'dropout_rate': 0.5,
                'batch_norm': True,
                'activation': 'relu',
                'is_classification': True
            }
        }
    
    # ResNet 默認配置
    elif model_type == 'resnet':
        return {
            'type': 'resnet',
            'parameters': {
                'model_name': 'resnet50',
                'pretrained': True,
                'input_channels': 3,
                'num_classes': 5,
                'dropout_rate': 0.3,
                'is_classification': True
            },
            'backbone': {
                'freeze': False,
                'unfreeze_layers': 2
            }
        }
    
    # Swin Transformer 默認配置
    elif model_type == 'swin_transformer':
        return {
            'type': 'swin_transformer',
            'parameters': {
                'model_name': 'swin_tiny_patch4_window7_224',
                'pretrained': True,
                'input_channels': 3,
                'input_size': (224, 224),
                'num_classes': 5,
                'dropout_rate': 0.1,
                'is_classification': True
            },
            'visual_prompting': {
                'enabled': False,
                'prompt_size': (16, 16),
                'prompt_dropout': 0.1
            },
            'backbone': {
                'freeze': False,
                'unfreeze_layers': 2
            }
        }
    
    # FCNN (全連接神經網絡) 默認配置
    elif model_type == 'fcnn':
        return {
            'type': 'fcnn',
            'parameters': {
                'input_dim': 1024,
                'hidden_layers': [512, 256, 128, 64],
                'num_classes': 5,
                'dropout_rate': 0.5,
                'activation': 'relu',
                'is_classification': True,
                'batch_norm': True
            }
        }
    
    # 未識別的模型類型
    else:
        raise ValueError(f"不支持的模型類型: {model_type}")

def create_model_config(
    model_type: str,
    num_classes: int,
    is_classification: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """創建模型配置，基於默認配置並修改關鍵參數
    
    Args:
        model_type: 模型類型
        num_classes: 類別數量或輸出維度
        is_classification: 是否為分類任務
        **kwargs: 其他要修改的參數
    
    Returns:
        Dict[str, Any]: 修改後的模型配置
    """
    # 獲取默認配置
    config = get_default_model_config(model_type)
    
    # 修改關鍵參數
    config['parameters']['num_classes'] = num_classes
    config['parameters']['is_classification'] = is_classification
    
    # 修改其他參數
    for key, value in kwargs.items():
        # 參數可能位於不同層級
        if key in config:
            config[key] = value
        elif key in config.get('parameters', {}):
            config['parameters'][key] = value
        elif 'backbone' in config and key in config['backbone']:
            config['backbone'][key] = value
        elif 'visual_prompting' in config and key in config['visual_prompting']:
            config['visual_prompting'][key] = value
            
    return config 