"""
模型結構信息模組：用於提取、保存和加載模型結構信息

此模組提供了 ModelStructureInfo 類，用於從 PyTorch 模型中提取結構信息，
以便於 SBP_analyzer 等外部工具進行分析和可視化。
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Union, Set
import json
import os
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

class ModelStructureInfo:
    """模型結構信息工具類，用於提取和保存模型架構信息"""
    
    def __init__(self, model: nn.Module = None, model_type: str = None):
        """初始化模型結構信息工具類
        
        Args:
            model: PyTorch 模型實例
            model_type: 模型類型，如果未提供則嘗試從模型屬性獲取
        """
        self.model = model
        self.model_type = model_type
        self.structure_info = None
        
        if model is not None:
            self.extract_structure()
    
    def extract_structure(self) -> Dict[str, Any]:
        """從模型中提取結構信息
        
        Returns:
            Dict[str, Any]: 模型結構信息字典
        """
        if self.model is None:
            raise ValueError("未設置模型，無法提取結構信息")
        
        self.structure_info = self.extract_structure_info(self.model, self.model_type)
        return self.structure_info
    
    def get_structure_dict(self) -> Dict[str, Any]:
        """獲取模型結構信息字典
        
        Returns:
            Dict[str, Any]: 模型結構信息字典
        """
        if self.structure_info is None and self.model is not None:
            self.extract_structure()
            
        if self.structure_info is None:
            raise ValueError("未提取結構信息，請先調用 extract_structure() 方法")
            
        return self.structure_info
    
    @staticmethod
    def extract_structure_info(model: nn.Module, model_type: str = None) -> Dict[str, Any]:
        """從模型中提取結構信息
        
        Args:
            model: PyTorch 模型實例
            model_type: 模型類型，如果未提供則嘗試從模型屬性獲取
            
        Returns:
            Dict[str, Any]: 模型結構信息字典
        """
        # 如果模型自己提供了結構信息方法，優先使用
        if hasattr(model, 'get_structure_info'):
            return model.get_structure_info()
        
        # 確定模型類型
        if not model_type:
            model_type = getattr(model, 'model_type', type(model).__name__)
        
        # 獲取模型摘要信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 分析模型層結構
        layers_info = []
        for name, module in model.named_modules():
            if name:  # 排除根模型自身
                params = sum(p.numel() for p in module.parameters())
                trainable_params_layer = sum(p.numel() for p in module.parameters() if p.requires_grad)
                
                layers_info.append({
                    'name': name,
                    'type': type(module).__name__,
                    'parameters': params,
                    'trainable_parameters': trainable_params_layer
                })
        
        # 識別關鍵層
        key_layers = ModelStructureInfo._identify_key_layers(model, model_type, layers_info)
        
        # 獲取參數信息
        parameters_info = []
        for name, param in model.named_parameters():
            parameters_info.append({
                'name': name,
                'shape': list(param.shape),
                'size': param.numel(),
                'trainable': param.requires_grad
            })
        
        # 識別關鍵參數
        key_parameters = ModelStructureInfo._identify_key_parameters(model, model_type, parameters_info)
        
        # 構建完整結構信息
        structure_info = {
            'model_type': model_type,
            'model_class': type(model).__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'layers': layers_info,
            'parameters': parameters_info,
            'key_layers': key_layers,
            'key_parameters': key_parameters
        }
        
        return structure_info

    @staticmethod
    def _identify_key_layers(model: nn.Module, model_type: str, layers_info: List[Dict[str, Any]]) -> Dict[str, str]:
        """識別模型中的關鍵層
        
        根據模型類型和層結構識別關鍵層，例如卷積層、Transformer塊等
        
        Args:
            model: PyTorch 模型實例
            model_type: 模型類型
            layers_info: 層信息列表
            
        Returns:
            Dict[str, str]: 關鍵層名稱映射
        """
        key_layers = {}
        
        # 基於模型類型和已知的層結構模式進行識別
        if 'swin_transformer' in model_type.lower():
            # 識別 Swin Transformer 的關鍵層
            for layer in layers_info:
                name = layer['name']
                if 'patch_embed' in name and len(name.split('.')) <= 2:
                    key_layers['embedding'] = name
                elif 'layers.0.blocks.0' in name and name.endswith('blocks.0'):
                    key_layers['first_block'] = name
                elif 'layers' in name and 'blocks' in name and name.split('.')[-2] == 'blocks' and name.split('.')[-1] == '0':
                    block_idx = name.split('.')[-3]
                    key_layers[f'layer_{block_idx}_first_block'] = name
                elif 'norm' in name and len(name.split('.')) <= 2:
                    key_layers['norm'] = name
                elif name == 'head' or name.endswith('.head'):
                    key_layers['head'] = name
        
        elif 'resnet' in model_type.lower():
            # 識別 ResNet 的關鍵層
            for layer in layers_info:
                name = layer['name']
                if 'conv1' in name and len(name.split('.')) <= 2:
                    key_layers['first_conv'] = name
                elif 'layer1' in name and len(name.split('.')) == 2:
                    key_layers['layer1'] = name
                elif 'layer2' in name and len(name.split('.')) == 2:
                    key_layers['layer2'] = name
                elif 'layer3' in name and len(name.split('.')) == 2:
                    key_layers['layer3'] = name
                elif 'layer4' in name and len(name.split('.')) == 2:
                    key_layers['layer4'] = name
                elif name == 'fc' or name.endswith('.fc'):
                    key_layers['fc'] = name
        
        elif 'cnn' in model_type.lower():
            # 識別 CNN 的關鍵層
            conv_count = 0
            fc_count = 0
            
            for layer in layers_info:
                name = layer['name']
                layer_type = layer['type']
                
                if 'Conv' in layer_type:
                    key_layers[f'conv_{conv_count}'] = name
                    conv_count += 1
                elif 'Linear' in layer_type:
                    key_layers[f'fc_{fc_count}'] = name
                    fc_count += 1
        
        # 如果沒有識別出關鍵層，嘗試通用方法
        if not key_layers:
            # 識別第一層和最後一層
            module_list = list(model.named_modules())
            if len(module_list) > 1:
                key_layers['first_layer'] = module_list[1][0]  # 跳過根模型
                key_layers['last_layer'] = module_list[-1][0]
        
        return key_layers

    @staticmethod
    def _identify_key_parameters(model: nn.Module, model_type: str, parameters_info: List[Dict[str, Any]]) -> Dict[str, str]:
        """識別模型中的關鍵參數
        
        根據模型類型和參數結構識別關鍵參數，例如權重矩陣、偏置等
        
        Args:
            model: PyTorch 模型實例
            model_type: 模型類型
            parameters_info: 參數信息列表
            
        Returns:
            Dict[str, str]: 關鍵參數名稱映射
        """
        key_parameters = {}
        
        # 基於模型類型和已知的參數結構模式進行識別
        if 'swin_transformer' in model_type.lower():
            for param_info in parameters_info:
                name = param_info['name']
                if 'patch_embed.proj.weight' in name:
                    key_parameters['embedding_weights'] = name
                elif 'head.weight' in name:
                    key_parameters['head_weights'] = name
                elif 'layers.0.blocks.0.mlp.fc1.weight' in name:
                    key_parameters['first_mlp_weights'] = name
        
        elif 'resnet' in model_type.lower():
            for param_info in parameters_info:
                name = param_info['name']
                if 'conv1.weight' in name:
                    key_parameters['first_conv_weights'] = name
                elif 'fc.weight' in name:
                    key_parameters['fc_weights'] = name
        
        # 如果沒有識別出關鍵參數，嘗試通用方法
        if not key_parameters:
            # 識別最大的幾個參數
            largest_params = sorted(parameters_info, key=lambda x: x['size'], reverse=True)[:3]
            for i, param in enumerate(largest_params):
                key_parameters[f'largest_param_{i}'] = param['name']
        
        return key_parameters
    
    @staticmethod
    def save_structure_info(model: nn.Module, save_path: str, model_type: str = None):
        """提取並保存模型結構信息到文件
        
        Args:
            model: PyTorch 模型實例
            save_path: 保存路徑
            model_type: 模型類型，如果未提供則嘗試從模型屬性獲取
        """
        structure_info = ModelStructureInfo.extract_structure_info(model, model_type)
        
        # 確保目錄存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存為 JSON 文件
        with open(save_path, 'w') as f:
            json.dump(structure_info, f, indent=2)
        
        logger.info(f"模型結構信息已保存到 {save_path}")
        
        return structure_info
    
    @staticmethod
    def load_structure_info(load_path: str) -> Dict[str, Any]:
        """從文件加載模型結構信息
        
        Args:
            load_path: 加載路徑
            
        Returns:
            Dict[str, Any]: 模型結構信息字典
        """
        with open(load_path, 'r') as f:
            structure_info = json.load(f)
        
        return structure_info

# 模型結構註冊表，用於儲存常見模型的結構信息
MODEL_STRUCTURES = {
    'swin_transformer': {
        'key_layers': {
            'embedding': 'model.patch_embed',
            'first_block': 'model.layers.0.blocks.0',
            'last_block': 'model.layers.3.blocks.1',
            'norm': 'model.norm',
            'head': 'head'
        },
        'key_parameters': {
            'embedding_weights': 'model.patch_embed.proj.weight',
            'head_weights': 'head.weight',
            'first_mlp_weights': 'model.layers.0.blocks.0.mlp.fc1.weight'
        }
    },
    'resnet': {
        'key_layers': {
            'first_conv': 'model.conv1',
            'layer1': 'model.layer1',
            'layer2': 'model.layer2',
            'layer3': 'model.layer3',
            'layer4': 'model.layer4',
            'fc': 'model.fc'
        },
        'key_parameters': {
            'first_conv_weights': 'model.conv1.weight',
            'fc_weights': 'model.fc.weight'
        }
    },
    'cnn': {
        'key_layers': {
            'first_conv': 'conv_layers.0',
            'last_conv': 'conv_layers.2',
            'first_fc': 'fc_layers.0',
            'last_fc': 'fc_layers.2'
        },
        'key_parameters': {
            'first_conv_weights': 'conv_layers.0.weight',
            'last_fc_weights': 'fc_layers.2.weight'
        }
    },
    'fcnn': {
        'key_layers': {
            'first_fc': 'layers.0',
            'last_fc': 'layers.2',
            'output': 'output'
        },
        'key_parameters': {
            'first_fc_weights': 'layers.0.weight',
            'output_weights': 'output.weight'
        }
    }
}

def get_registered_model_structure(model_type: str) -> Dict[str, Any]:
    """獲取註冊的模型結構信息
    
    Args:
        model_type: 模型類型
        
    Returns:
        Dict[str, Any]: 模型結構信息，如果未找到則返回空字典
    """
    return MODEL_STRUCTURES.get(model_type.lower(), {})

# 中文註解：這是model_structure.py的Minimal Executable Unit，檢查ModelStructureInfo能否正確提取模型結構，並測試錯誤情境時的優雅報錯
if __name__ == "__main__":
    """
    Description: Minimal Executable Unit for model_structure.py，檢查ModelStructureInfo能否正確提取模型結構，並測試錯誤情境時的優雅報錯。
    Args: None
    Returns: None
    References: 無
    """
    import torch
    import torch.nn as nn
    import logging
    logging.basicConfig(level=logging.INFO)
    # 不要再import自己
    # from models.model_structure import ModelStructureInfo

    # 測試正確模型
    try:
        class DummyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 2)
            def forward(self, x):
                return self.fc(x)
        model = DummyNet()
        info = ModelStructureInfo(model)
        struct = info.get_structure_dict()
        print(f"ModelStructureInfo測試成功，key_layers: {struct.get('key_layers', {})}")
    except Exception as e:
        print(f"ModelStructureInfo遇到錯誤（預期行為）: {e}")

    # 測試未設置模型
    try:
        info = ModelStructureInfo()
        struct = info.get_structure_dict()
        print(f"ModelStructureInfo未設置模型測試，key_layers: {struct.get('key_layers', {})}")
    except Exception as e:
        print(f"ModelStructureInfo遇到錯誤（預期行為）: {e}") 