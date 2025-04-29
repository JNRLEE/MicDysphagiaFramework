"""
此模組實現了損失函數工廠，用於創建和管理各種損失函數。
包括PyTorch標準損失函數和自定義排序損失函數。
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Union, Optional, List, Callable

logger = logging.getLogger(__name__)

class LossFactory:
    """
    損失函數工廠類，用於創建和管理各種損失函數。
    支持標準PyTorch損失函數和自定義排序損失函數。
    """
    _loss_registry = {}

    @classmethod
    def register_loss(cls, name: str) -> Callable:
        """
        裝飾器，用於註冊損失函數到工廠
        
        Args:
            name: 損失函數的名稱
            
        Returns:
            裝飾器函數
        """
        def decorator(loss_cls):
            cls._loss_registry[name] = loss_cls
            logger.debug(f"註冊損失函數: {name}")
            return loss_cls
        return decorator

    @classmethod
    def get_loss(cls, loss_config: Dict[str, Any]) -> nn.Module:
        """
        根據配置創建損失函數
        
        Args:
            loss_config: 損失函數的配置
            
        Returns:
            創建的損失函數
            
        Raises:
            ValueError: 如果損失函數類型不存在
        """
        loss_type = loss_config.get('type')
        loss_params = loss_config.get('parameters', {})
        
        # 處理標準PyTorch損失函數
        if loss_type in cls._loss_registry:
            logger.info(f"創建損失函數: {loss_type}")
            return cls._loss_registry[loss_type](**loss_params)
            
        # 如果不是註冊的損失函數，嘗試從torch.nn中獲取
        elif hasattr(nn, loss_type):
            logger.info(f"使用PyTorch內建損失函數: {loss_type}")
            loss_cls = getattr(nn, loss_type)
            return loss_cls(**loss_params)
        
        # 如果什麼都沒找到，拋出錯誤
        else:
            available_losses = list(cls._loss_registry.keys()) + [
                name for name in dir(nn) if isinstance(getattr(nn, name), type) 
                and issubclass(getattr(nn, name), nn.Module) 
                and 'Loss' in name
            ]
            raise ValueError(f"未知的損失函數類型: {loss_type}。可用的損失函數: {available_losses}")
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> nn.Module:
        """
        從配置創建損失函數
        
        Args:
            config: 損失函數配置，可能包含多個損失函數
            
        Returns:
            創建的損失函數
        """
        if 'combined' in config:
            # 如果是組合損失函數，則使用CombinedLoss
            from .combined_losses import CombinedLoss
            
            losses = {}
            weights = {}
            
            for loss_name, loss_detail in config['combined'].items():
                loss_weight = loss_detail.get('weight', 1.0)
                loss_config = {
                    'type': loss_detail.get('type'),
                    'parameters': loss_detail.get('parameters', {})
                }
                
                losses[loss_name] = cls.get_loss(loss_config)
                weights[loss_name] = loss_weight
                
            return CombinedLoss(losses, weights)
        else:
            # 單一損失函數
            return cls.get_loss(config)
    
    @classmethod
    def list_available_losses(cls) -> List[str]:
        """
        列出所有可用的損失函數
        
        Returns:
            可用損失函數列表
        """
        # 從PyTorch獲取所有標準損失函數
        pytorch_losses = [
            name for name in dir(nn) if isinstance(getattr(nn, name), type) 
            and issubclass(getattr(nn, name), nn.Module) 
            and 'Loss' in name
        ]
        
        # 合併註冊的損失函數和PyTorch損失函數
        return list(cls._loss_registry.keys()) + pytorch_losses

# 註冊標準PyTorch損失函數的包裝器
@LossFactory.register_loss("MSELoss")
class MSELossWrapper(nn.MSELoss):
    """MSE損失函數的包裝器"""
    pass

@LossFactory.register_loss("L1Loss")
class L1LossWrapper(nn.L1Loss):
    """L1損失函數的包裝器"""
    pass

@LossFactory.register_loss("CrossEntropyLoss")
class CrossEntropyLossWrapper(nn.CrossEntropyLoss):
    """交叉熵損失函數的包裝器"""
    pass

@LossFactory.register_loss("BCELoss")
class BCELossWrapper(nn.BCELoss):
    """二元交叉熵損失函數的包裝器"""
    pass

@LossFactory.register_loss("BCEWithLogitsLoss")
class BCEWithLogitsLossWrapper(nn.BCEWithLogitsLoss):
    """帶Logits的二元交叉熵損失函數的包裝器"""
    pass

@LossFactory.register_loss("NLLLoss")
class NLLLossWrapper(nn.NLLLoss):
    """負對數似然損失函數的包裝器"""
    pass

@LossFactory.register_loss("KLDivLoss")
class KLDivLossWrapper(nn.KLDivLoss):
    """KL散度損失函數的包裝器"""
    pass

@LossFactory.register_loss("SmoothL1Loss")
class SmoothL1LossWrapper(nn.SmoothL1Loss):
    """平滑L1損失函數的包裝器"""
    pass

@LossFactory.register_loss("HuberLoss")
class HuberLossWrapper(nn.HuberLoss):
    """Huber損失函數的包裝器"""
    pass

# 其他需要的標準PyTorch損失函數可以在這裡註冊...

# 中文註解：這是loss_factory.py的Minimal Executable Unit，檢查LossFactory能否正確創建損失函數，並測試錯誤type時的優雅報錯
if __name__ == "__main__":
    """
    Description: Minimal Executable Unit for loss_factory.py，檢查LossFactory能否正確創建損失函數，並測試錯誤type時的優雅報錯。
    Args: None
    Returns: None
    References: 無
    """
    import torch
    import logging
    logging.basicConfig(level=logging.INFO)
    # 不要再import自己
    # from losses.loss_factory import LossFactory

    # 測試正確type
    try:
        config = {"type": "MSELoss", "parameters": {}}
        loss_fn = LossFactory.get_loss(config)
        pred = torch.randn(4, 3)
        target = torch.randn(4, 3)
        loss = loss_fn(pred, target)
        print(f"MSELoss測試成功，loss: {loss.item()}")
    except Exception as e:
        print(f"遇到錯誤（預期行為）: {e}")

    # 測試錯誤type
    try:
        config = {"type": "NotExistLoss", "parameters": {}}
        loss_fn = LossFactory.get_loss(config)
    except Exception as e:
        print(f"遇到錯誤type時的報錯（預期行為）: {e}") 