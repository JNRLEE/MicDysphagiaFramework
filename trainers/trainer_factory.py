"""
訓練器工廠模組：根據配置創建適當的訓練器
功能：
1. 支持PyTorch框架
2. 創建適合不同模型和任務的訓練器
3. 處理訓練配置和回調設置
4. 提供統一的訓練接口
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class TrainerFactory:
    """訓練器工廠類，根據配置創建訓練器"""
    
    @staticmethod
    def create_trainer(config: Dict[str, Any], model, dataloaders: Tuple) -> Any:
        """創建訓練器
        
        Args:
            config: 配置字典，包含 'model' 和 'training' 部分
            model: 模型實例
            dataloaders: 包含訓練、驗證和測試數據加載器的元組
            
        Returns:
            Any: 創建的訓練器實例
        """
        return TrainerFactory._create_pytorch_trainer(config, model, dataloaders)
    
    @staticmethod
    def _create_pytorch_trainer(config: Dict[str, Any], model, dataloaders: Tuple) -> Any:
        """創建PyTorch訓練器
        
        Args:
            config: 配置字典
            model: PyTorch模型實例
            dataloaders: 包含訓練、驗證和測試數據加載器的元組
            
        Returns:
            Any: 創建的PyTorch訓練器實例
        """
        from .pytorch_trainer import PyTorchTrainer
        
        train_loader, val_loader, test_loader = dataloaders
        
        # 獲取訓練配置
        training_config = config['training']
        model_config = config['model']
        loss_config = training_config.get('loss', {})
        optimizer_config = training_config.get('optimizer', {})
        evaluation_config = config.get('evaluation', {})
        
        # 確定任務類型
        model_params = model_config.get('parameters', {})
        is_classification = model_params.get('is_classification', True)
        
        # 創建訓練器 - 只傳遞配置和模型
        trainer = PyTorchTrainer(
            config=config,
            model=model
        )
        
        # 使用訓練方法訓練模型
        result = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        )
        
        return trainer


def create_trainer(config: Dict[str, Any], model, dataloaders: Tuple) -> Any:
    """便捷函數，創建訓練器
    
    Args:
        config: 配置字典
        model: 模型實例
        dataloaders: 包含訓練、驗證和測試數據加載器的元組
        
    Returns:
        Any: 創建的訓練器實例
    """
    return TrainerFactory.create_trainer(config, model, dataloaders) 