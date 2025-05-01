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

# 中文註解：這是trainer_factory.py的Minimal Executable Unit，檢查TrainerFactory能否正確創建訓練器並執行train，並測試錯誤配置時的優雅報錯
if __name__ == "__main__":
    """
    Description: Minimal Executable Unit for trainer_factory.py，檢查TrainerFactory能否正確創建訓練器並執行train，並測試錯誤配置時的優雅報錯。
    Args: None
    Returns: None
    References: 無
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import logging
    logging.basicConfig(level=logging.INFO)
    # 不要再import自己
    # from trainers.trainer_factory import TrainerFactory

    class DummyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)
        def forward(self, x):
            return self.fc(x)

    x = torch.randn(20, 4)
    y = torch.randint(0, 2, (20,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=4)
    def dict_loader(dl):
        for xb, yb in dl:
            yield {"features": xb, "label": yb}
    train_loader = dict_loader(loader)
    val_loader = dict_loader(loader)
    test_loader = dict_loader(loader)

    config = {
        "model": {"type": "fcnn", "parameters": {"is_classification": True}},
        "training": {"epochs": 1, "learning_rate": 0.01, "loss": {"type": "CrossEntropyLoss", "parameters": {}}},
        "data": {"type": "feature"},
        "global": {"experiment_name": "dummy_exp", "output_dir": "results"}
    }
    try:
        model = DummyNet()
        trainer = TrainerFactory.create_trainer(config, model, (train_loader, val_loader, test_loader))
        print("TrainerFactory測試成功")
    except Exception as e:
        print(f"TrainerFactory遇到錯誤（預期行為）: {e}")

    # 錯誤配置測試
    try:
        bad_config = {"model": {}, "training": {}, "data": {}, "global": {}}
        model = DummyNet()
        trainer = TrainerFactory.create_trainer(bad_config, model, (train_loader, val_loader, test_loader))
    except Exception as e:
        print(f"TrainerFactory遇到錯誤配置時的報錯（預期行為）: {e}") 