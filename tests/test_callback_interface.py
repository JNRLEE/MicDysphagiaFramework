"""
回調接口測試模塊：驗證回調接口和模型鉤子功能

這個模塊測試 CallbackInterface 的各種回調方法是否被正確調用，
以及模型鉤子能否成功獲取模型內部狀態。
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import unittest
import logging
from typing import Dict, Any, List

# 添加項目根目錄到路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 引入要測試的模塊
from utils.callback_interface import CallbackInterface
from models.hook_bridge import SimpleModelHookManager as ModelHookManager, SimpleActivationHook as ActivationHook, SimpleGradientHook as GradientHook
from models.model_factory import create_model
from trainers.pytorch_trainer import PyTorchTrainer

# 配置日誌
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestCallback(CallbackInterface):
    """用於測試的回調類
    
    記錄各回調方法的調用次數和傳遞的參數，用於驗證回調機制正常工作。
    """
    
    def __init__(self):
        """初始化測試回調"""
        self.call_counts = {
            'on_train_begin': 0,
            'on_epoch_begin': 0,
            'on_batch_begin': 0,
            'on_batch_end': 0,
            'on_epoch_end': 0,
            'on_train_end': 0,
            'on_evaluation_begin': 0,
            'on_evaluation_end': 0
        }
        self.last_logs = {}
        
    def on_train_begin(self, model: nn.Module, logs: Dict[str, Any] = None) -> None:
        """訓練開始時調用"""
        self.call_counts['on_train_begin'] += 1
        self.last_logs['on_train_begin'] = logs or {}
        logger.info(f"on_train_begin 被調用 - 當前調用次數: {self.call_counts['on_train_begin']}")
        
    def on_epoch_begin(self, epoch: int, model: nn.Module, logs: Dict[str, Any] = None) -> None:
        """每個 epoch 開始時調用"""
        self.call_counts['on_epoch_begin'] += 1
        self.last_logs['on_epoch_begin'] = logs or {}
        logger.info(f"on_epoch_begin 被調用 - epoch: {epoch}, 當前調用次數: {self.call_counts['on_epoch_begin']}")
    
    def on_batch_begin(self, batch: int, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, logs: Dict[str, Any] = None) -> None:
        """每個批次開始時調用"""
        self.call_counts['on_batch_begin'] += 1
        self.last_logs['on_batch_begin'] = logs or {}
        logger.info(f"on_batch_begin 被調用 - batch: {batch}, 當前調用次數: {self.call_counts['on_batch_begin']}")
    
    def on_batch_end(self, batch: int, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, 
                    outputs: torch.Tensor, loss: torch.Tensor, logs: Dict[str, Any] = None) -> None:
        """每個批次結束時調用"""
        self.call_counts['on_batch_end'] += 1
        self.last_logs['on_batch_end'] = logs or {}
        logger.info(f"on_batch_end 被調用 - batch: {batch}, loss: {loss.item():.4f}, 當前調用次數: {self.call_counts['on_batch_end']}")
    
    def on_epoch_end(self, epoch: int, model: nn.Module, train_logs: Dict[str, Any], 
                   val_logs: Dict[str, Any], logs: Dict[str, Any] = None) -> None:
        """每個 epoch 結束時調用"""
        self.call_counts['on_epoch_end'] += 1
        self.last_logs['on_epoch_end'] = logs or {}
        logger.info(f"on_epoch_end 被調用 - epoch: {epoch}, train_loss: {train_logs.get('loss', 'NA')}, val_loss: {val_logs.get('loss', 'NA')}, 當前調用次數: {self.call_counts['on_epoch_end']}")
    
    def on_train_end(self, model: nn.Module, history: Dict[str, List], logs: Dict[str, Any] = None) -> None:
        """訓練結束時調用"""
        self.call_counts['on_train_end'] += 1
        self.last_logs['on_train_end'] = logs or {}
        logger.info(f"on_train_end 被調用 - 當前調用次數: {self.call_counts['on_train_end']}")

    def on_evaluation_begin(self, model: nn.Module, logs: Dict[str, Any] = None) -> None:
        """評估開始時調用"""
        self.call_counts['on_evaluation_begin'] += 1
        self.last_logs['on_evaluation_begin'] = logs or {}
        logger.info(f"on_evaluation_begin 被調用 - 當前調用次數: {self.call_counts['on_evaluation_begin']}")

    def on_evaluation_end(self, model: nn.Module, results: Dict[str, Any], logs: Dict[str, Any] = None) -> None:
        """評估結束時調用"""
        self.call_counts['on_evaluation_end'] += 1
        self.last_logs['on_evaluation_end'] = logs or {}
        logger.info(f"on_evaluation_end 被調用 - results: {results}, 當前調用次數: {self.call_counts['on_evaluation_end']}")


class ModelHookCallback(CallbackInterface):
    """模型鉤子回調
    
    使用 ModelHookManager 在訓練過程中收集和記錄模型內部狀態
    """
    
    def __init__(self, log_freq: int = 1):
        """初始化模型鉤子回調
        
        Args:
            log_freq: 記錄頻率，每隔多少批次記錄一次
        """
        self.hook_manager = None
        self.log_freq = log_freq
        self.batch_counter = 0
        
    def on_train_begin(self, model: nn.Module, logs: Dict[str, Any] = None) -> None:
        """訓練開始時初始化鉤子管理器"""
        logger.info("初始化模型鉤子管理器")
        self.hook_manager = ModelHookManager(model)
        
        # 記錄模型概要
        model_summary = self.hook_manager.get_model_summary()
        logger.info(f"模型參數總數: {model_summary['num_parameters']}")
        logger.info(f"可訓練參數數量: {model_summary['num_trainable_parameters']}")
        
    def on_batch_end(self, batch: int, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, 
                    outputs: torch.Tensor, loss: torch.Tensor, logs: Dict[str, Any] = None) -> None:
        """每個批次結束時收集模型內部狀態"""
        if self.hook_manager is None:
            return
            
        self.batch_counter += 1
        
        # 更新批次數據
        self.hook_manager.update_batch_data(inputs, outputs, targets, loss)
        
        # 按頻率記錄
        if self.batch_counter % self.log_freq == 0:
            # 獲取梯度統計數據
            grad_stats = self.hook_manager.get_gradient_statistics()
            
            # 獲取權重統計數據
            weight_stats = self.hook_manager.get_weight_statistics()
            
            # 獲取層輸出統計數據
            activation_stats = self.hook_manager.get_layer_output_statistics()
            
            # 記錄一些統計數據（這裡僅做示範，實際使用時可能需要進一步處理）
            logger.info(f"批次 {batch} 模型狀態:")
            
            # 記錄梯度信息
            for name, stats in list(grad_stats.items())[:2]:  # 只記錄前兩個參數的梯度，避免輸出過多
                logger.info(f"  梯度 {name}: 均值={stats.get('mean', 'NA'):.6f}, 範數={stats.get('norm', 'NA'):.6f}")
            
            # 記錄權重信息
            for name, stats in list(weight_stats.items())[:2]:  # 只記錄前兩個參數的權重
                logger.info(f"  權重 {name}: 均值={stats.get('mean', 'NA'):.6f}, 標準差={stats.get('std', 'NA'):.6f}")
            
            # 記錄激活信息
            for name, stats in list(activation_stats.items())[:2]:  # 只記錄前兩個層的激活
                logger.info(f"  激活 {name}: 均值={stats.get('mean', 'NA'):.6f}, 激活比例={stats.get('activation_ratio', 'NA'):.4f}")
    
    def on_train_end(self, model: nn.Module, history: Dict[str, List], logs: Dict[str, Any] = None) -> None:
        """訓練結束時清理鉤子"""
        if self.hook_manager is not None:
            logger.info("移除模型鉤子")
            self.hook_manager.remove_hooks()
            self.hook_manager = None


def test_callbacks():
    """測試回調接口功能
    
    創建一個簡單的模型和數據集，使用 PyTorchTrainer 進行訓練，
    並通過測試回調驗證各個回調方法是否被正確調用。
    """
    logger.info("開始測試回調接口...")
    
    # 創建測試數據集
    X = torch.randn(100, 1, 28, 28)
    y = torch.randint(0, 10, (100,))
    train_dataset = TensorDataset(X[:80], y[:80])
    val_dataset = TensorDataset(X[80:90], y[80:90])
    test_dataset = TensorDataset(X[90:], y[90:])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # 創建簡單的模型配置
    config = {
        'model': {
            'type': 'cnn',
            'parameters': {
                'input_channels': 1,
                'num_classes': 10,
                'is_classification': True
            }
        },
        'data': {
            'type': 'image',
            'dataloader': {
                'batch_size': 16,
                'num_workers': 0,
                'pin_memory': False
            }
        },
        'training': {
            'epochs': 2,
            'loss': {
                'type': 'CrossEntropyLoss'
            },
            'optimizer': {
                'type': 'Adam',
                'parameters': {
                    'lr': 0.001
                }
            }
        },
        'output_dir': 'tests/output'
    }
    
    # 創建模型
    model = create_model(config)
    
    # 創建測試回調
    test_callback = TestCallback()
    hook_callback = ModelHookCallback(log_freq=5)
    
    # 創建訓練器並添加回調
    trainer = PyTorchTrainer(config, model)
    trainer.add_callbacks([test_callback, hook_callback])
    
    # 修改 train 方法以避免調用 adapt_datasets_to_model
    original_train = trainer.train
    original_prepare_batch = trainer._prepare_batch
    
    # 添加處理TensorDataset輸出列表的能力
    def patched_prepare_batch(batch):
        # 對於TensorDataset，batch是一個列表
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            # 第一個元素是輸入，第二個是標籤
            return {
                'image': batch[0].to(trainer.device),  # 使用 'image' 作為鍵，與 _forward_pass 方法一致
                'label': batch[1].to(trainer.device)   # 使用 'label' 作為鍵，與 _compute_loss 方法一致
            }
        else:
            # 對於其他類型的批次，使用原始方法
            return original_prepare_batch(batch)
    
    def patched_train(*args, **kwargs):
        # 保存原來的方法
        from utils.data_adapter import adapt_datasets_to_model
        original_adapt = adapt_datasets_to_model
        
        # 替換為無操作的函數
        def noop_adapt(*args, **kwargs):
            return args[2], args[3], args[4]  # 直接返回原始數據加載器
        
        # 打補丁
        import utils.data_adapter
        utils.data_adapter.adapt_datasets_to_model = noop_adapt
        
        # 替換_prepare_batch方法
        trainer._prepare_batch = patched_prepare_batch
        
        try:
            # 調用原始方法
            result = original_train(*args, **kwargs)
        finally:
            # 恢復原始函數
            utils.data_adapter.adapt_datasets_to_model = original_adapt
            trainer._prepare_batch = original_prepare_batch
            
        return result
    
    # 應用補丁
    trainer.train = patched_train
    
    # 訓練模型
    logger.info("開始訓練...")
    trainer.train(train_loader, val_loader, test_loader)
    
    # 驗證回調調用次數
    logger.info("回調調用統計:")
    for method, count in test_callback.call_counts.items():
        logger.info(f"  {method}: {count} 次")
    
    # 簡單驗證
    epochs = config['training']['epochs']
    assert test_callback.call_counts['on_train_begin'] == 1, "訓練開始回調應該被調用一次"
    assert test_callback.call_counts['on_epoch_begin'] == epochs, f"Epoch 開始回調應該被調用 {epochs} 次"
    assert test_callback.call_counts['on_epoch_end'] == epochs, f"Epoch 結束回調應該被調用 {epochs} 次"
    assert test_callback.call_counts['on_batch_begin'] > 0, "批次開始回調應該被調用多次"
    assert test_callback.call_counts['on_batch_end'] > 0, "批次結束回調應該被調用多次"
    assert test_callback.call_counts['on_train_end'] == 1, "訓練結束回調應該被調用一次"
    assert test_callback.call_counts['on_evaluation_begin'] >= 1, "評估開始回調應該被調用至少一次"
    assert test_callback.call_counts['on_evaluation_end'] >= 1, "評估結束回調應該被調用至少一次"
    
    logger.info("回調測試通過！")
    
    return test_callback.call_counts


if __name__ == "__main__":
    test_callbacks() 