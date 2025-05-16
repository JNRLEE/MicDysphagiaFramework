"""
Hook 功能測試：測試評估結果捕獲和激活值捕獲功能

測試日期: 2024-05-13
測試目的: 驗證新增的評估結果捕獲和激活值捕獲 Hook 功能是否正常工作
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Dict, Any
import logging
import sys
import shutil
from datetime import datetime

# 將當前目錄添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 導入需要測試的模塊
from models.hook_bridge import EvaluationResultsHook, ActivationCaptureHook
from utils.save_manager import SaveManager
from utils.callback_interface import CallbackInterface

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DummyModel(nn.Module):
    """測試用的簡單模型
    
    Args:
        input_dim: 輸入維度
        hidden_dim: 隱藏層維度
        output_dim: 輸出維度
        
    Returns:
        None
        
    Description:
        一個簡單的前饋神經網絡，用於測試 Hook 功能
        
    References:
        無
    """
    def __init__(self, input_dim: int = 4, hidden_dim: int = 8, output_dim: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.final_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        features = self.encoder(x)
        output = self.final_layer(features)
        return output

class DummyTrainer:
    """測試用的簡單訓練器
    
    Args:
        model: 要訓練的模型
        output_dir: 輸出目錄
        
    Returns:
        None
        
    Description:
        一個簡單的訓練器，用於測試 Hook 功能
        
    References:
        無
    """
    def __init__(self, model: nn.Module, output_dir: str = 'test_outputs'):
        self.model = model
        self.output_dir = output_dir
        self.save_manager = SaveManager(output_dir, create_subdirs=True)
        self.callbacks = []
    
    def add_callback(self, callback: CallbackInterface):
        """添加回調
        
        Args:
            callback: 回調實例
            
        Returns:
            None
        """
        self.callbacks.append(callback)
    
    def _call_callback_method(self, method_name: str, **kwargs):
        """調用回調方法
        
        Args:
            method_name: 方法名稱
            **kwargs: 其他參數
            
        Returns:
            None
        """
        for callback in self.callbacks:
            if hasattr(callback, method_name):
                method = getattr(callback, method_name)
                method(**kwargs)
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """評估模型
        
        Args:
            test_loader: 測試數據加載器
            
        Returns:
            Dict[str, Any]: 評估結果
        """
        self.model.eval()
        
        # 通知回調評估開始
        self._call_callback_method('on_evaluation_begin', model=self.model)
        
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                # 前向傳播
                outputs = self.model(inputs)
                
                # 保存結果
                all_outputs.append(outputs)
                all_targets.append(targets)
                
                # 通知回調批次結束
                batch_logs = {'phase': 'validation'}
                self._call_callback_method(
                    'on_batch_end',
                    batch=batch_idx,
                    model=self.model,
                    inputs=inputs,
                    targets=targets,
                    outputs=outputs,
                    logs=batch_logs
                )
        
        # 計算結果
        outputs = torch.cat(all_outputs)
        targets = torch.cat(all_targets)
        
        # 計算準確率（如果是分類任務）
        if outputs.shape[1] > 1:
            preds = outputs.argmax(dim=1)
            accuracy = (preds == targets).float().mean().item()
        else:
            accuracy = 0.0
        
        # 評估結果
        results = {
            'accuracy': accuracy
        }
        
        # 通知回調評估結束
        self._call_callback_method('on_evaluation_end', model=self.model, results=results)
        
        return results

def test_evaluation_results_hook():
    """測試評估結果捕獲 Hook
    
    Args:
        None
        
    Returns:
        None
        
    Description:
        測試 EvaluationResultsHook 是否能正確捕獲和保存評估結果
        
    References:
        無
    """
    logger.info("開始測試 EvaluationResultsHook...")
    
    # 創建輸出目錄
    output_dir = os.path.join('tests', 'test_outputs', f'hook_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 創建模型和測試數據
    model = DummyModel(input_dim=4, output_dim=2)
    model.eval()
    
    # 創建測試數據
    X = torch.randn(20, 4)
    y = torch.randint(0, 2, (20,))
    dataset = TensorDataset(X, y)
    test_loader = DataLoader(dataset, batch_size=4)
    
    # 創建存檔管理器和評估結果捕獲鉤子
    save_manager = SaveManager(output_dir, create_subdirs=True)
    evaluation_hook = EvaluationResultsHook(save_manager)
    
    # 創建訓練器並添加鉤子
    trainer = DummyTrainer(model, output_dir)
    trainer.add_callback(evaluation_hook)
    
    # 執行評估
    results = trainer.evaluate(test_loader)
    
    # 檢查結果文件是否存在
    hooks_dir = os.path.join(output_dir, 'hooks')
    result_file = os.path.join(hooks_dir, 'evaluation_results_test.pt')
    
    if os.path.exists(result_file):
        logger.info(f"成功生成評估結果文件: {result_file}")
        
        # 讀取結果檢查內容
        data = torch.load(result_file)
        logger.info(f"評估結果文件包含以下鍵: {list(data.keys())}")
        
        if 'targets' in data and 'predictions' in data:
            logger.info(f"目標形狀: {data['targets'].shape}")
            logger.info(f"預測形狀: {data['predictions'].shape}")
            logger.info("EvaluationResultsHook 測試成功!")
        else:
            logger.error("評估結果文件格式不正確")
    else:
        logger.error(f"評估結果文件不存在: {result_file}")
    
    return output_dir

def test_activation_capture_hook():
    """測試激活值捕獲 Hook
    
    Args:
        None
        
    Returns:
        None
        
    Description:
        測試 ActivationCaptureHook 是否能正確捕獲和保存特定層的激活值
        
    References:
        無
    """
    logger.info("開始測試 ActivationCaptureHook...")
    
    # 創建輸出目錄
    output_dir = os.path.join('tests', 'test_outputs', f'hook_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 創建模型和測試數據
    model = DummyModel(input_dim=4, output_dim=2)
    model.eval()
    
    # 創建測試數據
    X = torch.randn(20, 4)
    y = torch.randint(0, 2, (20,))
    dataset = TensorDataset(X, y)
    test_loader = DataLoader(dataset, batch_size=4)
    
    # 創建存檔管理器和激活值捕獲鉤子
    save_manager = SaveManager(output_dir, create_subdirs=True)
    target_layers = ['encoder', 'final_layer']
    activation_hook = ActivationCaptureHook(model, target_layers, save_manager)
    
    # 創建訓練器並添加鉤子
    trainer = DummyTrainer(model, output_dir)
    trainer.add_callback(activation_hook)
    
    # 執行評估
    results = trainer.evaluate(test_loader)
    
    # 檢查結果文件是否存在
    hooks_dir = os.path.join(output_dir, 'hooks')
    success = True
    
    for layer_name in target_layers:
        result_file = os.path.join(hooks_dir, f'test_set_activations_{layer_name.replace(".", "_")}.pt')
        
        if os.path.exists(result_file):
            logger.info(f"成功生成層 '{layer_name}' 的激活值文件: {result_file}")
            
            # 讀取結果檢查內容
            data = torch.load(result_file)
            logger.info(f"激活值文件包含以下鍵: {list(data.keys())}")
            
            if 'layer_name' in data and 'activations' in data:
                logger.info(f"層名稱: {data['layer_name']}")
                logger.info(f"激活值形狀: {data['activations'].shape}")
            else:
                logger.error(f"層 '{layer_name}' 的激活值文件格式不正確")
                success = False
        else:
            logger.error(f"層 '{layer_name}' 的激活值文件不存在: {result_file}")
            success = False
    
    if success:
        logger.info("ActivationCaptureHook 測試成功!")
    
    return output_dir

def run_all_tests():
    """運行所有測試
    
    Args:
        None
        
    Returns:
        None
        
    Description:
        依次運行所有 Hook 測試，並清理測試目錄
        
    References:
        無
    """
    test_dirs = []
    
    # 運行評估結果捕獲 Hook 測試
    output_dir1 = test_evaluation_results_hook()
    test_dirs.append(output_dir1)
    
    # 運行激活值捕獲 Hook 測試
    output_dir2 = test_activation_capture_hook()
    test_dirs.append(output_dir2)
    
    # 清理測試目錄（可選）
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            # shutil.rmtree(test_dir)
            pass

if __name__ == "__main__":
    """
    Description: 運行 Hook 功能測試
    測試日期: 2024-05-13
    測試目的: 驗證新增的評估結果捕獲和激活值捕獲 Hook 功能是否正常工作
    Args: None
    Returns: None
    References: 無
    """
    run_all_tests() 