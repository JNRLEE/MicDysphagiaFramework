"""
ActivationCaptureHook特徵向量保存修復測試

針對ActivationCaptureHook特徵向量保存的直接測試,模擬實際使用場景
"""

import os
import sys
import torch
import torch.nn as nn
import unittest
import shutil
import logging
from datetime import datetime
from typing import Dict, Any, List, Set, Optional
import numpy as np

# 將項目根目錄添加到系統路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.hook_bridge import ActivationCaptureHook
from utils.save_manager import SaveManager

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleModel(nn.Module):
    """測試用簡單模型"""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(3, 16, 3, padding=1)
        self.layer2 = nn.Conv2d(16, 32, 3, padding=1)
        self.layer3 = nn.Linear(32 * 8 * 8, 10)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

class DebugHook(ActivationCaptureHook):
    """用於調試的激活值捕獲鉤子"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_logs = []
        self.processed_epochs = set()
        
    def on_evaluation_end(self, model: nn.Module, results: Dict[str, Any] = None, 
                        logs: Dict[str, Any] = None) -> None:
        """重寫評估結束方法以便追蹤處理流程"""
        epoch = logs.get('epoch', 0) if logs else 0
        self.debug_logs.append(f"on_evaluation_end called for epoch {epoch}, captured_epochs={self.captured_epochs}")
        
        # 捕獲特徵向量的邏輯
        # 檢查當前epoch是否已被處理過
        if epoch in self.captured_epochs:
            self.debug_logs.append(f"Epoch {epoch} already processed, skipping")
            return
            
        # 檢查是否應該處理當前epoch
        should_process = self._should_process_epoch(epoch)
        self.debug_logs.append(f"_should_process_epoch({epoch}) = {should_process}")
        
        if not should_process:
            self.debug_logs.append(f"Skipping epoch {epoch}")
            return
            
        # 標記此epoch已處理，並進行特徵向量處理
        self.processed_epochs.add(epoch)
        self.debug_logs.append(f"Processing features for epoch {epoch}")
        
        # 模擬特徵向量處理
        feature_dir = self.save_manager.get_path('feature_vectors', '')
        os.makedirs(feature_dir, exist_ok=True)
        
        # 創建epoch特定目錄
        epoch_dir = os.path.join(feature_dir, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        # 為每個監控層保存一個假的特徵向量
        for layer_name in self.layer_names:
            feature_path = os.path.join(epoch_dir, f'layer_{layer_name.replace(".", "_")}_features.pt')
            # 創建一個假的特徵向量數據
            fake_data = {
                'layer_name': layer_name,
                'activations': torch.randn(10, 100),  # 假設有10個樣本，每個100維
                'timestamp': datetime.now().isoformat(),
                'epoch': epoch
            }
            torch.save(fake_data, feature_path)
            self.debug_logs.append(f"Saved fake feature vector for layer {layer_name} at epoch {epoch}")
        
        # 標記這個epoch已經處理過，避免重複處理
        self.captured_epochs.add(epoch)
        self.debug_logs.append(f"Added epoch {epoch} to captured_epochs: {self.captured_epochs}")

class TestActivationHookDebug(unittest.TestCase):
    """ActivationCaptureHook調試測試"""
    
    def setUp(self):
        """設置測試環境"""
        self.test_dir = os.path.join(project_root, "tests", "debug_test_dir")
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)
        
        self.save_manager = SaveManager(self.test_dir)
        self.model = SimpleModel()
        
    def tearDown(self):
        """清理測試環境"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_captured_epochs_tracking(self):
        """測試captured_epochs是否正確追蹤已處理的epoch"""
        # 創建鉤子，每個epoch都保存
        hook = DebugHook(
            model=self.model,
            layer_names=['layer1', 'layer2', 'layer3'],
            save_manager=self.save_manager,
            save_frequency=1
        )
        
        # 啟動訓練
        hook.on_train_begin(self.model, {'config': {'training': {'epochs': 3}}})
        
        # 檢查初始狀態
        self.assertEqual(hook.captured_epochs, set(), "初始captured_epochs應為空集合")
        
        # 模擬多個epoch的評估
        for epoch in range(3):
            # 評估開始
            hook.on_evaluation_begin(self.model, {'epoch': epoch})
            
            # 評估結束
            hook.on_evaluation_end(self.model, None, {'epoch': epoch})
            
            # 檢查此epoch是否被標記為已處理
            self.assertIn(epoch, hook.captured_epochs, f"Epoch {epoch}應該被標記為已處理")
            
            # 檢查特徵向量是否保存
            feature_dir = os.path.join(self.test_dir, "feature_vectors", f"epoch_{epoch}")
            self.assertTrue(os.path.exists(feature_dir), f"特徵向量目錄 {feature_dir} 應該存在")
            
            # 檢查每層的特徵向量文件
            for layer_name in ['layer1', 'layer2', 'layer3']:
                feature_file = os.path.join(feature_dir, f"layer_{layer_name}_features.pt")
                self.assertTrue(os.path.exists(feature_file), f"特徵向量文件 {feature_file} 應該存在")
        
        # 驗證所有epoch都被處理了
        self.assertEqual(hook.captured_epochs, {0, 1, 2}, "所有epoch應該都被處理")
        self.assertEqual(hook.processed_epochs, {0, 1, 2}, "所有epoch應該都被處理")
        
        # 打印調試日誌
        for log in hook.debug_logs:
            logger.info(log)
    
    def test_same_epoch_twice(self):
        """測試同一個epoch處理兩次的情況"""
        # 創建鉤子
        hook = DebugHook(
            model=self.model,
            layer_names=['layer1'],
            save_manager=self.save_manager,
            save_frequency=1
        )
        
        # 啟動訓練
        hook.on_train_begin(self.model, {'config': {'training': {'epochs': 2}}})
        
        # 第一次處理epoch 0
        hook.on_evaluation_begin(self.model, {'epoch': 0})
        hook.on_evaluation_end(self.model, None, {'epoch': 0})
        
        # 檢查epoch 0是否被標記為已處理
        self.assertIn(0, hook.captured_epochs, "Epoch 0應該被標記為已處理")
        
        # 再次處理epoch 0
        hook.on_evaluation_begin(self.model, {'epoch': 0})
        hook.on_evaluation_end(self.model, None, {'epoch': 0})
        
        # 打印調試日誌
        for log in hook.debug_logs:
            logger.info(log)
        
        # 檢查processed_epochs，應該只包含一個epoch 0
        self.assertEqual(hook.processed_epochs, {0}, "Epoch 0應該只被處理一次")
        
        # 檢查特徵向量目錄
        feature_dir = os.path.join(self.test_dir, "feature_vectors", "epoch_0")
        self.assertTrue(os.path.exists(feature_dir), "特徵向量目錄應該存在")
    
    def test_modified_captured_epochs_handling(self):
        """修改captured_epochs後的處理測試"""
        # 創建鉤子
        hook = DebugHook(
            model=self.model,
            layer_names=['layer1'],
            save_manager=self.save_manager,
            save_frequency=1
        )
        
        # 手動設置一些已捕獲的epoch
        hook.captured_epochs = {0, 2}
        
        # 模擬訓練開始
        hook.on_train_begin(self.model, {'config': {'training': {'epochs': 4}}})
        
        # 處理各個epoch
        for epoch in range(4):
            hook.on_evaluation_begin(self.model, {'epoch': epoch})
            hook.on_evaluation_end(self.model, None, {'epoch': epoch})
            
            # epoch 0和2應該被跳過
            if epoch in {0, 2}:
                self.assertNotIn(epoch, hook.processed_epochs, f"Epoch {epoch}應該被跳過")
            else:
                self.assertIn(epoch, hook.processed_epochs, f"Epoch {epoch}應該被處理")
        
        # 打印調試日誌
        for log in hook.debug_logs:
            logger.info(log)
        
        # 檢查最終結果
        self.assertEqual(hook.captured_epochs, {0, 1, 2, 3}, "所有epoch應該都在captured_epochs中")
        self.assertEqual(hook.processed_epochs, {1, 3}, "只有epoch 1和3應該被處理")

if __name__ == "__main__":
    unittest.main() 