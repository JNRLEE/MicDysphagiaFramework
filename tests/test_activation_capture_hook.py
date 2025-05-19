"""
測試ActivationCaptureHook的特徵向量保存功能

此測試腳本用來檢測ActivationCaptureHook是否正確地保存了每個epoch的特徵向量
"""

import os
import sys
import torch
import torch.nn as nn
import unittest
import shutil
from typing import Dict, Any, List, Set, Optional
import logging

# 將項目根目錄添加到系統路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.hook_bridge import ActivationCaptureHook
from utils.save_manager import SaveManager

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleModel(nn.Module):
    """測試用的簡單模型"""
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # backbone.0
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # backbone.4
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1)  # backbone.7
        )
        self.head = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),  # head.0
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.head(x)

class TestActivationCaptureHook(unittest.TestCase):
    """測試ActivationCaptureHook的功能"""
    
    def setUp(self):
        """設置測試環境"""
        self.test_dir = os.path.join(project_root, "tests", "test_output")
        os.makedirs(self.test_dir, exist_ok=True)
        self.save_manager = SaveManager(self.test_dir)
        self.model = SimpleModel()
        
        # 設置監控層
        self.layer_names = ["backbone.0", "backbone.4", "backbone.7", "head.0"]
        
    def tearDown(self):
        """清理測試環境"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_activation_capture_with_save_frequency(self):
        """測試使用save_frequency參數時的特徵向量保存功能"""
        # 創建ActivationCaptureHook，設置save_frequency=1
        hook = ActivationCaptureHook(
            model=self.model,
            layer_names=self.layer_names,
            save_manager=self.save_manager,
            dataset_name="test",
            save_frequency=1,
            compute_similarity=True,
            compute_tsne=False  # 禁用t-SNE以加速測試
        )
        
        # 打印初始狀態
        logger.info(f"初始captured_epochs: {hook.captured_epochs}")
        
        # 模擬訓練過程中的多個epoch
        total_epochs = 3
        
        # 檢查hook的save_frequency
        logger.info(f"Hook的save_frequency: {hook.save_frequency}")
        self.assertEqual(hook.save_frequency, 1, "save_frequency應該為1")
        
        # 模擬on_train_begin
        hook.on_train_begin(self.model, {"config": {"training": {"epochs": total_epochs}}})
        
        # 對於每個epoch
        for epoch in range(total_epochs):
            # 模擬on_evaluation_begin
            hook.on_evaluation_begin(self.model, {"epoch": epoch})
            
            # 模擬批次處理
            batch_size = 4
            for batch in range(2):  # 模擬2個批次
                # 創建假數據
                inputs = torch.randn(batch_size, 3, 32, 32)
                targets = torch.randint(0, 10, (batch_size,))
                outputs = self.model(inputs)
                
                # 模擬on_batch_end
                hook.on_batch_end(
                    batch=batch,
                    model=self.model,
                    inputs=inputs,
                    targets=targets,
                    outputs=outputs,
                    logs={"epoch": epoch}
                )
            
            # 模擬on_evaluation_end
            hook.on_evaluation_end(
                model=self.model,
                results={"predictions": torch.randint(0, 10, (batch_size,))},
                logs={"epoch": epoch}
            )
            
            # 打印當前狀態
            logger.info(f"After epoch {epoch}, captured_epochs: {hook.captured_epochs}")
            
            # 檢查該epoch的特徵向量是否已保存
            feature_dir = os.path.join(self.test_dir, "feature_vectors", f"epoch_{epoch}")
            logger.info(f"檢查目錄: {feature_dir}")
            
            # 檢查目錄是否存在
            self.assertTrue(os.path.exists(feature_dir), f"目錄 {feature_dir} 不存在")
            
            # 檢查是否為每個監控層保存了特徵向量
            for layer_name in self.layer_names:
                layer_name_safe = layer_name.replace(".", "_")
                feature_file = os.path.join(feature_dir, f"layer_{layer_name_safe}_features.pt")
                logger.info(f"檢查文件: {feature_file}")
                self.assertTrue(os.path.exists(feature_file), f"文件 {feature_file} 不存在")
    
    def test_should_process_epoch(self):
        """直接測試_should_process_epoch方法的邏輯"""
        # 創建ActivationCaptureHook，設置save_frequency=1
        hook = ActivationCaptureHook(
            model=self.model,
            layer_names=self.layer_names,
            save_manager=self.save_manager,
            dataset_name="test",
            save_frequency=1
        )
        
        # 測試已捕獲的epoch
        hook.captured_epochs = {0}
        self.assertFalse(hook._should_process_epoch(0), "已捕獲的epoch不應該被處理")
        
        # 測試save_frequency
        hook.captured_epochs = set()
        hook.save_frequency = 2
        hook.total_epochs = 10
        self.assertTrue(hook._should_process_epoch(0), "save_frequency=2時，epoch 0應該被處理(mod=0)")
        self.assertFalse(hook._should_process_epoch(1), "不是save_frequency倍數的epoch不應該被處理")
        self.assertTrue(hook._should_process_epoch(2), "是save_frequency倍數的epoch應該被處理")
        self.assertFalse(hook._should_process_epoch(9), "不是save_frequency倍數的epoch不應該被處理")
    
    def test_on_evaluation_end_logic(self):
        """測試on_evaluation_end方法中的特徵向量處理邏輯"""
        # 創建一個特殊的ActivationCaptureHook，使我們能夠檢查關鍵變量
        class TestableActivationCaptureHook(ActivationCaptureHook):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.process_feature_called_epochs = []
            
            def _process_features(self, epoch):
                self.process_feature_called_epochs.append(epoch)
                super()._process_features(epoch)
        
        # 創建hook實例
        hook = TestableActivationCaptureHook(
            model=self.model,
            layer_names=self.layer_names,
            save_manager=self.save_manager,
            dataset_name="test",
            save_frequency=1,
            compute_similarity=True,
            compute_tsne=False  # 禁用t-SNE以加速測試
        )
        
        # 模擬on_train_begin
        hook.on_train_begin(self.model, {"config": {"training": {"epochs": 3}}})
        
        # 模擬兩個連續的epoch評估
        for epoch in range(2):
            # 重置激活值
            hook.all_activations = {layer_name: [] for layer_name in self.layer_names}
            hook.all_targets = []
            
            # 模擬on_evaluation_begin
            hook.on_evaluation_begin(self.model, {"epoch": epoch})
            
            # 添加一些假的激活值
            for layer_name in self.layer_names:
                hook.all_activations[layer_name] = [torch.randn(4, 64)]
            hook.all_targets = [torch.randint(0, 10, (4,))]
            
            # 模擬on_evaluation_end
            hook.on_evaluation_end(
                model=self.model,
                results={"predictions": torch.randint(0, 10, (4,))},
                logs={"epoch": epoch}
            )
            
            # 檢查captured_epochs是否正確更新
            self.assertIn(epoch, hook.captured_epochs, f"epoch {epoch} 應該在captured_epochs中")
            
            # 檢查_process_features是否被調用
            self.assertIn(epoch, hook.process_feature_called_epochs, 
                         f"_process_features應該被調用於epoch {epoch}")
        
        # 檢查是否為每個epoch都保存了特徵向量
        for epoch in range(2):
            feature_dir = os.path.join(self.test_dir, "feature_vectors", f"epoch_{epoch}")
            self.assertTrue(os.path.exists(feature_dir), f"目錄 {feature_dir} 不存在")

if __name__ == "__main__":
    unittest.main() 