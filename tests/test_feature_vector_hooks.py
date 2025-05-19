"""
專門測試特徵向量處理和保存的邏輯

此測試腳本集中檢測on_evaluation_end中的特徵向量處理邏輯
模擬實際訓練情況，檢查多個epoch的特徵向量是否都被正確保存
"""

import os
import sys
import torch
import torch.nn as nn
import unittest
import shutil
import logging
from typing import Dict, Any, List, Set, Optional
import time

# 將項目根目錄添加到系統路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.hook_bridge import ActivationCaptureHook
from utils.save_manager import SaveManager

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DebugActivationCaptureHook(ActivationCaptureHook):
    """用於調試的ActivationCaptureHook子類"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_calls = {
            "_should_process_epoch": [],
            "on_evaluation_begin": [],
            "on_evaluation_end": [],
            "_process_features": []
        }
    
    def _should_process_epoch(self, epoch: int) -> bool:
        result = super()._should_process_epoch(epoch)
        self.debug_calls["_should_process_epoch"].append((epoch, result))
        logger.info(f"_should_process_epoch({epoch}) = {result}")
        logger.info(f"  captured_epochs={self.captured_epochs}")
        logger.info(f"  save_frequency={self.save_frequency}")
        return result
    
    def on_evaluation_begin(self, model: nn.Module, logs: Dict[str, Any] = None) -> None:
        self.debug_calls["on_evaluation_begin"].append(logs.get("epoch", None) if logs else None)
        logger.info(f"on_evaluation_begin: logs={logs}")
        return super().on_evaluation_begin(model, logs)
    
    def on_evaluation_end(self, model: nn.Module, results: Dict[str, Any] = None, logs: Dict[str, Any] = None) -> None:
        epoch = logs.get("epoch", None) if logs else None
        self.debug_calls["on_evaluation_end"].append(epoch)
        logger.info(f"on_evaluation_end: epoch={epoch}, captured_epochs={self.captured_epochs}")
        return super().on_evaluation_end(model, results, logs)
    
    def _process_features(self, epoch: int) -> None:
        self.debug_calls["_process_features"].append(epoch)
        logger.info(f"_process_features({epoch})")
        return super()._process_features(epoch)

class SimpleTestModel(nn.Module):
    """測試用的簡單模型"""
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # backbone.0
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1)  # backbone.1
        )
        self.head = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64),  # head.0
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.head(x)

class TestFeatureVectorProcessing(unittest.TestCase):
    """專門測試特徵向量處理邏輯的測試類"""
    
    def setUp(self):
        """設置測試環境"""
        self.test_dir = os.path.join(project_root, "tests", "test_feature_vectors")
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)
        
        self.save_manager = SaveManager(self.test_dir)
        self.model = SimpleTestModel()
        
        # 記錄開始時間
        self.start_time = time.time()
    
    def tearDown(self):
        """清理測試環境並報告執行時間"""
        execution_time = time.time() - self.start_time
        logger.info(f"測試執行時間: {execution_time:.2f}秒")
        
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_process_features_all_epochs(self):
        """測試是否為所有應該處理的epoch保存特徵向量"""
        # 設置監控層
        layer_names = ["backbone.0", "backbone.1", "head.0"]
        
        # 創建帶調試功能的Hook
        hook = DebugActivationCaptureHook(
            model=self.model,
            layer_names=layer_names,
            save_manager=self.save_manager,
            dataset_name="test",
            save_frequency=1,  # 每個epoch都保存
            compute_similarity=False,  # 不計算相似度以加速測試
            compute_tsne=False  # 不計算t-SNE以加速測試
        )
        
        # 模擬on_train_begin
        total_epochs = 3
        hook.on_train_begin(self.model, {"config": {"training": {"epochs": total_epochs}}})
        
        # 檢查save_frequency
        logger.info(f"Hook的save_frequency: {hook.save_frequency}")
        self.assertEqual(hook.save_frequency, 1, "save_frequency應該為1")
        
        # 模擬多個epoch的評估
        for epoch in range(total_epochs):
            # 重置激活值
            hook.all_activations = {layer_name: [] for layer_name in layer_names}
            hook.all_targets = []
            
            # 模擬on_evaluation_begin
            hook.on_evaluation_begin(self.model, {"epoch": epoch})
            
            # 模擬收集激活值
            batch_size = 4
            for layer_name in layer_names:
                # 根據層名稱創建不同大小的假激活值
                if "backbone" in layer_name:
                    hook.all_activations[layer_name] = [torch.randn(batch_size, 32, 8, 8)]
                else:
                    hook.all_activations[layer_name] = [torch.randn(batch_size, 64)]
            
            # 添加目標
            hook.all_targets = [torch.randint(0, 10, (batch_size,))]
            
            # 模擬on_evaluation_end
            hook.on_evaluation_end(
                model=self.model,
                results={"predictions": torch.randint(0, 10, (batch_size,))},
                logs={"epoch": epoch}
            )
            
            # 檢查captured_epochs是否正確更新
            self.assertIn(epoch, hook.captured_epochs, f"epoch {epoch} 應該在captured_epochs中")
            
            # 檢查特徵向量文件是否存在
            feature_dir = os.path.join(self.test_dir, "feature_vectors", f"epoch_{epoch}")
            self.assertTrue(os.path.exists(feature_dir), f"目錄 {feature_dir} 不存在")
            
            # 檢查每個層的特徵向量文件
            for layer_name in layer_names:
                layer_name_safe = layer_name.replace(".", "_")
                feature_file = os.path.join(feature_dir, f"layer_{layer_name_safe}_features.pt")
                self.assertTrue(os.path.exists(feature_file), f"特徵向量文件 {feature_file} 不存在")
        
        # 檢查函數調用記錄
        logger.info(f"_should_process_epoch調用記錄: {hook.debug_calls['_should_process_epoch']}")
        logger.info(f"on_evaluation_begin調用記錄: {hook.debug_calls['on_evaluation_begin']}")
        logger.info(f"on_evaluation_end調用記錄: {hook.debug_calls['on_evaluation_end']}")
        logger.info(f"_process_features調用記錄: {hook.debug_calls['_process_features']}")
        
        # 確認_process_features被調用了正確的次數
        self.assertEqual(len(hook.debug_calls["_process_features"]), total_epochs, 
                         f"_process_features應該被調用{total_epochs}次")
        
        # 確認每個epoch都被處理了
        processed_epochs = set(hook.debug_calls["_process_features"])
        self.assertEqual(processed_epochs, set(range(total_epochs)), 
                         f"應該處理所有epoch {set(range(total_epochs))}，實際處理了 {processed_epochs}")
    
    def test_hook_with_real_workflow(self):
        """測試在更接近實際工作流程的情況下特徵向量的處理"""
        layer_names = ["backbone.0", "head.0"]
        
        # 創建Hook
        hook = DebugActivationCaptureHook(
            model=self.model,
            layer_names=layer_names,
            save_manager=self.save_manager,
            dataset_name="test",
            save_frequency=1
        )
        
        # 模擬on_train_begin
        total_epochs = 2
        hook.on_train_begin(self.model, {"config": {"training": {"epochs": total_epochs}}})
        
        # 模擬訓練循環
        for epoch in range(total_epochs):
            # 訓練階段
            # ...
            
            # 每個epoch進行兩次評估：一次驗證集，一次測試集
            for dataset in ["val", "test"]:
                # 重置激活值
                hook.all_activations = {layer_name: [] for layer_name in layer_names}
                hook.all_targets = []
                
                # 模擬on_evaluation_begin
                hook.on_evaluation_begin(self.model, {"epoch": epoch, "dataset": dataset})
                
                # 模擬批次處理，收集激活值
                batch_size = 4
                inputs = torch.randn(batch_size, 3, 8, 8)
                targets = torch.randint(0, 10, (batch_size,))
                
                # 前向傳播
                outputs = self.model(inputs)
                
                # 手動觸發鉤子，模擬批次處理
                hook.on_batch_end(
                    batch=0,
                    model=self.model,
                    inputs=inputs,
                    targets=targets,
                    outputs=outputs,
                    logs={"epoch": epoch, "dataset": dataset}
                )
                
                # 模擬on_evaluation_end
                hook.on_evaluation_end(
                    model=self.model,
                    results={"predictions": torch.randint(0, 10, (batch_size,))},
                    logs={"epoch": epoch, "dataset": dataset}
                )
            
            # 檢查該epoch的特徵向量是否已保存
            feature_dir = os.path.join(self.test_dir, "feature_vectors", f"epoch_{epoch}")
            self.assertTrue(os.path.exists(feature_dir), f"目錄 {feature_dir} 不存在")
            
            # 檢查層的特徵向量
            for layer_name in layer_names:
                layer_name_safe = layer_name.replace(".", "_")
                feature_file = os.path.join(feature_dir, f"layer_{layer_name_safe}_features.pt")
                self.assertTrue(os.path.exists(feature_file), f"特徵向量文件 {feature_file} 不存在")
        
        # 檢查captured_epochs
        self.assertEqual(hook.captured_epochs, set(range(total_epochs)),
                        f"captured_epochs應該包含所有epoch，實際為 {hook.captured_epochs}")
        
        # 檢查處理的epoch數量
        processed_epochs = set(hook.debug_calls["_process_features"])
        self.assertEqual(processed_epochs, set(range(total_epochs)),
                        f"應該處理所有epoch，實際處理了 {processed_epochs}")
        
        # 檢查總共調用了多少次_process_features
        # 對於每個epoch，測試集和驗證集的處理只會計入一次
        self.assertEqual(len(hook.debug_calls["_process_features"]), total_epochs,
                        f"_process_features應該被調用{total_epochs}次，實際調用了{len(hook.debug_calls['_process_features'])}次")

if __name__ == "__main__":
    unittest.main() 