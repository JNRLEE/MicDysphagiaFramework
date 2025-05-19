#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
測試特徵向量儲存機制的功能

此測試腳本用於驗證activation_capture_hook中的save_frequency和target_epochs配置是否正確生效，
以及在不同訓練輪數下是否能正確地儲存特徵向量。
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import logging
import tempfile
import shutil
from datetime import datetime

# 添加專案根目錄到路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 引入需要測試的模組
from models.hook_bridge import ActivationCaptureHook
from utils.save_manager import SaveManager
from utils.callback_interface import CallbackInterface

# 設定日誌
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 創建一個簡單的測試模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# 創建一個簡單的訓練器模擬類
class SimpleDummyTrainer:
    def __init__(self, model, total_epochs=10):
        self.model = model
        self.total_epochs = total_epochs
        self.callbacks = []
        self.current_epoch = 0
    
    def add_callback(self, callback):
        self.callbacks.append(callback)
    
    def train_and_evaluate(self, dataloader):
        # 模擬訓練過程
        logger.info(f"開始模擬訓練，總共 {self.total_epochs} 個輪次")
        
        # 調用 on_train_begin 回調
        train_begin_logs = {
            'config': {
                'training': {
                    'epochs': self.total_epochs
                }
            }
        }
        for callback in self.callbacks:
            callback.on_train_begin(self.model, train_begin_logs)
        
        # 模擬每個輪次的訓練和評估
        for epoch in range(self.total_epochs):
            self.current_epoch = epoch
            logger.info(f"Epoch {epoch}/{self.total_epochs-1}")
            
            # 調用 epoch 開始回調
            epoch_logs = {'epoch': epoch}
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch, self.model, epoch_logs)
            
            # 模擬訓練
            self.model.train()
            # ... 訓練過程 ...
            
            # 模擬評估
            self.model.eval()
            
            # 調用評估開始回調
            eval_logs = {'phase': 'validation', 'epoch': epoch}
            for callback in self.callbacks:
                callback.on_evaluation_begin(self.model, eval_logs)
            
            # 模擬評估過程
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(dataloader):
                    # 前向傳播
                    outputs = self.model(inputs)
                    
                    # 調用批次結束回調
                    batch_logs = {
                        'phase': 'validation',
                        'epoch': epoch,
                        'is_last_batch': batch_idx == len(dataloader) - 1
                    }
                    for callback in self.callbacks:
                        callback.on_batch_end(
                            batch_idx, self.model, inputs, targets, outputs, 
                            torch.tensor(0.1), batch_logs
                        )
            
            # 調用評估結束回調
            eval_end_logs = {'phase': 'validation', 'epoch': epoch}
            for callback in self.callbacks:
                callback.on_evaluation_end(self.model, {}, eval_end_logs)
            
            # 調用 epoch 結束回調
            epoch_end_logs = {'epoch': epoch}
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, self.model, {}, {}, epoch_end_logs)
        
        # 調用訓練結束回調
        train_end_logs = {}
        for callback in self.callbacks:
            callback.on_train_end(self.model, {}, train_end_logs)
        
        logger.info("模擬訓練完成")

def test_activation_capture_with_save_frequency():
    """測試使用save_frequency參數的特徵向量儲存"""
    logger.info("測試 save_frequency 參數...")
    
    # 創建臨時目錄
    temp_dir = tempfile.mkdtemp()
    try:
        # 創建模型和數據
        model = SimpleModel()
        X = torch.randn(20, 10)
        y = torch.randint(0, 5, (20,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=4)
        
        # 創建SaveManager
        save_manager = SaveManager(temp_dir)
        
        # 創建ActivationCaptureHook，使用save_frequency
        target_layers = ['layer1', 'layer2']
        save_frequency = 2
        hook = ActivationCaptureHook(
            model=model,
            layer_names=target_layers,
            save_manager=save_manager,
            dataset_name='test',
            save_frequency=save_frequency,
            save_first_last=True
        )
        
        # 創建訓練器並添加鉤子
        trainer = SimpleDummyTrainer(model, total_epochs=10)
        trainer.add_callback(hook)
        
        # 執行模擬訓練
        trainer.train_and_evaluate(dataloader)
        
        # 檢查結果目錄結構
        feature_vectors_dir = os.path.join(temp_dir, 'feature_vectors')
        assert os.path.exists(feature_vectors_dir), "特徵向量目錄不存在"
        
        # 檢查是否正確儲存了輪次 0, 2, 4, 6, 8, 9 (first, last, 和按照頻率)
        expected_epochs = [0, 2, 4, 6, 8, 9]  # 0是第一個，9是最後一個，2,4,6,8是按照頻率
        
        for epoch in expected_epochs:
            epoch_dir = os.path.join(feature_vectors_dir, f'epoch_{epoch}')
            assert os.path.exists(epoch_dir), f"Epoch {epoch} 的特徵向量目錄不存在"
            logger.info(f"確認 epoch_{epoch} 目錄存在")
            
            # 對於每一層，檢查特徵向量文件
            for layer in target_layers:
                layer_file = os.path.join(epoch_dir, f'layer_{layer}_features.pt')
                assert os.path.exists(layer_file), f"層 {layer} 在 epoch {epoch} 的特徵向量文件不存在"
                logger.info(f"確認 epoch_{epoch}/layer_{layer}_features.pt 文件存在")
        
        # 檢查未按照頻率的輪次是否被正確跳過
        skipped_epochs = [1, 3, 5, 7]
        for epoch in skipped_epochs:
            epoch_dir = os.path.join(feature_vectors_dir, f'epoch_{epoch}')
            assert not os.path.exists(epoch_dir), f"不應存在 Epoch {epoch} 的特徵向量目錄"
            logger.info(f"確認 epoch_{epoch} 目錄不存在（按照預期跳過）")
        
        logger.info("save_frequency 參數測試通過！")
        return True
    
    except AssertionError as e:
        logger.error(f"測試失敗: {e}")
        return False
    except Exception as e:
        logger.error(f"測試過程中發生錯誤: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        # 清理臨時目錄
        shutil.rmtree(temp_dir)

def test_activation_capture_with_target_epochs():
    """測試使用target_epochs參數的特徵向量儲存"""
    logger.info("測試 target_epochs 參數...")
    
    # 創建臨時目錄
    temp_dir = tempfile.mkdtemp()
    try:
        # 創建模型和數據
        model = SimpleModel()
        X = torch.randn(20, 10)
        y = torch.randint(0, 5, (20,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=4)
        
        # 創建SaveManager
        save_manager = SaveManager(temp_dir)
        
        # 創建ActivationCaptureHook，使用target_epochs
        target_layers = ['layer1', 'layer2']
        target_epochs = {0, 3, 5, 9}
        hook = ActivationCaptureHook(
            model=model,
            layer_names=target_layers,
            save_manager=save_manager,
            dataset_name='test',
            target_epochs=target_epochs
        )
        
        # 創建訓練器並添加鉤子
        trainer = SimpleDummyTrainer(model, total_epochs=10)
        trainer.add_callback(hook)
        
        # 執行模擬訓練
        trainer.train_and_evaluate(dataloader)
        
        # 檢查結果目錄結構
        feature_vectors_dir = os.path.join(temp_dir, 'feature_vectors')
        assert os.path.exists(feature_vectors_dir), "特徵向量目錄不存在"
        
        # 檢查是否正確儲存了指定的輪次
        expected_epochs = [0, 3, 5, 9]
        
        for epoch in expected_epochs:
            epoch_dir = os.path.join(feature_vectors_dir, f'epoch_{epoch}')
            assert os.path.exists(epoch_dir), f"Epoch {epoch} 的特徵向量目錄不存在"
            logger.info(f"確認 epoch_{epoch} 目錄存在")
            
            # 對於每一層，檢查特徵向量文件
            for layer in target_layers:
                layer_file = os.path.join(epoch_dir, f'layer_{layer}_features.pt')
                assert os.path.exists(layer_file), f"層 {layer} 在 epoch {epoch} 的特徵向量文件不存在"
                logger.info(f"確認 epoch_{epoch}/layer_{layer}_features.pt 文件存在")
        
        # 檢查未指定的輪次是否被正確跳過
        skipped_epochs = [1, 2, 4, 6, 7, 8]
        for epoch in skipped_epochs:
            epoch_dir = os.path.join(feature_vectors_dir, f'epoch_{epoch}')
            assert not os.path.exists(epoch_dir), f"不應存在 Epoch {epoch} 的特徵向量目錄"
            logger.info(f"確認 epoch_{epoch} 目錄不存在（按照預期跳過）")
        
        logger.info("target_epochs 參數測試通過！")
        return True
    
    except AssertionError as e:
        logger.error(f"測試失敗: {e}")
        return False
    except Exception as e:
        logger.error(f"測試過程中發生錯誤: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        # 清理臨時目錄
        shutil.rmtree(temp_dir)

def test_activation_capture_target_epochs_precedence():
    """測試當同時指定target_epochs和save_frequency時的優先級"""
    logger.info("測試 target_epochs 和 save_frequency 同時存在時的優先級...")
    
    # 創建臨時目錄
    temp_dir = tempfile.mkdtemp()
    try:
        # 創建模型和數據
        model = SimpleModel()
        X = torch.randn(20, 10)
        y = torch.randint(0, 5, (20,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=4)
        
        # 創建SaveManager
        save_manager = SaveManager(temp_dir)
        
        # 創建ActivationCaptureHook，同時指定target_epochs和save_frequency
        target_layers = ['layer1', 'layer2']
        target_epochs = {1, 4, 7}  # 應該優先使用這個
        save_frequency = 2  # 這個應該被忽略
        hook = ActivationCaptureHook(
            model=model,
            layer_names=target_layers,
            save_manager=save_manager,
            dataset_name='test',
            target_epochs=target_epochs,
            save_frequency=save_frequency,
            save_first_last=True  # 這個應該也被忽略，因為有target_epochs
        )
        
        # 創建訓練器並添加鉤子
        trainer = SimpleDummyTrainer(model, total_epochs=10)
        trainer.add_callback(hook)
        
        # 執行模擬訓練
        trainer.train_and_evaluate(dataloader)
        
        # 檢查結果目錄結構
        feature_vectors_dir = os.path.join(temp_dir, 'feature_vectors')
        assert os.path.exists(feature_vectors_dir), "特徵向量目錄不存在"
        
        # 檢查是否只儲存了target_epochs中指定的輪次
        expected_epochs = [1, 4, 7]
        
        for epoch in expected_epochs:
            epoch_dir = os.path.join(feature_vectors_dir, f'epoch_{epoch}')
            assert os.path.exists(epoch_dir), f"Epoch {epoch} 的特徵向量目錄不存在"
            logger.info(f"確認 epoch_{epoch} 目錄存在")
        
        # 檢查save_frequency和save_first_last相關的輪次是否被正確跳過
        skipped_epochs = [0, 2, 6, 8, 9]  # 0和9應該被跳過，因為save_first_last應該被忽略
        for epoch in skipped_epochs:
            epoch_dir = os.path.join(feature_vectors_dir, f'epoch_{epoch}')
            assert not os.path.exists(epoch_dir), f"不應存在 Epoch {epoch} 的特徵向量目錄"
            logger.info(f"確認 epoch_{epoch} 目錄不存在（按照預期跳過）")
        
        logger.info("優先級測試通過，target_epochs 優先於 save_frequency!")
        return True
    
    except AssertionError as e:
        logger.error(f"測試失敗: {e}")
        return False
    except Exception as e:
        logger.error(f"測試過程中發生錯誤: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        # 清理臨時目錄
        shutil.rmtree(temp_dir)

def find_and_fix_issues():
    """檢查代碼中的問題並提出修復方案"""
    issues = []
    
    # 檢查 on_train_begin 回調是否正確初始化 target_epochs
    issues.append({
        "description": "檢查ActivationCaptureHook的on_train_begin是否正確接收config並設置total_epochs",
        "status": "需要檢查",
        "file": "models/hook_bridge.py",
        "function": "on_train_begin"
    })
    
    # 檢查 on_evaluation_begin 和 on_evaluation_end 是否正確使用 target_epochs
    issues.append({
        "description": "檢查on_evaluation_begin和on_evaluation_end是否一致地檢查epoch條件",
        "status": "需要檢查",
        "file": "models/hook_bridge.py",
        "functions": ["on_evaluation_begin", "on_evaluation_end"]
    })
    
    # 檢查 trainer 是否正確傳遞 epoch 信息到回調
    issues.append({
        "description": "檢查PyTorchTrainer是否在驗證階段正確傳遞epoch參數到回調函數",
        "status": "需要檢查",
        "file": "trainers/pytorch_trainer.py",
        "functions": ["_validate_epoch", "evaluate"]
    })
    
    # 檢查 run_experiments.py 是否正確啟用 eval_epoch_tracking
    issues.append({
        "description": "檢查run_experiments.py是否正確啟用eval_epoch_tracking",
        "status": "需要檢查",
        "file": "scripts/run_experiments.py",
        "function": "run_experiment"
    })
    
    # 檢查配置文件中的 target_epochs 是否合理
    issues.append({
        "description": "檢查配置文件中的target_epochs設置是否符合預期",
        "status": "需要檢查",
        "file": "config/example_feature_vectors.yaml",
        "section": "hooks.activation_capture"
    })
    
    return issues

if __name__ == "__main__":
    logger.info("開始執行特徵向量儲存機制測試...")
    
    # 第一個測試: 使用save_frequency
    test_result1 = test_activation_capture_with_save_frequency()
    
    # 第二個測試: 使用target_epochs
    test_result2 = test_activation_capture_with_target_epochs()
    
    # 第三個測試: 測試優先級
    test_result3 = test_activation_capture_target_epochs_precedence()
    
    # 查找並修復問題
    issues = find_and_fix_issues()
    
    # 輸出總結
    logger.info("\n========== 測試結果摘要 ==========")
    logger.info(f"save_frequency 測試: {'通過' if test_result1 else '失敗'}")
    logger.info(f"target_epochs 測試: {'通過' if test_result2 else '失敗'}")
    logger.info(f"優先級測試: {'通過' if test_result3 else '失敗'}")
    logger.info("\n待檢查問題:")
    for i, issue in enumerate(issues):
        logger.info(f"{i+1}. {issue['description']} - {issue['status']}")
    logger.info("==================================") 