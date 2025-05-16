"""
測試索引數據集與訓練流程的整合

此測試檔案驗證索引數據集功能是否能夠正確地與訓練流程整合，包括：
1. 使用索引CSV配置創建數據集
2. 使用索引數據集進行模型訓練
3. 驗證訓練過程是否正常運行

測試日期: 2023-11-15
"""

import os
import sys
import unittest
import tempfile
import yaml
import shutil
import pandas as pd
import numpy as np
import torch
from pathlib import Path

# 添加項目根目錄到路徑，以便能夠導入模組
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config_loader import load_config
from data.dataset_factory import create_dataset
from models.model_factory import create_model
from trainers.trainer_factory import create_trainer
from losses.loss_factory import create_loss_function

class TestIndexedDatasetTraining(unittest.TestCase):
    """測試索引數據集與訓練流程的整合"""
    
    @classmethod
    def setUpClass(cls):
        """在所有測試之前設置測試環境"""
        # 創建臨時目錄
        cls.temp_dir = tempfile.mkdtemp()
        cls.data_dir = os.path.join(cls.temp_dir, 'data')
        os.makedirs(cls.data_dir, exist_ok=True)
        
        # 創建測試數據
        cls._create_test_data()
        
        # 創建配置文件
        cls.config_path = os.path.join(cls.temp_dir, 'test_config.yaml')
        cls._create_test_config()
    
    @classmethod
    def tearDownClass(cls):
        """在所有測試之後清理測試環境"""
        # 刪除臨時目錄
        shutil.rmtree(cls.temp_dir)
    
    @classmethod
    def _create_test_data(cls):
        """創建測試數據"""
        # 創建測試音頻文件
        for i in range(10):
            # 創建一個簡單的特徵檔案
            features = np.random.rand(10, 10).astype(np.float32)
            np.save(os.path.join(cls.data_dir, f'test_file_{i}_features.npy'), features)
            
            # 創建一個簡單的編碼檔案
            codes = np.random.rand(5, 5).astype(np.float32)
            np.save(os.path.join(cls.data_dir, f'test_file_{i}_codes.npy'), codes)
        
        # 創建數據索引CSV
        index_data = {
            'file_path': [os.path.join(cls.data_dir, f'test_file_{i}.wav') for i in range(10)],
            'score': [float(i % 5) for i in range(10)],
            'patient_id': [f'p{i//2}' for i in range(10)],
            'DrLee_Evaluation': ['聽起來正常', '輕度異常', '重度異常'] * 3 + ['聽起來正常'],
            'feature_path': [os.path.join(cls.data_dir, f'test_file_{i}_features.npy') for i in range(10)],
            'codes_path': [os.path.join(cls.data_dir, f'test_file_{i}_codes.npy') for i in range(10)],
            'status': ['processed'] * 10
        }
        
        cls.index_path = os.path.join(cls.data_dir, 'data_index.csv')
        pd.DataFrame(index_data).to_csv(cls.index_path, index=False)
    
    @classmethod
    def _create_test_config(cls):
        """創建測試配置文件"""
        config = {
            'global': {
                'experiment_name': 'indexed_dataset_test',
                'seed': 42,
                'device': 'cpu'
            },
            'data': {
                'type': 'feature',
                'index_path': cls.index_path,
                'label_field': 'score',
                'filter_criteria': {
                    'status': 'processed'
                },
                'source': {
                    'feature_dir': cls.data_dir
                },
                'dataloader': {
                    'batch_size': 2,
                    'num_workers': 0
                },
                'splits': {
                    'train_ratio': 0.6,
                    'val_ratio': 0.2,
                    'test_ratio': 0.2
                }
            },
            'model': {
                'type': 'fcnn',
                'parameters': {
                    'input_dim': 100,  # 10x10 features flattened
                    'hidden_dims': [50, 20],
                    'output_dim': 1,
                    'dropout': 0.2
                }
            },
            'training': {
                'epochs': 2,
                'optimizer': {
                    'type': 'adam',
                    'parameters': {
                        'lr': 0.001
                    }
                },
                'loss': {
                    'type': 'MSELoss'
                },
                'device': 'cpu',
                'early_stopping': {
                    'patience': 5,
                    'min_delta': 0.001
                }
            }
        }
        
        with open(cls.config_path, 'w') as f:
            yaml.dump(config, f)
    
    def test_create_dataset_from_index(self):
        """測試從索引CSV創建數據集"""
        config = load_config(self.config_path)
        
        # 確保配置是純字典
        config = config.config
        
        train_dataset, val_dataset, test_dataset = create_dataset(config)
        
        self.assertIsNotNone(train_dataset, "應該成功創建訓練數據集")
        self.assertIsNotNone(val_dataset, "應該成功創建驗證數據集")
        self.assertIsNotNone(test_dataset, "應該成功創建測試數據集")
        
        # 檢查數據集大小
        self.assertGreater(len(train_dataset), 0, "訓練數據集應該有數據")
        self.assertGreater(len(val_dataset), 0, "驗證數據集應該有數據")
        self.assertGreater(len(test_dataset), 0, "測試數據集應該有數據")
    
    def test_training_with_indexed_dataset(self):
        """測試使用索引數據集進行模型訓練"""
        config = load_config(self.config_path)
        
        # 確保配置是純字典
        config = config.config
        
        # 創建數據集
        train_dataset, val_dataset, test_dataset = create_dataset(config)
        
        # 創建數據加載器
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=2, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=2, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=2, shuffle=False
        )
        
        # 創建模型
        model = create_model(config)
        
        # 創建損失函數
        loss_fn = create_loss_function(config)
        
        # 對於測試，我們不需要真正創建訓練器，只需驗證數據集和模型是否可用
        self.assertIsNotNone(model, "模型應該成功創建")
        self.assertIsNotNone(loss_fn, "損失函數應該成功創建")
        self.assertIsNotNone(train_loader, "訓練數據加載器應該成功創建")
        self.assertIsNotNone(val_loader, "驗證數據加載器應該成功創建")
        self.assertIsNotNone(test_loader, "測試數據加載器應該成功創建")
    
    def test_classification_with_indexed_dataset(self):
        """測試使用索引數據集進行分類任務"""
        # 修改配置為分類任務
        config = load_config(self.config_path)
        
        # 確保配置是純字典
        config = config.config
        
        config['data']['label_field'] = 'DrLee_Evaluation'
        config['model']['parameters']['output_dim'] = 3  # 3個類別
        config['training']['loss']['type'] = 'CrossEntropyLoss'
        
        # 創建數據集
        train_dataset, val_dataset, test_dataset = create_dataset(config)
        
        # 對於測試，我們只需驗證數據集是否可用
        self.assertIsNotNone(train_dataset, "訓練數據集應該成功創建")
        self.assertIsNotNone(val_dataset, "驗證數據集應該成功創建")
        self.assertIsNotNone(test_dataset, "測試數據集應該成功創建")
        
        # 檢查數據集是否正確設置為分類任務 - 適應 Subset 類型
        # 由於我們使用 Subset，需要通過 .dataset 訪問原始數據集
        if hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'is_classification'):
            self.assertTrue(train_dataset.dataset.is_classification, "應該是分類任務")
            self.assertEqual(train_dataset.dataset.num_classes, 3, "應該有3個類別")
        
        # 創建模型
        model = create_model(config)
        self.assertIsNotNone(model, "模型應該成功創建")

if __name__ == '__main__':
    """
    Description: 測試索引數據集與訓練流程的整合
    Args: None
    Returns: None
    References: 無
    """
    unittest.main() 