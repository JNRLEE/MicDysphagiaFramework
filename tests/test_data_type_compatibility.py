"""
數據類型兼容性測試：驗證訓練過程中數據類型是否兼容

功能：
1. 測試不同數據集、模型和任務類型的數據類型兼容性
2. 驗證標籤數據類型是否正確轉換為float32
3. 模擬損失計算過程，確保類型匹配

Description:
    測試數據類型轉換機制，確保不會出現"Found dtype Long but expected Float"的錯誤。

References:
    None
"""

import os
import sys
import torch
import logging
import unittest
from pathlib import Path

# 添加項目根目錄到路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.audio_dataset import AudioDataset
from data.spectrogram_dataset import SpectrogramDataset
from utils.data_adapter import DataAdapter
from models.model_factory import create_model
from losses.loss_factory import LossFactory
from utils.constants import SELECTION_TYPES, LABEL_TO_INDEX

class TestDataTypeCompatibility(unittest.TestCase):
    """測試數據類型兼容性"""
    
    def setUp(self):
        """設置測試環境"""
        # 設置日誌
        logging.basicConfig(level=logging.INFO)
        
        # 測試數據目錄
        self.test_data_dir = Path('tests/dataloader_test/dataset_test')
        if not self.test_data_dir.exists():
            logging.warning(f"測試數據目錄不存在: {self.test_data_dir}")
        
        # 基本配置
        self.base_config = {
            'global': {
                'seed': 42,
                'device': 'cpu'
            },
            'data': {
                'preprocessing': {
                    'audio': {
                        'sr': 16000,
                        'duration': 5,
                        'normalize': True
                    },
                    'spectrogram': {
                        'method': 'mel',
                        'n_mels': 128,
                        'n_fft': 1024,
                        'hop_length': 512
                    }
                },
                'filtering': {
                    'score_thresholds': {
                        'normal': 0,
                        'patient': 9
                    },
                    'class_config': {
                        'NoMovement': 1,
                        'DrySwallow': 1,
                        'Cracker': 1,
                        'Jelly': 1,
                        'WaterDrinking': 1
                    },
                    'subject_source': {
                        'normal': {
                            'include_N': 1,
                            'include_P': 1
                        },
                        'patient': {
                            'include_N': 0,
                            'include_P': 1
                        }
                    }
                }
            },
            'model': {}
        }
    
    def _create_dataset(self, dataset_class, task_type):
        """創建指定類型的數據集"""
        config = self.base_config.copy()
        config['data']['filtering']['task_type'] = task_type
        
        # 根據數據集類型使用對應的參數名稱
        if dataset_class.__name__ == 'SpectrogramDataset':
            dataset = dataset_class(
                data_path=self.test_data_dir,
                config=config
            )
        else:
            dataset = dataset_class(
                root_dir=self.test_data_dir,
                config=config
            )
        
        return dataset
    
    def _create_model_config(self, model_type, is_classification):
        """創建模型配置"""
        model_config = self.base_config.copy()
        
        if model_type == 'swin_transformer':
            model_config['model'] = {
                'type': 'swin_transformer',
                'parameters': {
                    'model_name': 'swin_tiny_patch4_window7_224',
                    'pretrained': False,
                    'num_classes': 10 if is_classification else 1,
                    'input_channels': 3,
                    'input_size': [224, 224],
                    'is_classification': is_classification
                }
            }
        elif model_type == 'fcnn':
            model_config['model'] = {
                'type': 'fcnn',
                'parameters': {
                    'input_dim': 1289,
                    'hidden_layers': [512, 256],
                    'num_classes': 10 if is_classification else 1,
                    'dropout_rate': 0.2,
                    'is_classification': is_classification
                }
            }
        
        return model_config
    
    def _create_loss(self, is_classification):
        """創建損失函數"""
        if is_classification:
            loss_config = {
                'type': 'CrossEntropyLoss',
                'parameters': {}
            }
        else:
            loss_config = {
                'type': 'MSELoss',
                'parameters': {}
            }
        
        loss_factory = LossFactory()
        return loss_factory.get_loss(loss_config)
    
    def _test_complete_data_flow(self, dataset_class, model_type, task_type):
        """測試完整數據流程，確認類型兼容性"""
        is_classification = task_type == 'classification'
        
        # 創建數據集
        dataset = self._create_dataset(dataset_class, task_type)
        
        # 確保有數據
        self.assertGreater(len(dataset), 0, "數據集為空")
        
        # 獲取一個樣本
        sample = dataset[0]
        
        # 檢查標籤類型
        self.assertTrue('label' in sample, "樣本中沒有label字段")
        self.assertEqual(sample['label'].dtype, torch.float32, 
                        f"標籤類型錯誤: 預期 torch.float32, 實際 {sample['label'].dtype}")
        
        # 創建一個小批次
        batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v] for k, v in sample.items()}
        
        # 創建模型
        model_config = self._create_model_config(model_type, is_classification)
        model = create_model(model_config)
        
        # 創建損失函數
        criterion = self._create_loss(is_classification)
        
        # 數據適配
        adapted_batch = DataAdapter.adapt_batch(batch, model_type, model_config)
        
        # 確定輸入數據
        input_data = None
        if 'spectrogram' in adapted_batch:
            input_data = adapted_batch['spectrogram']
        elif 'image' in adapted_batch:
            input_data = adapted_batch['image']
        elif 'features' in adapted_batch:
            input_data = adapted_batch['features']
        elif 'audio' in adapted_batch:
            input_data = adapted_batch['audio']
        
        self.assertIsNotNone(input_data, "無法找到模型輸入數據")
        
        # 獲取標籤
        label = adapted_batch['label']
        
        # 記錄當前的dtype
        logging.info(f"輸入數據類型: {input_data.dtype}")
        logging.info(f"標籤數據類型: {label.dtype}")
        
        # 前向傳播
        output = model(input_data)
        
        # 打印輸出形狀和類型
        logging.info(f"模型輸出形狀: {output.shape}")
        logging.info(f"模型輸出類型: {output.dtype}")
        
        # 計算損失
        try:
            loss = criterion(output, label)
            logging.info(f"損失計算成功: {loss.item()}")
            
            # 測試反向傳播
            loss.backward()
            logging.info("反向傳播成功")
            
            # 測試通過
            return True
        except Exception as e:
            logging.error(f"損失計算或反向傳播失敗: {str(e)}")
            self.fail(f"數據類型不兼容: {str(e)}")
            return False
        
    def test_audio_swin_classification(self):
        """測試音頻數據集與Swin Transformer在分類任務上的兼容性"""
        result = self._test_complete_data_flow(AudioDataset, 'swin_transformer', 'classification')
        self.assertTrue(result, "音頻-Swin-分類測試失敗")
        
    def test_audio_swin_regression(self):
        """測試音頻數據集與Swin Transformer在回歸任務上的兼容性"""
        result = self._test_complete_data_flow(AudioDataset, 'swin_transformer', 'regression')
        self.assertTrue(result, "音頻-Swin-回歸測試失敗")
        
    def test_spectrogram_fcnn_classification(self):
        """測試頻譜圖數據集與FCNN在分類任務上的兼容性"""
        result = self._test_complete_data_flow(SpectrogramDataset, 'fcnn', 'classification')
        self.assertTrue(result, "頻譜圖-FCNN-分類測試失敗")
        
    def test_spectrogram_fcnn_regression(self):
        """測試頻譜圖數據集與FCNN在回歸任務上的兼容性"""
        result = self._test_complete_data_flow(SpectrogramDataset, 'fcnn', 'regression')
        self.assertTrue(result, "頻譜圖-FCNN-回歸測試失敗")


def run_tests():
    """運行所有測試"""
    unittest.main()


if __name__ == '__main__':
    run_tests() 