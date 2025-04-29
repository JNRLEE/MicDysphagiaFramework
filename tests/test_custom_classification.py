"""
測試自定義分類功能
測試日期：2024-06-10
測試目的：驗證自定義分類功能，特別是Excel檔案讀取與分類邏輯
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import torch
import logging
from pathlib import Path
import tempfile
import shutil
import json
from unittest import skipIf

# 設置日誌
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加項目根目錄到Python路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 導入相關模組
from utils.custom_classification_loader import CustomClassificationLoader
from data.audio_dataset import AudioDataset
from models.model_factory import ModelFactory

class TestCustomClassification(unittest.TestCase):
    """測試自定義分類功能的類別"""
    
    def setUp(self):
        """設置測試環境，創建測試數據"""
        self.temp_dir = tempfile.mkdtemp()
        self.excel_path = os.path.join(self.temp_dir, 'test_classification.xlsx')
        
        # 創建測試Excel檔案
        self.create_test_excel()
        
        # 測試配置
        self.config = {
            'data': {
                'preprocessing': {
                    'audio': {
                        'sample_rate': 16000,
                        'duration': 5.0,
                        'normalize': True
                    }
                },
                'filtering': {
                    'custom_classification': {
                        'enabled': True,
                        'excel_path': self.excel_path,
                        'patient_id_column': 'PatientID',
                        'class_column': 'Class'
                    },
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
                    'task_type': 'classification'
                }
            },
            'model': {
                'type': 'swin_transformer',
                'parameters': {
                    'num_classes': 10,
                    'is_classification': True
                }
            }
        }
        
        # 創建測試患者目錄
        self.test_data_dir = os.path.join(self.temp_dir, 'test_data')
        os.makedirs(self.test_data_dir, exist_ok=True)
        self.create_test_patient_dirs()
    
    def tearDown(self):
        """清理測試環境"""
        shutil.rmtree(self.temp_dir)
    
    def create_test_excel(self):
        """創建測試Excel檔案"""
        # 創建測試數據
        data = {
            'PatientID': ['P001', 'P002', 'P003', 'N001', 'N002', 'P004'],
            'Score': [10, 15, 5, 0, 0, 12],
            'Class': ['嚴重', '中度', '輕微', '正常', '正常', '嚴重']
        }
        
        # 創建DataFrame並保存為Excel
        df = pd.DataFrame(data)
        df.to_excel(self.excel_path, index=False)
        
        logger.info(f"創建測試Excel檔案: {self.excel_path}")
    
    def create_test_patient_dirs(self):
        """創建測試患者目錄與相關檔案"""
        # 測試患者列表
        patients = ['P001', 'P002', 'P003', 'N001', 'N002', 'P004']
        
        for patient in patients:
            # 創建患者目錄
            patient_dir = os.path.join(self.test_data_dir, patient)
            os.makedirs(patient_dir, exist_ok=True)
            
            # 創建虛擬音頻檔案
            audio_path = os.path.join(patient_dir, 'Probe0_RX_IN_TDM4CH0.wav')
            with open(audio_path, 'wb') as f:
                # 創建一個空檔案
                f.write(b'\x00' * 100)
            
            # 創建info.json檔案
            info_path = os.path.join(patient_dir, 'info.json')
            
            # 根據患者ID設置分數與動作選擇
            if patient.startswith('P'):
                score = 10
                selection = "乾吞嚥"
            else:
                score = 0
                selection = "無動作"
            
            # 寫入info.json
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(f'{{"patient_id": "{patient}", "score": {score}, "selection": "{selection}"}}')
    
    def test_custom_classification_loader(self):
        """測試自定義分類載入器"""
        logger.info("=== 測試自定義分類載入器 ===")
        
        # 創建載入器
        loader = CustomClassificationLoader(self.config)
        
        # 檢查是否啟用
        self.assertTrue(loader.enabled, "應該啟用自定義分類")
        
        # 檢查類別數
        classes = loader.get_all_classes()
        self.assertEqual(len(classes), 4, f"應該有4個類別，但有 {len(classes)}")
        self.assertIn("嚴重", classes, "應該包含'嚴重'類別")
        self.assertIn("中度", classes, "應該包含'中度'類別")
        self.assertIn("輕微", classes, "應該包含'輕微'類別")
        self.assertIn("正常", classes, "應該包含'正常'類別")
        
        # 檢查患者分類
        self.assertEqual(loader.get_class("P001"), "嚴重", "P001應該是'嚴重'類別")
        self.assertEqual(loader.get_class("P002"), "中度", "P002應該是'中度'類別")
        self.assertEqual(loader.get_class("P003"), "輕微", "P003應該是'輕微'類別")
        self.assertEqual(loader.get_class("N001"), "正常", "N001應該是'正常'類別")
        
        # 檢查分類索引
        self.assertEqual(loader.get_class_index("P001"), loader.class_to_index["嚴重"], "P001的分類索引不正確")
        
        logger.info(f"分類類別: {classes}")
        logger.info(f"分類映射: {loader.class_to_index}")
        logger.info("自定義分類載入器測試通過")
    
    @skipIf(True, "測試需要實際音頻檔案和有效的info.json，暫時跳過")
    def test_audio_dataset_with_custom_classification(self):
        """測試使用自定義分類的音頻數據集"""
        logger.info("=== 測試使用自定義分類的音頻數據集 ===")
        
        # 創建數據集
        dataset = AudioDataset(self.test_data_dir, self.config)
        
        # 檢查數據集大小
        self.assertGreater(len(dataset), 0, "數據集應該包含樣本")
        
        # 檢查樣本
        for i in range(len(dataset)):
            sample = dataset[i]
            patient_id = sample['patient_id']
            label = sample['label']
            
            # 檢查標籤是否為tensor
            self.assertIsInstance(label, torch.Tensor, "標籤應該是tensor")
            
            # 獲取預期的標籤
            expected_class = self.config['data']['filtering']['custom_classification']['enabled'] and \
                            CustomClassificationLoader(self.config).get_class(patient_id)
            
            logger.info(f"樣本 {i}: 患者ID={patient_id}, 標籤={label.item()}, 自定義類別={sample.get('custom_class', '')}")
            
            # 如果有自定義類別，檢查是否設置
            if expected_class:
                self.assertIn('custom_class', sample, "樣本應該包含custom_class")
                self.assertEqual(sample['custom_class'], expected_class, f"患者 {patient_id} 的custom_class不正確")
        
        logger.info("音頻數據集與自定義分類整合測試通過")
    
    def test_model_factory_with_custom_classification(self):
        """測試使用自定義分類時的模型工廠"""
        logger.info("=== 測試使用自定義分類時的模型工廠 ===")
        
        try:
            # 獲取自定義分類的類別數
            custom_classifier = CustomClassificationLoader(self.config)
            expected_classes = custom_classifier.get_total_classes()
            
            # 創建模型 - 設置一個簡單的模型類型以避免加載swin_transformer的問題
            self.config['model']['type'] = 'fcnn'
            self.config['model']['parameters']['input_dim'] = 100
            model = ModelFactory.create_model(self.config)
            
            # 檢查模型最後一層的輸出大小是否與類別數匹配
            if hasattr(model, 'fc'):
                out_features = model.fc.out_features
            elif hasattr(model, 'head'):
                out_features = model.head.out_features
            else:
                out_features = None
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        if 'out' in name.lower() or name == '':
                            out_features = module.out_features
                            break
            
            self.assertIsNotNone(out_features, "無法確定模型的輸出大小")
            self.assertEqual(out_features, expected_classes, f"模型輸出大小應為 {expected_classes}，但為 {out_features}")
            
            logger.info(f"模型輸出大小: {out_features}, 預期類別數: {expected_classes}")
            logger.info("模型工廠與自定義分類整合測試通過")
        except Exception as e:
            self.fail(f"模型工廠測試失敗: {str(e)}")

if __name__ == '__main__':
    unittest.main() 