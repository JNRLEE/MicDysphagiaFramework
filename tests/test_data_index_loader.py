"""
測試數據索引加載器模組的功能

此測試文件測試utils/data_index_loader.py中的DataIndexLoader類的所有主要功能，包括：
1. 基本加載功能
2. 數據篩選功能
3. 標籤映射功能
4. 按患者ID拆分數據功能
5. 錯誤處理機制

測試日期: 2023-11-15
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

# 添加項目根目錄到路徑，以便能夠導入模組
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_index_loader import DataIndexLoader

class TestDataIndexLoader(unittest.TestCase):
    """測試DataIndexLoader類"""
    
    def setUp(self):
        """在每個測試之前設置測試環境"""
        # 創建臨時測試數據
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w')
        self.test_data = pd.DataFrame({
            'file_path': ['/path/to/file1.wav', '/path/to/file2.wav', '/path/to/file3.wav', '/path/to/file4.wav'],
            'score': [15, 25, 5, 30],
            'patient_id': ['p001', 'p002', 'p001', 'p003'],
            'DrLee_Evaluation': ['聽起來正常', '輕度異常', '重度異常', '聽起來正常'],
            'DrTai_Evaluation': ['正常', '無OR 輕微吞嚥障礙', '重度吞嚥障礙', '正常'],
            'selection': ['乾吞嚥1口', '吞水10ml', '餅乾1塊', '乾吞嚥2口'],
            'status': ['processed', 'processed', 'raw', 'processed']
        })
        self.test_data.to_csv(self.temp_file.name, index=False)
        self.test_csv_path = self.temp_file.name
    
    def tearDown(self):
        """在每個測試之後清理測試環境"""
        # 刪除臨時文件
        self.temp_file.close()
        os.unlink(self.temp_file.name)
    
    def test_basic_loading(self):
        """測試基本加載功能"""
        loader = DataIndexLoader(self.test_csv_path, verify_paths=False)
        self.assertEqual(len(loader.index_df), 4, "應該加載4條記錄")
        self.assertIn('file_path', loader.index_df.columns, "應該包含file_path列")
        self.assertIn('score', loader.index_df.columns, "應該包含score列")
    
    def test_filter_by_criteria(self):
        """測試篩選功能"""
        loader = DataIndexLoader(self.test_csv_path, verify_paths=False)
        
        # 測試單一條件篩選
        filtered = loader.filter_by_criteria({'status': 'processed'})
        self.assertEqual(len(filtered), 3, "應該篩選出3條processed記錄")
        
        # 測試多條件篩選
        filtered = loader.filter_by_criteria({'status': 'processed', 'patient_id': 'p001'})
        self.assertEqual(len(filtered), 1, "應該篩選出1條符合兩個條件的記錄")
        
        # 測試無匹配篩選
        filtered = loader.filter_by_criteria({'status': 'invalid_status'})
        self.assertEqual(len(filtered), 0, "應該沒有匹配的記錄")
        
        # 測試空條件篩選
        filtered = loader.filter_by_criteria({})
        self.assertEqual(len(filtered), 4, "空條件應該返回所有記錄")
        
        # 測試None條件篩選
        filtered = loader.filter_by_criteria(None)
        self.assertEqual(len(filtered), 4, "None條件應該返回所有記錄")
        
        # 測試條件中包含None值
        filtered = loader.filter_by_criteria({'status': None, 'patient_id': 'p001'})
        self.assertEqual(len(filtered), 2, "應該忽略None值條件")
    
    def test_get_labels(self):
        """測試獲取標籤功能"""
        loader = DataIndexLoader(self.test_csv_path, verify_paths=False)
        
        # 測試獲取score標籤
        labels = loader.get_labels('score')
        self.assertEqual(len(labels), 4, "應該有4個標籤")
        self.assertEqual(labels.iloc[0], 15, "第一個標籤應該是15")
        
        # 測試獲取分類標籤
        labels = loader.get_labels('DrLee_Evaluation')
        self.assertEqual(len(labels), 4, "應該有4個標籤")
        self.assertEqual(labels.iloc[0], '聽起來正常', "第一個標籤應該是'聽起來正常'")
        
        # 測試獲取不存在的標籤欄位
        with self.assertRaises(ValueError):
            loader.get_labels('non_existent_field')
    
    def test_get_item_by_path(self):
        """測試根據路徑獲取數據記錄功能"""
        loader = DataIndexLoader(self.test_csv_path, verify_paths=False)
        
        # 測試獲取存在的記錄
        item = loader.get_item_by_path('/path/to/file1.wav')
        self.assertIsNotNone(item, "應該找到記錄")
        self.assertEqual(item['score'], 15, "記錄的score應該是15")
        
        # 測試獲取不存在的記錄
        item = loader.get_item_by_path('/path/to/non_existent_file.wav')
        self.assertIsNone(item, "不存在的路徑應該返回None")
    
    def test_get_mapping_dict(self):
        """測試獲取標籤映射字典功能"""
        loader = DataIndexLoader(self.test_csv_path, verify_paths=False)
        
        # 測試獲取DrLee_Evaluation映射
        mapping = loader.get_mapping_dict('DrLee_Evaluation')
        self.assertEqual(len(mapping), 3, "應該有3個唯一值")
        self.assertIn('聽起來正常', mapping, "映射應該包含'聽起來正常'")
        self.assertIn('輕度異常', mapping, "映射應該包含'輕度異常'")
        self.assertIn('重度異常', mapping, "映射應該包含'重度異常'")
        
        # 測試獲取不存在的欄位映射
        with self.assertRaises(ValueError):
            loader.get_mapping_dict('non_existent_field')
    
    def test_get_num_classes(self):
        """測試獲取類別數量功能"""
        loader = DataIndexLoader(self.test_csv_path, verify_paths=False)
        
        # 測試獲取score類別數量
        num_classes = loader.get_num_classes('score')
        self.assertEqual(num_classes, 41, "score應該有41個可能值(0-40)")
        
        # 測試獲取DrLee_Evaluation類別數量
        num_classes = loader.get_num_classes('DrLee_Evaluation')
        self.assertEqual(num_classes, 3, "DrLee_Evaluation應該有3個類別")
        
        # 測試獲取不存在的欄位類別數量
        with self.assertRaises(ValueError):
            loader.get_num_classes('non_existent_field')
    
    def test_get_files_by_patient(self):
        """測試獲取指定患者文件功能"""
        loader = DataIndexLoader(self.test_csv_path, verify_paths=False)
        
        # 測試獲取存在的患者文件
        files = loader.get_files_by_patient('p001')
        self.assertEqual(len(files), 2, "患者p001應該有2個文件")
        
        # 測試獲取不存在的患者文件
        files = loader.get_files_by_patient('non_existent_patient')
        self.assertEqual(len(files), 0, "不存在的患者應該返回空DataFrame")
    
    def test_split_by_patient(self):
        """測試按患者ID拆分數據功能"""
        loader = DataIndexLoader(self.test_csv_path, verify_paths=False)
        
        # 設置隨機種子以確保結果可重現
        np.random.seed(42)
        
        # 測試默認拆分
        train_indices, val_indices, test_indices = loader.split_by_patient()
        self.assertIsNotNone(train_indices, "應該返回訓練集索引")
        self.assertIsNotNone(val_indices, "應該返回驗證集索引")
        self.assertIsNotNone(test_indices, "應該返回測試集索引")
        
        # 測試自定義拆分比例
        train_indices, val_indices, test_indices = loader.split_by_patient(train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
        self.assertIsNotNone(train_indices, "應該返回訓練集索引")
        self.assertIsNotNone(val_indices, "應該返回驗證集索引")
        self.assertIsNotNone(test_indices, "應該返回測試集索引")
    
    def test_file_not_found(self):
        """測試文件不存在錯誤處理"""
        with self.assertRaises(FileNotFoundError):
            DataIndexLoader('non_existent_file.csv')
    
    def test_missing_columns(self):
        """測試缺少必要列的錯誤處理"""
        # 創建缺少必要列的測試數據
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as temp_file:
            invalid_data = pd.DataFrame({
                'file_path': ['/path/to/file1.wav', '/path/to/file2.wav'],
                # 缺少score列
            })
            invalid_data.to_csv(temp_file.name, index=False)
            invalid_csv_path = temp_file.name
        
        try:
            with self.assertRaises(ValueError):
                DataIndexLoader(invalid_csv_path)
        finally:
            os.unlink(invalid_csv_path)

if __name__ == '__main__':
    unittest.main() 