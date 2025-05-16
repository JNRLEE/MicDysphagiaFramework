"""
測試索引數據集基類的功能

此測試文件測試data/indexed_dataset.py中的IndexedDatasetBase類的所有主要功能，包括：
1. 基本加載功能
2. 標籤處理功能
3. 數據獲取功能
4. 退化機制功能
5. 按患者ID拆分數據功能

測試日期: 2023-11-15
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import tempfile
import torch
from pathlib import Path

# 添加項目根目錄到路徑，以便能夠導入模組
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.indexed_dataset import IndexedDatasetBase

class TestIndexedDataset(IndexedDatasetBase):
    """用於測試的索引數據集子類"""
    
    def __init__(self, **kwargs):
        """初始化測試數據集"""
        super().__init__(**kwargs)
    
    def setup_direct_mode(self):
        """設置直接加載模式"""
        self.direct_data = [(np.array([i]), float(i)) for i in range(10)]
        self.labels = np.array([float(i) for i in range(10)])
        self.num_classes = 1  # 回歸任務
    
    def load_data(self, data_row):
        """從數據行加載數據"""
        # 模擬從文件路徑加載數據
        # 在實際應用中，這裡會加載實際的數據文件
        # 這裡我們只是將文件路徑轉換為一個簡單的數組
        path = data_row['file_path']
        # 從路徑中提取一個數字作為模擬數據
        try:
            num = int(path.split('file')[1].split('.')[0])
        except:
            num = 0
        return np.array([num])
    
    def direct_getitem(self, idx):
        """直接加載模式的數據獲取"""
        return self.direct_data[idx]
    
    def direct_len(self):
        """直接加載模式的數據集大小"""
        return len(self.direct_data)

class TestIndexedDatasetBase(unittest.TestCase):
    """測試IndexedDatasetBase類"""
    
    def setUp(self):
        """在每個測試之前設置測試環境"""
        # 創建臨時測試數據
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w')
        self.test_data = pd.DataFrame({
            'file_path': ['/path/to/file1.wav', '/path/to/file2.wav', '/path/to/file3.wav', '/path/to/file4.wav'],
            'score': [15.0, 25.0, 5.0, 30.0],
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
        dataset = TestIndexedDataset(
            index_path=self.test_csv_path,
            label_field='score'
        )
        self.assertEqual(len(dataset), 4, "應該加載4條記錄")
        self.assertFalse(dataset.is_direct_mode, "應該是索引模式")
        self.assertEqual(dataset.num_classes, 1, "回歸任務應該有1個輸出")
    
    def test_classification_label(self):
        """測試分類標籤處理"""
        dataset = TestIndexedDataset(
            index_path=self.test_csv_path,
            label_field='DrLee_Evaluation'
        )
        self.assertEqual(dataset.num_classes, 3, "應該有3個分類")
        self.assertTrue(dataset.is_classification, "應該是分類任務")
        
        # 檢查標籤映射
        label_map = dataset.get_label_map()
        self.assertIsNotNone(label_map, "應該有標籤映射")
        self.assertEqual(len(label_map), 3, "應該有3個唯一標籤")
        
        # 檢查第一個樣本的標籤
        _, label = dataset[0]
        self.assertEqual(label, 0, "第一個樣本的標籤應該映射為0")
    
    def test_regression_label(self):
        """測試回歸標籤處理"""
        dataset = TestIndexedDataset(
            index_path=self.test_csv_path,
            label_field='score'
        )
        self.assertEqual(dataset.num_classes, 1, "回歸任務應該有1個輸出")
        self.assertFalse(dataset.is_classification, "應該是回歸任務")
        
        # 檢查標籤映射
        label_map = dataset.get_label_map()
        self.assertIsNone(label_map, "回歸任務不應該有標籤映射")
        
        # 檢查第一個樣本的標籤
        _, label = dataset[0]
        self.assertEqual(label, 15.0, "第一個樣本的標籤應該是15.0")
    
    def test_filter_criteria(self):
        """測試篩選條件功能"""
        # 測試單一條件篩選
        dataset = TestIndexedDataset(
            index_path=self.test_csv_path,
            label_field='score',
            filter_criteria={'status': 'processed'}
        )
        self.assertEqual(len(dataset), 3, "應該篩選出3條processed記錄")
        
        # 測試多條件篩選
        dataset = TestIndexedDataset(
            index_path=self.test_csv_path,
            label_field='score',
            filter_criteria={'status': 'processed', 'patient_id': 'p001'}
        )
        self.assertEqual(len(dataset), 1, "應該篩選出1條符合兩個條件的記錄")
        
        # 測試無匹配篩選
        dataset = TestIndexedDataset(
            index_path=self.test_csv_path,
            label_field='score',
            filter_criteria={'status': 'invalid_status'}
        )
        self.assertTrue(dataset.is_direct_mode, "無匹配記錄應該進入直接加載模式")
    
    def test_transform(self):
        """測試數據轉換功能"""
        # 定義簡單的轉換函數
        def transform(x):
            return x * 2
        
        dataset = TestIndexedDataset(
            index_path=self.test_csv_path,
            label_field='score',
            transform=transform
        )
        
        # 獲取第一個樣本
        data, _ = dataset[0]
        self.assertEqual(data[0], 2, "數據應該被轉換為原來的2倍")
    
    def test_fallback_mechanism(self):
        """測試退化機制"""
        # 測試文件不存在時的退化
        dataset = TestIndexedDataset(
            index_path='non_existent_file.csv',
            label_field='score',
            fallback_to_direct=True
        )
        self.assertTrue(dataset.is_direct_mode, "文件不存在應該進入直接加載模式")
        self.assertEqual(len(dataset), 10, "直接加載模式應該有10個樣本")
        
        # 測試禁用退化機制
        with self.assertRaises(FileNotFoundError):
            TestIndexedDataset(
                index_path='non_existent_file.csv',
                label_field='score',
                fallback_to_direct=False
            )
    
    def test_split_by_patient(self):
        """測試按患者ID拆分數據功能"""
        dataset = TestIndexedDataset(
            index_path=self.test_csv_path,
            label_field='score'
        )
        
        # 設置隨機種子以確保結果可重現
        np.random.seed(42)
        
        # 測試默認拆分
        train_indices, val_indices, test_indices = dataset.split_by_patient()
        self.assertIsNotNone(train_indices, "應該返回訓練集索引")
        self.assertIsNotNone(val_indices, "應該返回驗證集索引")
        self.assertIsNotNone(test_indices, "應該返回測試集索引")
        
        # 測試直接加載模式下的拆分
        direct_dataset = TestIndexedDataset(
            index_path='non_existent_file.csv',
            label_field='score',
            fallback_to_direct=True
        )
        train_indices, val_indices, test_indices = direct_dataset.split_by_patient()
        self.assertIsNone(train_indices, "直接加載模式應該返回None")
        self.assertIsNone(val_indices, "直接加載模式應該返回None")
        self.assertIsNone(test_indices, "直接加載模式應該返回None")
    
    def test_getitem(self):
        """測試數據獲取功能"""
        dataset = TestIndexedDataset(
            index_path=self.test_csv_path,
            label_field='score'
        )
        
        # 獲取第一個樣本
        data, label = dataset[0]
        self.assertIsInstance(data, np.ndarray, "數據應該是numpy數組")
        self.assertEqual(data[0], 1, "第一個樣本的數據應該是1")
        self.assertEqual(label, 15.0, "第一個樣本的標籤應該是15.0")
        
        # 獲取直接加載模式的樣本
        direct_dataset = TestIndexedDataset(
            index_path='non_existent_file.csv',
            label_field='score',
            fallback_to_direct=True
        )
        data, label = direct_dataset[0]
        self.assertIsInstance(data, np.ndarray, "數據應該是numpy數組")
        self.assertEqual(data[0], 0, "第一個樣本的數據應該是0")
        self.assertEqual(label, 0.0, "第一個樣本的標籤應該是0.0")
    
    def test_invalid_label_field(self):
        """測試無效標籤欄位的錯誤處理"""
        # 修正：當提供無效標籤欄位時，框架會自動退化到直接加載模式，而不是拋出異常
        # 因此，我們測試是否正確進入了直接加載模式
        dataset = TestIndexedDataset(
            index_path=self.test_csv_path,
            label_field='non_existent_field',
            fallback_to_direct=True
        )
        self.assertTrue(dataset.is_direct_mode, "無效標籤欄位應該進入直接加載模式")
        
        # 測試禁用退化機制時是否拋出異常
        with self.assertRaises(ValueError):
            TestIndexedDataset(
                index_path=self.test_csv_path,
                label_field='non_existent_field',
                fallback_to_direct=False
            )

if __name__ == '__main__':
    unittest.main() 