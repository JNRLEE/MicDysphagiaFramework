"""
索引數據集基類：為所有數據集類型提供統一的索引CSV支援
功能：
1. 提供基於數據索引CSV的數據加載接口
2. 處理標籤映射與轉換
3. 提供退化機制，當索引加載失敗時回退到原始加載方式
4. 支持按患者ID拆分數據集
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import logging
from typing import Dict, List, Tuple, Optional, Callable, Union, Any

from utils.data_index_loader import DataIndexLoader

logger = logging.getLogger(__name__)

class IndexedDatasetBase(Dataset):
    """
    基於數據索引CSV的數據集基類
    
    Args:
        index_path: 索引CSV文件的路徑
        label_field: 標籤欄位名稱，可以是 'score', 'DrLee_Evaluation', 'DrTai_Evaluation', 'selection'
        transform: 數據轉換函數
        filter_criteria: 篩選條件字典
        fallback_to_direct: 當索引加載失敗時是否退化到直接加載
        
    Attributes:
        index_path: 索引CSV文件路徑
        label_field: 使用的標籤欄位
        transform: 數據轉換函數
        filter_criteria: 篩選條件
        index_loader: 數據索引加載器
        data_df: 篩選後的數據框
        labels: 處理後的標籤數組
        num_classes: 分類任務的類別數量
    """
    
    def __init__(
        self, 
        index_path: str, 
        label_field: str = 'score',
        transform: Optional[Callable] = None,
        filter_criteria: Optional[Dict] = None,
        fallback_to_direct: bool = True
    ):
        """初始化索引數據集基類
        
        Args:
            index_path: 索引CSV文件的路徑
            label_field: 標籤欄位名稱，可以是 'score', 'DrLee_Evaluation', 'DrTai_Evaluation', 'selection'
            transform: 數據轉換函數
            filter_criteria: 篩選條件字典
            fallback_to_direct: 當索引加載失敗時是否退化到直接加載
        """
        self.index_path = index_path
        self.label_field = label_field
        self.transform = transform
        self.filter_criteria = filter_criteria or {}
        self.fallback_to_direct = fallback_to_direct
        
        # 初始化屬性
        self.index_loader = None
        self.data_df = None
        self.labels = None
        self.label_map = None
        self.num_classes = None
        self.is_direct_mode = False
        
        # 嘗試加載索引
        self._load_index()
    
    def _load_index(self) -> None:
        """加載索引CSV並設置數據"""
        if not self.index_path or not os.path.exists(self.index_path):
            if self.fallback_to_direct:
                logger.warning(f"索引文件不存在: {self.index_path}，將使用直接加載模式")
                self.is_direct_mode = True
                self.setup_direct_mode()
                return
            else:
                raise FileNotFoundError(f"索引文件不存在: {self.index_path}")
        
        try:
            # 加載索引
            self.index_loader = DataIndexLoader(self.index_path)
            self.data_df = self.index_loader.filter_by_criteria(self.filter_criteria)
            
            if len(self.data_df) == 0:
                logger.warning(f"篩選條件 {self.filter_criteria} 沒有匹配到任何記錄")
                if self.fallback_to_direct:
                    logger.warning("將使用直接加載模式")
                    self.is_direct_mode = True
                    self.setup_direct_mode()
                    return
                else:
                    raise ValueError(f"篩選條件 {self.filter_criteria} 沒有匹配到任何記錄")
            
            # 設置標籤
            self._setup_labels()
            
            logger.info(f"成功加載數據索引，共 {len(self.data_df)} 條記錄")
            
        except Exception as e:
            logger.error(f"索引加載失敗: {str(e)}")
            if self.fallback_to_direct:
                logger.warning("將使用直接加載模式")
                self.is_direct_mode = True
                self.setup_direct_mode()
            else:
                raise
    
    def _setup_labels(self) -> None:
        """設置標籤數據"""
        # 檢查標籤欄位是否存在
        if self.label_field not in self.data_df.columns:
            raise ValueError(f"數據索引中不存在標籤欄位: {self.label_field}")
        
        # 獲取標籤
        self.labels = self.data_df[self.label_field].values
        
        # 處理分類標籤
        if self.label_field in ['DrLee_Evaluation', 'DrTai_Evaluation', 'selection']:
            self.label_map = self.index_loader.get_mapping_dict(self.label_field)
            self.labels = np.array([self.label_map.get(label, 0) for label in self.labels])
            self.num_classes = len(self.label_map)
            logger.info(f"分類任務 {self.label_field}: {self.num_classes} 個類別，標籤映射 = {self.label_map}")
        else:
            # 回歸任務，如'score'
            self.labels = self.labels.astype(np.float32)
            self.num_classes = 1
            logger.info(f"回歸任務 {self.label_field}")
    
    def setup_direct_mode(self) -> None:
        """設置直接加載模式，由子類實現"""
        raise NotImplementedError("子類必須實現setup_direct_mode方法")
    
    def __len__(self) -> int:
        """返回數據集大小"""
        if self.is_direct_mode:
            return self.direct_len()
        return len(self.data_df)
    
    def __getitem__(self, idx: int) -> Tuple:
        """獲取指定索引的數據
        
        Args:
            idx: 數據索引
            
        Returns:
            Tuple: (數據, 標籤) 對
        """
        if self.is_direct_mode:
            return self.direct_getitem(idx)
        
        data_row = self.data_df.iloc[idx]
        
        # 獲取數據
        data = self.load_data(data_row)
        
        # 獲取標籤
        label = self.labels[idx]
        
        # 應用轉換
        if self.transform:
            data = self.transform(data)
        
        return data, label
    
    def load_data(self, data_row: pd.Series) -> Any:
        """加載數據，由子類實現
        
        Args:
            data_row: 數據行
            
        Returns:
            Any: 加載的數據
        """
        raise NotImplementedError("子類必須實現load_data方法")
    
    def direct_getitem(self, idx: int) -> Tuple:
        """直接加載模式的數據獲取，由子類實現
        
        Args:
            idx: 數據索引
            
        Returns:
            Tuple: (數據, 標籤) 對
        """
        raise NotImplementedError("子類必須實現direct_getitem方法")
    
    def direct_len(self) -> int:
        """直接加載模式的數據集大小，由子類實現
        
        Returns:
            int: 數據集大小
        """
        raise NotImplementedError("子類必須實現direct_len方法")
    
    def get_label_map(self) -> Optional[Dict]:
        """獲取標籤映射字典
        
        Returns:
            Optional[Dict]: 標籤映射字典，如果是回歸任務則返回None
        """
        return self.label_map
    
    def split_by_patient(self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
        """按照患者ID拆分數據為訓練、驗證和測試集
        
        Args:
            train_ratio: 訓練集比例
            val_ratio: 驗證集比例
            test_ratio: 測試集比例
            seed: 隨機種子
            
        Returns:
            Tuple[List[int], List[int], List[int]]: (訓練集索引, 驗證集索引, 測試集索引)
        """
        if self.is_direct_mode:
            logger.warning("直接加載模式不支持按患者ID拆分，將使用隨機拆分")
            return None, None, None
        
        if not self.index_loader:
            logger.warning("索引加載器未初始化，無法按患者ID拆分")
            return None, None, None
        
        return self.index_loader.split_by_patient(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed
        )
    
    @property
    def is_classification(self) -> bool:
        """是否為分類任務
        
        Returns:
            bool: 如果是分類任務返回True，否則返回False
        """
        return self.label_field in ['DrLee_Evaluation', 'DrTai_Evaluation', 'selection']


# 測試代碼
if __name__ == "__main__":
    """
    Description: Minimal Executable Unit for indexed_dataset.py，測試索引數據集基類的接口。
    Args: None
    Returns: None
    References: 無
    """
    
    import logging
    
    # 配置日誌
    logging.basicConfig(level=logging.INFO)
    
    # 創建一個簡單的測試子類
    class TestIndexedDataset(IndexedDatasetBase):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        
        def setup_direct_mode(self):
            self.direct_data = [(i, i) for i in range(10)]
        
        def load_data(self, data_row):
            return data_row['file_path']
        
        def direct_getitem(self, idx):
            return self.direct_data[idx]
        
        def direct_len(self):
            return len(self.direct_data)
    
    # 使用測試子類測試基類功能
    try:
        print("測試: 使用不存在的索引文件，應該進入直接加載模式")
        dataset = TestIndexedDataset(
            index_path="non_existent_file.csv",
            label_field="score",
            fallback_to_direct=True
        )
        print(f"成功: 直接加載模式, 數據集大小 = {len(dataset)}")
        print(f"數據樣本: {dataset[0]}")
        
        # 這個測試會失敗，因為索引文件不存在且未啟用直接加載模式
        # 不運行此測試以避免中斷其他測試
        # dataset_fail = TestIndexedDataset(
        #     index_path="non_existent_file.csv",
        #     label_field="score",
        #     fallback_to_direct=False
        # )
        
        print("\n所有測試通過！")
        
    except Exception as e:
        print(f"測試失敗: {str(e)}") 