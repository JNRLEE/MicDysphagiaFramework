"""
數據索引加載器模組：處理data_index.csv文件的讀取與解析
功能：
1. 加載和解析data_index.csv文件
2. 提供基於不同條件的數據篩選
3. 建立標籤映射與轉換
4. 驗證數據路徑與格式
"""

import pandas as pd
import os
import logging
from typing import Dict, List, Union, Optional, Any

logger = logging.getLogger(__name__)

class DataIndexLoader:
    """
    加載和處理data_index.csv的工具類
    
    Args:
        index_path: 索引CSV文件的路徑
        verify_paths: 是否驗證索引中的文件路徑是否存在
        
    Attributes:
        index_path: 索引CSV文件路徑
        index_df: 加載的pandas DataFrame
        label_maps: 不同標籤類型的映射字典
    """
    
    def __init__(self, index_path: str, verify_paths: bool = True):
        """初始化數據索引加載器
        
        Args:
            index_path: 索引CSV文件的路徑
            verify_paths: 是否驗證索引中的文件路徑是否存在
        """
        self.index_path = index_path
        self.verify_paths = verify_paths
        self.index_df = None
        self.label_maps = {}
        self.load_index()
    
    def load_index(self) -> None:
        """加載索引CSV文件並進行基本處理"""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"數據索引文件不存在: {self.index_path}")
        
        self.index_df = pd.read_csv(self.index_path)
        
        # 驗證必要的列是否存在
        required_columns = ['file_path', 'score']
        missing_columns = [col for col in required_columns if col not in self.index_df.columns]
        if missing_columns:
            raise ValueError(f"數據索引缺少必要的列: {missing_columns}")
        
        # 驗證文件路徑
        if self.verify_paths:
            self._verify_file_paths()
        
        # 創建標籤類別的映射
        self._create_label_maps()
        
        logger.info(f"成功加載數據索引: {self.index_path}, 共 {len(self.index_df)} 條記錄")
    
    def _verify_file_paths(self) -> None:
        """驗證索引中的文件路徑是否存在"""
        invalid_paths = []
        for idx, row in self.index_df.iterrows():
            if not os.path.exists(row['file_path']):
                invalid_paths.append((idx, row['file_path']))
        
        if invalid_paths:
            logger.warning(f"警告: 發現{len(invalid_paths)}個無效的文件路徑")
            for idx, path in invalid_paths[:10]:  # 只顯示前10個
                logger.warning(f"  行 {idx}: {path}")
            if len(invalid_paths) > 10:
                logger.warning(f"  ...以及其他{len(invalid_paths)-10}個路徑")
    
    def _create_label_maps(self) -> None:
        """為分類標籤創建映射字典"""
        # 為分類標籤創建映射字典
        categorical_labels = ['DrLee_Evaluation', 'DrTai_Evaluation', 'selection']
        for label in categorical_labels:
            if label in self.index_df.columns:
                unique_values = sorted(self.index_df[label].dropna().unique())
                self.label_maps[label] = {value: idx for idx, value in enumerate(unique_values)}
                logger.info(f"{label} 標籤映射: {self.label_maps[label]}")
    
    def get_all_data(self) -> pd.DataFrame:
        """獲取所有數據
        
        Returns:
            pd.DataFrame: 完整的數據索引DataFrame
        """
        return self.index_df
    
    def filter_by_criteria(self, criteria: Dict) -> pd.DataFrame:
        """根據條件篩選數據
        
        Args:
            criteria: 篩選條件字典，例如{'status': 'processed', 'selection': '乾吞嚥1口'}
        
        Returns:
            pd.DataFrame: 符合條件的數據框
        """
        if not criteria:
            return self.index_df
        
        # 移除None值的條件
        valid_criteria = {k: v for k, v in criteria.items() if v is not None}
        if not valid_criteria:
            return self.index_df
        
        # 構建查詢條件
        query_parts = []
        for k, v in valid_criteria.items():
            if k not in self.index_df.columns:
                logger.warning(f"篩選條件中的欄位 '{k}' 在數據索引中不存在，將忽略此條件")
                continue
                
            if isinstance(v, str):
                query_parts.append(f"{k} == '{v}'")
            else:
                query_parts.append(f"{k} == {v}")
        
        query = ' & '.join(query_parts)
        if not query:
            return self.index_df
        
        try:
            filtered_df = self.index_df.query(query)
            logger.info(f"篩選條件 {valid_criteria} 匹配到 {len(filtered_df)} 條記錄")
            return filtered_df
        except Exception as e:
            logger.error(f"篩選條件錯誤: {str(e)}")
            return self.index_df
    
    def get_labels(self, label_field: str = 'score') -> pd.Series:
        """獲取指定標籤欄位的數據
        
        Args:
            label_field: 標籤欄位名稱，默認為'score'
        
        Returns:
            pd.Series: 包含標籤的Series
        """
        if label_field not in self.index_df.columns:
            raise ValueError(f"數據索引中不存在標籤欄位: {label_field}")
        
        return self.index_df[label_field]
    
    def get_item_by_path(self, file_path: str) -> Optional[Dict]:
        """根據文件路徑獲取一條數據
        
        Args:
            file_path: 文件路徑
        
        Returns:
            Optional[Dict]: 對應的數據記錄，如果不存在則返回None
        """
        result = self.index_df[self.index_df['file_path'] == file_path]
        if result.empty:
            return None
        return result.iloc[0].to_dict()
    
    def get_mapping_dict(self, label_field: str) -> Dict:
        """獲取標籤映射字典
        
        Args:
            label_field: 標籤欄位名稱
            
        Returns:
            Dict: 標籤到索引的映射字典
        """
        if label_field in self.label_maps:
            return self.label_maps[label_field]
        
        if label_field not in self.index_df.columns:
            raise ValueError(f"數據索引中不存在標籤欄位: {label_field}")
        
        # 如果之前沒有創建過此欄位的映射，則創建
        unique_values = sorted(self.index_df[label_field].dropna().unique())
        self.label_maps[label_field] = {value: idx for idx, value in enumerate(unique_values)}
        return self.label_maps[label_field]
    
    def get_num_classes(self, label_field: str) -> int:
        """獲取標籤類別數量
        
        Args:
            label_field: 標籤欄位名稱
            
        Returns:
            int: 標籤類別數量
        """
        if label_field == 'score':
            # EAT-10問卷得分是連續值，但這裡可以返回最大可能值（0-40）
            return 41
        
        mapping = self.get_mapping_dict(label_field)
        return len(mapping)
    
    def get_files_by_patient(self, patient_id: str) -> pd.DataFrame:
        """獲取指定患者的所有文件
        
        Args:
            patient_id: 患者ID
            
        Returns:
            pd.DataFrame: 包含該患者所有文件的DataFrame
        """
        if 'patient_id' not in self.index_df.columns:
            logger.warning("數據索引中不存在patient_id欄位")
            return pd.DataFrame()
        
        return self.index_df[self.index_df['patient_id'] == patient_id]
    
    def split_by_patient(self, train_ratio: float = 0.7, val_ratio: float = 0.15, 
                        test_ratio: float = 0.15, seed: int = 42) -> tuple:
        """按照患者ID拆分數據為訓練、驗證和測試集，確保N開頭和P開頭的患者ID在各集合中均勻分佈
        
        Args:
            train_ratio: 訓練集比例
            val_ratio: 驗證集比例
            test_ratio: 測試集比例
            seed: 隨機種子
            
        Returns:
            tuple: 包含訓練、驗證和測試集索引的元組
        """
        if 'patient_id' not in self.index_df.columns:
            logger.warning("數據索引中不存在patient_id欄位，無法按患者ID拆分")
            return None, None, None
        
        import random
        random.seed(seed)
        
        # 獲取唯一的患者ID
        patient_ids = self.index_df['patient_id'].unique()
        
        # 分別處理N開頭和P開頭的患者ID
        n_prefixed_patients = [pid for pid in patient_ids if pid.startswith('N')]
        p_prefixed_patients = [pid for pid in patient_ids if pid.startswith('P')]
        other_patients = [pid for pid in patient_ids if not (pid.startswith('N') or pid.startswith('P'))]
        
        # 打亂各組患者ID
        random.shuffle(n_prefixed_patients)
        random.shuffle(p_prefixed_patients)
        random.shuffle(other_patients)
        
        logger.info(f"患者ID分佈: N開頭 {len(n_prefixed_patients)} 位, P開頭 {len(p_prefixed_patients)} 位, 其他 {len(other_patients)} 位")
        
        # 為每組計算分割點
        n_train_size = int(len(n_prefixed_patients) * train_ratio)
        n_val_size = int(len(n_prefixed_patients) * val_ratio)
        
        p_train_size = int(len(p_prefixed_patients) * train_ratio)
        p_val_size = int(len(p_prefixed_patients) * val_ratio)
        
        other_train_size = int(len(other_patients) * train_ratio)
        other_val_size = int(len(other_patients) * val_ratio)
        
        # 分割各組患者ID
        n_train = n_prefixed_patients[:n_train_size]
        n_val = n_prefixed_patients[n_train_size:n_train_size+n_val_size]
        n_test = n_prefixed_patients[n_train_size+n_val_size:]
        
        p_train = p_prefixed_patients[:p_train_size]
        p_val = p_prefixed_patients[p_train_size:p_train_size+p_val_size]
        p_test = p_prefixed_patients[p_train_size+p_val_size:]
        
        other_train = other_patients[:other_train_size]
        other_val = other_patients[other_train_size:other_train_size+other_val_size]
        other_test = other_patients[other_train_size+other_val_size:]
        
        # 合併各組患者ID
        train_patients = n_train + p_train + other_train
        val_patients = n_val + p_val + other_val
        test_patients = n_test + p_test + other_test
        
        # 記錄各集合的患者ID分佈
        logger.info(f"訓練集患者ID分佈: N開頭 {len(n_train)} 位, P開頭 {len(p_train)} 位, 其他 {len(other_train)} 位")
        logger.info(f"驗證集患者ID分佈: N開頭 {len(n_val)} 位, P開頭 {len(p_val)} 位, 其他 {len(other_val)} 位")
        logger.info(f"測試集患者ID分佈: N開頭 {len(n_test)} 位, P開頭 {len(p_test)} 位, 其他 {len(other_test)} 位")
        
        # 獲取每個集合的文件索引
        train_indices = self.index_df[self.index_df['patient_id'].isin(train_patients)].index.tolist()
        val_indices = self.index_df[self.index_df['patient_id'].isin(val_patients)].index.tolist()
        test_indices = self.index_df[self.index_df['patient_id'].isin(test_patients)].index.tolist()
        
        logger.info(f"按患者ID拆分數據: 訓練集 {len(train_indices)} 個文件，驗證集 {len(val_indices)} 個文件，測試集 {len(test_indices)} 個文件")
        
        return train_indices, val_indices, test_indices


# 單元測試，當腳本直接運行時執行
if __name__ == "__main__":
    """
    Description: Minimal Executable Unit for data_index_loader.py，測試數據索引加載器的主要功能。
    Args: None
    Returns: None
    References: 無
    """
    import logging
    import tempfile
    import pandas as pd
    import os
    
    # 配置日誌
    logging.basicConfig(level=logging.INFO)
    
    # 創建測試數據
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as temp_file:
        test_data = pd.DataFrame({
            'file_path': ['/path/to/file1.wav', '/path/to/file2.wav', '/path/to/file3.wav'],
            'score': [15, 25, 5],
            'patient_id': ['p001', 'p002', 'p001'],
            'DrLee_Evaluation': ['聽起來正常', '輕度異常', '重度異常'],
            'DrTai_Evaluation': ['正常', '無OR 輕微吞嚥障礙', '重度吞嚥障礙'],
            'selection': ['乾吞嚥1口', '吞水10ml', '餅乾1塊'],
            'status': ['processed', 'processed', 'raw']
        })
        test_data.to_csv(temp_file.name, index=False)
        test_csv_path = temp_file.name
    
    try:
        # 測試加載功能
        print("測試1: 基本加載功能")
        loader = DataIndexLoader(test_csv_path, verify_paths=False)
        print(f"成功: 加載了 {len(loader.index_df)} 條記錄")
        
        # 測試篩選功能
        print("\n測試2: 篩選功能")
        filtered = loader.filter_by_criteria({'status': 'processed'})
        print(f"成功: 篩選到 {len(filtered)} 條處理過的記錄")
        
        # 測試標籤映射
        print("\n測試3: 標籤映射")
        mapping = loader.get_mapping_dict('DrLee_Evaluation')
        print(f"成功: DrLee_Evaluation 標籤映射 = {mapping}")
        
        # 測試按患者拆分
        print("\n測試4: 按患者拆分")
        train, val, test = loader.split_by_patient(train_ratio=0.5, val_ratio=0.5, test_ratio=0)
        print(f"成功: 拆分數據為訓練集 {len(train)} 個文件，驗證集 {len(val)} 個文件，測試集 {len(test)} 個文件")
        
        print("\n所有測試通過！")
        
    except Exception as e:
        print(f"測試失敗: {str(e)}")
    
    finally:
        # 清理測試文件
        os.unlink(test_csv_path) 