"""
特徵數據集：支持從預處理過的特徵文件中讀取特徵
功能：
1. 從特徵文件（如JSON、CSV、NPZ等）中讀取預處理的特徵
2. 支持多種特徵組合和選擇
3. 支持按患者ID拆分數據集
4. 支持特徵標準化和轉換
5. 支持從索引CSV加載數據
6. 支持特徵向量置中填充
7. 支持PCA降維壓縮特徵向量

數據讀取邏輯：
1. NPZ特徵文件讀取:
   - 使用`np.load(file_path, allow_pickle=True)`讀取NPZ文件
   - 優先從'features'鍵獲取特徵數據
   - 如果沒有'features'鍵，嘗試提取所有非元數據字段
   - 處理不同維度的特徵向量，默認限制為10000維，過大時會自動截斷

2. JSON特徵文件讀取:
   - 使用json.load讀取文件內容
   - 支持單樣本和多樣本JSON文件格式
   - 從JSON中提取指定的特徵字段或所有非元數據字段

3. CSV特徵文件讀取:
   - 使用pandas.read_csv讀取表格數據
   - 支持提取特定行或全部數據
   - 從行數據中選擇所需的特徵列

4. 患者信息關聯:
   - 可以從特徵文件中獲取患者ID和分數
   - 也可以從文件名或相關的info.json文件中提取患者信息
   - 支持不同的患者ID和標籤列名配置

5. 特徵標準化:
   - 使用sklearn.preprocessing.StandardScaler對特徵進行標準化
   - 對於不同維度的特徵，使用最常見維度的特徵進行標準化器訓練
   - 處理標準化過程中可能出現的異常

6. 緩存機制:
   - 支持緩存已讀取的特徵以提高效率
   - 適用於反复讀取相同特徵的場景
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json
import os
import logging
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from pathlib import Path
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # 添加PCA引入

# 引入索引數據集基類
from data.indexed_dataset import IndexedDatasetBase

logger = logging.getLogger(__name__)

class FeatureDataset(IndexedDatasetBase):
    """特徵數據集類，用於從預處理過的特徵文件中讀取特徵
    
    主要功能：
    1. 從NPZ、JSON、CSV等格式讀取預處理的特徵數據
    2. 支持多種特徵選擇和過濾方式
    3. 支持特徵標準化和轉換
    4. 支持按患者ID拆分數據集
    5. 支持從索引CSV加載數據
    6. 支持PCA降維壓縮特徵
    
    數據讀取模式：
    1. 索引模式：使用data_index.csv加載數據
       - 提供index_path參數啟用此模式
       - 使用label_field指定標籤欄位
       - 使用filter_criteria篩選數據
       
    2. 直接模式：直接從目錄讀取特徵文件
       - 使用data_path參數指定特徵文件的根目錄
       - 從特徵文件或相關的info.json獲取標籤信息
       - 此模式與原始實現兼容
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        transform: Optional[Dict[str, Any]] = None,
        is_train: bool = True,
        cache_features: bool = True,
        # 新增索引模式參數
        index_path: Optional[str] = None,
        label_field: str = 'score',
        filter_criteria: Optional[Dict] = None,
        # 新增填充模式參數
        padding_mode: Optional[str] = None,
        # 新增降維參數
        compression_method: Optional[str] = None,
        target_dim: Optional[int] = None
    ):
        """初始化特徵數據集
        
        Args:
            data_path: 特徵數據路徑（直接模式必填）
            config: 配置字典（直接模式必填）
            transform: 轉換配置
            is_train: 是否為訓練模式
            cache_features: 是否緩存特徵
            index_path: 索引CSV文件路徑（索引模式必填）
            label_field: 標籤欄位名稱，可以是'score', 'DrLee_Evaluation', 'DrTai_Evaluation', 'selection'
            filter_criteria: 篩選條件字典
            padding_mode: 填充模式，'right'(默認)或'center'(置中填充)
            compression_method: 壓縮方法，'pca'(PCA降維)或None(不壓縮)
            target_dim: 目標維度，用於PCA降維
        """
        # 決定是否使用索引模式
        self.use_index_mode = index_path is not None and os.path.exists(index_path)
        
        # 初始化共用屬性
        self.is_train = is_train
        self.cache_features = cache_features
        self.feature_cache = {}
        self.scaler = None
        self.pca = None  # 初始化PCA為None
        self.samples = []  # 確保samples始終被初始化為空列表
        
        if self.use_index_mode:
            # 索引模式初始化
            self.data_path = None
            self.config = config or {}
            
            # 獲取特徵配置
            self.feature_config = self.config.get('data', {}).get('preprocessing', {}).get('features', {})
            self.normalize = self.feature_config.get('normalize', True)
            
            # 獲取padding_mode配置
            self.padding_mode = padding_mode or self.feature_config.get('padding_mode', 'right')
            self.max_feature_dim = self.feature_config.get('max_feature_dim', 1024)
            
            # 獲取壓縮方法配置
            self.compression_method = compression_method or self.feature_config.get('compression_method', None)
            self.target_dim = target_dim or self.feature_config.get('target_dim', self.max_feature_dim)
            
            # 呼叫父類初始化
            super().__init__(
                index_path=index_path,
                label_field=label_field,
                transform=transform,
                filter_criteria=filter_criteria,
                fallback_to_direct=True
            )
            
            # 從data_df創建samples列表，以便與直接模式兼容
            if hasattr(self, 'data_df') and self.data_df is not None:
                self.samples = []
                for idx, row in self.data_df.iterrows():
                    sample = row.to_dict()
                    # 確保樣本包含必要的字段
                    if 'patient_id' not in sample and 'file_path' in sample:
                        # 嘗試從文件路徑提取患者ID
                        file_path = sample['file_path']
                        patient_id = os.path.basename(os.path.dirname(file_path))
                        sample['patient_id'] = patient_id
                    
                    # 確保樣本包含標籤
                    if 'label' not in sample and hasattr(self, 'labels') and idx < len(self.labels):
                        sample['label'] = self.labels[idx]
                    
                    self.samples.append(sample)
                
                logger.info(f"從數據索引創建了 {len(self.samples)} 個樣本")
                
                # 如果使用置中填充，先掃描所有特徵以找出最大長度
                if self.padding_mode == 'center':
                    logger.info(f"使用置中填充模式，開始掃描最大特徵長度...")
                    self.max_feature_length = self._scan_max_feature_length()
                    logger.info(f"掃描完成，最大特徵長度: {self.max_feature_length}")
                
                # 如果使用PCA降維，初始化PCA模型
                if self.compression_method == 'pca' and is_train:
                    logger.info(f"使用PCA降維，初始化PCA模型 (目標維度: {self.target_dim})...")
                    self._init_pca()
            
            # 如果需要標準化並處於訓練模式，初始化標準化器
            if self.normalize and is_train and not self.is_direct_mode:
                self._init_scaler()
            
        else:
            # 直接模式初始化
            if data_path is None or config is None:
                raise ValueError("直接模式下需要提供data_path和config參數")
                
            self.data_path = Path(data_path)
            self.config = config
            self.is_direct_mode = True
            
            # 獲取特徵配置
            self.feature_config = config.get('data', {}).get('preprocessing', {}).get('features', {})
            
            # 獲取padding_mode配置
            self.padding_mode = padding_mode or self.feature_config.get('padding_mode', 'right')
            self.max_feature_dim = self.feature_config.get('max_feature_dim', 1024)
            
            # 獲取壓縮方法配置
            self.compression_method = compression_method or self.feature_config.get('compression_method', None)
            self.target_dim = target_dim or self.feature_config.get('target_dim', self.max_feature_dim)
            
            # 獲取feature_type，可能存在不同位置
            data_config = config.get('data', {})
            self.feature_type = data_config.get('type', 'feature')
            
            # 獲取擴展名配置（如果有）
            self.feature_extension = data_config.get('source', {}).get('feature_extension', None)
            
            # 是特徵數據集但沒有指定具體類型，根據擴展名判斷
            if self.feature_type == 'feature' and self.feature_extension:
                self.feature_type = self.feature_extension  # 使用擴展名作為類型
            
            # 確保兼容性
            if self.feature_type not in ['json', 'csv', 'npz', 'npy']:
                logger.warning(f"未識別的特徵類型 '{self.feature_type}'，使用 feature_extension '{self.feature_extension}' 代替")
                if self.feature_extension in ['json', 'csv', 'npz', 'npy']:
                    self.feature_type = self.feature_extension
                else:
                    logger.warning(f"無法確定特徵類型，將嘗試依次搜索 json, csv, npz, npy 格式的文件")
                    self.feature_type = 'auto'  # 自動檢測
                
            self.feature_names = self.feature_config.get('names', [])  # 要使用的特徵列表
            self.label_name = self.feature_config.get('label', 'score')  # 標籤列名
            self.patient_id_column = self.feature_config.get('patient_id_column', 'patient_id')  # 患者ID列名
            
            # 特徵標準化設置
            self.normalize = self.feature_config.get('normalize', True)
            
            # 收集樣本
            self.samples = self._collect_samples()
            
            logger.info(f"加載了 {len(self.samples)} 個特徵樣本")
            
            # 如果使用置中填充，先掃描所有特徵以找出最大長度
            if self.padding_mode == 'center':
                logger.info(f"使用置中填充模式，開始掃描最大特徵長度...")
                self.max_feature_length = self._scan_max_feature_length()
                logger.info(f"掃描完成，最大特徵長度: {self.max_feature_length}")
            
            # 如果使用PCA降維，初始化PCA模型
            if self.compression_method == 'pca' and is_train:
                logger.info(f"使用PCA降維，初始化PCA模型 (目標維度: {self.target_dim})...")
                self._init_pca()
            
            # 如果需要標準化並處於訓練模式，初始化標準化器
            if self.normalize and is_train:
                self._init_scaler()
    
    def setup_direct_mode(self) -> None:
        """設置直接加載模式
        
        當索引加載失敗時，如果fallback_to_direct為True，則調用此方法
        設置為直接模式，但由於沒有data_path和config，因此僅提供最小功能
        """
        logger.warning("索引加載失敗並退化到直接模式，但未提供data_path和config，將使用空數據集")
        self.samples = []
        self.feature_type = 'auto'
    
    def load_data(self, data_row: dict) -> torch.Tensor:
        """從數據行加載特徵數據
        
        Args:
            data_row: 數據行，包含file_path等字段
            
        Returns:
            torch.Tensor: 加載的特徵數據
        """
        # 優先使用features_path欄位，如果有的話
        if 'features_path' in data_row and os.path.exists(data_row['features_path']):
            file_path = data_row['features_path']
        else:
            # file_path可能是目錄，需要尋找特徵文件
            dir_path = data_row['file_path']
            if os.path.isdir(dir_path):
                # 嘗試查找命名慣例的特徵文件
                dir_name = os.path.basename(dir_path)
                feature_path = os.path.join(dir_path, f"{dir_name}_features.npy")
                
                if os.path.exists(feature_path):
                    file_path = feature_path
                else:
                    # 嘗試查找目錄中的任何.npy文件
                    npy_files = [f for f in os.listdir(dir_path) if f.endswith('_features.npy') or f.endswith('.npz') or f.endswith('.npy')]
                    if npy_files:
                        file_path = os.path.join(dir_path, npy_files[0])
                    else:
                        logger.error(f"在目錄 {dir_path} 中找不到特徵文件")
                        # 返回零張量作為後備
                        feature_dim = self._infer_feature_dim()
                        return torch.zeros(feature_dim, dtype=torch.float32)
            else:
                # 可能file_path本身就是文件
                file_path = dir_path
        
        # 檢查緩存
        if self.cache_features and file_path in self.feature_cache:
            return self.feature_cache[file_path]
        
        try:
            # 判斷文件類型並加載特徵
            features = self._load_feature_by_path(file_path)
            
            # 標準化特徵
            if self.normalize and self.scaler is not None:
                try:
                    features = self.scaler.transform([features])[0]
                except Exception as e:
                    logger.warning(f"特徵標準化失敗: {str(e)}")
            
            # 轉換為張量
            features_tensor = torch.tensor(features, dtype=torch.float32)
            
            # 緩存結果
            if self.cache_features:
                self.feature_cache[file_path] = features_tensor
            
            return features_tensor
            
        except Exception as e:
            logger.error(f"加載特徵文件失敗: {file_path}, 錯誤: {str(e)}")
            # 返回零張量作為後備
            
            # 如果使用置中填充模式，創建與max_feature_length相同長度的零張量
            if self.padding_mode == 'center' and hasattr(self, 'max_feature_length'):
                feature_dim = self.max_feature_length
            else:
                # 否則使用配置的max_feature_dim或推斷的特徵維度
                feature_dim = self._infer_feature_dim()
                
            return torch.zeros(feature_dim, dtype=torch.float32)
    
    def direct_getitem(self, idx: int) -> Tuple[torch.Tensor, Any]:
        """直接模式下獲取項目
        
        Args:
            idx: 索引
            
        Returns:
            Tuple[torch.Tensor, Any]: (特徵數據, 標籤)
        """
        if not hasattr(self, 'samples') or not self.samples:
            # 如果沒有樣本，返回空數據
            return torch.zeros(10, dtype=torch.float32), 0  # 默認10維特徵
            
        sample = self.samples[idx]
        
        # 加載特徵
        try:
            # 使用_load_features獲取特徵
            features = self._load_features(idx)
            
            # 轉換為張量
            features = torch.tensor(features, dtype=torch.float32)
            
            # 獲取標籤
            label = sample['label']
            
            return features, label
            
        except Exception as e:
            logger.error(f"處理樣本失敗，索引 {idx}: {str(e)}")
            # 返回零張量作為後備
            feature_dim = self._infer_feature_dim()
            return torch.zeros(feature_dim, dtype=torch.float32), sample.get('label', 0)
    
    def direct_len(self) -> int:
        """直接模式下的數據集大小
        
        Returns:
            int: 數據集大小
        """
        if hasattr(self, 'samples'):
            return len(self.samples)
        return 0
    
    def _infer_feature_dim(self) -> int:
        """推斷特徵維度
        
        嘗試從配置或現有樣本中推斷特徵向量的維度
        
        Returns:
            int: 推斷的特徵維度，如果無法推斷則返回默認值1024
        """
        # 從配置中獲取
        max_feature_dim = self.feature_config.get('max_feature_dim', None)
        if max_feature_dim is not None:
            return max_feature_dim
        
        # 嘗試從樣本中獲取
        if hasattr(self, 'samples') and len(self.samples) > 0:
            for sample in self.samples:
                try:
                    if 'features_path' in sample and os.path.exists(sample['features_path']):
                        # 嘗試加載第一個樣本的特徵以推斷維度
                        features = self._load_feature_by_path(sample['features_path'])
                        if features is not None:
                            return len(features)
                except Exception:
                    pass
        
        # 默認值
        return 1024
    
    def _load_feature_by_path(self, file_path: str) -> np.ndarray:
        """根據文件路徑加載特徵
        
        Args:
            file_path: 特徵文件路徑
            
        Returns:
            np.ndarray: 特徵數組
        """
        # 根據擴展名判斷文件類型
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext == '.npz':
                features = self._load_npz_feature(file_path)
            elif ext == '.npy':
                features = self._load_npy_feature(file_path)
            elif ext == '.json':
                features = self._load_json_feature(file_path)
            elif ext == '.csv':
                features = self._load_csv_feature(file_path)
            else:
                raise ValueError(f"不支持的特徵文件類型: {ext}")
            
            # 特徵處理邏輯 (先填充，再PCA降維)
            if self.padding_mode == 'center' and hasattr(self, 'max_feature_length'):
                # 步驟1: 使用置中填充
                features = self._center_pad_features(features, self.max_feature_length)
            
            # 步驟2: 如果啟用了PCA降維
            if self.compression_method == 'pca' and self.pca is not None:
                original_length = len(features)
                features = self._apply_pca(features)
                logger.debug(f"PCA降維: {original_length} -> {len(features)}")
            elif len(features) > self.max_feature_dim:
                # 如果未使用PCA，則使用傳統的截斷方法
                features = features[:self.max_feature_dim]
                
            return features
                
        except Exception as e:
            logger.error(f"加載特徵文件失敗: {file_path}, 錯誤: {str(e)}")
            raise ValueError(f"無法加載特徵文件 {file_path}: {str(e)}")
    
    def _load_npz_feature(self, file_path: str) -> np.ndarray:
        """加載NPZ格式的特徵文件
        
        Args:
            file_path: NPZ文件路徑
            
        Returns:
            np.ndarray: 特徵數組
        """
        try:
            data = np.load(file_path, allow_pickle=True)
            
            # 優先從'features'鍵獲取特徵
            if 'features' in data:
                features = data['features']
                if isinstance(features, np.ndarray):
                    return features
            
            # 否則嘗試提取所有非元數據字段
            feature_arrays = []
            for key in data.keys():
                if key != 'metadata':
                    arr = data[key]
                    if isinstance(arr, np.ndarray):
                        feature_arrays.append(arr)
            
            if feature_arrays:
                # 如果有多個特徵數組，連接它們
                return np.concatenate(feature_arrays, axis=0)
            
            raise ValueError(f"NPZ文件 {file_path} 中找不到有效的特徵數據")
            
        except Exception as e:
            raise ValueError(f"無法加載NPZ文件 {file_path}: {str(e)}")
    
    def _load_npy_feature(self, file_path: str) -> np.ndarray:
        """加載NPY格式的特徵文件
        
        Args:
            file_path: NPY文件路徑
            
        Returns:
            np.ndarray: 特徵數組
        """
        try:
            # 直接使用numpy.load加載.npy文件
            features = np.load(file_path)
            
            # 確保特徵是一維數組
            if features.ndim > 1:
                # 記錄原始形狀以便日誌
                original_shape = features.shape
                features = features.flatten()
                logger.debug(f"將特徵從形狀 {original_shape} 展平為 {features.shape}")
            
            return features
            
        except Exception as e:
            raise ValueError(f"無法加載NPY文件 {file_path}: {str(e)}")
    
    def _load_json_feature(self, file_path: str) -> np.ndarray:
        """加載JSON格式的特徵文件
        
        Args:
            file_path: JSON文件路徑
            
        Returns:
            np.ndarray: 特徵數組
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                return np.array(list(data.values()))
            elif isinstance(data, list) and len(data) > 0:
                return np.array(data)
            else:
                raise ValueError(f"不支持的JSON格式: {type(data)}")
            
        except Exception as e:
            raise ValueError(f"無法加載JSON文件 {file_path}: {str(e)}")
    
    def _collect_samples(self) -> List[Dict[str, Any]]:
        """收集所有樣本
        
        Returns:
            List[Dict[str, Any]]: 樣本列表
        """
        samples = []
        
        # 日誌配置信息
        logger.info(f"開始收集樣本，資料目錄: {self.data_path}")
        logger.info(f"特徵類型: {self.feature_type}")
        logger.info(f"特徵擴展名: {self.feature_extension}")
        
        # 檢查目錄是否存在和可訪問
        try:
            if not os.path.exists(self.data_path):
                logger.error(f"數據目錄不存在: {self.data_path}")
                return samples
            elif not os.path.isdir(self.data_path):
                logger.error(f"路徑不是目錄: {self.data_path}")
                return samples
            else:
                # 列出目錄內容
                dir_content = os.listdir(self.data_path)
                logger.info(f"目錄 {self.data_path} 中有 {len(dir_content)} 個項目")
        except Exception as e:
            logger.error(f"訪問目錄 {self.data_path} 出錯: {str(e)}")
            return samples
        
        # 如果是auto模式，嘗試找到所有類型的文件
        if self.feature_type == 'auto':
            # 嘗試所有支持的格式
            logger.info("使用自動類型檢測模式")
            for ext in ['json', 'npz', 'npy', 'csv']:
                file_pattern = f"**/*.{ext}"
                files = list(self.data_path.glob(file_pattern))
                if files:
                    logger.info(f"找到 {len(files)} 個 {ext} 文件，使用 {ext} 作為特徵類型")
                    self.feature_type = ext
                    break
            
            if self.feature_type == 'auto':
                logger.error(f"沒有在目錄 {self.data_path} 中找到支持的特徵文件 (json, npz, npy, csv)")
                # 查找所有文件和子目錄
                all_files = list(self.data_path.glob("**/*"))
                logger.info(f"目錄中所有項目 ({len(all_files)} 個):")
                for file in all_files:
                    logger.info(f"- {file} ({os.path.getsize(file) if os.path.isfile(file) else '目錄'})")
                return samples
        
        # 讀取正確的特徵擴展名
        feature_extension = self.feature_extension
        
        # 特定於任務類型的映射
        task_name_mapping = {
            'NoMovement': ['無動作', 'nomove', 'nomovement'],
            'DrySwallow': ['乾吞嚥', 'dryswallow', 'dry'],
            'Cracker': ['餅乾', 'cracker'],
            'Jelly': ['果凍', 'jelly'],
            'WaterDrinking': ['吞水', 'water', 'drinking']
        }
        
        # 根據特徵類型收集樣本
        if self.feature_type == 'json' or (feature_extension and feature_extension.lower() == 'json'):
            # 查找所有JSON特徵文件
            feature_files = list(self.data_path.glob("**/*.json"))
            logger.info(f"找到 {len(feature_files)} 個JSON特徵文件")
            
            # 輸出所有找到的文件路徑以便檢查
            for i, file_path in enumerate(feature_files):
                logger.info(f"JSON文件 {i+1}: {file_path}")
            
            for feature_file in feature_files:
                try:
                    logger.info(f"嘗試讀取JSON文件: {feature_file}")
                    with open(feature_file, 'r') as f:
                        data = json.load(f)
                    
                    logger.info(f"成功讀取JSON文件: {feature_file}, 數據類型: {type(data)}")
                    if isinstance(data, dict):
                        logger.info(f"字典結構數據, 鍵: {list(data.keys())}")
                    elif isinstance(data, list) and len(data) > 0:
                        logger.info(f"列表結構數據, 長度: {len(data)}")
                        if isinstance(data[0], dict):
                            logger.info(f"列表的第一個元素是字典, 鍵: {list(data[0].keys())}")
                    
                    # 檢查是否是單個樣本的特徵文件
                    if isinstance(data, dict) and self.patient_id_column in data:
                        # 單個樣本情況
                        sample = {
                            'feature_path': str(feature_file),
                            'patient_id': data.get(self.patient_id_column),
                            'score': data.get(self.label_name, -1)
                        }
                        samples.append(sample)
                        logger.info(f"從 {feature_file} 加載單樣本: ID={sample['patient_id']}, 分數={sample['score']}")
                    elif isinstance(data, list):
                        # 多個樣本情況
                        for item in data:
                            if isinstance(item, dict) and self.patient_id_column in item:
                                sample = {
                                    'feature_path': str(feature_file),
                                    'index_in_file': data.index(item),
                                    'patient_id': item.get(self.patient_id_column),
                                    'score': item.get(self.label_name, -1)
                                }
                                samples.append(sample)
                                logger.info(f"從 {feature_file} 加載多樣本之一: ID={sample['patient_id']}, 分數={sample['score']}")
                except Exception as e:
                    logger.error(f"讀取 {feature_file} 失敗: {str(e)}，跳過")
                    
        elif self.feature_type == 'csv' or (feature_extension and feature_extension.lower() == 'csv'):
            # 查找所有CSV特徵文件
            feature_files = list(self.data_path.glob("**/*.csv"))
            logger.info(f"找到 {len(feature_files)} 個CSV特徵文件")
            
            # 輸出所有找到的文件路徑以便檢查
            for i, file_path in enumerate(feature_files):
                logger.info(f"CSV文件 {i+1}: {file_path}")
            
            for feature_file in feature_files:
                try:
                    logger.info(f"嘗試讀取CSV文件: {feature_file}")
                    # 檢查CSV是否包含多個樣本或單個樣本
                    df = pd.read_csv(feature_file)
                    logger.info(f"CSV文件 {feature_file} 列名: {list(df.columns)}")
                    
                    if self.patient_id_column in df.columns:
                        # 多個樣本的CSV
                        for idx, row in df.iterrows():
                            sample = {
                                'feature_path': str(feature_file),
                                'index_in_file': idx,
                                'patient_id': row[self.patient_id_column],
                                'score': row.get(self.label_name, -1)
                            }
                            samples.append(sample)
                            logger.info(f"從 {feature_file} 加載CSV樣本: ID={sample['patient_id']}, 分數={sample['score']}")
                    else:
                        # 假設這是單個患者的特徵文件
                        patient_id = feature_file.stem.split('_')[0]  # 從文件名推測患者ID
                        sample = {
                            'feature_path': str(feature_file),
                            'patient_id': patient_id,
                            'score': -1  # 需要從其他地方獲取分數
                        }
                        samples.append(sample)
                        logger.info(f"從 {feature_file} 加載單CSV樣本: ID={sample['patient_id']}, 分數={sample['score']}")
                except Exception as e:
                    logger.error(f"讀取 {feature_file} 失敗: {str(e)}，跳過")
                    
        elif self.feature_type == 'npz' or (feature_extension and feature_extension.lower() == 'npz'):
            # 查找所有NPZ特徵文件
            feature_files = list(self.data_path.glob("**/*.npz"))
            logger.info(f"找到 {len(feature_files)} 個NPZ特徵文件")
            
            # 輸出所有找到的文件路徑以便檢查
            for i, file_path in enumerate(feature_files):
                logger.info(f"NPZ文件 {i+1}: {file_path}")
            
            for feature_file in feature_files:
                try:
                    logger.info(f"嘗試讀取NPZ文件: {feature_file}")
                    data = np.load(feature_file, allow_pickle=True)
                    logger.info(f"NPZ文件 {feature_file} 包含欄位: {list(data.keys())}")
                    
                    # 從文件路徑提取患者ID
                    # 正確從路徑中提取患者ID和任務類型
                    file_path_str = str(feature_file)
                    
                    # 提取患者ID (例如 P001, N021 等)
                    patient_id_match = None
                    for part in file_path_str.split('/'):
                        if part.startswith('P') or part.startswith('N'):
                            # 檢查是否匹配格式 P\d+ 或 N\d+
                            import re
                            match = re.match(r'([NP]\d+).*', part)
                            if match:
                                patient_id_match = match.group(1)
                                break
                    
                    # 如果沒有找到，嘗試從目錄名稱獲取
                    if not patient_id_match:
                        dir_name = os.path.dirname(file_path_str)
                        for part in dir_name.split('/'):
                            if part.startswith('P') or part.startswith('N'):
                                match = re.match(r'([NP]\d+).*', part)
                                if match:
                                    patient_id_match = match.group(1)
                                    break
                    
                    patient_id = patient_id_match if patient_id_match else "Unknown"
                    
                    # 從分數數據庫或默認值中獲取分數
                    # 這裡我們使用默認值, 實際應用中可以從外部數據庫獲取
                    score = -1
                    if patient_id.startswith('N'):
                        # 假設所有N開頭的都是正常人
                        score = 0
                    elif patient_id.startswith('P'):
                        # 假設所有P開頭的都是病人，分數大於9
                        score = 10
                    
                    # 確定任務類型 (NoMovement, Cracker, WaterDrinking等)
                    selection_type = None
                    file_path_lower = file_path_str.lower()
                    
                    for task_name, keywords in task_name_mapping.items():
                        for keyword in keywords:
                            if keyword.lower() in file_path_lower:
                                selection_type = task_name
                                logger.info(f"從路徑 '{file_path_str}' 識別到任務類型: {selection_type} (關鍵詞: {keyword})")
                                break
                        if selection_type:
                            break
                    
                    # 如果沒有識別到任務類型，嘗試從檔案內容獲取
                    if selection_type is None and 'selection' in data:
                        selection_data = data['selection']
                        if isinstance(selection_data, str):
                            logger.info(f"從NPZ文件內容中發現選擇類型: {selection_data}")
                            # 將文件中的selection映射到任務類型
                            for task_name, keywords in task_name_mapping.items():
                                if any(keyword.lower() in selection_data.lower() for keyword in keywords):
                                    selection_type = task_name
                                    break
                    
                    # 檢查NPZ文件是否包含必要的字段
                    if self.patient_id_column in data:
                        patient_ids = data[self.patient_id_column]
                        logger.info(f"找到患者ID欄位: {self.patient_id_column}, 值類型: {type(patient_ids)}, 形狀: {patient_ids.shape if hasattr(patient_ids, 'shape') else '未知'}")
                        
                        if self.label_name in data:
                            scores = data[self.label_name]
                            logger.info(f"找到標籤欄位: {self.label_name}, 值類型: {type(scores)}, 形狀: {scores.shape if hasattr(scores, 'shape') else '未知'}")
                        else:
                            scores = [-1] * len(patient_ids)
                            logger.warning(f"NPZ文件 {feature_file} 不包含標籤欄位 {self.label_name}，使用預設值 -1")
                            
                        # 創建樣本
                        for i, (file_patient_id, file_score) in enumerate(zip(patient_ids, scores)):
                            sample = {
                                'feature_path': str(feature_file),
                                'index_in_file': i,
                                'patient_id': file_patient_id,
                                'score': file_score,
                                'selection_type': selection_type
                            }
                            samples.append(sample)
                            logger.info(f"從 {feature_file} 加載NPZ樣本: ID={file_patient_id}, 分數={file_score}, 選擇類型={selection_type}")
                    else:
                        # 單個患者的特徵文件，使用從文件路徑推測的ID
                        sample = {
                            'feature_path': str(feature_file),
                            'patient_id': patient_id,
                            'score': score,
                            'selection_type': selection_type
                        }
                        samples.append(sample)
                        logger.info(f"從 {feature_file} 加載單NPZ樣本: ID={patient_id}, 分數={score}, 選擇類型={selection_type}")
                except Exception as e:
                    logger.error(f"讀取 {feature_file} 失敗: {str(e)}，跳過")
                    
        elif self.feature_type == 'npy' or (feature_extension and feature_extension.lower() == 'npy'):
            # 查找所有NPY特徵文件
            feature_files = list(self.data_path.glob("**/*.npy"))
            logger.info(f"找到 {len(feature_files)} 個NPY特徵文件")
            
            # 輸出所有找到的文件路徑以便檢查
            for i, file_path in enumerate(feature_files):
                logger.info(f"NPY文件 {i+1}: {file_path}")
            
            for feature_file in feature_files:
                try:
                    logger.info(f"嘗試讀取NPY文件: {feature_file}")
                    
                    # 從文件路徑提取患者ID
                    file_path_str = str(feature_file)
                    
                    # 提取患者ID (例如 P001, N021 等)
                    patient_id_match = None
                    for part in file_path_str.split('/'):
                        if part.startswith('P') or part.startswith('N'):
                            # 檢查是否匹配格式 P\d+ 或 N\d+
                            import re
                            match = re.match(r'([NP]\d+).*', part)
                            if match:
                                patient_id_match = match.group(1)
                                break
            
                    # 如果沒有找到，嘗試從目錄名稱獲取
                    if not patient_id_match:
                        dir_name = os.path.dirname(file_path_str)
                        for part in dir_name.split('/'):
                            if part.startswith('P') or part.startswith('N'):
                                match = re.match(r'([NP]\d+).*', part)
                                if match:
                                    patient_id_match = match.group(1)
                                    break
            
                    patient_id = patient_id_match if patient_id_match else "Unknown"
                    
                    # 從分數數據庫或默認值中獲取分數
                    # 這裡我們使用默認值, 實際應用中可以從外部數據庫獲取
                    score = -1
                    if patient_id.startswith('N'):
                        # 假設所有N開頭的都是正常人
                        score = 0
                    elif patient_id.startswith('P'):
                        # 假設所有P開頭的都是病人，分數大於9
                        score = 10
                    
                    # 確定任務類型 (NoMovement, Cracker, WaterDrinking等)
                    selection_type = None
                    file_path_lower = file_path_str.lower()
                    
                    for task_name, keywords in task_name_mapping.items():
                        for keyword in keywords:
                            if keyword.lower() in file_path_lower:
                                selection_type = task_name
                                logger.info(f"從路徑 '{file_path_str}' 識別到任務類型: {selection_type} (關鍵詞: {keyword})")
                                break
                        if selection_type:
                            break
                    
                    # 創建樣本
                    sample = {
                        'feature_path': str(feature_file),
                        'patient_id': patient_id,
                        'score': score,
                        'selection_type': selection_type
                    }
                    samples.append(sample)
                    logger.info(f"從 {feature_file} 加載NPY樣本: ID={patient_id}, 分數={score}, 選擇類型={selection_type}")
                    
                except Exception as e:
                    logger.error(f"讀取 {feature_file} 失敗: {str(e)}，跳過")
                    
        else:
            logger.error(f"不支持的特徵類型: {self.feature_type}")
            
        # 檢查目錄是否存在
        if len(samples) == 0:
            logger.error(f"數據目錄 {self.data_path} 中沒有找到有效的 {self.feature_type} 特徵文件。")
            # 檢查目錄中的所有文件
            all_files = list(self.data_path.glob("**/*"))
            logger.info(f"目錄中的所有文件 ({len(all_files)} 個):")
            for file in all_files:
                if os.path.isfile(file):
                    logger.info(f"- {file} ({os.path.getsize(file)} 字節)")
                else:
                    logger.info(f"- {file} (目錄)")
            
        # 輸出收集到的原始樣本數
        logger.info(f"收集到 {len(samples)} 個原始樣本，開始過濾...")
            
        # 從配置中獲取過濾設定
        filtering_config = self.config.get('data', {}).get('filtering', {})
        task_type = filtering_config.get('task_type', '{}')
        class_config = filtering_config.get('class_config', {})
        
        # 定義空的標籤映射（如果需要的話）
        LABEL_TO_INDEX = {}
        
        # 只在分類任務中使用的配置
        if task_type == 'classification':
            score_thresholds = filtering_config.get('score_thresholds', {})
            subject_source = filtering_config.get('subject_source', {})
            normal_threshold = score_thresholds.get('normal', {})
            patient_threshold = score_thresholds.get('patient', {})
            logger.info(f"分類任務: 正常人閾值 <= {normal_threshold}，患者閾值 >= {patient_threshold}")
            logger.info(f"來源設定: 正常人組: {subject_source.get('normal', {})}，患者組: {subject_source.get('patient', {})}")
            
            # 根據配置生成標籤映射
            selection_types = [k for k, v in class_config.items() if v == 1]
            logger.info(f"啟用的選擇類型: {selection_types}")
            
            # 創建標籤映射表
            label_idx = 0
            for group in ['Normal', 'Patient']:
                for selection in selection_types:
                    class_name = f"{group}-{selection}"
                    LABEL_TO_INDEX[class_name] = label_idx
                    label_idx += 1
            
            logger.info(f"標籤映射表: {LABEL_TO_INDEX}")
        
        # 過濾樣本
        filtered_samples = []
        skipped_samples = 0
        
        for sample in samples:
            score = sample['score']
            patient_id = sample['patient_id']
            sample_info = f"樣本 ID={patient_id}, 分數={score}, 檔案={sample['feature_path']}"
            
            if task_type == 'classification':
                # 分類任務特有的過濾邏輯
                patient_group = None
                selection_type = sample.get('selection_type')
                
                # 如果樣本中已有selection_type，使用該值
                if selection_type is None:
                    # 從文件路徑中獲取選擇類型
                    file_path = sample['feature_path']
                    file_path_lower = file_path.lower()
                    
                    for task_name, keywords in task_name_mapping.items():
                        for keyword in keywords:
                            if keyword.lower() in file_path_lower:
                                selection_type = task_name
                                break
                        if selection_type:
                            break
                
                if selection_type is None:
                    logger.info(f"{sample_info} - 跳過: 無法從文件路徑識別有效的選擇類型")
                    skipped_samples += 1
                    continue
                
                if class_config.get(selection_type, 0) != 1:
                    logger.info(f"{sample_info} - 跳過: 選擇類型 {selection_type} 已禁用")
                    skipped_samples += 1
                    continue
                
                # 基於分數判斷病人組別
                if score <= normal_threshold:
                    patient_group = 'normal'
                elif score >= patient_threshold:
                    patient_group = 'patient'
                
                if patient_group is None:
                    logger.info(f"{sample_info} - 跳過: 分數 {score} 不在正常人 (<={normal_threshold}) 或患者 (>={patient_threshold}) 範圍內")
                    skipped_samples += 1
                    continue
                
                # 檢查ID前綴
                id_prefix = patient_id[0].upper() if patient_id else ''
                if id_prefix in ['N', 'P']:
                    include_config = subject_source.get(patient_group, {})
                    if id_prefix == 'N' and include_config.get('include_N', 1) != 1:
                        logger.info(f"{sample_info} - 跳過: N前綴{patient_group}已禁用")
                        skipped_samples += 1
                        continue
                    if id_prefix == 'P' and include_config.get('include_P', 1) != 1:
                        logger.info(f"{sample_info} - 跳過: P前綴{patient_group}已禁用")
                        skipped_samples += 1
                        continue
                
                # 生成分類標籤
                class_name = f"{patient_group.capitalize()}-{selection_type}"
                class_idx = LABEL_TO_INDEX.get(class_name)
                
                if class_idx is None or class_idx >= 10:
                    logger.info(f"{sample_info} - 跳過: 無效的類別標籤: {class_name}")
                    skipped_samples += 1
                    continue
                    
                label = torch.tensor(class_idx, dtype=torch.float32)
                sample['label'] = label
                sample['patient_group'] = patient_group
                
                logger.info(f"{sample_info} - 接受: 類別={class_name}, 標籤索引={class_idx}")
            else:
                # 回歸任務：直接使用分數作為標籤
                if score < 0 or score > 40:
                    logger.info(f"{sample_info} - 跳過: 分數 {score} 超出有效範圍 [0, 40]")
                    skipped_samples += 1
                    continue
                
                label = torch.tensor(score, dtype=torch.float32)
                sample['label'] = label
                logger.info(f"{sample_info} - 接受: 回歸任務，標籤={score}")
            
            filtered_samples.append(sample)
        
        logger.info(f"過濾完成: 接受 {len(filtered_samples)} 個樣本，跳過 {skipped_samples} 個樣本")
        
        return filtered_samples
        
    def _init_scaler(self):
        """初始化特徵標準化器
        
        在訓練模式下初始化標準化器，用於對特徵進行標準化處理
        """
        logger.info("初始化特徵標準化器...")
        
        # 在索引模式下，我們暫時不做標準化
        if self.use_index_mode and not self.is_direct_mode:
            logger.warning("索引模式下暫不支持特徵標準化，跳過初始化標準化器")
            return
            
        # 檢查是否有樣本
        if not hasattr(self, 'samples') or not self.samples:
            logger.warning("沒有樣本可用於初始化標準化器")
            return
            
        try:
            # 收集特徵維度統計
            dimensions = {}
            feature_samples = []
            max_samples = min(len(self.samples), 100)  # 最多使用100個樣本初始化
            
            for i in range(min(max_samples, len(self.samples))):
                try:
                    features = self._load_features(i)
                    dim = len(features)
                    dimensions[dim] = dimensions.get(dim, 0) + 1
                    feature_samples.append(features)
                except Exception as e:
                    logger.warning(f"加載特徵失敗 (樣本 {i}): {str(e)}")
                    continue
            
            if not feature_samples:
                logger.warning("無法加載任何特徵樣本，跳過初始化標準化器")
                return
                
            # 使用最常見的維度
            most_common_dim = max(dimensions, key=dimensions.get)
            logger.info(f"使用最常見的特徵維度 {most_common_dim} 初始化標準化器")
            
            # 只使用具有最常見維度的樣本
            valid_samples = [sample for sample in feature_samples if len(sample) == most_common_dim]
            
            # 初始化並訓練標準化器
            self.scaler = StandardScaler()
            self.scaler.fit(valid_samples)
            
            logger.info("標準化器初始化完成")
            
        except Exception as e:
            logger.error(f"初始化標準化器失敗: {str(e)}")
            self.scaler = None
    
    def _load_features(self, idx: int) -> np.ndarray:
        """根據樣本索引加載特徵
        
        Args:
            idx (int): 樣本索引
            
        Returns:
            np.ndarray: 特徵向量
        """
        # 獲取樣本
        if not hasattr(self, 'samples') or len(self.samples) <= idx:
            logger.error(f"無法加載索引 {idx} 的特徵，samples不存在或索引超出範圍")
            return np.zeros(10)  # 返回默認值

        sample = self.samples[idx]
        
        # 索引模式使用load_data方法
        if not self.is_direct_mode:
            try:
                # 從load_data方法獲取特徵張量
                features_tensor = self.load_data(sample)
                
                # 確保張量是一維的
                if features_tensor.dim() > 1:
                    features_tensor = features_tensor.flatten()
                
                # 轉換為numpy數組並返回
                return features_tensor.numpy()
            except Exception as e:
                logger.error(f"索引模式下加載特徵失敗，錯誤: {str(e)}")
                return np.zeros(self._infer_feature_dim())  # 返回默認值
        
        # 直接模式
        feature_path = sample.get('feature_path')
        
        if not feature_path or not os.path.exists(feature_path):
            logger.warning(f"特徵文件 {feature_path} 不存在")
            return np.zeros(10)  # 返回默認值
            
        # 根據文件類型選擇適當的加載方法
        ext = os.path.splitext(feature_path)[1].lower()
        
        try:
            if ext == '.npz':
                return self._load_npz_feature(feature_path)
            elif ext == '.npy':
                return self._load_npy_feature(feature_path)
            elif ext == '.json':
                return self._load_json_feature(feature_path)
            elif ext == '.csv':
                return self._load_csv_feature(feature_path)
            else:
                logger.warning(f"未知的特徵文件擴展名: {ext}")
                return np.zeros(10)  # 返回默認值
                
        except Exception as e:
            raise ValueError(f"無法加載JSON文件 {file_path}: {str(e)}")
    
    def _load_csv_feature(self, file_path: str) -> np.ndarray:
        """加載CSV格式的特徵文件
        
        Args:
            file_path: CSV文件路徑
            
        Returns:
            np.ndarray: 特徵數組
        """
        try:
            df = pd.read_csv(file_path)
            
            # 優先使用配置中指定的特徵列
            if self.feature_names:
                # 檢查所有指定的列是否存在
                valid_names = [name for name in self.feature_names if name in df.columns]
                if valid_names:
                    return df[valid_names].values.flatten()
            
            # 否則使用除了ID和標籤以外的所有列
            exclude_cols = [self.patient_id_column, self.label_name]
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            if feature_cols:
                return df[feature_cols].values.flatten()
            
            raise ValueError(f"CSV文件 {file_path} 中找不到有效的特徵列")
            
        except Exception as e:
            raise ValueError(f"無法加載CSV文件 {file_path}: {str(e)}")
    
    def _scan_max_feature_length(self) -> int:
        """掃描數據集中所有特徵的最大長度
        
        在使用置中填充模式時，需要提前知道最大特徵長度，以便正確填充
        
        Returns:
            int: 數據集中特徵向量的最大長度
        """
        max_length = 0
        scanned_count = 0
        error_count = 0
        
        logger.info(f"開始掃描特徵最大長度，共有 {len(self.samples)} 個樣本")
        
        # 限制掃描的最大樣本數，避免過長時間
        max_scan_samples = min(len(self.samples), 1000)  # 最多掃描1000個樣本
        
        for i, sample in enumerate(self.samples[:max_scan_samples]):
            try:
                # 使用features_path欄位（如果有）
                if 'features_path' in sample and os.path.exists(sample['features_path']):
                    features = self._load_feature_by_path_raw(sample['features_path'])
                    if features is not None:
                        feature_length = len(features)
                        max_length = max(max_length, feature_length)
                        scanned_count += 1
                
                # 或嘗試加載file_path中的特徵文件
                elif 'file_path' in sample:
                    file_path = sample['file_path']
                    if os.path.isdir(file_path):
                        # 查找特徵文件
                        dir_name = os.path.basename(file_path)
                        feature_path = os.path.join(file_path, f"{dir_name}_features.npy")
                        
                        if os.path.exists(feature_path):
                            features = self._load_feature_by_path_raw(feature_path)
                            if features is not None:
                                feature_length = len(features)
                                max_length = max(max_length, feature_length)
                                scanned_count += 1
                
                # 每100個樣本打印一次進度
                if (i+1) % 100 == 0:
                    logger.info(f"已掃描 {i+1}/{len(self.samples[:max_scan_samples])} 個樣本，當前最大長度: {max_length}")
            
            except Exception as e:
                error_count += 1
                if error_count <= 5:  # 只打印前5個錯誤
                    logger.warning(f"掃描特徵長度時出錯: {str(e)}")
                continue
        
        # 如果沒有成功掃描到特徵，使用配置中的max_feature_dim
        if max_length == 0:
            max_length = self.max_feature_dim
            logger.warning(f"未能掃描到任何有效特徵，使用配置中的max_feature_dim: {max_length}")
        
        logger.info(f"特徵長度掃描完成，成功掃描 {scanned_count} 個樣本，最大長度: {max_length}")
        return max_length
    
    def _load_feature_by_path_raw(self, file_path: str) -> np.ndarray:
        """原始加載特徵文件，不進行任何裁剪或填充
        
        用於掃描特徵長度時加載原始特徵
        
        Args:
            file_path: 特徵文件路徑
            
        Returns:
            np.ndarray: 原始特徵數組
        """
        # 根據擴展名判斷文件類型
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext == '.npz':
                data = np.load(file_path, allow_pickle=True)
                if 'features' in data:
                    return data['features']
                return next((data[k] for k in data.keys() if k != 'metadata'), np.array([]))
                
            elif ext == '.npy':
                features = np.load(file_path)
                if features.ndim > 1:
                    features = features.flatten()
                return features
                
            elif ext == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return np.array(list(data.values()))
                elif isinstance(data, list) and len(data) > 0:
                    return np.array(data)
                return np.array([])
                
            elif ext == '.csv':
                df = pd.read_csv(file_path)
                exclude_cols = [self.patient_id_column, self.label_name]
                feature_cols = [col for col in df.columns if col not in exclude_cols]
                if feature_cols:
                    return df[feature_cols].values.flatten()
                return np.array([])
            
            else:
                return np.array([])
        
        except Exception as e:
            logger.debug(f"加載原始特徵失敗 {file_path}: {str(e)}")
            return np.array([])
    
    def _center_pad_features(self, features: np.ndarray, target_length: int) -> np.ndarray:
        """將特徵向量置中並填充至目標長度
        
        Args:
            features: 特徵向量 (numpy array)
            target_length: 目標長度
            
        Returns:
            numpy array: 置中填充後的特徵向量
        """
        current_length = len(features)
        
        # 如果當前長度已達到或超過目標長度，則居中截斷
        if current_length >= target_length:
            start = (current_length - target_length) // 2
            return features[start:start+target_length]
        
        # 計算需要填充的總長度
        padding = target_length - current_length
        
        # 計算左右填充量
        left_pad = padding // 2
        right_pad = padding - left_pad
        
        # 創建結果數組並填充
        result = np.zeros(target_length, dtype=features.dtype)
        result[left_pad:left_pad+current_length] = features
        
        return result
    
    def _init_pca(self):
        """初始化PCA模型
        
        在訓練模式下收集樣本並訓練PCA模型，用於降維
        """
        logger.info("初始化PCA模型...")
        
        # 檢查是否有樣本
        if not hasattr(self, 'samples') or not self.samples:
            logger.warning("沒有樣本可用於初始化PCA模型")
            return
        
        try:
            # 收集特徵向量用於訓練PCA
            features_list = []
            max_samples = min(len(self.samples), 500)  # 增加訓練樣本數量，限制在500個
            processed_count = 0
            
            logger.info(f"收集訓練PCA的樣本數據 (最多{max_samples}個樣本)...")
            
            # 首先掃描特徵長度，確保能夠對齊
            max_feature_length = 0
            raw_features = []
            feature_paths = []
            
            # 掃描最大特徵長度
            for i in range(min(max_samples, len(self.samples))):
                try:
                    # 加載原始特徵（不經過PCA處理）
                    feature_path = None
                    if 'features_path' in self.samples[i] and os.path.exists(self.samples[i]['features_path']):
                        feature_path = self.samples[i]['features_path']
                    elif 'file_path' in self.samples[i]:
                        file_path = self.samples[i]['file_path']
                        if os.path.isdir(file_path):
                            # 尋找特徵文件
                            dir_name = os.path.basename(file_path)
                            possible_feature_path = os.path.join(file_path, f"{dir_name}_features.npy")
                            if os.path.exists(possible_feature_path):
                                feature_path = possible_feature_path
                    
                    if feature_path:
                        # 加載原始特徵
                        features = self._load_feature_by_path_raw(feature_path)
                        if features is not None and len(features) > 0:
                            # 記錄原始特徵和路徑
                            raw_features.append(features)
                            feature_paths.append(feature_path)
                            max_feature_length = max(max_feature_length, len(features))
                            processed_count += 1
                            
                            # 每50個樣本打印一次進度
                            if processed_count % 50 == 0:
                                logger.info(f"已掃描 {processed_count}/{max_samples} 個樣本")
                
                except Exception as e:
                    logger.warning(f"掃描樣本 {i} 特徵長度時出錯: {str(e)}")
                    continue
            
            if not raw_features:
                logger.warning("無法加載任何特徵樣本，跳過初始化PCA模型")
                return
                
            logger.info(f"掃描完成，共找到 {len(raw_features)} 個有效樣本，最大特徵長度: {max_feature_length}")
            
            # 對齊所有特徵長度
            aligned_features = []
            for i, feat in enumerate(raw_features):
                if self.padding_mode == 'center':
                    # 使用置中填充
                    aligned_feat = self._center_pad_features(feat, max_feature_length)
                else:
                    # 使用右側填充
                    aligned_feat = np.zeros(max_feature_length)
                    aligned_feat[:len(feat)] = feat
                
                aligned_features.append(aligned_feat)
                
                # 每50個樣本打印一次進度
                if (i+1) % 50 == 0:
                    logger.info(f"已對齊 {i+1}/{len(raw_features)} 個樣本")
            
            # 將特徵列表轉換為二維數組，每行一個樣本
            features_array = np.vstack(aligned_features)
            logger.info(f"成功收集並對齊了 {features_array.shape[0]} 個樣本，每個樣本維度為 {features_array.shape[1]}")
            
            # 調整目標維度，確保不超過樣本數量
            n_samples, n_features = features_array.shape
            max_components = min(n_samples, n_features)
            target_dim = min(self.target_dim, max_components)
            
            if target_dim < self.target_dim:
                logger.warning(f"由於樣本數量限制，PCA目標維度從 {self.target_dim} 調整為 {target_dim}")
            
            # 初始化並訓練PCA模型
            self.pca = PCA(n_components=target_dim)
            self.pca.fit(features_array)
            
            # 計算解釋方差
            explained_variance = sum(self.pca.explained_variance_ratio_) * 100
            logger.info(f"PCA模型初始化完成，{target_dim}個主成分解釋了{explained_variance:.2f}%的方差")
            
        except Exception as e:
            logger.error(f"初始化PCA模型失敗: {str(e)}")
            self.pca = None
    
    def _apply_pca(self, features):
        """應用PCA降維到特徵向量
        
        Args:
            features: 原始特徵向量
            
        Returns:
            np.ndarray: 降維後的特徵向量
        """
        if self.pca is None:
            logger.warning("PCA模型未初始化，跳過降維")
            # 如果PCA未初始化，則使用截斷方式處理
            return features[:self.target_dim] if len(features) > self.target_dim else features
        
        try:
            # 需要確保特徵長度與訓練PCA時一致
            feature_length = len(features)
            expected_length = self.pca.n_features_in_
            
            # 如果特徵長度不符合PCA期望長度，首先對齊特徵長度
            if feature_length != expected_length:
                logger.debug(f"特徵長度 ({feature_length}) 與PCA期望長度 ({expected_length}) 不一致，進行對齊")
                
                if self.padding_mode == 'center':
                    # 使用置中填充
                    features = self._center_pad_features(features, expected_length)
                else:
                    # 使用右側填充
                    aligned_features = np.zeros(expected_length)
                    aligned_features[:feature_length] = features
                    features = aligned_features
            
            # 確保特徵是二維的 (PCA需要的格式)
            features_reshaped = features.reshape(1, -1)
            
            # 應用PCA轉換
            transformed = self.pca.transform(features_reshaped)
            
            # 返回一維數組
            return transformed.flatten()
            
        except Exception as e:
            logger.error(f"應用PCA降維失敗: {str(e)}")
            # 失敗時使用截斷作為備選方案
            return features[:self.target_dim] if len(features) > self.target_dim else features

# 中文註解：這是feature_dataset.py的Minimal Executable Unit，檢查能否正確初始化與錯誤路徑時的優雅報錯
if __name__ == "__main__":
    """
    Description: Minimal Executable Unit for feature_dataset.py，測試特徵數據集的索引模式和直接模式。
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
    
    # 創建測試索引CSV
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as temp_file:
        test_data = pd.DataFrame({
            'file_path': ['/path/to/feature1.npz', '/path/to/feature2.npz', '/path/to/feature3.npz'],
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
        # 測試索引模式
        print("測試1: 索引模式 (預期會創建空數據集)")
        config = {
            'data': {
                'preprocessing': {
                    'features': {
                        'normalize': False  # 關閉標準化，避免嘗試初始化標準化器
                    }
                }
            }
        }
        
        dataset = FeatureDataset(
            index_path=test_csv_path,
            label_field="score",
            config=config,
            is_train=False  # 設為False以避免初始化標準化器
        )
        print(f"索引模式數據集大小: {len(dataset)}")
        
        # 測試直接模式配置
        print("\n測試2: 直接模式 (預期會創建空數據集或報錯)")
        try:
            direct_config = {
                'data': {
                    'preprocessing': {
                        'features': {
                            'normalize': False  # 關閉標準化
                        }
                    }
                }
            }
            direct_dataset = FeatureDataset(
                data_path="./",  # 使用當前目錄
                config=direct_config,
                is_train=False  # 設為False以避免初始化標準化器
            )
            print(f"直接模式數據集大小: {len(direct_dataset)}")
        except Exception as e:
            print(f"創建直接模式數據集失敗 (預期行為): {str(e)}")
        
        print("\n測試結束")
        
    except Exception as e:
        print(f"測試失敗: {str(e)}")
    
    finally:
        # 清理測試文件
        if os.path.exists(test_csv_path):
            os.unlink(test_csv_path) 