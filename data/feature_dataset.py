"""
特徵數據集：支持從預處理過的特徵文件中讀取特徵
功能：
1. 從特徵文件（如JSON、CSV、NPZ等）中讀取預處理的特徵
2. 支持多種特徵組合和選擇
3. 支持按患者ID拆分數據集
4. 支持特徵標準化和轉換
5. 支持從索引CSV加載數據

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
        filter_criteria: Optional[Dict] = None
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
        """
        # 決定是否使用索引模式
        self.use_index_mode = index_path is not None and os.path.exists(index_path)
        
        # 初始化共用屬性
        self.is_train = is_train
        self.cache_features = cache_features
        self.feature_cache = {}
        self.scaler = None
        
        if self.use_index_mode:
            # 索引模式初始化
            self.data_path = None
            self.config = config or {}
            
            # 獲取特徵配置
            self.feature_config = self.config.get('data', {}).get('preprocessing', {}).get('features', {})
            self.normalize = self.feature_config.get('normalize', True)
            
            # 呼叫父類初始化
            super().__init__(
                index_path=index_path,
                label_field=label_field,
                transform=transform,
                filter_criteria=filter_criteria,
                fallback_to_direct=True
            )
            
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
            
            # 設置日誌級別
            logging.basicConfig(level=logging.INFO)
            
            # 獲取特徵配置
            self.feature_config = config.get('data', {}).get('preprocessing', {}).get('features', {})
            
            # 獲取feature_type，可能存在不同位置
            data_config = config.get('data', {})
            self.feature_type = data_config.get('type', 'feature')
            
            # 獲取擴展名配置（如果有）
            self.feature_extension = data_config.get('source', {}).get('feature_extension', None)
            
            # 是特徵數據集但沒有指定具體類型，根據擴展名判斷
            if self.feature_type == 'feature' and self.feature_extension:
                self.feature_type = self.feature_extension  # 使用擴展名作為類型
            
            # 確保兼容性
            if self.feature_type not in ['json', 'csv', 'npz']:
                logger.warning(f"未識別的特徵類型 '{self.feature_type}'，使用 feature_extension '{self.feature_extension}' 代替")
                if self.feature_extension in ['json', 'csv', 'npz']:
                    self.feature_type = self.feature_extension
                else:
                    logger.warning(f"無法確定特徵類型，將嘗試依次搜索 json, csv, npz 格式的文件")
                    self.feature_type = 'auto'  # 自動檢測
                
            self.feature_names = self.feature_config.get('names', [])  # 要使用的特徵列表
            self.label_name = self.feature_config.get('label', 'score')  # 標籤列名
            self.patient_id_column = self.feature_config.get('patient_id_column', 'patient_id')  # 患者ID列名
            
            # 特徵標準化設置
            self.normalize = self.feature_config.get('normalize', True)
            
            # 收集樣本
            self.samples = self._collect_samples()
            
            logger.info(f"加載了 {len(self.samples)} 個特徵樣本")
            
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
        file_path = data_row['file_path']
        
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
            # 嘗試獲取一個合理的特徵維度
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
        
        Returns:
            int: 特徵維度，默認為10
        """
        # 如果有樣本，嘗試加載第一個樣本獲取維度
        if hasattr(self, 'samples') and self.samples:
            try:
                features = self._load_features(0)
                return len(features)
            except:
                pass
        
        return 10  # 默認值
    
    def _load_feature_by_path(self, file_path: str) -> np.ndarray:
        """根據文件路徑加載特徵
        
        Args:
            file_path: 特徵文件路徑
            
        Returns:
            np.ndarray: 特徵數組
        """
        # 根據擴展名判斷文件類型
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.npz':
            return self._load_npz_feature(file_path)
        elif ext == '.json':
            return self._load_json_feature(file_path)
        elif ext == '.csv':
            return self._load_csv_feature(file_path)
        else:
            raise ValueError(f"不支持的特徵文件類型: {ext}")
    
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
            for ext in ['json', 'npz', 'csv']:
                file_pattern = f"**/*.{ext}"
                files = list(self.data_path.glob(file_pattern))
                if files:
                    logger.info(f"找到 {len(files)} 個 {ext} 文件，使用 {ext} 作為特徵類型")
                    self.feature_type = ext
                    break
            
            if self.feature_type == 'auto':
                logger.error(f"沒有在目錄 {self.data_path} 中找到支持的特徵文件 (json, npz, csv)")
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
        """加載特徵數據
        
        加載步驟：
        1. 獲取樣本的特徵文件路徑
        2. 根據文件類型選擇對應的加載方法:
           a. NPZ文件:
              - 使用np.load(file_path, allow_pickle=True)加載
              - 優先從'features'字段提取特徵
              - 如果不存在，則從所有非元數據字段提取
           b. JSON文件:
              - 使用json.load加載
              - 支持單樣本或多樣本格式
           c. CSV文件:
              - 使用pandas.read_csv加載
              - 提取特定行或列
        3. 對特徵數據進行後處理:
           - 確保特徵是一維向量
           - 限制特徵維度(默認10000)，截斷過大的特徵
           - 如果啟用標準化，使用預先訓練的scaler進行標準化
        
        注意事項:
        - 支持緩存機制，如果idx已在緩存中直接返回
        - 處理特徵加載過程中可能出現的異常
        - 標準化可能因特徵維度不一致而失敗，此時返回原始特徵
        
        Args:
            idx (int): 樣本索引
            
        Returns:
            np.ndarray: 特徵數組，如果加載失敗則返回空數組
        """
        sample = self.samples[idx]
        
        # 如果已緩存，則直接返回
        if self.cache_features and idx in self.feature_cache:
            return self.feature_cache[idx]
            
        feature_path = sample['feature_path']
        
        try:
            if self.feature_type == 'json':
                with open(feature_path, 'r') as f:
                    data = json.load(f)
                    
                if 'index_in_file' in sample:
                    # 從集合中提取特定樣本
                    data = data[sample['index_in_file']]
                    
                # 提取特徵
                if self.feature_names:
                    # 僅使用指定的特徵
                    features = np.array([data.get(name, 0.0) for name in self.feature_names])
                else:
                    # 使用所有非元數據特徵
                    metadata_fields = [self.patient_id_column, self.label_name, 'timestamp', 'selection']
                    features = np.array([v for k, v in data.items() 
                                       if k not in metadata_fields and isinstance(v, (int, float))])
                
            elif self.feature_type == 'csv':
                df = pd.read_csv(feature_path)
                
                if 'index_in_file' in sample:
                    # 從DataFrame中提取特定行
                    row = df.iloc[sample['index_in_file']]
                else:
                    # 單個樣本的CSV
                    row = df.iloc[0]
                    
                # 提取特徵
                if self.feature_names:
                    # 僅使用指定的特徵
                    features = np.array([row.get(name, 0.0) for name in self.feature_names])
                else:
                    # 使用所有非元數據列
                    metadata_fields = [self.patient_id_column, self.label_name, 'timestamp', 'selection']
                    features = np.array([v for k, v in row.items() 
                                       if k not in metadata_fields and isinstance(v, (int, float))])
                    
            elif self.feature_type == 'npz':
                data = np.load(feature_path, allow_pickle=True)
                
                if 'index_in_file' in sample:
                    # 多個樣本的NPZ
                    if self.feature_names:
                        # 僅使用指定的特徵
                        features = np.array([data[name][sample['index_in_file']] 
                                           if name in data else 0.0 
                                           for name in self.feature_names])
                    elif 'features' in data:
                        # 使用專門的特徵數組
                        features = data['features'][sample['index_in_file']]
                    else:
                        # 嘗試從所有非元數據字段中提取
                        metadata_fields = [self.patient_id_column, self.label_name, 'timestamp', 'selection']
                        features = np.array([data[k][sample['index_in_file']] 
                                           for k in data.keys() 
                                           if k not in metadata_fields and isinstance(data[k], np.ndarray)])
                else:
                    # 單個樣本的NPZ
                    if self.feature_names:
                        # 僅使用指定的特徵
                        features = np.array([data[name] if name in data else 0.0 for name in self.feature_names])
                    elif 'features' in data:
                        # 使用專門的特徵數組
                        features = data['features']
                    else:
                        # 嘗試從所有非元數據字段中提取
                        metadata_fields = [self.patient_id_column, self.label_name, 'timestamp', 'selection']
                        features = np.array([data[k] for k in data.keys() 
                                           if k not in metadata_fields and isinstance(data[k], np.ndarray)])
            else:
                logger.error(f"不支持的特徵類型: {self.feature_type}")
                return None
                
            # 確保特徵是一維數組
            if features.ndim > 1:
                # 記錄原始形狀以便日誌
                original_shape = features.shape
                features = features.flatten()
                logger.debug(f"將特徵從形狀 {original_shape} 展平為 {features.shape}")
            
            # 對於NPZ文件中的離散代碼等，限制特徵維度以避免過大
            max_feature_dim = self.feature_config.get('max_feature_dim', 10000)
            if len(features) > max_feature_dim:
                logger.warning(f"特徵維度 {len(features)} 超過限制 {max_feature_dim}，進行截斷")
                features = features[:max_feature_dim]
            
            # 標準化特徵
            if self.normalize and self.scaler is not None:
                try:
                    features = self.scaler.transform(features.reshape(1, -1)).flatten()
                except ValueError as e:
                    logger.warning(f"標準化特徵失敗: {str(e)}，使用未標準化的特徵")
            
            # 緩存特徵
            if self.cache_features:
                self.feature_cache[idx] = features
                
            return features
            
        except Exception as e:
            logger.error(f"加載特徵 {feature_path} 失敗: {str(e)}")
            return np.array([])
            
    def __len__(self) -> int:
        """獲取數據集長度
        
        Returns:
            int: 數據集長度
        """
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """獲取指定索引的樣本
        
        Args:
            idx (int): 樣本索引
            
        Returns:
            Dict[str, Any]: 樣本字典，包含 'features', 'label', 'score' 等
        """
        sample = self.samples[idx]
        
        # 加載特徵
        features = self._load_features(idx)
        
        # 將特徵轉換為張量
        if features is None or len(features) == 0:
            # 如果特徵加載失敗，使用零向量代替
            features = np.zeros(len(self.feature_names) if self.feature_names else 10)
            
        features_tensor = torch.FloatTensor(features)
        
        # 獲取標籤
        label = sample.get('label', None)
        score = sample.get('score', -1)
        
        # 創建輸出字典
        output = {
            'features': features_tensor,
            'patient_id': sample['patient_id']
        }
        
        # 確保同時提供 label 和 score，無論是分類還是回歸任務
        if label is not None:
            output['label'] = label
            # 如果沒有明確的分數，也將標籤用作分數
            if score == -1:
                output['score'] = label
            else:
                output['score'] = torch.tensor(score, dtype=torch.float32)
        else:
            # 對於回歸任務，score 被用作標籤
            score_tensor = torch.tensor(score, dtype=torch.float32)
            output['score'] = score_tensor
            output['label'] = score_tensor  # 同時設置為 label，確保兼容性
        
        return output
        
    def split_by_patient(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Tuple[List[int], List[int], List[int]]:
        """按患者ID拆分數據集
        
        Args:
            train_ratio (float, optional): 訓練集比例. 預設為 0.7.
            val_ratio (float, optional): 驗證集比例. 預設為 0.15.
            test_ratio (float, optional): 測試集比例. 預設為 0.15.
            seed (int, optional): 隨機種子. 預設為 42.
            
        Returns:
            Tuple[List[int], List[int], List[int]]: 訓練、驗證和測試樣本的索引
        """
        # 設置隨機種子
        random.seed(seed)
        
        # 獲取唯一的患者ID
        patient_ids = list(set(sample['patient_id'] for sample in self.samples))
        
        # 打亂患者ID
        random.shuffle(patient_ids)
        
        # 計算分割點
        train_size = int(len(patient_ids) * train_ratio)
        val_size = int(len(patient_ids) * val_ratio)
        
        # 分割患者ID
        train_patients = patient_ids[:train_size]
        val_patients = patient_ids[train_size:train_size + val_size]
        test_patients = patient_ids[train_size + val_size:]
        
        # 按患者ID分組樣本索引
        train_indices = []
        val_indices = []
        test_indices = []
        
        for i, sample in enumerate(self.samples):
            patient_id = sample['patient_id']
            
            if patient_id in train_patients:
                train_indices.append(i)
            elif patient_id in val_patients:
                val_indices.append(i)
            elif patient_id in test_patients:
                test_indices.append(i)
                
        return train_indices, val_indices, test_indices
        
    def get_feature_dim(self) -> int:
        """獲取特徵維度
        
        Returns:
            int: 特徵維度
        """
        # 加載第一個樣本以確定特徵維度
        if len(self.samples) > 0:
            features = self._load_features(0)
            if features is not None:
                return len(features)
                
        # 如果沒有樣本或加載失敗，返回默認值
        return len(self.feature_names) if self.feature_names else 0 

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
            
            # 如果是字典，尋找特徵字段
            if isinstance(data, dict):
                # 優先使用配置中指定的特徵名
                if self.feature_names:
                    features = []
                    for name in self.feature_names:
                        if name in data:
                            features.extend(data[name])
                    
                    if features:
                        return np.array(features, dtype=np.float32)
                
                # 否則嘗試找到features字段
                if 'features' in data:
                    features = data['features']
                    if isinstance(features, list):
                        return np.array(features, dtype=np.float32)
                
                # 否則嘗試提取所有數值字段
                features = []
                for key, value in data.items():
                    if isinstance(value, (list, np.ndarray)) and key != 'metadata':
                        features.extend(value)
                
                if features:
                    return np.array(features, dtype=np.float32)
            
            # 如果是列表，直接使用
            elif isinstance(data, list):
                return np.array(data, dtype=np.float32)
            
            raise ValueError(f"JSON文件 {file_path} 中找不到有效的特徵數據")
            
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