"""
頻譜圖數據集：支持讀取和處理頻譜圖數據
功能：
1. 讀取頻譜圖文件
2. 支持數據增強和轉換
3. 支持按患者ID拆分數據集
4. 支持從索引CSV加載數據
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import logging
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from pathlib import Path
import random
from PIL import Image
import torchvision.transforms as transforms
from utils.constants import SELECTION_TYPES, LABEL_TO_INDEX
from utils.patient_info_loader import load_patient_info, list_patient_dirs

# 引入索引數據集基類
from data.indexed_dataset import IndexedDatasetBase

logger = logging.getLogger(__name__)

class SpectrogramDataset(IndexedDatasetBase):
    """頻譜圖數據集類，支持讀取和處理頻譜圖數據
    
    主要功能：
    1. 從PNG文件讀取頻譜圖圖像
    2. 支持各種圖像轉換和數據增強
    3. 支持按患者ID拆分數據集
    4. 支持從索引CSV加載數據
    
    數據讀取模式：
    1. 索引模式：使用data_index.csv加載數據
       - 提供index_path參數啟用此模式
       - 使用label_field指定標籤欄位
       - 使用filter_criteria篩選數據
       
    2. 直接模式：直接從目錄讀取頻譜圖文件
       - 使用root_dir參數指定頻譜圖文件的根目錄
       - 從患者info.json獲取標籤信息
       - 此模式與原始實現兼容
    """
    
    def __init__(
        self,
        root_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        transform: Optional[Dict[str, Any]] = None,
        is_train: bool = True,
        cache_spectrograms: bool = False,
        # 新增索引模式參數
        index_path: Optional[str] = None,
        label_field: str = 'score',
        filter_criteria: Optional[Dict] = None,
        generate_spectrograms: bool = False,
        save_dir: Optional[str] = None
    ):
        """初始化頻譜圖數據集
        
        Args:
            root_dir: 頻譜圖文件的根目錄（直接模式必填）
            config: 配置字典（直接模式必填）
            transform: 數據轉換配置
            is_train: 是否為訓練模式
            cache_spectrograms: 是否緩存頻譜圖
            index_path: 索引CSV文件路徑（索引模式必填）
            label_field: 標籤欄位名稱，可以是'score', 'DrLee_Evaluation', 'DrTai_Evaluation', 'selection'
            filter_criteria: 篩選條件字典
            generate_spectrograms: 是否從音頻生成頻譜圖
            save_dir: 頻譜圖保存目錄
        """
        # 決定是否使用索引模式
        self.use_index_mode = index_path is not None and os.path.exists(index_path)
        
        # 初始化共用屬性
        self.is_train = is_train
        self.cache_spectrograms = cache_spectrograms
        self.image_cache = {}
        self.transform_config = transform or {}
        self.generate_spectrograms = generate_spectrograms
        
        if self.use_index_mode:
            # 索引模式初始化
            self.root_dir = None
            self.config = config or {}
            self.save_dir = save_dir
            
            # 讀取頻譜圖配置
            spec_config = self.config.get('data', {}).get('preprocessing', {}).get('spectrogram', {})
            self.resize = spec_config.get('resize', (224, 224))
            
            # 構建轉換
            self.transform = self._build_transforms(self.transform_config)
            
            # 呼叫父類初始化
            super().__init__(
                index_path=index_path,
                label_field=label_field,
                transform=None,  # 暫不設置轉換，我們使用自己的轉換機制
                filter_criteria=filter_criteria,
                fallback_to_direct=True
            )
            
        else:
            # 直接模式初始化
            if root_dir is None or config is None:
                raise ValueError("直接模式下需要提供root_dir和config參數")
                
            self.root_dir = Path(root_dir) if root_dir else None
            self.config = config
            self.save_dir = save_dir
            self.is_direct_mode = True
            
            # 獲取頻譜圖處理參數
            spec_config = config.get('data', {}).get('preprocessing', {}).get('spectrogram', {})
            self.resize = spec_config.get('resize', (224, 224))
            
            # 構建轉換
            self.transform = self._build_transforms(self.transform_config)
            
            # 獲取所有患者目錄
            self.patient_dirs = list_patient_dirs(self.root_dir)
            
            # 收集所有樣本
            self.samples = self._collect_samples()
            
            logger.info(f"加載了 {len(self.samples)} 個頻譜圖樣本")
    
    def setup_direct_mode(self) -> None:
        """設置直接加載模式
        
        當索引加載失敗時，如果fallback_to_direct為True，則調用此方法
        設置為直接模式，但由於沒有root_dir和config，因此僅提供最小功能
        """
        logger.warning("索引加載失敗並退化到直接模式，但未提供root_dir和config，將使用空數據集")
        self.samples = []
        self.patient_dirs = []
    
    def load_data(self, data_row: dict) -> torch.Tensor:
        """從數據行加載頻譜圖數據
        
        Args:
            data_row: 數據行，包含file_path等字段
            
        Returns:
            torch.Tensor: 加載的頻譜圖數據
        """
        # 優先使用spectrogram_path欄位，如果有的話
        if 'spectrogram_path' in data_row and os.path.exists(data_row['spectrogram_path']):
            spec_path = data_row['spectrogram_path']
        else:
            # file_path可能是目錄，需要尋找頻譜圖文件
            dir_path = data_row['file_path']
            if os.path.isdir(dir_path):
                # 嘗試查找標準頻譜圖文件名
                spec_path = os.path.join(dir_path, 'spectrogram.png')
                
                if not os.path.exists(spec_path):
                    # 嘗試查找目錄中的任何.png文件
                    png_files = [f for f in os.listdir(dir_path) if f.endswith('.png')]
                    if png_files:
                        spec_path = os.path.join(dir_path, png_files[0])
                    else:
                        logger.warning(f"在目錄 {dir_path} 中找不到頻譜圖文件，將返回空張量")
                        # 返回全黑圖像作為後備
                        return torch.zeros(3, self.resize[0], self.resize[1])
            else:
                # 可能file_path本身就是文件
                spec_path = dir_path
        
        # 檢查文件是否存在
        if not os.path.exists(spec_path):
            logger.warning(f"頻譜圖文件不存在: {spec_path}，將返回空張量")
            # 返回全黑圖像作為後備
            return torch.zeros(3, self.resize[0], self.resize[1])
        
        # 加載頻譜圖
        try:
            # 檢查緩存
            if self.cache_spectrograms and spec_path in self.image_cache:
                return self.image_cache[spec_path]
            
            # 加載圖像
            image = Image.open(spec_path).convert('RGB')
            
            # 應用轉換
            if self.transform:
                image = self.transform(image)
            
            # 緩存結果
            if self.cache_spectrograms:
                self.image_cache[spec_path] = image
            
            return image
        except Exception as e:
            logger.error(f"加載頻譜圖文件失敗: {spec_path}, 錯誤: {str(e)}")
            # 返回全黑圖像作為後備
            return torch.zeros(3, self.resize[0], self.resize[1])
    
    def direct_getitem(self, idx: int) -> Tuple[torch.Tensor, Any]:
        """直接模式下獲取項目
        
        Args:
            idx: 索引
            
        Returns:
            Tuple[torch.Tensor, Any]: (頻譜圖數據, 標籤)
        """
        if not hasattr(self, 'samples') or not self.samples:
            # 如果沒有樣本，返回空數據
            dummy_image = torch.zeros(3, self.resize[0], self.resize[1])
            return dummy_image, 0
            
        sample = self.samples[idx]
        
        # 加載頻譜圖
        try:
            spec_path = sample['spec_path']
            
            # 檢查緩存
            if self.cache_spectrograms and spec_path in self.image_cache:
                image = self.image_cache[spec_path]
            else:
                # 加載圖像
                image = Image.open(spec_path).convert('RGB')
                
                # 應用轉換
                if self.transform:
                    image = self.transform(image)
                
                # 緩存結果
                if self.cache_spectrograms:
                    self.image_cache[spec_path] = image
            
            # 獲取標籤
            label = sample['label']
            
            return image, label
            
        except Exception as e:
            logger.error(f"處理樣本失敗，索引 {idx}: {str(e)}")
            # 返回全黑圖像作為後備
            dummy_image = torch.zeros(3, self.resize[0], self.resize[1])
            return dummy_image, sample.get('label', 0)
    
    def direct_len(self) -> int:
        """直接模式下的數據集大小
        
        Returns:
            int: 數據集大小
        """
        if hasattr(self, 'samples'):
            return len(self.samples)
        return 0
        
    def _collect_samples(self) -> List[Dict[str, Any]]:
        """收集所有頻譜圖樣本
        
        收集步驟：
        1. 遍歷所有患者目錄
        2. 處理每個患者的info.json文件，獲取patient_id, score, selection等
        3. 查找有效的頻譜圖文件
        4. 根據過濾配置過濾不需要的樣本
        
        Returns:
            List[Dict[str, Any]]: 收集到的樣本列表，每個樣本包含patient_id, spec_path, score等信息
        """
        samples = []
        
        # 獲取配置
        filtering_config = self.config.get('data', {}).get('filtering', {})
        task_type = filtering_config.get('task_type', 'classification')
        class_config = filtering_config.get('class_config', {})
        
        # 只在分類任務中使用的配置
        if task_type == 'classification':
            score_thresholds = filtering_config.get('score_thresholds', {})
            normal_threshold = score_thresholds.get('normal', 0)
            patient_threshold = score_thresholds.get('patient', 9)
            subject_source = filtering_config.get('subject_source', {})
        else:
            # 回歸任務中，不需要使用分數閾值過濾
            normal_threshold = -float('inf')  # 所有分數都大於這個值
            patient_threshold = float('inf')  # 所有分數都小於這個值
            subject_source = {}
        
        for patient_dir in self.patient_dirs:
            # 使用patient_info_loader讀取患者信息
            patient_info = load_patient_info(patient_dir)
            if patient_info is None:
                logger.warning(f"在 {patient_dir} 找不到患者信息文件，跳過")
                continue
                
            # 獲取患者ID、分數和選擇
            patient_id = patient_info['patient_id']
            score = patient_info['score']
            selection = patient_info['selection']
            raw_info = patient_info['raw_info']  # 獲取原始info內容，用於查找spectrograms字段
                
            # 查找頻譜圖文件
            spec_paths = []
            if 'spectrograms' in raw_info:
                # 如果患者信息中有頻譜圖路徑
                for spec_data in raw_info['spectrograms']:
                    if isinstance(spec_data, str):
                        spec_path = patient_dir / spec_data
                    elif isinstance(spec_data, dict) and 'path' in spec_data:
                        spec_path = patient_dir / spec_data['path']
                    else:
                        continue
                        
                    if spec_path.exists():
                        spec_paths.append(spec_path)
            else:
                # 否則查找目錄中的所有PNG文件
                spec_paths = list(patient_dir.glob("*.png"))
                
            # 為每個頻譜圖創建樣本
            for spec_path in spec_paths:
                # 映射選擇類型
                selection_type = None
                for std_type, selections in SELECTION_TYPES.items():
                    if any(s in selection for s in selections):
                        selection_type = std_type
                        break
                
                if selection_type is None:
                    logger.debug(f"無法映射選擇類型: {selection}, 跳過")
                    continue
                
                # 檢查是否啟用此選擇類型
                if class_config.get(selection_type, 0) != 1:
                    logger.debug(f"選擇類型 {selection_type} 已禁用，跳過")
                    continue
                
                # 基於分數確定患者分組
                if score <= normal_threshold:
                    patient_group = 'normal'
                elif score >= patient_threshold:
                    patient_group = 'patient'
                else:
                    logger.debug(f"分數 {score} 不在正常或患者範圍內，跳過")
                    continue
                
                # 檢查ID前綴
                id_prefix = patient_id[0].upper() if patient_id else ''
                if id_prefix in ['N', 'P']:
                    include_config = subject_source.get(patient_group, {})
                    if id_prefix == 'N' and include_config.get('include_N', 1) != 1:
                        logger.debug(f"N前綴正常人已禁用，跳過 {patient_id}")
                        continue
                    if id_prefix == 'P' and include_config.get('include_P', 1) != 1:
                        logger.debug(f"P前綴正常人已禁用，跳過 {patient_id}")
                        continue
                
                # 生成標籤
                if task_type == 'classification':
                    # 分類任務：生成類別標籤
                    class_name = f"{patient_group.capitalize()}-{selection_type}"
                    class_idx = LABEL_TO_INDEX.get(class_name, 0)
                    label = torch.tensor(class_idx, dtype=torch.long)  # 分類任務用 long
                else:
                    # 回歸任務：使用分數作為標籤
                    label = torch.tensor(score, dtype=torch.float32)  # 回歸任務用 float32
                
                # 創建樣本字典
                sample = {
                    'patient_id': patient_id,
                    'spec_path': str(spec_path),
                    'score': torch.tensor(score, dtype=torch.float32),  # 確保score也是float32
                    'selection': selection,
                    'selection_type': selection_type,
                    'patient_group': patient_group,
                    'label': label
                }
                
                samples.append(sample)
                
        logger.info(f"收集了 {len(samples)} 個頻譜圖樣本")
        return samples
        
    def _build_transforms(self, transform_config: Optional[Dict[str, Any]] = None) -> transforms.Compose:
        """構建轉換管道
        
        Args:
            transform_config: 轉換配置
            
        Returns:
            transforms.Compose: 轉換管道
        """
        transform_list = []
        
        # 首先添加調整大小的轉換
        transform_list.append(transforms.Resize(self.resize))
        
        # 如果在訓練模式並且指定了數據增強
        if self.is_train and transform_config and transform_config.get('augment', False):
            # 隨機裁剪
            if transform_config.get('random_crop', False):
                crop_size = transform_config.get('crop_size', self.resize)
                transform_list.append(transforms.RandomCrop(crop_size))
                
            # 隨機水平翻轉
            if transform_config.get('random_horizontal_flip', False):
                prob = transform_config.get('horizontal_flip_prob', 0.5)
                transform_list.append(transforms.RandomHorizontalFlip(prob))
                
            # 隨機旋轉
            if transform_config.get('random_rotation', False):
                degrees = transform_config.get('rotation_degrees', 10)
                transform_list.append(transforms.RandomRotation(degrees))
                
            # 隨機顏色抖動
            if transform_config.get('color_jitter', False):
                brightness = transform_config.get('brightness', 0.1)
                contrast = transform_config.get('contrast', 0.1)
                saturation = transform_config.get('saturation', 0.1)
                hue = transform_config.get('hue', 0.05)
                transform_list.append(
                    transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
                )
                
        # 轉換為張量
        transform_list.append(transforms.ToTensor())
        
        # 標準化
        if transform_config and transform_config.get('normalize', True):
            mean = transform_config.get('mean', [0.485, 0.456, 0.406])
            std = transform_config.get('std', [0.229, 0.224, 0.225])
            transform_list.append(transforms.Normalize(mean=mean, std=std))
        
        return transforms.Compose(transform_list)

# 測試代碼
if __name__ == "__main__":
    """
    Description: Minimal Executable Unit for spectrogram_dataset.py，測試頻譜圖數據集的索引模式和直接模式。
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
            'file_path': ['/path/to/spec1.png', '/path/to/spec2.png', '/path/to/spec3.png'],
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
        # 測試索引模式 (由於文件路徑不存在，應該回退到直接模式)
        print("測試1: 索引模式 (回退到直接模式)")
        dataset = SpectrogramDataset(
            index_path=test_csv_path,
            label_field="score"
        )
        print(f"模式: {'索引模式' if not dataset.is_direct_mode else '直接模式'}")
        print(f"數據集大小: {len(dataset)}")
        
        # 測試直接模式的初始化 (使用空數據)
        print("\n測試2: 直接模式")
        config = {
            'data': {
                'preprocessing': {
                    'spectrogram': {
                        'resize': (224, 224)
                    }
                },
                'filtering': {
                    'task_type': 'classification',
                    'class_config': {}
                }
            }
        }
        direct_dataset = SpectrogramDataset(
            root_dir="./",  # 使用當前目錄
            config=config
        )
        print(f"直接模式數據集大小: {len(direct_dataset)}")
        
        print("\n所有測試通過！")
        
    except Exception as e:
        print(f"測試失敗: {str(e)}")
    
    finally:
        # 清理測試文件
        os.unlink(test_csv_path) 