"""
頻譜圖數據集：支持讀取和處理頻譜圖數據
功能：
1. 讀取頻譜圖文件
2. 支持數據增強和轉換
3. 支持按患者ID拆分數據集
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import random
from PIL import Image
import torchvision.transforms as transforms
from utils.constants import SELECTION_TYPES, LABEL_TO_INDEX
from utils.patient_info_loader import load_patient_info, list_patient_dirs

logger = logging.getLogger(__name__)

class SpectrogramDataset(Dataset):
    """頻譜圖數據集類，支持讀取和處理頻譜圖數據
    
    主要功能：
    1. 從PNG文件讀取頻譜圖圖像
    2. 支持各種圖像轉換和數據增強
    3. 支持按患者ID拆分數據集
    
    數據讀取邏輯：
    1. 頻譜圖文件讀取:
       - 在患者資料夾中查找PNG格式的頻譜圖文件
       - 首先檢查info.json中的'spectrograms'字段獲取頻譜圖路徑
       - 如果沒有指定路徑，搜索資料夾中所有.png文件
       - 使用PIL.Image.open()讀取圖像並轉換為RGB格式
    
    2. 患者信息讀取:
       - 使用utils.patient_info_loader模組自動查找並解析患者目錄中的info.json
       - 獲取標準化的患者ID、分數和選擇信息
       - 如果無法找到有效的info.json，則跳過該目錄
    
    3. 圖像處理:
       - 調整圖像大小為配置中指定的尺寸(默認224x224)
       - 訓練模式下可以應用數據增強(隨機裁剪、翻轉、旋轉、色彩抖動等)
       - 標準化圖像數據，轉換為PyTorch張量格式
    
    4. 緩存機制:
       - 可選啟用緩存機制，將讀取過的圖像保存在內存中
       - 適合數據集較小或內存充足的情況
    
    5. 數據集拆分:
       - 使用split_by_patient方法，按患者ID將數據集拆分為訓練、驗證和測試集
       - 確保同一患者的所有數據僅出現在一個數據集中
    """
    
    def __init__(
        self,
        data_path: str,
        config: Dict[str, Any],
        transform: Optional[Dict[str, Any]] = None,
        is_train: bool = True,
        cache_spectrograms: bool = False
    ):
        """初始化頻譜圖數據集
        
        Args:
            data_path (str): 數據路徑
            config (Dict[str, Any]): 配置字典
            transform (Optional[Dict[str, Any]], optional): 轉換配置. 預設為 None.
            is_train (bool, optional): 是否為訓練模式. 預設為 True.
            cache_spectrograms (bool, optional): 是否緩存頻譜圖. 預設為 False.
        """
        self.data_path = Path(data_path)
        self.config = config
        self.is_train = is_train
        self.cache_spectrograms = cache_spectrograms
        self.image_cache = {}
        
        # 解析轉換配置
        self.transform = self._build_transforms(transform)
        
        # 獲取所有患者目錄
        self.patient_dirs = list_patient_dirs(self.data_path)
        
        # 收集所有樣本
        self.samples = self._collect_samples()
        
        logger.info(f"加載了 {len(self.samples)} 個頻譜圖樣本")
        
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
                    label = torch.tensor(class_idx, dtype=torch.float32)  # 注意：使用float32而非long
                else:
                    # 回歸任務：使用分數作為標籤
                    label = torch.tensor(score, dtype=torch.float32)
                
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
            transform_config (Optional[Dict[str, Any]], optional): 轉換配置. 預設為 None.
            
        Returns:
            transforms.Compose: 轉換管道
        """
        transform_list = []
        
        # 首先添加調整大小的轉換
        resize = self.config.get('resize', (224, 224))
        transform_list.append(transforms.Resize(resize))
        
        # 如果在訓練模式並且指定了數據增強
        if self.is_train and transform_config and transform_config.get('augment', False):
            # 隨機裁剪
            if transform_config.get('random_crop', False):
                crop_size = transform_config.get('crop_size', resize)
                transform_list.append(transforms.RandomCrop(crop_size))
                
            # 隨機水平翻轉
            if transform_config.get('random_horizontal_flip', False):
                prob = transform_config.get('horizontal_flip_prob', 0.5)
                transform_list.append(transforms.RandomHorizontalFlip(prob))
                
            # 隨機旋轉
            if transform_config.get('random_rotation', False):
                degrees = transform_config.get('rotation_degrees', 10)
                transform_list.append(transforms.RandomRotation(degrees))
                
            # 色彩抖動
            if transform_config.get('color_jitter', False):
                brightness = transform_config.get('brightness', 0.1)
                contrast = transform_config.get('contrast', 0.1)
                saturation = transform_config.get('saturation', 0.1)
                hue = transform_config.get('hue', 0.05)
                transform_list.append(
                    transforms.ColorJitter(brightness, contrast, saturation, hue)
                )
                
        # 轉換為張量
        transform_list.append(transforms.ToTensor())
        
        # 標準化
        if transform_config and transform_config.get('normalize', False):
            mean = transform_config.get('mean', [0.485, 0.456, 0.406])
            std = transform_config.get('std', [0.229, 0.224, 0.225])
            transform_list.append(transforms.Normalize(mean, std))
            
        return transforms.Compose(transform_list)
    
    def _load_image(self, spec_path: str) -> torch.Tensor:
        """加載頻譜圖圖像
        
        Args:
            spec_path (str): 頻譜圖路徑
            
        Returns:
            torch.Tensor: 轉換後的圖像張量
            
        Raises:
            FileNotFoundError: 如果文件不存在
            ValueError: 如果無法加載圖像
        """
        # 如果已緩存，則直接返回
        if self.cache_spectrograms and spec_path in self.image_cache:
            return self.image_cache[spec_path]
            
        # 檢查文件是否存在
        if not os.path.exists(spec_path):
            raise FileNotFoundError(f"頻譜圖文件不存在: {spec_path}")
            
        try:
            # 使用PIL加載圖像
            image = Image.open(spec_path).convert('RGB')
            
            # 應用轉換
            tensor = self.transform(image)
            
            # 緩存圖像（如果啟用）
            if self.cache_spectrograms:
                self.image_cache[spec_path] = tensor
                
            return tensor
            
        except Exception as e:
            raise ValueError(f"無法加載圖像 {spec_path}: {str(e)}")
    
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
            Dict[str, Any]: 樣本字典，包含 'image', 'score' 等
        """
        sample = self.samples[idx]
        
        # 加載頻譜圖
        image = self._load_image(sample['spec_path'])
        
        # 創建輸出字典
        output = {
            'image': image,
            'score': sample['score'],  # float32
            'label': sample['label'],  # float32
            'patient_id': sample['patient_id'],
            'selection': sample.get('selection', ''),
            'selection_type': sample.get('selection_type', ''),
            'patient_group': sample.get('patient_group', '')
        }
            
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

# 中文註解：這是spectrogram_dataset.py的Minimal Executable Unit，檢查能否正確初始化與錯誤路徑時的優雅報錯
if __name__ == "__main__":
    """
    Description: Minimal Executable Unit for spectrogram_dataset.py，檢查SpectrogramDataset能否正確初始化與錯誤路徑時的優雅報錯。
    Args: None
    Returns: None
    References: 無
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    from data.spectrogram_dataset import SpectrogramDataset

    # 測試錯誤路徑
    try:
        dummy_config = {
            "data": {
                "type": "spectrogram",
                "source": {"spectrogram_dir": "not_exist_dir"},
                "preprocessing": {"spectrogram": {"normalize": True}}
            }
        }
        dataset = SpectrogramDataset(data_path="not_exist_dir", config=dummy_config)
        print(f"資料集長度: {len(dataset)}")
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"第一筆資料: {sample}")
        else:
            print("沒有資料可供測試")
    except Exception as e:
        print(f"遇到錯誤（預期行為）: {e}") 