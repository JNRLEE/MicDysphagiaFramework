"""
音頻數據集模組：讀取WAV文件並提取相關信息
功能：
1. 讀取音頻WAV文件
2. 從info.json中提取患者信息
3. 提供數據增強選項
4. 支持按患者ID拆分數據集
5. 支持從索引CSV加載數據
"""

import os
import glob
import json
import random
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import soundfile as sf
import warnings
from utils.constants import SELECTION_TYPES, LABEL_TO_INDEX
from utils.patient_info_loader import load_patient_info, list_patient_dirs
from utils.custom_classification_loader import CustomClassificationLoader
from utils.audio_feature_extractor import extract_features_from_config

# 引入索引數據集基類
from data.indexed_dataset import IndexedDatasetBase

logger = logging.getLogger(__name__)

class AudioDataset(IndexedDatasetBase):
    """音頻數據集類，用於讀取和處理音頻數據
    
    主要功能：
    1. 加載WAV音頻文件和對應的元數據
    2. 支持音頻數據的預處理和增強
    3. 支持按患者ID拆分數據集
    4. 支持從索引CSV加載數據
    
    數據讀取模式：
    1. 索引模式：使用data_index.csv加載數據
       - 提供index_path參數啟用此模式
       - 使用label_field指定標籤欄位
       - 使用filter_criteria篩選數據
       
    2. 直接模式：直接從目錄讀取WAV文件
       - 使用root_dir參數指定音頻文件的根目錄
       - 從患者info.json獲取標籤信息
       - 此模式與原始實現兼容
    """
    
    def __init__(
        self,
        root_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        transform: Optional[Dict[str, Any]] = None,
        is_train: bool = True,
        # 新增索引模式參數
        index_path: Optional[str] = None,
        label_field: str = 'score',
        filter_criteria: Optional[Dict] = None,
        sample_rate: int = 16000,
        duration: Optional[float] = None
    ):
        """初始化音頻數據集
        
        Args:
            root_dir: WAV文件的根目錄（直接模式必填）
            config: 配置字典（直接模式必填）
            transform: 數據轉換配置
            is_train: 是否為訓練模式
            index_path: 索引CSV文件路徑（索引模式必填）
            label_field: 標籤欄位名稱，可以是'score', 'DrLee_Evaluation', 'DrTai_Evaluation', 'selection'
            filter_criteria: 篩選條件字典
            sample_rate: 音頻採樣率，索引模式時使用
            duration: 音頻長度（秒），索引模式時使用
        """
        # 決定是否使用索引模式
        self.use_index_mode = index_path is not None and os.path.exists(index_path)
        
        # 初始化共用屬性
        self.transform_config = transform or {}
        self.is_train = is_train
        
        if self.use_index_mode:
            # 索引模式初始化
            self.root_dir = None
            self.config = config or {}
            self.sample_rate = sample_rate
            self.duration = duration
            self.max_len = int(self.sample_rate * self.duration) if duration else None
            self.normalize_audio = True
            
            # 呼叫父類初始化
            super().__init__(
                index_path=index_path,
                label_field=label_field,
                transform=None,  # 暫不設置轉換，後續會設置
                filter_criteria=filter_criteria,
                fallback_to_direct=True
            )
            
            # 設置轉換函數
            self.set_transforms(self.transform_config)
            
        else:
            # 直接模式初始化
            if root_dir is None or config is None:
                raise ValueError("直接模式下需要提供root_dir和config參數")
                
            self.root_dir = Path(root_dir) if root_dir else None
            self.config = config
            self.is_direct_mode = True
            
            # 獲取音頻處理參數
            audio_config = config.get('data', {}).get('preprocessing', {}).get('audio', {})
            self.sample_rate = audio_config.get('sample_rate', 16000)
            self.duration = audio_config.get('duration', 5.0)  # 秒
            self.max_len = int(self.sample_rate * self.duration)
            self.normalize_audio = audio_config.get('normalize', True)
            self.target_wav_file = audio_config.get('target_file', 'Probe0_RX_IN_TDM4CH0.wav')
            
            # 初始化轉換函數 (稍後在 set_transforms 中設置)
            self.transforms = {}
            
            # 載入自定義分類配置
            self.custom_classifier = CustomClassificationLoader(config)
            
            # 檢查是否需要特徵提取
            self.enable_feature_extraction = config.get('data', {}).get('preprocessing', {}).get('features', {}).get('method', None) is not None
            if self.enable_feature_extraction:
                logger.info(f"啟用特徵提取: {config.get('data', {}).get('preprocessing', {}).get('features', {}).get('method', '未指定方法')}")
            
            # 加載並過濾數據，生成最終的樣本列表 self.samples
            self.samples = self._load_and_filter_data()

            # 設置轉換函數 (現在基於 is_train)
            self.set_transforms(self.transform_config)
    
    def setup_direct_mode(self) -> None:
        """設置直接加載模式
        
        當索引加載失敗時，如果fallback_to_direct為True，則調用此方法
        設置為直接模式，但由於沒有root_dir和config，因此僅提供最小功能
        """
        logger.warning("索引加載失敗並退化到直接模式，但未提供root_dir和config，將使用空數據集")
        self.samples = []
        self.transforms = {}
    
    def load_data(self, data_row: dict) -> torch.Tensor:
        """從數據行加載音頻數據
        
        Args:
            data_row: 數據行，包含file_path等字段
            
        Returns:
            torch.Tensor: 加載的音頻數據
        """
        wav_path = data_row['file_path']
        
        try:
            # 加載WAV文件
            waveform = self._load_audio(wav_path)
            return waveform
        except Exception as e:
            logger.error(f"加載音頻文件失敗: {wav_path}, 錯誤: {str(e)}")
            # 返回零張量作為後備
            if self.max_len:
                return torch.zeros(1, self.max_len)
            else:
                return torch.zeros(1, int(self.sample_rate * 5))  # 默認5秒
    
    def direct_getitem(self, idx: int) -> Tuple[torch.Tensor, Any]:
        """直接模式下獲取項目（與原始__getitem__相同）
        
        Args:
            idx: 索引
            
        Returns:
            Tuple[torch.Tensor, Any]: (音頻數據, 標籤)
        """
        if not hasattr(self, 'samples') or not self.samples:
            # 如果沒有樣本，返回空數據
            dummy_audio = torch.zeros(1, int(self.sample_rate * 5))
            return dummy_audio, 0
            
        sample = self.samples[idx]
        
        # 加載音頻
        try:
            audio_path = sample['audio_path']
            waveform = self._load_audio(audio_path)
            
            # 應用轉換
            if self.is_train and 'train' in self.transforms:
                waveform = self._apply_transforms(waveform, self.transforms['train'])
            elif not self.is_train and 'val' in self.transforms:
                waveform = self._apply_transforms(waveform, self.transforms['val'])
                
            # 獲取標籤
            label = sample['label']
            
            return waveform, label
            
        except Exception as e:
            logger.error(f"處理樣本失敗，索引 {idx}: {str(e)}")
            # 返回零張量作為後備
            dummy_audio = torch.zeros(1, self.max_len)
            return dummy_audio, sample.get('label', 0)
    
    def direct_len(self) -> int:
        """直接模式下的數據集大小
        
        Returns:
            int: 數據集大小
        """
        if hasattr(self, 'samples'):
            return len(self.samples)
        return 0
    
    # 保留原有的方法（直接模式使用）
    def _load_and_filter_data(self) -> List[Dict[str, Any]]:
        """加載、過濾並收集所有有效的音頻樣本
        
        收集步驟：
        1. 遍歷所有患者目錄
        2. 處理每個患者的info.json文件和音頻文件
        3. 根據task_type執行不同的過濾邏輯 (包括自定義分類)
        4. 生成對應的標籤
        5. 返回通過所有過濾的樣本列表
        
        Returns:
            List[Dict[str, Any]]: 收集到的有效樣本列表
        """
        logger.info(f"正在從 {self.root_dir} 加載並過濾音頻數據...")
        samples = []
        
        # 檢查是否是目錄
        if not self.root_dir.is_dir():
            logger.error(f"{self.root_dir} 不是有效的目錄")
            return samples
        
        # 獲取所有患者目錄
        patient_dirs = list_patient_dirs(self.root_dir)
        
        if not patient_dirs:
            logger.warning(f"在 {self.root_dir} 未找到任何患者目錄")
            return samples

        # 獲取配置
        filtering_config = self.config.get('data', {}).get('filtering', {})
        task_type = filtering_config.get('task_type', 'classification')
        class_config = filtering_config.get('class_config', {})
        
        # 檢查是否啟用自定義分類
        use_custom_classification = self.custom_classifier.enabled
        
        # 只在標準分類任務中使用的配置
        normal_threshold = -float('inf')
        patient_threshold = float('inf')
        subject_source = {}
        if task_type == 'classification' and not use_custom_classification:
            score_thresholds = filtering_config.get('score_thresholds', {})
            subject_source = filtering_config.get('subject_source', {})
            normal_threshold = score_thresholds.get('normal', {})
            patient_threshold = score_thresholds.get('patient', {})

        # 記錄過濾統計
        stats = {
            'total_potential': 0,
            'no_info': 0,
            'no_audio': 0,
            'filtered_score': 0,
            'filtered_selection': 0,
            'filtered_id': 0,
            'filtered_custom': 0,
            'accepted': 0
        }

        # 遍歷所有患者目錄
        for patient_dir in patient_dirs:
            stats['total_potential'] += 1
            
            # 使用patient_info_loader讀取患者信息
            info = load_patient_info(patient_dir)
            if info is None:
                logger.warning(f"在 {patient_dir} 找不到有效的info.json")
                stats['no_info'] += 1
                continue
            
            # 查找目標音頻文件
            audio_path = patient_dir / self.target_wav_file
            if not audio_path.exists():
                logger.warning(f"在 {patient_dir} 找不到目標音頻文件 {self.target_wav_file}")
                stats['no_audio'] += 1
                continue

            # 獲取基本信息
            patient_id = info['patient_id']
            score = info['score']
            selection = info['selection']
            
            # 映射選擇類型
            selection_type = None
            for std_type, selections in SELECTION_TYPES.items():
                if any(s in selection for s in selections):
                    selection_type = std_type
                    break
            
            if selection_type is None:
                logger.debug(f"無法映射選擇類型: {selection}, 跳過 {patient_id}")
                stats['filtered_selection'] += 1
                continue
            
            # 過濾邏輯
            label = None
            custom_class_name = None
            patient_group = None

            if task_type == 'classification':
                if use_custom_classification:
                    # --- 自定義分類邏輯 ---
                    custom_class = self.custom_classifier.get_class(patient_id)
                    custom_class_idx = self.custom_classifier.get_class_index(patient_id)
                    
                    if custom_class is None or custom_class_idx is None:
                        logger.debug(f"在自定義分類中找不到患者 {patient_id} 或無效分類，跳過")
                        stats['filtered_custom'] += 1
                        continue
                    
                    # 如果分類是 'nan' 或其他需要排除的值，則跳過
                    # Check if the class itself should be excluded (e.g., 'nan')
                    if self.custom_classifier.is_class_excluded(custom_class):
                       logger.debug(f"患者 {patient_id} 的自定義分類為 '{custom_class}'，已被排除，跳過")
                       stats['filtered_custom'] += 1
                       continue

                    # *** ADDED: Check if the action type is allowed based on custom_classification.class_config ***
                    if not self.custom_classifier.is_action_allowed(selection_type):
                        logger.debug(f"患者 {patient_id} (目錄 {patient_dir.name}) 的動作類型 {selection_type} 不在允許列表中 (基於 custom_classification.class_config)，跳過")
                        # Use filtered_selection stat or create a new one if needed
                        stats['filtered_selection'] += 1 
                        continue
                    # *** END ADDED CHECK ***

                    label = torch.tensor(custom_class_idx, dtype=torch.long) # 分類標籤用 Long
                    custom_class_name = custom_class # 記錄一下名字

                else:
                    # --- 原始分類邏輯 ---
                    # 檢查選擇類型是否啟用
                    if class_config.get(selection_type, 0) != 1:
                        logger.debug(f"選擇類型 {selection_type} 已禁用，跳過 {patient_id}")
                        stats['filtered_selection'] += 1
                        continue

                    # 基於分數確定患者分組
                    if score <= normal_threshold:
                        patient_group = 'normal'
                    elif score >= patient_threshold:
                        patient_group = 'patient'
                    
                    if patient_group is None:
                        logger.debug(f"分數 {score} 不在正常或患者範圍內，跳過 {patient_id}")
                        stats['filtered_score'] += 1
                        continue
                    
                    # 檢查ID前綴
                    id_prefix = patient_id[0].upper() if patient_id else ''
                    if id_prefix in ['N', 'P']:
                        include_config = subject_source.get(patient_group, {})
                        if id_prefix == 'N' and include_config.get('include_N', 1) != 1:
                            logger.debug(f"N前綴正常人已禁用，跳過 {patient_id}")
                            stats['filtered_id'] += 1
                            continue
                        if id_prefix == 'P' and include_config.get('include_P', 1) != 1:
                            logger.debug(f"P前綴正常人已禁用，跳過 {patient_id}")
                            stats['filtered_id'] += 1
                            continue
                    
                    # 生成分類標籤
                    class_name = f"{patient_group.capitalize()}-{selection_type}"
                    class_idx = LABEL_TO_INDEX.get(class_name)
                    
                    if class_idx is None: # or class_idx >= 10: # 移除上限檢查，讓其更通用
                        logger.warning(f"無效的類別標籤: {class_name}, 跳過 {patient_id}")
                        stats['filtered_selection'] += 1
                        continue
                        
                    label = torch.tensor(class_idx, dtype=torch.long) # 分類標籤用 Long
                    
            else:  # regression
                # --- 回歸任務邏輯 ---
                # 可以添加分數範圍檢查
                valid_score_range = filtering_config.get('valid_score_range', [0, 40])
                if not (valid_score_range[0] <= score <= valid_score_range[1]):
                    logger.debug(f"分數 {score} 超出有效範圍 {valid_score_range}，跳過 {patient_id}")
                    stats['filtered_score'] += 1
                    continue
                
                label = torch.tensor(score, dtype=torch.float32) # 回歸標籤用 Float

            # --- 如果通過所有過濾，則創建樣本 ---
            if label is not None:
                sample = {
                    'patient_id': patient_id,
                    'audio_path': str(audio_path),
                    'score': torch.tensor(score, dtype=torch.float32), # 分數始終記錄為 float
                    'selection': selection,
                    'selection_type': selection_type,
                    'label': label # 標籤根據任務類型決定 (Long for classification, Float for regression)
                }
                
                # 根據分類方式添加額外的信息
                if task_type == 'classification':
                    if use_custom_classification:
                        sample['custom_class'] = custom_class_name
                    else:
                        sample['patient_group'] = patient_group
                
                samples.append(sample)
                stats['accepted'] += 1
        
        # 輸出過濾統計
        logger.info("=== 數據加載與過濾統計 ===")
        logger.info(f"任務類型: {task_type}")
        logger.info(f"使用自定義分類: {use_custom_classification}")
        logger.info(f"總潛在患者目錄數: {stats['total_potential']}")
        logger.info(f"缺少info.json文件: {stats['no_info']}")
        logger.info(f"缺少目標音頻文件: {stats['no_audio']}")
        logger.info(f"因分數範圍/閾值被過濾: {stats['filtered_score']}")
        logger.info(f"因選擇類型被過濾: {stats['filtered_selection']}")
        if task_type == 'classification':
            if not use_custom_classification:
                logger.info(f"因ID前綴被過濾: {stats['filtered_id']}")
            else:
                logger.info(f"因自定義分類未找到或無效被過濾: {stats['filtered_custom']}")
                logger.info(f"自定義分類類別數: {self.custom_classifier.get_total_classes()}")
                logger.info(f"自定義分類類別: {self.custom_classifier.get_all_classes()}")
        logger.info(f"最終接受的樣本數: {stats['accepted']}")
        
        return samples
        
    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """加載音頻文件並進行預處理
        
        Args:
            audio_path (str): 音頻文件路徑
            
        Returns:
            torch.Tensor: 處理後的音頻張量，形狀為 [1, num_samples]
        """
        try:
            # 使用librosa加載音頻
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # 轉換為張量並重塑為 [1, num_samples]
            waveform = torch.tensor(waveform).float().unsqueeze(0)
            
            # 處理音頻長度
            if self.max_len is not None:
                if waveform.shape[1] > self.max_len:
                    # 截斷
                    waveform = waveform[:, :self.max_len]
                elif waveform.shape[1] < self.max_len:
                    # 填充
                    padding = torch.zeros(1, self.max_len - waveform.shape[1])
                    waveform = torch.cat([waveform, padding], dim=1)
            
            # 歸一化
            if self.normalize_audio:
                with torch.no_grad():
                    waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
                    
            return waveform
            
        except Exception as e:
            logger.error(f"加載音頻失敗 {audio_path}: {str(e)}")
            if self.max_len is not None:
                return torch.zeros(1, self.max_len)
            else:
                return torch.zeros(1, int(self.sample_rate * 5))  # 默認5秒

    def set_transforms(self, transform_config):
        """設置音頻轉換函數
        
        Args:
            transform_config (Dict): 轉換配置字典
        """
        self.transforms = {}
        
        if not transform_config:
            return
        
        # 為訓練和驗證設置不同的轉換
        for mode in ['train', 'val']:
            if mode in transform_config:
                transforms_list = []
                mode_config = transform_config[mode]
                
                # 時間偏移
                if mode_config.get('time_shift', {}).get('enabled', False):
                    max_shift_sec = mode_config['time_shift'].get('max_shift_sec', 0.5)
                    
                    def time_shift(x):
                        shift_amt = int(random.random() * max_shift_sec * self.sample_rate)
                        return torch.roll(x, shifts=shift_amt, dims=1)
                    
                    transforms_list.append(time_shift)
                
                # 音量縮放
                if mode_config.get('volume_scale', {}).get('enabled', False):
                    min_scale = mode_config['volume_scale'].get('min_scale', 0.5)
                    max_scale = mode_config['volume_scale'].get('max_scale', 2.0)
                    
                    def volume_scale(x):
                        scale = random.uniform(min_scale, max_scale)
                        return x * scale
                    
                    transforms_list.append(volume_scale)
                
                # 時間拉伸 (簡化，不實際拉伸音頻)
                if mode_config.get('time_stretch', {}).get('enabled', False):
                    # 這裡僅作為示例，真正的時間拉伸需要更複雜的處理
                    def time_stretch(x):
                        # 簡化：隨機選擇一個速度因子，但不實際拉伸，避免複雜性
                        # 在實際應用中應使用librosa.effects.time_stretch實現
                        return x
                    
                    transforms_list.append(time_stretch)
                
                self.transforms[mode] = transforms_list
    
    def _apply_transforms(self, x: torch.Tensor, transforms: List[Callable]) -> torch.Tensor:
        """應用一系列轉換到音頻張量
        
        Args:
            x (torch.Tensor): 輸入音頻張量
            transforms (List[Callable]): 轉換函數列表
            
        Returns:
            torch.Tensor: 轉換後的音頻張量
        """
        x_transformed = x.clone()
        for transform in transforms:
            x_transformed = transform(x_transformed)
        return x_transformed
    
    def split_by_patient(
        self, 
        train_ratio: float = 0.7, 
        val_ratio: float = 0.15, 
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Tuple[List[int], List[int], List[int]]:
        """按患者ID拆分數據集 (基於過濾後的 self.samples)
        
        Args:
            train_ratio (float, optional): 訓練集比例. 預設為 0.7.
            val_ratio (float, optional): 驗證集比例. 預設為 0.15.
            test_ratio (float, optional): 測試集比例. 預設為 0.15.
            seed (int, optional): 隨機種子. 預設為 42.
            
        Returns:
            Tuple[List[int], List[int], List[int]]: 訓練、驗證和測試樣本在 self.samples 中的索引列表
        """
        # 確保比例總和約等於 1
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0):
             logger.warning(f"拆分比例之和 ({total}) 不為 1，將重新歸一化。")
             train_ratio /= total
             val_ratio /= total
             test_ratio /= total
        
        # 設置隨機種子
        random.seed(seed)
        
        # 建立患者ID到 self.samples 索引的映射
        patient_to_sample_indices = {}
        for i, sample in enumerate(self.samples):
            patient_id = sample['patient_id']
            if patient_id not in patient_to_sample_indices:
                patient_to_sample_indices[patient_id] = []
            patient_to_sample_indices[patient_id].append(i)
            
        # 獲取唯一的患者ID列表
        unique_patient_ids = list(patient_to_sample_indices.keys())
        
        # 打亂患者ID列表
        random.shuffle(unique_patient_ids)
        
        # 計算每個集合的患者數量
        num_patients = len(unique_patient_ids)
        train_patient_size = int(num_patients * train_ratio)
        val_patient_size = int(num_patients * val_ratio)
        # test_patient_size = num_patients - train_patient_size - val_patient_size # 確保所有患者都被分配

        # 分割患者ID
        train_patients = unique_patient_ids[:train_patient_size]
        val_patients = unique_patient_ids[train_patient_size : train_patient_size + val_patient_size]
        test_patients = unique_patient_ids[train_patient_size + val_patient_size:]
        
        # 根據分配的患者ID收集 self.samples 的索引
        train_indices = []
        val_indices = []
        test_indices = []
        
        for patient_id in train_patients:
            train_indices.extend(patient_to_sample_indices[patient_id])
        for patient_id in val_patients:
            val_indices.extend(patient_to_sample_indices[patient_id])
        for patient_id in test_patients:
            test_indices.extend(patient_to_sample_indices[patient_id])
            
        # 隨機打亂每個集合內的索引（可選，但通常是好的做法）
        random.shuffle(train_indices)
        random.shuffle(val_indices)
        random.shuffle(test_indices)

        logger.info(f"數據集按患者ID拆分 (基於 {len(self.samples)} 個過濾後樣本) - "
                    f"訓練: {len(train_indices)} 樣本 ({len(train_patients)} 患者), "
                    f"驗證: {len(val_indices)} 樣本 ({len(val_patients)} 患者), "
                    f"測試: {len(test_indices)} 樣本 ({len(test_patients)} 患者)")
            
        return train_indices, val_indices, test_indices

    def extract_mel_spectrogram(self, audio: np.ndarray, n_mels: int = 128) -> np.ndarray:
        """提取Mel頻譜圖 (示例，實際可能在Dataset外部或Adapter中完成)
        
        Args:
            audio (np.ndarray): 音頻數據
            n_mels (int, optional): Mel濾波器數量. 預設為 128.
            
        Returns:
            np.ndarray: Mel頻譜圖
        """
        try:
            # 使用 librosa 提取 Mel 頻譜圖
             mel_spec = librosa.feature.melspectrogram(
                 y=audio, 
                 sr=self.sample_rate, 
                 n_mels=n_mels,
                 n_fft=self.config.get('data', {}).get('preprocessing', {}).get('spectrogram', {}).get('n_fft', 1024),
                 hop_length=self.config.get('data', {}).get('preprocessing', {}).get('spectrogram', {}).get('hop_length', 512),
                 power=self.config.get('data', {}).get('preprocessing', {}).get('spectrogram', {}).get('power', 2.0)
             )
             # 轉換為分貝單位
             mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
             return mel_spec_db
        except Exception as e:
             logger.error(f"提取 Mel 頻譜圖時出錯: {e}")
             # 返回一個形狀正確的零數組
             time_steps = int(np.ceil(len(audio) / self.config.get('data', {}).get('preprocessing', {}).get('spectrogram', {}).get('hop_length', 512)))
             return np.zeros((n_mels, time_steps))


# 中文註解：這是audio_dataset.py的Minimal Executable Unit，檢查能否正確初始化與取樣，並測試錯誤路徑時的優雅報錯行為
if __name__ == "__main__":
    """
    Description: Minimal Executable Unit for audio_dataset.py，測試音頻數據集的索引模式和直接模式。
    Args: None
    Returns: None
    References: 無
    """
    import logging
    import tempfile
    import pandas as pd
    import os
    import numpy as np
    
    # 配置日誌
    logging.basicConfig(level=logging.INFO)
    
    # 創建測試數據
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as temp_file:
        test_data = pd.DataFrame({
            'file_path': ['/path/to/audio1.wav', '/path/to/audio2.wav', '/path/to/audio3.wav'],
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
        dataset = AudioDataset(
            index_path=test_csv_path,
            label_field="score",
            sample_rate=16000,
            duration=5.0
        )
        print(f"模式: {'索引模式' if not dataset.is_direct_mode else '直接模式'}")
        print(f"數據集大小: {len(dataset)}")
        
        # 測試直接模式的初始化 (使用空數據)
        print("\n測試2: 直接模式")
        config = {
            'data': {
                'preprocessing': {
                    'audio': {
                        'sample_rate': 16000,
                        'duration': 5.0,
                        'normalize': True
                    }
                }
            }
        }
        direct_dataset = AudioDataset(
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