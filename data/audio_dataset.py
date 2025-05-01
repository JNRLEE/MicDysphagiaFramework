"""
音頻數據集模組：讀取WAV文件並提取相關信息
功能：
1. 讀取音頻WAV文件
2. 從info.json中提取患者信息
3. 提供數據增強選項
4. 支持按患者ID拆分數據集
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

logger = logging.getLogger(__name__)

class AudioDataset(Dataset):
    """音頻數據集類，用於讀取和處理音頻數據
    
    主要功能：
    1. 加載WAV音頻文件和對應的元數據
    2. 支持音頻數據的預處理和增強
    3. 支持按患者ID拆分數據集
    
    數據讀取邏輯：
    1. WAV文件讀取:
       - 在每個患者資料夾中查找名為"Probe0_RX_IN_TDM4CH0.wav"的音頻文件(可通過config指定)
       - 使用librosa.load加載音頻，根據設定的采樣率和時長進行處理
       - 如果音頻長度不足，則填充零；如果過長，則截斷
    
    2. 患者信息(info.json)讀取:
       - 使用utils.patient_info_loader模組讀取患者信息
       - 從患者目錄中自動查找並解析info.json文件
       - 獲取患者ID、EAT-10分數和動作選擇等標準化信息
       - 如果無法找到有效的info.json文件，則跳過該患者資料夾
    
    3. 數據集拆分:
       - 通過split_by_patient方法按患者ID將數據集拆分為訓練、驗證和測試集
       - 確保同一患者的所有數據僅出現在一個集合中，避免數據洩漏
    """
    
    def __init__(
        self,
        root_dir: str,
        config: Dict[str, Any],
        transform: Optional[Dict[str, Any]] = None,
        is_train: bool = True
    ):
        """初始化音頻數據集
        
        Args:
            root_dir (str): WAV文件的根目錄
            config (Dict[str, Any]): 配置字典
            transform (Optional[Dict[str, Any]], optional): 數據轉換配置. 預設為 None.
            is_train (bool, optional): 是否為訓練模式. 預設為 True.
        """
        self.root_dir = Path(root_dir)
        self.config = config
        self.transform_config = transform or {} # Store transform config
        self.is_train = is_train
        
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
        
    def _load_audio(self, audio_path: str) -> torch.Tensor: # 返回 Tensor
        """加載並預處理音頻
        
        Args:
            audio_path (str): 音頻文件路徑
            
        Returns:
            torch.Tensor: 處理後的音頻數據張量
        """
        try:
            # 使用 soundfile 加載，更穩定
            audio, orig_sr = sf.read(audio_path, dtype='float32') 
            
            # 轉換為單聲道（如果需要）
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            # 重採樣到目標採樣率 (使用 librosa)
            if orig_sr != self.sample_rate:
                 with warnings.catch_warnings(): # 抑制 librosa 可能的警告
                    warnings.simplefilter("ignore")
                    audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sample_rate)

            # 確保音頻長度一致
            if len(audio) > self.max_len:
                # 如果音頻過長，截斷
                audio = audio[:self.max_len]
            elif len(audio) < self.max_len:
                # 如果音頻過短，填充
                padding = np.zeros(self.max_len - len(audio), dtype=np.float32)
                audio = np.concatenate((audio, padding))
            
            # 標準化音頻
            if self.normalize_audio:
                 # 檢查 audio 是否全為 0
                if np.all(audio == 0):
                    logger.warning(f"音頻文件 {audio_path} 標準化前全為 0")
                else:
                    max_val = np.max(np.abs(audio))
                    if max_val > 1e-8: # 避免除以零
                        audio = audio / max_val
                    else:
                        logger.warning(f"音頻文件 {audio_path} 標準化時最大絕對值過小 ({max_val})，跳過標準化")

            return torch.from_numpy(audio) # 轉換為 Tensor
            
        except Exception as e:
            logger.error(f"加載音頻文件 {audio_path} 時出錯: {str(e)}")
            # 返回全零張量
            return torch.zeros(self.max_len, dtype=torch.float32)
    
    def __len__(self) -> int:
        """返回數據集大小（基於過濾後的樣本）"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """獲取指定索引的樣本（基於過濾後的樣本）
        
        Args:
            idx (int): 樣本索引 (相對於 self.samples)
            
        Returns:
            Dict[str, Any]: 包含audio、features和label的字典
        """
        if idx >= len(self.samples):
             # 增加保護，理論上不應發生，但有助於調試
            logger.error(f"索引 {idx} 超出範圍，樣本數為 {len(self.samples)}")
            raise IndexError(f"索引 {idx} 超出範圍，樣本數為 {len(self.samples)}")

        sample_info = self.samples[idx]
        
        # 加載音頻
        audio_tensor = self._load_audio(sample_info['audio_path'])

        # 應用數據增強 (如果配置了)
        if self.is_train and self.transforms:
             audio_tensor = self._apply_transforms(audio_tensor, self.transforms.get('audio', []))
        
        # 提取特徵 (如果啟用)
        features_tensor = None
        if self.enable_feature_extraction:
            try:
                features_tensor = extract_features_from_config(audio_tensor, self.config)
                logger.debug(f"從音頻 {sample_info['audio_path']} 提取特徵，形狀為 {features_tensor.shape}")
            except Exception as e:
                logger.error(f"特徵提取失敗: {e}，將使用原始音頻")
                features_tensor = audio_tensor.unsqueeze(0)  # 確保是 2D

        # 返回字典
        output_dict = {
            'audio': audio_tensor, # Tensor
            'score': sample_info['score'], # Tensor (float32)
            'label': sample_info['label'], # Tensor (long or float32)
            'patient_id': sample_info['patient_id'], # str
            'selection': sample_info.get('selection', ''), # str
            'selection_type': sample_info.get('selection_type', ''), # str
        }
        
        # 如果特徵提取成功，則添加到輸出中
        if features_tensor is not None:
            output_dict['features'] = features_tensor
            
        # 添加額外信息
        if 'custom_class' in sample_info:
            output_dict['custom_class'] = sample_info['custom_class']
        if 'patient_group' in sample_info:
             output_dict['patient_group'] = sample_info['patient_group']
            
        return output_dict
    
    def set_transforms(self, transform_config: Dict[str, Any]):
        """設置數據轉換 (根據 is_train 狀態)
        
        Args:
            transform_config (Dict[str, Any]): 轉換配置
        """
        self.transforms = {'audio': []} # 重置
        
        if not self.is_train: # 非訓練模式不應用增強
             return

        # 從存儲的 transform_config 中獲取配置
        config = self.transform_config.get('audio', {}) if self.transform_config else {}

        # 音頻轉換 (只在訓練時應用)
        if config.get('add_noise', False):
            noise_level = config.get('noise_level', 0.005)
            self.transforms['audio'].append(
                lambda x: x + torch.randn_like(x) * noise_level
            )
        
        if config.get('time_shift', False):
            shift_range = config.get('shift_range', 0.1)  # 最大偏移比例
            max_shift = int(self.max_len * shift_range)
            
            def time_shift(x):
                 if max_shift == 0: return x # 如果 max_len 或 shift_range 為 0
                 shift = random.randint(-max_shift, max_shift)
                 if shift > 0:
                     # 向右移，左邊補零
                     shifted = torch.cat([torch.zeros(shift, dtype=x.dtype, device=x.device), x[:-shift]])
                 elif shift < 0:
                     # 向左移，右邊補零
                     shifted = torch.cat([x[-shift:], torch.zeros(-shift, dtype=x.dtype, device=x.device)])
                 else:
                     shifted = x
                 return shifted

            self.transforms['audio'].append(time_shift)
        
        if config.get('time_stretch', False):
             # 注意：時間拉伸比較耗時，且需要在numpy上操作
             # 這裡提供一個簡化的佔位符，實際實現可能需要更複雜的處理
             # 或者考慮使用 torchaudio.transforms.TimeStretch
             stretch_rate = config.get('stretch_rate', (0.9, 1.1))
             logger.warning("時間拉伸轉換 (time_stretch) 目前是簡化實現，可能影響性能或效果。")
             
             def time_stretch(x):
                 # 簡化：隨機選擇一個速度因子，但不實際拉伸，避免複雜性
                 rate = random.uniform(stretch_rate[0], stretch_rate[1])
                 # Placeholder: return x # 實際應用中需要替換為拉伸邏輯
                 return x 
             self.transforms['audio'].append(time_stretch)
             
        logger.info(f"已為 {'訓練' if self.is_train else '驗證/測試'} 模式設置 {len(self.transforms['audio'])} 個音頻轉換。")

    def _apply_transforms(self, x: torch.Tensor, transforms: List[Callable]) -> torch.Tensor:
        """應用轉換列表到數據
        
        Args:
            x (torch.Tensor): 輸入張量
            transforms (List[Callable]): 轉換函數列表
            
        Returns:
            torch.Tensor: 轉換後的張量
        """
        for transform in transforms:
             try:
                 x = transform(x)
             except Exception as e:
                 logger.error(f"應用轉換 {transform.__name__ if hasattr(transform, '__name__') else transform} 時出錯: {e}")
        return x
    
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
    Description: Minimal Executable Unit for audio_dataset.py，檢查dataset能否正確初始化與取樣，並測試錯誤路徑時的報錯行為。
    Args: None
    Returns: None
    References: 無
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    from utils.config_loader import load_config # 需要 config_loader

    # 測試正常初始化
    try:
        # 使用一個存在的測試配置或創建一個簡單的 dummy config
        # config_path = Path(__file__).parent.parent / 'config' / 'boss_custom_classification.yaml'
        # if not config_path.exists():
        #     raise FileNotFoundError("測試需要 config/boss_custom_classification.yaml")
        # config = load_config(str(config_path))

        # 或者創建一個 dummy config
        dummy_config = {
            "data": {
                "type": "audio",
                "source": {
                    # 使用一個保證存在的路徑，例如 tests/dataloader_test/dataset_test
                    "wav_dir": str(Path(__file__).parent.parent / 'tests' / 'dataloader_test' / 'dataset_test')
                 },
                "preprocessing": {
                    "audio": {"sample_rate": 16000, "duration": 5, "normalize": True},
                    "features": {
                        "method": "mel_spectrogram",
                        "n_mels": 128,
                        "n_fft": 1024,
                        "hop_length": 512,
                        "log_mel": True,
                        "target_duration_sec": 5,
                        "scaling_method": "standard"
                    }
                },
                "filtering": { # 添加空的 filtering 配置避免 KeyErrors
                     "task_type": "classification",
                     "custom_classification": {"enabled": False}, # 禁用自定義分類測試
                     "score_thresholds": {},
                     "class_config": {},
                     "subject_source": {}
                },
                "splits": {} # 添加空的 splits
            }
        }

        # 確保測試數據目錄存在
        test_data_dir = Path(dummy_config["data"]["source"]["wav_dir"])
        if not test_data_dir.exists():
             logger.error(f"測試數據目錄不存在: {test_data_dir}")
             raise FileNotFoundError(f"測試數據目錄不存在: {test_data_dir}")
        if not any(test_data_dir.iterdir()):
             logger.warning(f"測試數據目錄為空: {test_data_dir}")
             # 如果目錄為空，可能無法進行有效測試，但仍可測試初始化

        logger.info(f"使用配置進行測試: {dummy_config}")
        dataset = AudioDataset(root_dir=dummy_config["data"]["source"]["wav_dir"], config=dummy_config)
        
        logger.info(f"數據集初始化成功，樣本數: {len(dataset)}")
        
        if len(dataset) > 0:
            logger.info("嘗試獲取第一個樣本...")
            sample = dataset[0]
            logger.info(f"成功獲取樣本，鍵: {sample.keys()}")
            logger.info(f"音頻張量形狀: {sample['audio'].shape}")
            if 'features' in sample:
                logger.info(f"特徵張量形狀: {sample['features'].shape}")
            logger.info(f"標籤: {sample['label']}")
            
            logger.info("嘗試拆分數據集...")
            train_idx, val_idx, test_idx = dataset.split_by_patient()
            logger.info(f"數據集拆分成功 - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        else:
            logger.warning("數據集為空，無法測試獲取樣本和拆分功能。請檢查數據路徑和過濾條件。")

    except Exception as e:
        logger.error(f"AudioDataset 初始化或基本操作測試失敗: {e}", exc_info=True)

    # 測試錯誤路徑
    try:
        logger.info("測試無效根目錄...")
        error_config = { "data": { "source": {"wav_dir": "non_existent_directory"}, "preprocessing": {}, "filtering": {}, "splits": {} } }
        dataset_err = AudioDataset(root_dir="non_existent_directory", config=error_config)
        logger.info(f"無效路徑測試，數據集長度: {len(dataset_err)}") # 應該為 0
        assert len(dataset_err) == 0, "對於無效路徑，數據集長度應為 0"
        logger.info("無效路徑測試通過。")
    except Exception as e:
        logger.error(f"AudioDataset 無效路徑測試失敗: {e}", exc_info=True)