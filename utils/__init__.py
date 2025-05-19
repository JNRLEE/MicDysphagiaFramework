"""
工具函數模組
提供各種輔助功能支持整個框架的運行：

1. 配置系統：
   - 統一配置加載與解析

2. 數據管理：
   - 數據索引集加載
   - 患者信息擷取
   - 音頻處理與特徵提取

3. 模型管理：
   - 模型保存管理
   - 訓練日誌記錄

4. 回調接口：
   - 標準化回調機制
   - 訓練過程監控點
"""

# 配置相關
from .config_loader import load_config

# 數據索引與患者信息
from .data_index_loader import DataIndexLoader
from .patient_info_loader import load_patient_info, list_patient_dirs

# 存檔與加載
from .save_manager import SaveManager

# 回調系統
from .callback_interface import CallbackInterface

# 音頻處理與特徵提取
from .audio_feature_extractor import extract_features_from_config, preprocess_audio, extract_mel_spectrogram

# 常量定義
from .constants import SELECTION_TYPES, CLASS_LABELS, LABEL_TO_INDEX, INDEX_TO_LABEL

__all__ = [
    # 配置相關
    'load_config',
    
    # 數據索引與患者信息
    'DataIndexLoader',
    'load_patient_info',
    'list_patient_dirs',
    
    # 存檔與加載
    'SaveManager',
    
    # 回調系統
    'CallbackInterface',
    
    # 音頻處理與特徵提取
    'extract_features_from_config',
    'preprocess_audio',
    'extract_mel_spectrogram',
    
    # 常量定義
    'SELECTION_TYPES',
    'CLASS_LABELS',
    'LABEL_TO_INDEX',
    'INDEX_TO_LABEL'
] 