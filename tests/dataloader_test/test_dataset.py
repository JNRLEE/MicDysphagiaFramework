"""
數據集測試腳本：測試AudioDataset、FeatureDataset和SpectrogramDataset的功能
"""

import os
import sys
import json
import logging
import yaml
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np

# 設置日誌
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 導入數據集類
from MicDysphagiaFramework.data.audio_dataset import AudioDataset
from MicDysphagiaFramework.data.feature_dataset import FeatureDataset
from MicDysphagiaFramework.data.spectrogram_dataset import SpectrogramDataset

def create_test_config() -> Dict[str, Any]:
    """創建測試配置
    
    Returns:
        Dict[str, Any]: 測試配置
    """
    return {
        'data': {
            'preprocessing': {
                'audio': {
                    'sample_rate': 16000,
                    'duration': 5.0,
                    'normalize': True,
                    'target_file': 'Probe0_RX_IN_TDM4CH0.wav'
                },
                'spectrogram': {
                    'type': 'mel',
                    'n_fft': 1024,
                    'hop_length': 512,
                    'n_mels': 64
                }
            },
            'features': {
                'type': 'npz',
                'names': [],
                'label': 'score',
                'patient_id_column': 'patient_id',
                'normalize': True
            }
        },
        'model': {
            'parameters': {
                'is_classification': False
            }
        },
        'resize': (224, 224)
    }

def test_audio_dataset(data_path: str, config: Dict[str, Any]):
    """測試AudioDataset
    
    Args:
        data_path (str): 數據路徑
        config (Dict[str, Any]): 配置
    """
    logger.info(f"\n{'='*20} 測試 AudioDataset {'='*20}")
    try:
        # 創建數據集
        dataset = AudioDataset(data_path, config)
        logger.info(f"成功加載 {len(dataset)} 個音頻樣本")
        
        # 檢查第一個樣本
        if len(dataset) > 0:
            sample = dataset[0]
            logger.info(f"第一個樣本: 患者ID={sample['patient_id']}, 分數={sample['score']}, 選擇={sample['selection']}")
            logger.info(f"音頻張量形狀: {sample['audio'].shape}")
            
            # 測試拆分功能
            if len(dataset) >= 3:
                train_indices, val_indices, test_indices = dataset.split_by_patient(
                    train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42
                )
                logger.info(f"數據集拆分: 訓練={len(train_indices)}, 驗證={len(val_indices)}, 測試={len(test_indices)}")
        
        logger.info("音頻數據集測試成功!")
    except Exception as e:
        logger.error(f"音頻數據集測試失敗: {str(e)}")
        import traceback
        traceback.print_exc()

def test_feature_dataset(data_path: str, config: Dict[str, Any]):
    """測試FeatureDataset
    
    Args:
        data_path (str): 數據路徑
        config (Dict[str, Any]): 配置
    """
    logger.info(f"\n{'='*20} 測試 FeatureDataset {'='*20}")
    try:
        # 查找NPZ文件
        npz_files = list(Path(data_path).glob("**/*.npz"))
        if not npz_files:
            logger.warning(f"在 {data_path} 中沒有找到NPZ文件，無法測試FeatureDataset")
            return
            
        logger.info(f"找到 {len(npz_files)} 個NPZ文件")
        
        # 創建數據集
        dataset = FeatureDataset(data_path, config)
        logger.info(f"成功加載 {len(dataset)} 個特徵樣本")
        
        # 檢查第一個樣本
        if len(dataset) > 0:
            sample = dataset[0]
            logger.info(f"第一個樣本: 患者ID={sample['patient_id']}, 分數={sample['score']}")
            logger.info(f"特徵張量形狀: {sample['features'].shape}")
            
            # 獲取特徵維度
            feature_dim = dataset.get_feature_dim()
            logger.info(f"特徵維度: {feature_dim}")
            
            # 測試拆分功能
            if len(dataset) >= 3:
                train_indices, val_indices, test_indices = dataset.split_by_patient(
                    train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42
                )
                logger.info(f"數據集拆分: 訓練={len(train_indices)}, 驗證={len(val_indices)}, 測試={len(test_indices)}")
        
        logger.info("特徵數據集測試成功!")
    except Exception as e:
        logger.error(f"特徵數據集測試失敗: {str(e)}")
        import traceback
        traceback.print_exc()

def test_spectrogram_dataset(data_path: str, config: Dict[str, Any]):
    """測試SpectrogramDataset
    
    Args:
        data_path (str): 數據路徑
        config (Dict[str, Any]): 配置
    """
    logger.info(f"\n{'='*20} 測試 SpectrogramDataset {'='*20}")
    try:
        # 查找PNG文件
        png_files = list(Path(data_path).glob("**/*.png"))
        if not png_files:
            logger.warning(f"在 {data_path} 中沒有找到PNG文件，嘗試使用示例測試")
            
        # 創建數據集
        dataset = SpectrogramDataset(data_path, config)
        logger.info(f"成功加載 {len(dataset)} 個頻譜圖樣本")
        
        # 檢查第一個樣本
        if len(dataset) > 0:
            sample = dataset[0]
            logger.info(f"第一個樣本: 患者ID={sample['patient_id']}, 分數={sample['score']}")
            logger.info(f"圖像張量形狀: {sample['image'].shape}")
            
            # 測試拆分功能
            if len(dataset) >= 3:
                train_indices, val_indices, test_indices = dataset.split_by_patient(
                    train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42
                )
                logger.info(f"數據集拆分: 訓練={len(train_indices)}, 驗證={len(val_indices)}, 測試={len(test_indices)}")
        
        logger.info("頻譜圖數據集測試成功!")
    except Exception as e:
        logger.error(f"頻譜圖數據集測試失敗: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """主函數"""
    # 獲取命令行參數
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        # 默認數據路徑
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                               "WavData", "test")
    
    logger.info(f"使用數據路徑: {data_path}")
    
    # 創建測試配置
    config = create_test_config()
    
    # 測試音頻數據集
    test_audio_dataset(data_path, config)
    
    # 測試特徵數據集
    test_feature_dataset(data_path, config)
    
    # 測試頻譜圖數據集
    test_spectrogram_dataset(data_path, config)

if __name__ == "__main__":
    main() 