"""
數據集工廠模組：根據配置動態創建數據集
功能：
1. 提供統一的數據集加載接口
2. 支持不同類型的數據：音頻、頻譜圖、特徵向量
3. 根據配置選擇適當的數據加載方式
4. 處理數據拆分和轉換
"""

import os
import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

logger = logging.getLogger(__name__)

class DatasetFactory:
    """數據集工廠類，根據配置創建數據集和數據加載器"""
    
    @staticmethod
    def create_dataset(config: Dict[str, Any]) -> Tuple[Dataset, Dataset, Dataset]:
        """根據配置創建訓練、驗證和測試數據集
        
        Args:
            config: 配置字典，至少包含 'data' 部分
            
        Returns:
            Tuple[Dataset, Dataset, Dataset]: 訓練、驗證和測試數據集
            
        Raises:
            ValueError: 如果配置中指定了不支持的數據類型
        """
        data_config = config['data']
        data_type = data_config.get('type', 'audio')
        
        if data_type == 'audio':
            return DatasetFactory._create_audio_dataset(config)
        elif data_type == 'spectrogram':
            return DatasetFactory._create_spectrogram_dataset(config)
        elif data_type == 'feature':
            return DatasetFactory._create_feature_dataset(config)
        else:
            raise ValueError(f"不支持的數據類型: {data_type}")
    
    @staticmethod
    def _create_audio_dataset(config: Dict[str, Any]) -> Tuple[Dataset, Dataset, Dataset]:
        """創建音頻數據集
        
        Args:
            config: 配置字典
            
        Returns:
            Tuple[Dataset, Dataset, Dataset]: 訓練、驗證和測試數據集
        """
        from .audio_dataset import AudioDataset
        
        data_config = config['data']
        wav_dir = data_config['source']['wav_dir']
        
        # 創建數據集
        full_dataset = AudioDataset(
            root_dir=wav_dir,
            config=config
        )
        
        # 拆分數據集
        return DatasetFactory._split_dataset(full_dataset, config)
    
    @staticmethod
    def _create_spectrogram_dataset(config: Dict[str, Any]) -> Tuple[Dataset, Dataset, Dataset]:
        """創建頻譜圖數據集
        
        Args:
            config: 配置字典
            
        Returns:
            Tuple[Dataset, Dataset, Dataset]: 訓練、驗證和測試數據集
        """
        from .spectrogram_dataset import SpectrogramDataset
        
        data_config = config['data']
        spectrogram_dir = data_config['source']['spectrogram_dir']
        
        # 如果沒有指定頻譜圖目錄但指定了音頻目錄，則使用音頻生成頻譜圖
        if not spectrogram_dir and 'wav_dir' in data_config['source'] and data_config['source']['wav_dir']:
            logger.info("未提供頻譜圖目錄，將從音頻生成頻譜圖")
            wav_dir = data_config['source']['wav_dir']
            
            # 創建頻譜圖保存目錄
            save_dir = data_config.get('preprocessing', {}).get('spectrogram', {}).get('save_dir')
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            
            # 創建數據集
            full_dataset = SpectrogramDataset(
                root_dir=wav_dir,
                save_dir=save_dir,
                config=config,
                generate_spectrograms=True
            )
        else:
            # 使用已有的頻譜圖
            full_dataset = SpectrogramDataset(
                root_dir=spectrogram_dir,
                config=config,
                generate_spectrograms=False
            )
        
        # 拆分數據集
        return DatasetFactory._split_dataset(full_dataset, config)
    
    @staticmethod
    def _create_feature_dataset(config: Dict[str, Any]) -> Tuple[Dataset, Dataset, Dataset]:
        """創建特徵數據集
        
        Args:
            config: 配置字典
            
        Returns:
            Tuple[Dataset, Dataset, Dataset]: 訓練、驗證和測試數據集
        """
        from .feature_dataset import FeatureDataset
        
        data_config = config['data']
        feature_dir = data_config['source']['feature_dir']
        
        # 創建數據集
        full_dataset = FeatureDataset(
            data_path=feature_dir,
            config=config
        )
        
        # 拆分數據集
        return DatasetFactory._split_dataset(full_dataset, config)
    
    @staticmethod
    def _split_dataset(dataset: Dataset, config: Dict[str, Any]) -> Tuple[Dataset, Dataset, Dataset]:
        """將數據集拆分為訓練、驗證和測試集
        
        Args:
            dataset: 完整數據集
            config: 配置字典
            
        Returns:
            Tuple[Dataset, Dataset, Dataset]: 訓練、驗證和測試數據集
        """
        from torch.utils.data import random_split, Subset
        
        data_config = config['data']
        splits = data_config.get('splits', {})
        
        train_ratio = splits.get('train_ratio', 0.7)
        val_ratio = splits.get('val_ratio', 0.15)
        test_ratio = splits.get('test_ratio', 0.15)
        
        # 確保比例總和為 1
        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total
        
        # 計算每個集合的樣本數
        dataset_size = len(dataset)
        train_size = int(dataset_size * train_ratio)
        val_size = int(dataset_size * val_ratio)
        test_size = dataset_size - train_size - val_size
        
        # 檢查是否按患者 ID 拆分
        split_by_patient = splits.get('split_by_patient', True)
        
        if hasattr(dataset, 'split_by_patient') and split_by_patient:
            logger.info("按患者 ID 拆分數據集")
            train_indices, val_indices, test_indices = dataset.split_by_patient(
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=splits.get('split_seed', 42)
            )
            
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
            test_dataset = Subset(dataset, test_indices)
        else:
            # 隨機拆分
            logger.info("隨機拆分數據集")
            train_dataset, val_dataset, test_dataset = random_split(
                dataset, 
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(splits.get('split_seed', 42))
            )
        
        logger.info(f"數據集拆分完成: 訓練集 {len(train_dataset)}，驗證集 {len(val_dataset)}，測試集 {len(test_dataset)}")
        
        # 為每個數據集設置適當的轉換
        if hasattr(dataset, 'set_transforms') and 'transforms' in data_config:
            transforms = data_config['transforms']
            
            if hasattr(train_dataset, 'dataset'):
                # 處理 Subset 對象
                train_dataset.dataset.set_transforms(transforms.get('train', {}))
                val_dataset.dataset.set_transforms(transforms.get('val', {}))
                test_dataset.dataset.set_transforms(transforms.get('test', {}))
            else:
                # 直接處理 Dataset 對象
                train_dataset.set_transforms(transforms.get('train', {}))
                val_dataset.set_transforms(transforms.get('val', {}))
                test_dataset.set_transforms(transforms.get('test', {}))
        
        return train_dataset, val_dataset, test_dataset
    
    @staticmethod
    def create_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """創建數據加載器
        
        Args:
            config: 配置字典
            
        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: 訓練、驗證和測試數據加載器
        """
        # 創建數據集
        train_dataset, val_dataset, test_dataset = DatasetFactory.create_dataset(config)
        
        # 獲取數據加載器配置
        dataloader_config = config['data'].get('dataloader', {})
        batch_size = dataloader_config.get('batch_size', 32)
        num_workers = dataloader_config.get('num_workers', 4)
        pin_memory = dataloader_config.get('pin_memory', True)
        drop_last = dataloader_config.get('drop_last', False)
        
        # 創建數據加載器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        
        return train_loader, val_loader, test_loader


def create_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """便捷函數，創建數據加載器
    
    Args:
        config: 配置字典
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: 訓練、驗證和測試數據加載器
    """
    return DatasetFactory.create_dataloaders(config)

def create_datasets_and_loaders(config: Dict[str, Any]) -> Tuple[Dataset, Dataset, Dataset, DataLoader, DataLoader, DataLoader]:
    """創建數據集和數據加載器
    
    此函數結合了 create_dataset 和 create_dataloaders 的功能，
    返回數據集和數據加載器，以符合 run_experiments.py 的調用要求。
    
    Args:
        config: 配置字典
        
    Returns:
        Tuple[Dataset, Dataset, Dataset, DataLoader, DataLoader, DataLoader]: 
            訓練數據集、驗證數據集、測試數據集、訓練數據加載器、驗證數據加載器、測試數據加載器
    """
    # 創建數據集
    train_dataset, val_dataset, test_dataset = DatasetFactory.create_dataset(config)
    
    # 創建數據加載器
    train_loader, val_loader, test_loader = DatasetFactory.create_dataloaders(config)
    
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader

# 中文註解：這是dataset_factory.py的Minimal Executable Unit，檢查能否正確初始化與錯誤type/路徑時的優雅報錯
if __name__ == "__main__":
    """
    Description: Minimal Executable Unit for dataset_factory.py，檢查create_dataset能否正確初始化與錯誤type/路徑時的優雅報錯。
    Args: None
    Returns: None
    References: 無
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    # 不要再import自己
    # from data.audio_dataset import AudioDataset

    try:
        dummy_config = {
            "data": {
                "type": "not_exist_type",
                "source": {"wav_dir": "not_exist_dir"}
            }
        }
        DatasetFactory.create_dataset(dummy_config)
    except Exception as e:
        print(f"遇到錯誤type時的報錯（預期行為）: {e}")

    # 測試正確type但錯誤路徑
    try:
        dummy_config = {
            "data": {
                "type": "audio",
                "source": {"wav_dir": "not_exist_dir"}
            }
        }
        train, val, test = DatasetFactory.create_dataset(dummy_config)
        print(f"train: {len(train)}, val: {len(val)}, test: {len(test)}")
    except Exception as e:
        print(f"遇到錯誤路徑時的報錯（預期行為）: {e}") 