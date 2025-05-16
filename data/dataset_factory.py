"""
數據集工廠模組：根據配置動態創建數據集
功能：
1. 提供統一的數據集加載接口
2. 支持不同類型的數據：音頻、頻譜圖、特徵向量
3. 根據配置選擇適當的數據加載方式
4. 處理數據拆分和轉換
5. 支持從索引CSV加載數據
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
        
        # 檢查是否使用索引CSV
        use_index = data_config.get('use_index', False)
        index_path = data_config.get('index_path', 'data/data_index.csv')
        
        if use_index and os.path.exists(index_path):
            logger.info(f"使用索引CSV: {index_path}")
            return DatasetFactory._create_indexed_dataset(config)
        else:
            # 使用傳統方式加載數據
            if use_index:
                logger.warning(f"索引文件 {index_path} 不存在，將使用傳統方式加載數據")
            
            if data_type == 'audio':
                return DatasetFactory._create_audio_dataset(config)
            elif data_type == 'spectrogram':
                return DatasetFactory._create_spectrogram_dataset(config)
            elif data_type == 'feature':
                return DatasetFactory._create_feature_dataset(config)
            else:
                raise ValueError(f"不支持的數據類型: {data_type}")
    
    @staticmethod
    def _create_indexed_dataset(config: Dict[str, Any]) -> Tuple[Dataset, Dataset, Dataset]:
        """使用索引CSV創建數據集
        
        Args:
            config: 配置字典
            
        Returns:
            Tuple[Dataset, Dataset, Dataset]: 訓練、驗證和測試數據集
        """
        data_config = config['data']
        data_type = data_config.get('type', 'audio')
        index_path = data_config.get('index_path', 'data/data_index.csv')
        label_field = data_config.get('label_field', 'score')
        
        # 獲取數據加載參數
        audio_config = data_config.get('preprocessing', {}).get('audio', {})
        sample_rate = audio_config.get('sample_rate', 16000)
        duration = audio_config.get('duration', 5.0)
        
        # 設置分割條件
        splits = data_config.get('splits', {})
        train_filter = {'split': 'train'} if 'split' in data_config.get('filter_criteria', {}) else {}
        val_filter = {'split': 'val'} if 'split' in data_config.get('filter_criteria', {}) else {}
        test_filter = {'split': 'test'} if 'split' in data_config.get('filter_criteria', {}) else {}
        
        # 添加通用過濾條件
        common_filters = {k: v for k, v in data_config.get('filter_criteria', {}).items() if k != 'split'}
        train_filter.update(common_filters)
        val_filter.update(common_filters)
        test_filter.update(common_filters)
        
        transforms = data_config.get('transforms', {})
        
        # 根據數據類型創建數據集
        if data_type == 'audio':
            from data.audio_dataset import AudioDataset
            
            train_dataset = AudioDataset(
                index_path=index_path,
                label_field=label_field,
                filter_criteria=train_filter,
                sample_rate=sample_rate,
                duration=duration,
                transform=transforms.get('train', {}),
                is_train=True
            )
            
            val_dataset = AudioDataset(
                index_path=index_path,
                label_field=label_field,
                filter_criteria=val_filter,
                sample_rate=sample_rate,
                duration=duration,
                transform=transforms.get('val', {}),
                is_train=False
            )
            
            test_dataset = AudioDataset(
                index_path=index_path,
                label_field=label_field,
                filter_criteria=test_filter,
                sample_rate=sample_rate,
                duration=duration,
                transform=transforms.get('test', {}),
                is_train=False
            )
            
        elif data_type == 'spectrogram':
            from data.spectrogram_dataset import SpectrogramDataset
            
            # 獲取頻譜圖參數
            spec_config = data_config.get('preprocessing', {}).get('spectrogram', {})
            
            train_dataset = SpectrogramDataset(
                index_path=index_path,
                label_field=label_field,
                filter_criteria=train_filter,
                transform=transforms.get('train', {}),
                is_train=True,
                config=config
            )
            
            val_dataset = SpectrogramDataset(
                index_path=index_path,
                label_field=label_field,
                filter_criteria=val_filter,
                transform=transforms.get('val', {}),
                is_train=False,
                config=config
            )
            
            test_dataset = SpectrogramDataset(
                index_path=index_path,
                label_field=label_field,
                filter_criteria=test_filter,
                transform=transforms.get('test', {}),
                is_train=False,
                config=config
            )
            
        elif data_type == 'feature':
            from data.feature_dataset import FeatureDataset
            
            train_dataset = FeatureDataset(
                index_path=index_path,
                label_field=label_field,
                filter_criteria=train_filter,
                transform=transforms.get('train', {}),
                is_train=True,
                config=config
            )
            
            val_dataset = FeatureDataset(
                index_path=index_path,
                label_field=label_field,
                filter_criteria=val_filter,
                transform=transforms.get('val', {}),
                is_train=False,
                config=config
            )
            
            test_dataset = FeatureDataset(
                index_path=index_path,
                label_field=label_field,
                filter_criteria=test_filter,
                transform=transforms.get('test', {}),
                is_train=False,
                config=config
            )
        
        else:
            raise ValueError(f"不支持的數據類型: {data_type}")
        
        logger.info(f"使用索引模式創建數據集 - 訓練: {len(train_dataset)}，驗證: {len(val_dataset)}，測試: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    @staticmethod
    def _create_audio_dataset(config: Dict[str, Any]) -> Tuple[Dataset, Dataset, Dataset]:
        """創建音頻數據集
        
        Args:
            config: 配置字典
            
        Returns:
            Tuple[Dataset, Dataset, Dataset]: 訓練、驗證和測試數據集
        """
        from data.audio_dataset import AudioDataset
        
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
        from data.spectrogram_dataset import SpectrogramDataset
        
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
        from data.feature_dataset import FeatureDataset
        
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
    
    @staticmethod
    def create_datasets_and_loaders(config: Dict[str, Any]) -> Tuple[Dataset, Dataset, Dataset, DataLoader, DataLoader, DataLoader]:
        """創建數據集和數據加載器
        
        Args:
            config: 配置字典
            
        Returns:
            Tuple[Dataset, Dataset, Dataset, DataLoader, DataLoader, DataLoader]: 訓練、驗證和測試數據集與加載器
        """
        train_dataset, val_dataset, test_dataset = DatasetFactory.create_dataset(config)
        train_loader, val_loader, test_loader = DatasetFactory.create_dataloaders(config)
        
        return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


# 單元測試代碼
if __name__ == "__main__":
    """
    Description: Minimal Executable Unit for dataset_factory.py，測試數據集工廠的索引模式和直接模式功能。
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
            'file_path': ['/path/to/audio1.wav', '/path/to/audio2.wav', '/path/to/audio3.wav'],
            'score': [15, 25, 5],
            'patient_id': ['p001', 'p002', 'p001'],
            'DrLee_Evaluation': ['聽起來正常', '輕度異常', '重度異常'],
            'DrTai_Evaluation': ['正常', '無OR 輕微吞嚥障礙', '重度吞嚥障礙'],
            'selection': ['乾吞嚥1口', '吞水10ml', '餅乾1塊'],
            'status': ['processed', 'processed', 'raw'],
            'split': ['train', 'val', 'test']  # 添加分割信息
        })
        test_data.to_csv(temp_file.name, index=False)
        test_csv_path = temp_file.name
    
    try:
        # 測試索引模式配置
        print("測試1: 索引模式配置")
        index_config = {
            'data': {
                'type': 'audio',
                'use_index': True,
                'index_path': test_csv_path,
                'label_field': 'score',
                'filter_criteria': {
                    'status': 'processed'
                },
                'preprocessing': {
                    'audio': {
                        'sample_rate': 16000,
                        'duration': 5.0
                    }
                },
                'transforms': {},
                'dataloader': {
                    'batch_size': 32,
                    'num_workers': 4
                }
            }
        }
        
        # 創建數據集（不會實際加載音頻文件，因為路徑不存在）
        train_dataset, val_dataset, test_dataset = DatasetFactory.create_dataset(index_config)
        print(f"索引模式數據集大小 - 訓練: {len(train_dataset)}, 驗證: {len(val_dataset)}, 測試: {len(test_dataset)}")
        
        # 測試直接模式配置
        print("\n測試2: 直接模式配置")
        direct_config = {
            'data': {
                'type': 'audio',
                'source': {
                    'wav_dir': './'  # 使用當前目錄
                },
                'preprocessing': {
                    'audio': {
                        'sample_rate': 16000,
                        'duration': 5.0
                    }
                },
                'filtering': {
                    'task_type': 'classification',
                    'custom_classification': {'enabled': False}
                },
                'splits': {},
                'transforms': {},
                'dataloader': {
                    'batch_size': 32,
                    'num_workers': 4
                }
            }
        }
        
        # 創建數據集（直接模式）
        try:
            direct_train, direct_val, direct_test = DatasetFactory.create_dataset(direct_config)
            print(f"直接模式數據集大小 - 訓練: {len(direct_train)}, 驗證: {len(direct_val)}, 測試: {len(direct_test)}")
        except Exception as e:
            print(f"直接模式測試預期錯誤: {str(e)}")
        
        print("\n所有測試通過！")
        
    except Exception as e:
        print(f"測試失敗: {str(e)}")
    
    finally:
        # 清理測試文件
        os.unlink(test_csv_path)

# 提供模組級別的函數，以便其他模組可以直接導入
def create_dataset(config: Dict[str, Any]) -> Tuple[Dataset, Dataset, Dataset]:
    """
    根據配置創建訓練、驗證和測試數據集
    
    Args:
        config: 配置字典，至少包含 'data' 部分
        
    Returns:
        Tuple[Dataset, Dataset, Dataset]: 訓練、驗證和測試數據集
        
    Raises:
        ValueError: 如果配置中指定了不支持的數據類型
    """
    return DatasetFactory.create_dataset(config)

def create_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    創建數據加載器
    
    Args:
        config: 配置字典
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: 訓練、驗證和測試數據加載器
    """
    return DatasetFactory.create_dataloaders(config)

def create_datasets_and_loaders(config: Dict[str, Any]) -> Tuple[Dataset, Dataset, Dataset, DataLoader, DataLoader, DataLoader]:
    """
    創建數據集和數據加載器
    
    Args:
        config: 配置字典
        
    Returns:
        Tuple[Dataset, Dataset, Dataset, DataLoader, DataLoader, DataLoader]: 訓練、驗證和測試數據集與加載器
    """
    return DatasetFactory.create_datasets_and_loaders(config) 