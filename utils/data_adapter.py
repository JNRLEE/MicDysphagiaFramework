"""
數據適配器模組：處理不同數據格式與模型輸入需求之間的轉換
功能：
1. 檢測數據與模型之間的格式不匹配
2. 提供音頻到頻譜圖的轉換
3. 提供特徵向量的維度調整
4. 支持批量處理的數據轉換
"""

import torch
import numpy as np
import logging
from typing import Dict, Any, Tuple, Union, List, Optional
import librosa
import torchaudio
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

class DataAdapter:
    """數據適配器類，用於處理不同數據格式與模型輸入需求之間的轉換"""
    
    @staticmethod
    def adapt_batch(batch: Dict[str, Any], model_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """根據模型類型調整批次數據格式
        
        Args:
            batch: 數據批次，字典格式
            model_type: 模型類型，如 'swin_transformer', 'fcnn', 'cnn', 'resnet'
            config: 配置字典
            
        Returns:
            Dict[str, Any]: 調整後的批次
        """
        # 複製批次，避免修改原始數據
        adapted_batch = {k: v for k, v in batch.items()}
        
        # 獲取主要數據字段
        data_fields = ['audio', 'spectrogram', 'features']
        main_data = None
        data_type = None
        
        for field in data_fields:
            if field in batch and batch[field] is not None:
                main_data = batch[field]
                data_type = field
                break
        
        if main_data is None or data_type is None:
            logger.warning("無法確定批次中的主要數據字段")
            return batch
        
        # 檢查數據與模型是否兼容，並進行必要的轉換
        if model_type in ['swin_transformer', 'cnn', 'resnet']:
            # 這些模型需要圖像數據 [B, C, H, W]
            if data_type == 'audio':
                # 轉換音頻為頻譜圖
                logger.info("將音頻轉換為頻譜圖以適應視覺模型")
                spectrograms = DataAdapter.convert_audio_to_spectrogram(
                    main_data,
                    config.get('data', {}).get('preprocessing', {}).get('spectrogram', {})
                )
                
                # 確保是3通道（視覺模型需要）
                if spectrograms.size(1) == 1:
                    logger.info("將單通道頻譜圖轉換為3通道")
                    spectrograms = spectrograms.repeat(1, 3, 1, 1)
                
                adapted_batch['spectrogram'] = spectrograms
            
            elif data_type == 'features':
                # 標準化特徵輸入格式
                # 處理各種可能的特徵形狀，確保其為 [batch_size, feature_dim]
                if len(main_data.shape) > 2:
                    logger.info(f"將特徵從 {main_data.shape} 調整為 2D 形狀")
                    if len(main_data.shape) == 3:  # [B, L, D]
                        # 如果是序列數據，可以取平均或者展平
                        main_data = main_data.reshape(main_data.size(0), -1)
                    elif len(main_data.shape) == 4:  # [B, C, H, W]
                        # 如果是圖像數據，展平為特徵向量
                        main_data = main_data.reshape(main_data.size(0), -1)
                    
                # 嘗试将特征转换为适合CNN的形状
                logger.info("將特徵向量重塑為2D格式以適應視覺模型")
                reshaped_features = DataAdapter.reshape_features_to_2d(
                    main_data,
                    config.get('data', {}).get('preprocessing', {}).get('reshape', {})
                )
                
                # 確保是3通道（視覺模型需要）
                if reshaped_features.size(1) == 1:
                    logger.info("將單通道特徵圖轉換為3通道")
                    reshaped_features = reshaped_features.repeat(1, 3, 1, 1)
                
                adapted_batch['spectrogram'] = reshaped_features
            
            elif data_type == 'spectrogram':
                # 確保頻譜圖是3通道的
                if main_data.size(1) != 3:
                    logger.info("將頻譜圖調整為3通道")
                    # 如果是單通道，複製到3通道
                    if main_data.size(1) == 1:
                        adapted_batch['spectrogram'] = main_data.repeat(1, 3, 1, 1)
                    # 如果是其他通道數，使用前3個通道或複製到3個通道
                    else:
                        if main_data.size(1) < 3:
                            # 少於3通道，複製到3通道
                            adapted_batch['spectrogram'] = main_data.repeat(1, 3 // main_data.size(1) + 1, 1, 1)[:, :3, :, :]
                        else:
                            # 多於3通道，取前3個通道
                            adapted_batch['spectrogram'] = main_data[:, :3, :, :]
        
        elif model_type == 'fcnn':
            # FCNN需要特徵向量 [B, D] 或 [B, L, D]
            if data_type == 'audio':
                # 提取音頻特徵
                logger.info("提取音頻特徵以適應FCNN模型")
                features = DataAdapter.extract_audio_features(
                    main_data,
                    config.get('data', {}).get('preprocessing', {}).get('features', {})
                )
                adapted_batch['features'] = features
            
            elif data_type == 'spectrogram':
                # 平展頻譜圖為向量
                logger.info("平展頻譜圖為向量以適應FCNN模型")
                flattened = DataAdapter.flatten_spectrogram(main_data)
                adapted_batch['features'] = flattened
            
            elif data_type == 'features':
                # 標準化特徵輸入格式
                # 處理各種可能的特徵形狀，確保其為 [batch_size, feature_dim]
                logger.info(f"標準化特徵輸入格式：原形狀 {main_data.shape}")
                if len(main_data.shape) > 2:
                    if len(main_data.shape) == 3:  # [B, L, D]
                        # 如果是序列數據，可以取平均或者展平
                        main_data = main_data.reshape(main_data.size(0), -1)
                        logger.info(f"將3D特徵展平為2D: {main_data.shape}")
                    elif len(main_data.shape) == 4:  # [B, C, H, W]
                        # 如果是圖像數據，展平為特徵向量
                        main_data = main_data.reshape(main_data.size(0), -1)
                        logger.info(f"將4D特徵展平為2D: {main_data.shape}")
                
                adapted_batch['features'] = main_data
            
            # 檢查特徵維度是否與模型匹配
            if 'features' in adapted_batch:
                # 獲取模型配置中的輸入維度
                input_dim = config.get('model', {}).get('parameters', {}).get('input_dim')
                if input_dim and adapted_batch['features'].size(1) != input_dim:
                    logger.warning(f"特徵維度不匹配：模型期望 {input_dim}，實際為 {adapted_batch['features'].size(1)}")
                    # 調整特徵維度（截斷或填充）
                    if adapted_batch['features'].size(1) > input_dim:
                        logger.info(f"截斷特徵: {adapted_batch['features'].size(1)} -> {input_dim}")
                        adapted_batch['features'] = adapted_batch['features'][:, :input_dim]
                    else:
                        logger.info(f"填充特徵: {adapted_batch['features'].size(1)} -> {input_dim}")
                        padding = torch.zeros(adapted_batch['features'].size(0), 
                                             input_dim - adapted_batch['features'].size(1),
                                             device=adapted_batch['features'].device)
                        adapted_batch['features'] = torch.cat([adapted_batch['features'], padding], dim=1)
        
        return adapted_batch
    
    @staticmethod
    def convert_audio_to_spectrogram(
        audio_batch: torch.Tensor,
        spec_config: Dict[str, Any] = None
    ) -> torch.Tensor:
        """將音頻批次轉換為頻譜圖批次
        
        Args:
            audio_batch: 音頻批次，形狀為 [batch_size, audio_length]
            spec_config: 頻譜圖配置
            
        Returns:
            torch.Tensor: 頻譜圖批次，形狀為 [batch_size, 1, height, width]
        """
        if spec_config is None:
            spec_config = {}
        
        n_fft = spec_config.get('n_fft', 1024)
        hop_length = spec_config.get('hop_length', 512)
        n_mels = spec_config.get('n_mels', 128)
        power = spec_config.get('power', 2.0)
        normalized = spec_config.get('normalized', True)
        
        # 檢查設備並移動數據
        device = audio_batch.device
        
        # 創建mel頻譜圖轉換器
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,  # 默認採樣率
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=power,
            normalized=normalized
        ).to(device)
        
        # 對數縮放
        amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(device)
        
        # 轉換每個音頻
        spectrograms = []
        for audio in audio_batch:
            # 確保音頻是2D的，形狀為 [1, audio_length]
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            
            # 生成頻譜圖
            spec = mel_spectrogram(audio)
            spec = amplitude_to_db(spec)
            
            # 將通道維度移到最前面
            if spec.dim() == 2:
                spec = spec.unsqueeze(0)
            
            spectrograms.append(spec)
        
        # 堆疊為批次
        return torch.stack(spectrograms)
    
    @staticmethod
    def extract_audio_features(
        audio_batch: torch.Tensor,
        feature_config: Dict[str, Any] = None
    ) -> torch.Tensor:
        """從音頻批次中提取特徵
        
        Args:
            audio_batch: 音頻批次，形狀為 [batch_size, audio_length]
            feature_config: 特徵配置
            
        Returns:
            torch.Tensor: 特徵批次，形狀為 [batch_size, feature_dim]
        """
        if feature_config is None:
            feature_config = {}
        
        # 轉換為NumPy進行處理
        device = audio_batch.device
        audio_numpy = audio_batch.cpu().numpy()
        
        feature_type = feature_config.get('type', 'mfcc')
        n_mfcc = feature_config.get('n_mfcc', 40)
        
        features = []
        for audio in audio_numpy:
            if feature_type == 'mfcc':
                # 提取MFCC特徵
                mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=n_mfcc)
                # 計算統計特徵（均值和標準差）
                mfcc_mean = np.mean(mfcc, axis=1)
                mfcc_std = np.std(mfcc, axis=1)
                feature = np.concatenate([mfcc_mean, mfcc_std])
            elif feature_type == 'spectral':
                # 提取頻譜特徵
                spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=16000).mean(axis=1)
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=16000).mean(axis=1)
                spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=16000).mean(axis=1)
                feature = np.concatenate([spectral_centroid, spectral_bandwidth, spectral_rolloff])
            else:
                # 默認使用原始音頻作為特徵
                feature = audio
            
            features.append(feature)
        
        # 轉換為張量並返回到原始設備
        features_tensor = torch.tensor(np.array(features), dtype=torch.float32).to(device)
        return features_tensor
    
    @staticmethod
    def flatten_spectrogram(spec_batch: torch.Tensor) -> torch.Tensor:
        """將頻譜圖批次平展為特徵向量批次
        
        Args:
            spec_batch: 頻譜圖批次，形狀為 [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: 特徵向量批次，形狀為 [batch_size, channels*height*width]
        """
        batch_size = spec_batch.size(0)
        # 使用reshape而非view，以處理非連續張量
        return spec_batch.reshape(batch_size, -1)
    
    @staticmethod
    def reshape_features_to_2d(
        feature_batch: torch.Tensor,
        reshape_config: Dict[str, Any] = None
    ) -> torch.Tensor:
        """將特徵向量批次重塑為2D格式
        
        Args:
            feature_batch: 特徵向量批次，形狀為 [batch_size, feature_dim] 或 [batch_size, 1, feature_dim]
            reshape_config: 重塑配置
            
        Returns:
            torch.Tensor: 2D特徵批次，形狀為 [batch_size, 1, height, width]
        """
        if reshape_config is None:
            reshape_config = {}
        
        # 獲取批次大小
        batch_size = feature_batch.size(0)
        
        # 處理多維度特徵張量
        if feature_batch.dim() > 2:
            # 如果是[batch_size, 1, feature_dim]或更多維度的張量，平展為[batch_size, -1]
            feature_batch = feature_batch.reshape(batch_size, -1)
        
        # 計算特徵總數
        feature_dim = feature_batch.size(1)
        
        # 計算合適的高度和寬度
        height = reshape_config.get('height', int(np.sqrt(feature_dim)))
        width = reshape_config.get('width', feature_dim // height)
        
        # 確保高度和寬度的乘積不超過特徵維度
        if height * width > feature_dim:
            width = feature_dim // height
        
        # 如果還有剩餘的特徵，填充為正方形
        padding_size = height * width - feature_dim
        if padding_size > 0:
            padding = torch.zeros(batch_size, padding_size, device=feature_batch.device)
            feature_batch = torch.cat([feature_batch, padding], dim=1)
        elif padding_size < 0:
            # 如果填充後仍然超出，則截斷
            feature_batch = feature_batch[:, :height * width]
        
        # 重塑為 [batch_size, 1, height, width]
        return feature_batch.reshape(batch_size, 1, height, width)
    
    @staticmethod
    def check_model_data_compatibility(
        model_type: str,
        data_type: str
    ) -> Tuple[bool, Optional[str]]:
        """檢查模型類型和數據類型是否兼容
        
        Args:
            model_type: 模型類型
            data_type: 數據類型
            
        Returns:
            Tuple[bool, Optional[str]]: 是否兼容，以及不兼容時的轉換方法
        """
        compatibility_map = {
            'swin_transformer': {
                'audio': ('convert_audio_to_spectrogram', False),
                'spectrogram': (None, True),
                'feature': ('reshape_features_to_2d', False)
            },
            'cnn': {
                'audio': ('convert_audio_to_spectrogram', False),
                'spectrogram': (None, True),
                'feature': ('reshape_features_to_2d', False)
            },
            'resnet': {
                'audio': ('convert_audio_to_spectrogram', False),
                'spectrogram': (None, True),
                'feature': ('reshape_features_to_2d', False)
            },
            'fcnn': {
                'audio': ('extract_audio_features', False),
                'spectrogram': ('flatten_spectrogram', False),
                'feature': (None, True)
            }
        }
        
        if model_type not in compatibility_map:
            return False, None
        
        if data_type not in compatibility_map[model_type]:
            return False, None
        
        adapter_method, is_compatible = compatibility_map[model_type][data_type]
        return is_compatible, adapter_method


def adapt_datasets_to_model(
    model_type: str,
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """調整數據集以適應模型
    
    Args:
        model_type: 模型類型
        config: 配置字典
        train_loader: 訓練數據加載器
        val_loader: 驗證數據加載器
        test_loader: 測試數據加載器
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: 調整後的數據加載器
    """
    # 確定數據類型
    data_type = config['data']['type']
    
    # 檢查兼容性
    is_compatible, adapter_method = DataAdapter.check_model_data_compatibility(model_type, data_type)
    
    if is_compatible:
        logger.info(f"模型類型 '{model_type}' 與數據類型 '{data_type}' 兼容，無需調整")
        return train_loader, val_loader, test_loader
    
    # 需要調整數據
    logger.info(f"模型類型 '{model_type}' 與數據類型 '{data_type}' 不兼容，使用 '{adapter_method}' 進行調整")
    
    # 創建適配器包裝的數據加載器
    wrapped_train_loader = AdapterDataLoader(train_loader, model_type, config)
    wrapped_val_loader = AdapterDataLoader(val_loader, model_type, config)
    wrapped_test_loader = AdapterDataLoader(test_loader, model_type, config)
    
    return wrapped_train_loader, wrapped_val_loader, wrapped_test_loader


class AdapterDataLoader:
    """適配器數據加載器，在數據加載器的基礎上增加數據轉換功能"""
    
    def __init__(self, dataloader: DataLoader, model_type: str, config: Dict[str, Any]):
        """初始化適配器數據加載器
        
        Args:
            dataloader: 原始數據加載器
            model_type: 模型類型
            config: 配置字典
        """
        self.dataloader = dataloader
        self.model_type = model_type
        self.config = config
        self.iterator = None
    
    def __iter__(self):
        """返回迭代器"""
        self.iterator = iter(self.dataloader)
        return self
    
    def __next__(self):
        """獲取下一個批次"""
        batch = next(self.iterator)
        return DataAdapter.adapt_batch(batch, self.model_type, self.config)
    
    def __len__(self):
        """返回數據加載器長度"""
        return len(self.dataloader) 