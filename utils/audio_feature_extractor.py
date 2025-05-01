"""
音訊特徵提取模組：提供音訊數據的前處理與特徵提取工具
功能：
1. 音訊預處理（正規化、去噪、靜音檢測）
2. 特徵提取（Mel頻譜圖、MFCC）
3. 特徵處理（維度調整、縮放）
"""

import torch
import torchaudio
import numpy as np
import librosa
import logging
from typing import Dict, Any, Tuple, Optional, Union, List
import warnings

logger = logging.getLogger(__name__)

def preprocess_audio(
    audio_tensor: torch.Tensor, 
    sr: int = 16000, 
    normalize: bool = True, 
    remove_silence: bool = False, 
    silence_threshold: float = 0.01, 
    denoise: bool = False
) -> torch.Tensor:
    """音訊前處理：正規化、去噪、靜音檢測等
    
    Args:
        audio_tensor: 輸入音訊張量 [samples]
        sr: 採樣率
        normalize: 是否正規化音訊到 [-1, 1] 範圍
        remove_silence: 是否移除靜音片段
        silence_threshold: 靜音檢測閾值
        denoise: 是否進行簡單去噪處理
    
    Returns:
        處理後的音訊張量 [samples]
        
    References:
        https://librosa.org/doc/main/generated/librosa.effects.split.html
        https://librosa.org/doc/main/generated/librosa.util.normalize.html
    """
    # 確保音訊是一維張量
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.squeeze()
    
    # 轉為 numpy 進行處理
    audio_np = audio_tensor.numpy()
    
    # 靜音檢測與移除
    if remove_silence:
        try:
            # 使用 librosa 進行靜音檢測
            non_silent_intervals = librosa.effects.split(
                audio_np, 
                top_db=20,  # 相對於峰值的分貝值，越低越敏感
                frame_length=1024, 
                hop_length=512
            )
            
            if len(non_silent_intervals) > 0:
                # 提取並連接非靜音片段
                processed_signal = []
                for interval in non_silent_intervals:
                    start, end = interval
                    processed_signal.extend(audio_np[start:end])
                
                if len(processed_signal) > 0:
                    audio_np = np.array(processed_signal)
                else:
                    logger.warning("靜音檢測後沒有足夠的非靜音信號，保留原始信號")
            else:
                logger.warning("找不到非靜音片段，保留原始信號")
        except Exception as e:
            logger.warning(f"靜音檢測時發生錯誤: {e}，保留原始信號")
    
    # 去噪處理（這裡是簡單的軟閾值方法，僅作示例）
    if denoise:
        try:
            # 簡單的軟閾值去噪
            noise_threshold = silence_threshold * np.max(np.abs(audio_np))
            audio_np = np.where(
                np.abs(audio_np) < noise_threshold,
                0,
                audio_np * (1 - noise_threshold / np.maximum(np.abs(audio_np), 1e-10))
            )
        except Exception as e:
            logger.warning(f"去噪處理時發生錯誤: {e}，保留原始信號")
    
    # 正規化音訊
    if normalize:
        try:
            # 檢查音訊是否全零
            if np.all(audio_np == 0):
                logger.warning("正規化嘗試失敗：音訊全為零")
            else:
                max_val = np.max(np.abs(audio_np))
                if max_val > 1e-10:  # 避免除以零
                    audio_np = audio_np / max_val
        except Exception as e:
            logger.warning(f"正規化時發生錯誤: {e}，保留原始信號")
    
    # 轉回 PyTorch 張量
    return torch.from_numpy(audio_np).float()

def extract_mel_spectrogram(
    audio_tensor: torch.Tensor,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 128,
    power: float = 2.0,
    normalized: bool = True,
    log_mel: bool = True
) -> torch.Tensor:
    """從音訊提取 Mel 頻譜圖
    
    Args:
        audio_tensor: 輸入音訊張量 [samples]
        sr: 採樣率
        n_fft: FFT 窗口大小
        hop_length: 幀移動長度
        n_mels: Mel 濾波器數量
        power: 頻譜的指數，1 為能量，2 為功率
        normalized: 是否使用正規化 Mel 濾波器
        log_mel: 是否轉換為對數 Mel 頻譜
    
    Returns:
        Mel 頻譜圖張量 [n_mels, n_frames]
        
    References:
        https://pytorch.org/audio/stable/transforms.html#melspectrogram
    """
    try:
        # 確保音訊是一維張量
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.squeeze(0)
        
        # 使用 torchaudio 提取 Mel 頻譜圖
        mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=power,
            normalized=normalized
        )
        
        mel_spectrogram = mel_spectrogram_transform(audio_tensor)
        
        # 轉換為對數尺度
        if log_mel:
            # 添加小量值避免對數運算中的零值問題
            mel_spectrogram = torch.log(mel_spectrogram + 1e-9)
        
        return mel_spectrogram
    
    except Exception as e:
        logger.error(f"Mel 頻譜圖提取失敗: {e}")
        # 返回形狀正確的零張量
        time_frames = int(audio_tensor.size(0) // hop_length + 1)
        return torch.zeros((n_mels, time_frames))

def extract_mfcc(
    audio_tensor: torch.Tensor,
    sr: int = 16000,
    n_mfcc: int = 40,
    n_fft: int = 1024,
    hop_length: int = 512,
    log_mels: bool = True,
    melkwargs: Optional[Dict[str, Any]] = None
) -> torch.Tensor:
    """從音訊提取 MFCC 特徵
    
    Args:
        audio_tensor: 輸入音訊張量 [samples]
        sr: 採樣率
        n_mfcc: MFCC 係數數量
        n_fft: FFT 窗口大小
        hop_length: 幀移動長度
        log_mels: 是否在計算 MFCC 前先對 Mel 頻譜圖取對數
        melkwargs: Mel 頻譜圖提取的額外參數
    
    Returns:
        MFCC 特徵張量 [n_mfcc, n_frames]
        
    References:
        https://pytorch.org/audio/stable/transforms.html#mfcc
    """
    try:
        # 確保音訊是一維張量
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.squeeze(0)
            
        # 設置默認的 melkwargs
        if melkwargs is None:
            melkwargs = {
                'n_mels': 128,
                'normalized': True
            }
        
        # 使用 torchaudio 提取 MFCC
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            log_mels=log_mels,
            melkwargs=melkwargs
        )
        
        mfcc = mfcc_transform(audio_tensor)
        return mfcc
    
    except Exception as e:
        logger.error(f"MFCC 提取失敗: {e}")
        # 返回形狀正確的零張量
        time_frames = int(audio_tensor.size(0) // hop_length + 1)
        return torch.zeros((n_mfcc, time_frames))

def pad_features(
    features: torch.Tensor, 
    target_length: int, 
    mode: str = 'constant', 
    value: float = 0.0,
    dim: int = -1
) -> torch.Tensor:
    """將特徵填充或截斷至目標長度
    
    Args:
        features: 輸入特徵張量 [n_features, n_frames] 或變動大小
        target_length: 目標時間步長
        mode: 填充模式 ('constant', 'reflect', 'replicate', 'circular')
        value: 填充值（當 mode='constant' 時）
        dim: 要填充的維度，默認為最後一維（時間維度）
    
    Returns:
        填充後的特徵張量 [n_features, target_length]
        
    References:
        https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    """
    if features.size(dim) == target_length:
        return features
    
    if features.size(dim) > target_length:
        # 截斷至目標長度
        if dim == -1 or dim == features.dim() - 1:
            return features[..., :target_length]
        elif dim == 0:
            return features[:target_length]
        elif dim == 1 and features.dim() > 1:
            return features[:, :target_length]
    else:
        # 需要填充
        padding_size = target_length - features.size(dim)
        
        # 根據 dim 創建適當的填充元組
        pad_tuple = [0, 0] * features.dim()
        # 對應位置設置填充大小
        pad_idx = features.dim() * 2 - 2 * (dim % features.dim()) - 2
        pad_tuple[pad_idx + 1] = padding_size
        
        import torch.nn.functional as F
        return F.pad(features, tuple(pad_tuple), mode=mode, value=value)

def scale_features(
    features: torch.Tensor, 
    method: str = 'standard', 
    dim: int = 1, 
    eps: float = 1e-8
) -> torch.Tensor:
    """對特徵進行縮放
    
    Args:
        features: 輸入特徵張量 [n_features, n_frames] 或 [batch, n_features, n_frames]
        method: 縮放方法 ('standard': 標準化, 'minmax': 最小-最大縮放, None: 不縮放)
        dim: 計算統計量的維度，通常為頻率維度(1)或時間維度(2)
        eps: 數值穩定性的小值
    
    Returns:
        縮放後的特徵張量，形狀與輸入相同
        
    References:
        https://en.wikipedia.org/wiki/Standard_score
        https://en.wikipedia.org/wiki/Feature_scaling
    """
    if method is None or method.lower() == 'none':
        return features
    
    # 如果是單個樣本，添加 batch 維度便於處理
    add_batch_dim = False
    if features.dim() == 2:
        features = features.unsqueeze(0)
        add_batch_dim = True
    
    # 獲取維度索引，考慮負索引的情況
    dim_idx = dim if dim >= 0 else features.dim() + dim
    
    if method.lower() == 'standard':
        # 標準化: (x - mean) / std
        mean = torch.mean(features, dim=dim_idx, keepdim=True)
        std = torch.std(features, dim=dim_idx, keepdim=True) + eps
        scaled_features = (features - mean) / std
    
    elif method.lower() == 'minmax':
        # 最小-最大縮放: (x - min) / (max - min)
        min_val = torch.min(features, dim=dim_idx, keepdim=True)[0]
        max_val = torch.max(features, dim=dim_idx, keepdim=True)[0]
        range_val = max_val - min_val + eps
        scaled_features = (features - min_val) / range_val
    
    else:
        logger.warning(f"未知的縮放方法: {method}，返回原始特徵")
        scaled_features = features
    
    # 如果原始輸入是 2D，移除添加的 batch 維度
    if add_batch_dim:
        scaled_features = scaled_features.squeeze(0)
    
    return scaled_features

def extract_features_from_config(
    audio_tensor: torch.Tensor,
    config: Dict[str, Any]
) -> torch.Tensor:
    """根據配置從音訊中提取特徵
    
    Args:
        audio_tensor: 輸入音訊張量 [samples]
        config: 特徵提取配置字典
    
    Returns:
        提取的特徵張量 [n_features, n_frames] 或視配置而定的其他形狀
        
    Description:
        根據配置文件中的設定執行完整的特徵提取流程，包含：
        1. 預處理（正規化、去噪等）
        2. 特徵提取（Mel頻譜圖或MFCC）
        3. 填充到目標長度
        4. 特徵縮放（標準化或最小-最大縮放）
        
    References:
        無
    """
    # 獲取音訊預處理配置
    audio_config = config.get('data', {}).get('preprocessing', {}).get('audio', {})
    sr = audio_config.get('sr', 16000)
    normalize = audio_config.get('normalize', True)
    remove_silence = audio_config.get('remove_silence', False)
    silence_threshold = audio_config.get('silence_threshold', 0.01)
    denoise = audio_config.get('denoise', False)
    
    # 獲取特徵提取配置
    features_config = config.get('data', {}).get('preprocessing', {}).get('features', {})
    method = features_config.get('method', 'mel_spectrogram')
    
    # 音訊預處理
    processed_audio = preprocess_audio(
        audio_tensor=audio_tensor,
        sr=sr,
        normalize=normalize,
        remove_silence=remove_silence,
        silence_threshold=silence_threshold,
        denoise=denoise
    )
    
    # 特徵提取
    if method.lower() == 'mel_spectrogram':
        n_fft = features_config.get('n_fft', 1024)
        hop_length = features_config.get('hop_length', 512)
        n_mels = features_config.get('n_mels', 128)
        power = features_config.get('power', 2.0)
        normalized = features_config.get('normalized', True)
        log_mel = features_config.get('log_mel', True)
        
        features = extract_mel_spectrogram(
            audio_tensor=processed_audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=power,
            normalized=normalized,
            log_mel=log_mel
        )
    
    elif method.lower() == 'mfcc':
        n_fft = features_config.get('n_fft', 1024)
        hop_length = features_config.get('hop_length', 512)
        n_mfcc = features_config.get('n_mfcc', 40)
        log_mels = features_config.get('log_mels', True)
        
        melkwargs = {
            'n_mels': features_config.get('n_mels', 128),
            'n_fft': n_fft,
            'hop_length': hop_length,
            'normalized': features_config.get('normalized', True)
        }
        
        features = extract_mfcc(
            audio_tensor=processed_audio,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            log_mels=log_mels,
            melkwargs=melkwargs
        )
    
    else:
        logger.error(f"未知的特徵提取方法: {method}")
        # 返回原始音訊作為特徵（擴展為 2D）
        features = processed_audio.unsqueeze(0)
    
    # 填充到目標長度
    target_length = None
    target_duration_sec = features_config.get('target_duration_sec', None)
    
    if 'target_length' in features_config:
        target_length = features_config['target_length']
    elif target_duration_sec is not None:
        hop_length = features_config.get('hop_length', 512)
        target_length = int(np.ceil(sr * target_duration_sec / hop_length))
    
    if target_length is not None:
        features = pad_features(
            features=features,
            target_length=target_length,
            mode=features_config.get('pad_mode', 'constant'),
            value=features_config.get('pad_value', 0.0)
        )
    
    # 特徵縮放
    scaling_method = features_config.get('scaling_method', None)
    if scaling_method:
        features = scale_features(
            features=features,
            method=scaling_method,
            dim=features_config.get('scaling_dim', 1),
            eps=features_config.get('scaling_eps', 1e-8)
        )
    
    return features

# 中文註解：這是 audio_feature_extractor.py 的 Minimal Executable Unit，測試特徵提取功能
if __name__ == "__main__":
    """
    Description: Minimal Executable Unit for audio_feature_extractor.py，測試音訊特徵提取功能，包括預處理、Mel頻譜圖、MFCC等功能。
    Args: None
    Returns: None
    References: 無
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 創建測試音訊
    sample_rate = 16000
    duration = 3  # 秒
    t = torch.linspace(0, duration, int(sample_rate * duration))
    # 生成一個簡單的音訊信號（一個純音）
    audio_tensor = torch.sin(2 * np.pi * 440 * t)  # 440 Hz 正弦波
    
    logger.info(f"生成測試音訊信號: shape={audio_tensor.shape}, 採樣率={sample_rate}Hz")
    
    # 測試預處理
    processed_audio = preprocess_audio(audio_tensor, sr=sample_rate, normalize=True)
    logger.info(f"預處理後的音訊: shape={processed_audio.shape}")
    
    # 測試 Mel 頻譜圖提取
    mel_spec = extract_mel_spectrogram(
        processed_audio, 
        sr=sample_rate, 
        n_mels=128, 
        n_fft=1024, 
        hop_length=512
    )
    logger.info(f"Mel 頻譜圖: shape={mel_spec.shape}")
    
    # 測試 MFCC 提取
    mfcc = extract_mfcc(
        processed_audio,
        sr=sample_rate,
        n_mfcc=40,
        n_fft=1024,
        hop_length=512
    )
    logger.info(f"MFCC: shape={mfcc.shape}")
    
    # 測試特徵填充
    target_length = 100
    padded_features = pad_features(mel_spec, target_length=target_length)
    logger.info(f"填充後的特徵: shape={padded_features.shape}, 目標長度={target_length}")
    
    # 測試特徵縮放
    scaled_features = scale_features(mel_spec, method='standard')
    logger.info(f"縮放後的特徵: shape={scaled_features.shape}, 均值≈{scaled_features.mean().item():.4f}, 標準差≈{scaled_features.std().item():.4f}")
    
    # 測試從配置提取特徵
    test_config = {
        'data': {
            'preprocessing': {
                'audio': {
                    'sr': sample_rate,
                    'normalize': True
                },
                'features': {
                    'method': 'mel_spectrogram',
                    'n_mels': 128,
                    'n_fft': 1024,
                    'hop_length': 512,
                    'log_mel': True,
                    'target_length': 100,
                    'scaling_method': 'standard'
                }
            }
        }
    }
    
    features = extract_features_from_config(audio_tensor, test_config)
    logger.info(f"從配置提取的特徵: shape={features.shape}")
    logger.info("測試完成。") 