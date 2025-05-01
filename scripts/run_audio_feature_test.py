#!/usr/bin/env python
"""
音訊特徵提取測試執行腳本

此腳本執行音訊特徵提取的測試並展示如何使用特徵提取功能。
它生成合成音訊並處理，展示不同的特徵提取方法。

執行方式：
python scripts/run_audio_feature_test.py
"""

import os
import sys
import unittest
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import logging

# 將項目根目錄添加到系統路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# 導入所需的模組
from utils.audio_feature_extractor import (
    preprocess_audio, 
    extract_mel_spectrogram,
    extract_mfcc, 
    extract_features_from_config
)
from models.fcnn import FCNN
from tests.test_audio_feature_extraction import TestAudioFeatureExtraction

def run_unittest():
    """執行單元測試"""
    logger.info("=== 執行音訊特徵提取單元測試 ===")
    
    # 載入測試案例
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestAudioFeatureExtraction)
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_result = test_runner.run(test_suite)
    
    logger.info(f"測試結果: {test_result}")
    
    return test_result.wasSuccessful()

def visualize_features():
    """視覺化特徵提取過程與結果"""
    logger.info("=== 視覺化音訊特徵提取 ===")
    
    # 創建測試音訊 (複音 - 包含兩個頻率)
    sample_rate = 16000
    duration = 3  # 秒
    t = torch.linspace(0, duration, int(sample_rate * duration))
    audio_tensor = torch.sin(2 * np.pi * 440 * t) + 0.5 * torch.sin(2 * np.pi * 880 * t)
    
    # 預處理音訊
    processed_audio = preprocess_audio(audio_tensor, sr=sample_rate)
    
    # 提取特徵
    mel_spec = extract_mel_spectrogram(
        processed_audio, 
        sr=sample_rate, 
        n_mels=128, 
        n_fft=1024, 
        hop_length=512
    )
    
    mfcc = extract_mfcc(
        processed_audio,
        sr=sample_rate,
        n_mfcc=40,
        n_fft=1024,
        hop_length=512
    )
    
    # 創建測試配置
    config = {
        'data': {
            'preprocessing': {
                'audio': {
                    'sr': sample_rate,
                    'duration': duration,
                    'normalize': True
                },
                'features': {
                    'method': 'mel_spectrogram',
                    'n_mels': 128,
                    'n_fft': 1024,
                    'hop_length': 512,
                    'power': 2.0,
                    'normalized': True,
                    'log_mel': True,
                    'target_duration_sec': duration,
                    'scaling_method': 'standard'
                }
            }
        }
    }
    
    features = extract_features_from_config(audio_tensor, config)
    
    # 準備存儲路徑
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_dir = os.path.join(project_root, 'results', f'feature_extraction_{timestamp}')
    os.makedirs(plot_dir, exist_ok=True)
    
    # 保存實驗元數據
    metadata = {
        'timestamp': timestamp,
        'experiment': 'audio_feature_extraction',
        'sample_rate': sample_rate,
        'duration': duration,
        'feature_types': ['waveform', 'mel_spectrogram', 'mfcc', 'normalized_features'],
    }
    with open(os.path.join(plot_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 記錄到實驗日誌
    log_path = os.path.join(plot_dir, 'experiments.log')
    with open(log_path, 'a') as f:
        f.write(json.dumps(metadata) + '\n')
    
    # 繪製波形
    plt.figure(figsize=(12, 6))
    plt.plot(t.numpy(), audio_tensor.numpy())
    plt.title('Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{timestamp}_waveform.png"))
    
    # 繪製 Mel 頻譜圖
    plt.figure(figsize=(12, 6))
    plt.imshow(mel_spec.numpy(), aspect='auto', origin='lower')
    plt.title('Mel Spectrogram')
    plt.xlabel('Time Frame')
    plt.ylabel('Mel Frequency Bin')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{timestamp}_mel_spectrogram.png"))
    
    # 繪製 MFCC
    plt.figure(figsize=(12, 6))
    plt.imshow(mfcc.numpy(), aspect='auto', origin='lower')
    plt.title('MFCC')
    plt.xlabel('Time Frame')
    plt.ylabel('MFCC Coefficient')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{timestamp}_mfcc.png"))
    
    # 繪製標準化特徵
    plt.figure(figsize=(12, 6))
    plt.imshow(features.numpy(), aspect='auto', origin='lower')
    plt.title('Normalized Features from Config')
    plt.xlabel('Time Frame')
    plt.ylabel('Feature Dimension')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{timestamp}_normalized_features.png"))
    
    logger.info(f"特徵視覺化已保存到 {plot_dir}")
    
    return plot_dir

def test_fcnn_with_feature():
    """測試FCNN模型處理特徵的能力"""
    logger.info("=== 測試FCNN與特徵整合 ===")
    
    # 創建測試音訊
    sample_rate = 16000
    duration = 3  # 秒
    t = torch.linspace(0, duration, int(sample_rate * duration))
    audio_tensor = torch.sin(2 * np.pi * 440 * t)
    
    # 創建測試配置
    config = {
        'data': {
            'preprocessing': {
                'audio': {
                    'sr': sample_rate,
                    'duration': duration,
                    'normalize': True
                },
                'features': {
                    'method': 'mel_spectrogram',
                    'n_mels': 128,
                    'n_fft': 1024,
                    'hop_length': 512,
                    'log_mel': True,
                    'target_duration_sec': duration,
                    'scaling_method': 'standard'
                }
            }
        },
        'model': {
            'type': 'fcnn',
            'parameters': {
                'input_dim': 128,  # 對應 n_mels
                'hidden_layers': [64, 32],
                'num_classes': 2,
                'is_classification': True
            }
        }
    }
    
    # 提取特徵
    features = extract_features_from_config(audio_tensor, config)
    features_batch = features.unsqueeze(0)  # [1, n_mels, frames]
    
    # 創建模型
    model = FCNN(
        input_dim=config['model']['parameters']['input_dim'],
        hidden_layers=config['model']['parameters']['hidden_layers'],
        num_classes=config['model']['parameters']['num_classes'],
        is_classification=config['model']['parameters']['is_classification']
    )
    
    # 將特徵傳入模型
    with torch.no_grad():
        output = model(features_batch)
    
    logger.info(f"特徵形狀: {features_batch.shape}")
    logger.info(f"模型輸出形狀: {output.shape}")
    logger.info(f"模型輸出: {output}")
    
    return output

def main():
    """主函數"""
    # 執行單元測試
    test_success = run_unittest()
    
    if test_success:
        logger.info("單元測試通過，繼續視覺化特徵和測試模型...")
        
        # 視覺化特徵
        plot_dir = visualize_features()
        
        # 測試FCNN與特徵整合
        output = test_fcnn_with_feature()
        
        logger.info("=== 測試完成 ===")
        logger.info(f"特徵視覺化路徑: {plot_dir}")
    else:
        logger.error("單元測試失敗，請檢查錯誤並修復。")

if __name__ == "__main__":
    main() 