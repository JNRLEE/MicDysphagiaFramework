"""
音訊特徵提取與FCNN模型整合測試
功能：
1. 測試特徵提取器的功能
2. 測試數據流，從音訊到特徵再到模型預測
3. 確保FCNN模型可以正確處理3D特徵輸入
"""

import os
import sys
import unittest
import torch
import numpy as np
import tempfile
from pathlib import Path
import logging
import json
import yaml

# 將項目根目錄添加到系統路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 導入所需的模組
from utils.audio_feature_extractor import (
    preprocess_audio, 
    extract_mel_spectrogram,
    extract_mfcc, 
    extract_features_from_config
)
from models.fcnn import FCNN
from data.audio_dataset import AudioDataset
from utils.data_adapter import DataAdapter

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

class TestAudioFeatureExtraction(unittest.TestCase):
    """測試音訊特徵提取與FCNN模型整合"""
    
    def setUp(self):
        """設置測試環境"""
        # 創建測試音訊
        self.sample_rate = 16000
        self.duration = 3  # 秒
        t = torch.linspace(0, self.duration, int(self.sample_rate * self.duration))
        self.audio_tensor = torch.sin(2 * np.pi * 440 * t)  # 440 Hz 正弦波
        
        # 創建測試配置
        self.config = {
            'data': {
                'preprocessing': {
                    'audio': {
                        'sr': self.sample_rate,
                        'duration': self.duration,
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
                        'target_duration_sec': self.duration,
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
        
        # 創建模型
        self.model = FCNN(
            input_dim=self.config['model']['parameters']['input_dim'],
            hidden_layers=self.config['model']['parameters']['hidden_layers'],
            num_classes=self.config['model']['parameters']['num_classes'],
            is_classification=self.config['model']['parameters']['is_classification']
        )
    
    def test_feature_extraction_functions(self):
        """測試個別特徵提取函數"""
        # 測試音訊預處理
        processed_audio = preprocess_audio(self.audio_tensor, sr=self.sample_rate)
        self.assertEqual(processed_audio.shape, self.audio_tensor.shape)
        self.assertLessEqual(torch.max(processed_audio), 1.0)
        self.assertGreaterEqual(torch.min(processed_audio), -1.0)
        
        # 測試 Mel 頻譜圖提取
        mel_spec = extract_mel_spectrogram(
            processed_audio, 
            sr=self.sample_rate, 
            n_mels=128, 
            n_fft=1024, 
            hop_length=512
        )
        self.assertEqual(mel_spec.shape[0], 128)  # n_mels
        self.assertGreater(mel_spec.shape[1], 0)  # frames
        
        # 測試 MFCC 提取
        mfcc = extract_mfcc(
            processed_audio,
            sr=self.sample_rate,
            n_mfcc=40,
            n_fft=1024,
            hop_length=512
        )
        self.assertEqual(mfcc.shape[0], 40)  # n_mfcc
        self.assertGreater(mfcc.shape[1], 0)  # frames
    
    def test_extract_features_from_config(self):
        """測試從配置提取特徵"""
        features = extract_features_from_config(self.audio_tensor, self.config)
        
        # 檢查形狀
        self.assertEqual(features.shape[0], self.config['data']['preprocessing']['features']['n_mels'])
        self.assertGreater(features.shape[1], 0)
        
        # 使用MFCC測試
        mfcc_config = self.config.copy()
        mfcc_config['data']['preprocessing']['features']['method'] = 'mfcc'
        mfcc_config['data']['preprocessing']['features']['n_mfcc'] = 40
        
        mfcc_features = extract_features_from_config(self.audio_tensor, mfcc_config)
        self.assertEqual(mfcc_features.shape[0], 40)
        self.assertGreater(mfcc_features.shape[1], 0)
    
    def test_fcnn_with_features(self):
        """測試FCNN接收特徵輸入"""
        # 提取特徵
        features = extract_features_from_config(self.audio_tensor, self.config)
        
        # 添加批次維度
        features_batch = features.unsqueeze(0)  # [1, n_mels, frames]
        
        # 將特徵傳入模型
        output = self.model(features_batch)
        
        # 檢查輸出形狀
        self.assertEqual(output.shape, (1, self.config['model']['parameters']['num_classes']))
    
    def test_data_adapter_with_features(self):
        """測試DataAdapter處理特徵與FCNN的整合"""
        # 提取特徵
        features = extract_features_from_config(self.audio_tensor, self.config)
        features_batch = features.unsqueeze(0)  # [1, n_mels, frames]
        
        # 創建批次
        batch = {
            'features': features_batch,
            'label': torch.tensor([0])
        }
        
        # 使用DataAdapter適配批次
        adapted_batch = DataAdapter.adapt_batch(batch, 'fcnn', self.config)
        
        # 檢查適配後的批次
        self.assertIn('features', adapted_batch)
        
        # 確保模型可以處理適配後的批次
        output = self.model(adapted_batch['features'])
        self.assertEqual(output.shape, (1, self.config['model']['parameters']['num_classes']))
    
    @unittest.skip("需要真實數據目錄和配置進行完整測試")
    def test_audio_dataset_with_features(self):
        """測試AudioDataset的特徵提取功能（需要實際的音訊文件）"""
        # 這個測試需要實際的音訊文件和valid的配置
        # 如果沒有適當的環境，可以使用unittest.skip跳過
        
        # 創建臨時目錄
        with tempfile.TemporaryDirectory() as temp_dir:
            # 創建測試音訊文件
            wav_path = os.path.join(temp_dir, "test_audio.wav")
            info_path = os.path.join(temp_dir, "info.json")
            
            # 保存測試音訊
            import torchaudio
            torchaudio.save(wav_path, self.audio_tensor.unsqueeze(0), self.sample_rate)
            
            # 創建info.json
            info_data = {
                "patient_id": "test_patient",
                "score": 0,
                "selection": "jelly"
            }
            with open(info_path, 'w') as f:
                json.dump(info_data, f)
            
            # 修改配置
            test_config = self.config.copy()
            test_config['data']['source'] = {'wav_dir': temp_dir}
            test_config['data']['filtering'] = {
                'task_type': 'classification',
                'custom_classification': {'enabled': False},
                'score_thresholds': {'normal': 0, 'patient': 9},
                'class_config': {'Jelly': 1}
            }
            
            # 創建 Dataset
            dataset = AudioDataset(root_dir=temp_dir, config=test_config)
            
            # 檢查數據集大小
            self.assertGreater(len(dataset), 0)
            
            # 獲取第一個樣本
            sample = dataset[0]
            
            # 檢查是否包含特徵
            self.assertIn('features', sample)
            self.assertEqual(sample['features'].shape[0], self.config['data']['preprocessing']['features']['n_mels'])
    
    def test_yaml_config_loading(self):
        """測試YAML配置載入和解析特徵配置"""
        # 創建臨時YAML文件
        with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w+') as temp_file:
            # 寫入測試配置
            yaml.dump(self.config, temp_file)
            temp_file.flush()
            
            # 重新讀取配置
            temp_file.seek(0)
            loaded_config = yaml.safe_load(temp_file)
            
            # 檢查特徵配置是否正確
            self.assertEqual(
                loaded_config['data']['preprocessing']['features']['method'],
                self.config['data']['preprocessing']['features']['method']
            )
            self.assertEqual(
                loaded_config['data']['preprocessing']['features']['n_mels'],
                self.config['data']['preprocessing']['features']['n_mels']
            )
            
            # 檢查模型配置
            self.assertEqual(
                loaded_config['model']['parameters']['input_dim'],
                self.config['model']['parameters']['input_dim']
            )

if __name__ == '__main__':
    unittest.main() 