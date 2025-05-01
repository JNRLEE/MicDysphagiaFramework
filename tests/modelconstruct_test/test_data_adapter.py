"""
數據適配器測試模組
測試日期: 2024/04/09

功能說明:
此測試模組針對utils/data_adapter.py中的數據轉換方法進行單元測試，確保各種數據轉換功能正常運作。
測試內容包括:
1. 音頻到頻譜圖的轉換
2. 特徵向量到2D圖像的重塑
3. 頻譜圖到特徵向量的平展
4. 批次數據適配
5. 音頻特徵提取

測試數據源: tests/test_dataset/N002_2024-10-22_11-04-56/
"""

import os
import sys
import unittest
import torch
import numpy as np
from pathlib import Path
import logging
import librosa
import json
from PIL import Image

# 添加專案根目錄到系統路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 導入需要測試的模組
from utils.data_adapter import DataAdapter

# 設置日誌
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestDataAdapter(unittest.TestCase):
    """測試DataAdapter類的各項功能"""
    
    @classmethod
    def setUpClass(cls):
        """設置測試環境"""
        # 設定測試資料目錄
        cls.test_dir = Path('tests/test_dataset/N002_2024-10-22_11-04-56')
        
        # 確認測試資料存在
        assert cls.test_dir.exists(), f"測試目錄不存在: {cls.test_dir}"
        
        # 加載測試數據
        cls._load_test_data()
        
        logger.info("測試環境設置完成")
    
    @classmethod
    def _load_test_data(cls):
        """加載測試數據"""
        # 讀取音頻數據
        wav_path = cls.test_dir / 'Probe0_RX_IN_TDM4CH0.wav'
        cls.audio, cls.sample_rate = librosa.load(wav_path, sr=16000)
        cls.audio_tensor = torch.tensor(cls.audio).float()
        logger.info(f"加載音頻: {wav_path}, 形狀: {cls.audio_tensor.shape}")
        
        # 讀取頻譜圖
        spec_path = cls.test_dir / 'spectrogram.png'
        cls.spectrogram_image = Image.open(spec_path)
        cls.spectrogram_tensor = torch.tensor(np.array(cls.spectrogram_image)).float()
        # 調整為模型需要的形狀 [C, H, W]
        cls.spectrogram_tensor = cls.spectrogram_tensor.permute(2, 0, 1)
        logger.info(f"加載頻譜圖: {spec_path}, 形狀: {cls.spectrogram_tensor.shape}")
        
        # 讀取特徵數據
        feature_path = cls.test_dir / 'WavTokenizer_tokens.npy'
        cls.features = np.load(feature_path)
        cls.features_tensor = torch.tensor(cls.features).float()
        logger.info(f"加載特徵數據: {feature_path}, 形狀: {cls.features_tensor.shape}")
        
        # 準備模型輸入的批次數據
        cls.audio_batch = cls.audio_tensor.unsqueeze(0)  # [1, seq_len]
        cls.spectrogram_batch = cls.spectrogram_tensor.unsqueeze(0)  # [1, C, H, W]
        cls.features_batch = cls.features_tensor.unsqueeze(0)  # [1, feature_dim]
    
    def test_convert_audio_to_spectrogram(self):
        """測試音頻到頻譜圖的轉換"""
        logger.info("測試音頻到頻譜圖的轉換")
        
        # 使用DataAdapter轉換音頻為頻譜圖
        spec_config = {
            'n_fft': 1024,
            'hop_length': 512,
            'n_mels': 128
        }
        
        spectrogram = DataAdapter.convert_audio_to_spectrogram(
            self.audio_batch,
            spec_config
        )
        
        # 驗證轉換結果
        self.assertIsNotNone(spectrogram)
        self.assertEqual(spectrogram.dim(), 4)
        self.assertEqual(spectrogram.size(0), 1)
        self.assertEqual(spectrogram.size(1), 1)  # 單通道
        
        logger.info(f"音頻轉換為頻譜圖，形狀: {spectrogram.shape}")
        
        # 測試不同配置
        spec_config2 = {
            'n_fft': 2048,
            'hop_length': 1024,
            'n_mels': 64
        }
        
        spectrogram2 = DataAdapter.convert_audio_to_spectrogram(
            self.audio_batch,
            spec_config2
        )
        
        self.assertIsNotNone(spectrogram2)
        self.assertEqual(spectrogram2.size(1), 1)  # 單通道
        self.assertEqual(spectrogram2.size(2), 64)  # n_mels = 64
        
        logger.info(f"使用不同配置轉換音頻，形狀: {spectrogram2.shape}")
    
    def test_extract_audio_features(self):
        """測試音頻特徵提取"""
        logger.info("測試音頻特徵提取")
        
        # 使用DataAdapter提取音頻特徵
        feature_config = {
            'type': 'mfcc',
            'n_mfcc': 40
        }
        
        features = DataAdapter.extract_audio_features(
            self.audio_batch,
            feature_config
        )
        
        # 驗證轉換結果
        self.assertIsNotNone(features)
        self.assertEqual(features.dim(), 2)
        self.assertEqual(features.size(0), 1)
        
        logger.info(f"提取MFCC特徵，形狀: {features.shape}")
        
        # 測試不同特徵類型
        feature_config2 = {
            'type': 'spectral'
        }
        
        features2 = DataAdapter.extract_audio_features(
            self.audio_batch,
            feature_config2
        )
        
        self.assertIsNotNone(features2)
        self.assertEqual(features2.dim(), 2)
        
        logger.info(f"提取頻譜特徵，形狀: {features2.shape}")
    
    def test_flatten_spectrogram(self):
        """測試頻譜圖平展為特徵向量"""
        logger.info("測試頻譜圖平展為特徵向量")
        
        # 使用DataAdapter將頻譜圖平展為向量
        features = DataAdapter.flatten_spectrogram(self.spectrogram_batch)
        
        # 驗證轉換結果
        self.assertIsNotNone(features)
        self.assertEqual(features.dim(), 2)
        self.assertEqual(features.size(0), 1)
        
        # 檢查特徵向量維度是否正確
        expected_dim = self.spectrogram_batch.size(1) * self.spectrogram_batch.size(2) * self.spectrogram_batch.size(3)
        self.assertEqual(features.size(1), expected_dim)
        
        logger.info(f"頻譜圖平展為特徵向量，形狀: {features.shape}")
    
    def test_reshape_features_to_2d(self):
        """測試特徵向量重塑為2D圖像"""
        logger.info("測試特徵向量重塑為2D圖像")
        
        # 使用DataAdapter將特徵重塑為2D圖像
        reshape_config = {
            'height': 32,
            'width': 32
        }
        
        image = DataAdapter.reshape_features_to_2d(
            self.features_batch,
            reshape_config
        )
        
        # 驗證轉換結果
        self.assertIsNotNone(image)
        self.assertEqual(image.dim(), 4)
        self.assertEqual(image.size(0), 1)
        self.assertEqual(image.size(2), 32)  # 高度
        self.assertEqual(image.size(3), 32)  # 寬度
        
        logger.info(f"特徵向量重塑為2D圖像，形狀: {image.shape}")
        
        # 測試不同的高寬比
        reshape_config2 = {
            'height': 16,
            'width': 64
        }
        
        image2 = DataAdapter.reshape_features_to_2d(
            self.features_batch,
            reshape_config2
        )
        
        self.assertIsNotNone(image2)
        self.assertEqual(image2.size(2), 16)  # 高度
        self.assertEqual(image2.size(3), 64)  # 寬度
        
        logger.info(f"使用不同高寬比重塑特徵向量，形狀: {image2.shape}")
    
    def test_adapt_batch(self):
        """測試批次適配功能"""
        logger.info("測試批次適配功能")
        
        # 創建測試批次
        audio_batch = {
            'audio': self.audio_batch,
            'label': torch.tensor([[0.0]])
        }
        
        spectrogram_batch = {
            'spectrogram': self.spectrogram_batch,
            'label': torch.tensor([[0.0]])
        }
        
        features_batch = {
            'features': self.features_batch,
            'label': torch.tensor([[0.0]])
        }
        
        # 創建模型配置
        swin_config = {
            'model': {
                'type': 'swin_transformer'
            }
        }
        
        fcnn_config = {
            'model': {
                'type': 'fcnn'
            }
        }
        
        # 測試批次適配
        
        # 音頻 -> Swin Transformer (需要頻譜圖)
        adapted_batch1 = DataAdapter.adapt_batch(audio_batch, 'swin_transformer', swin_config)
        self.assertIn('spectrogram', adapted_batch1)
        logger.info("音頻到Swin Transformer適配成功")
        
        # 頻譜圖 -> FCNN (需要特徵向量)
        adapted_batch2 = DataAdapter.adapt_batch(spectrogram_batch, 'fcnn', fcnn_config)
        self.assertIn('features', adapted_batch2)
        logger.info("頻譜圖到FCNN適配成功")
        
        # 特徵向量 -> CNN (需要圖像)
        adapted_batch3 = DataAdapter.adapt_batch(features_batch, 'cnn', swin_config)
        self.assertIn('spectrogram', adapted_batch3)
        logger.info("特徵向量到CNN適配成功")
    
    def test_edge_cases(self):
        """測試邊界情況處理"""
        logger.info("測試邊界情況處理")
        
        # 創建配置
        swin_config = {
            'model': {
                'type': 'swin_transformer'
            }
        }
        
        # 處理空音頻
        empty_audio = torch.zeros(1, 0)
        
        try:
            # 嘗試轉換空音頻
            spec = DataAdapter.convert_audio_to_spectrogram(empty_audio)
            logger.info(f"空音頻轉換結果: {spec.shape if spec is not None else 'None'}")
        except Exception as e:
            logger.info(f"空音頻轉換異常處理: {str(e)}")
        
        # 處理異常維度的特徵
        odd_features = torch.rand(1, 1023)  # 不是整數平方的維度
        
        image = DataAdapter.reshape_features_to_2d(odd_features)
        self.assertIsNotNone(image)
        logger.info(f"處理非整數平方維度的特徵，輸出形狀: {image.shape}")
        
        # 處理空批次
        empty_batch = {}
        
        adapted = DataAdapter.adapt_batch(empty_batch, 'swin_transformer', swin_config)
        self.assertEqual(empty_batch, adapted)  # 應該保持不變
        logger.info("空批次處理成功")

if __name__ == '__main__':
    unittest.main() 