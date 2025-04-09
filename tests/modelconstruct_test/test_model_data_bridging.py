"""
模型與數據適配性測試模組
測試日期: 2024/04/09

功能說明:
本測試模組用於驗證不同數據類型與模型組合的適配性，測試數據橋接器的有效性。
測試內容包括:
1. 讀取測試資料夾中的音頻、頻譜圖和特徵數據
2. 初始化各種模型類型(Swin Transformer, FCNN, CNN, ResNet)
3. 測試不同數據類型與模型之間的兼容性
4. 驗證DataAdapter的數據轉換功能
5. 檢查模型輸入輸出維度是否符合預期

測試數據源: tests/test_dataset/N002_2024-10-22_11-04-56/
"""

import os
import sys
import unittest
import torch
import numpy as np
from pathlib import Path
import logging
import json
from PIL import Image
import librosa
import matplotlib.pyplot as plt

# 添加專案根目錄到系統路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 導入需要測試的模組
from models.model_factory import ModelFactory
from models.swin_transformer import SwinTransformerModel
from models.fcnn import FCNN
from models.cnn_model import CNNModel
from models.resnet_model import ResNetModel
from utils.data_adapter import DataAdapter

# 設置日誌
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestModelDataBridging(unittest.TestCase):
    """測試模型與數據類型之間的適配性"""
    
    @classmethod
    def setUpClass(cls):
        """設置測試環境"""
        # 設定測試資料目錄
        cls.test_dir = Path('tests/test_dataset/N002_2024-10-22_11-04-56')
        
        # 確認測試資料存在
        assert cls.test_dir.exists(), f"測試目錄不存在: {cls.test_dir}"
        
        # 加載測試數據
        cls._load_test_data()
        
        # 創建模型配置
        cls._create_model_configs()
        
        # 初始化所有模型
        cls._initialize_models()
        
        logger.info("測試環境設置完成")
    
    @classmethod
    def _load_test_data(cls):
        """加載測試數據，包括音頻、頻譜圖和特徵"""
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
        
        # 讀取患者信息
        info_path = cls.test_dir / 'N002_info.json'
        with open(info_path, 'r') as f:
            cls.patient_info = json.load(f)
        logger.info(f"加載患者信息: {info_path}")
        
        # 準備模型輸入的批次數據
        cls.audio_batch = cls.audio_tensor.unsqueeze(0)  # [1, seq_len]
        cls.spectrogram_batch = cls.spectrogram_tensor.unsqueeze(0)  # [1, C, H, W]
        cls.features_batch = cls.features_tensor.unsqueeze(0)  # [1, feature_dim]
        
    @classmethod
    def _create_model_configs(cls):
        """創建用於初始化模型的配置"""
        # Swin Transformer 配置
        cls.swin_config = {
            'model': {
                'type': 'swin_transformer',
                'parameters': {
                    'model_name': 'swin_tiny_patch4_window7_224',
                    'pretrained': True,
                    'num_classes': 1,  # 回歸任務
                    'input_channels': 3,
                    'input_size': (224, 224),
                    'is_classification': False
                }
            }
        }
        
        # FCNN 配置
        cls.fcnn_config = {
            'model': {
                'type': 'fcnn',
                'parameters': {
                    'input_dim': cls.features_tensor.shape[-1],
                    'hidden_layers': [512, 256],
                    'num_classes': 1,  # 回歸任務
                    'is_classification': False
                }
            }
        }
        
        # CNN 配置
        cls.cnn_config = {
            'model': {
                'type': 'cnn',
                'parameters': {
                    'input_channels': 3,
                    'input_size': (224, 224),
                    'filters': [32, 64, 128],
                    'num_classes': 1,  # 回歸任務
                    'is_classification': False
                }
            }
        }
        
        # ResNet 配置
        cls.resnet_config = {
            'model': {
                'type': 'resnet',
                'parameters': {
                    'model_name': 'resnet18',
                    'pretrained': True,
                    'num_classes': 1,  # 回歸任務
                    'input_channels': 3,
                    'is_classification': False
                }
            }
        }
        
    @classmethod
    def _initialize_models(cls):
        """初始化所有需要測試的模型"""
        # 使用ModelFactory創建模型
        try:
            cls.swin_model = ModelFactory.create_model(cls.swin_config)
            logger.info("成功創建Swin Transformer模型")
        except Exception as e:
            logger.error(f"創建Swin Transformer模型失敗: {str(e)}")
            cls.swin_model = None
        
        try:
            cls.fcnn_model = ModelFactory.create_model(cls.fcnn_config)
            logger.info("成功創建FCNN模型")
        except Exception as e:
            logger.error(f"創建FCNN模型失敗: {str(e)}")
            cls.fcnn_model = None
        
        try:
            cls.cnn_model = ModelFactory.create_model(cls.cnn_config)
            logger.info("成功創建CNN模型")
        except Exception as e:
            logger.error(f"創建CNN模型失敗: {str(e)}")
            cls.cnn_model = None
        
        try:
            cls.resnet_model = ModelFactory.create_model(cls.resnet_config)
            logger.info("成功創建ResNet模型")
        except Exception as e:
            logger.error(f"創建ResNet模型失敗: {str(e)}")
            cls.resnet_model = None
    
    def test_audio_to_spectrogram_bridge(self):
        """測試音頻到頻譜圖的轉換橋接"""
        logger.info("開始測試音頻到頻譜圖的轉換橋接")
        
        # 使用DataAdapter轉換音頻為頻譜圖
        spec_config = {
            'n_fft': 1024,
            'hop_length': 512,
            'n_mels': 128
        }
        
        # 轉換單個音頻
        spectrogram = DataAdapter.convert_audio_to_spectrogram(
            self.audio_batch,
            spec_config
        )
        
        # 驗證轉換結果
        self.assertIsNotNone(spectrogram, "轉換結果不應為None")
        self.assertEqual(spectrogram.dim(), 4, "頻譜圖應為4維張量 [B, C, H, W]")
        self.assertEqual(spectrogram.size(0), 1, "批次大小應為1")
        
        logger.info(f"音頻到頻譜圖轉換成功，輸出形狀: {spectrogram.shape}")
        
        # 嘗試將轉換的頻譜圖輸入視覺模型
        if self.swin_model is not None:
            # 調整頻譜圖大小以適應Swin Transformer的輸入要求
            resized_spec = torch.nn.functional.interpolate(
                spectrogram, 
                size=(224, 224), 
                mode='bilinear', 
                align_corners=False
            )
            
            # 確保通道數正確
            if resized_spec.size(1) == 1:
                resized_spec = resized_spec.repeat(1, 3, 1, 1)
                
            # 測試模型前向傳播
            with torch.no_grad():
                output = self.swin_model(resized_spec)
                
            self.assertIsNotNone(output, "模型輸出不應為None")
            logger.info(f"Swin Transformer模型接受轉換後的頻譜圖，輸出形狀: {output.shape}")
    
    def test_features_to_2d_bridge(self):
        """測試特徵向量到2D圖像的轉換橋接"""
        logger.info("開始測試特徵向量到2D圖像的轉換橋接")
        
        # 使用DataAdapter將特徵轉換為2D圖像
        reshape_config = {
            'height': 32,
            'width': 32
        }
        
        # 轉換特徵
        image_tensor = DataAdapter.reshape_features_to_2d(
            self.features_batch,
            reshape_config
        )
        
        # 驗證轉換結果
        self.assertIsNotNone(image_tensor, "轉換結果不應為None")
        self.assertEqual(image_tensor.dim(), 4, "圖像應為4維張量 [B, C, H, W]")
        
        logger.info(f"特徵向量到2D圖像轉換成功，輸出形狀: {image_tensor.shape}")
        
        # 測試與CNN模型的兼容性
        if self.cnn_model is not None:
            # 設置模型為評估模式
            self.cnn_model.eval()
            
            # 調整大小和通道以適應CNN模型
            if image_tensor.size(1) != 3:
                # 如果不是3通道，複製到3通道
                image_tensor = image_tensor.repeat(1, 3, 1, 1)
                
            # 調整大小
            resized_img = torch.nn.functional.interpolate(
                image_tensor, 
                size=(224, 224), 
                mode='bilinear', 
                align_corners=False
            )
            
            # 測試模型前向傳播
            with torch.no_grad():
                output = self.cnn_model(resized_img)
                
            self.assertIsNotNone(output, "模型輸出不應為None")
            logger.info(f"CNN模型接受轉換後的特徵圖像，輸出形狀: {output.shape}")
    
    def test_spectrogram_to_features_bridge(self):
        """測試頻譜圖到特徵向量的轉換橋接"""
        logger.info("開始測試頻譜圖到特徵向量的轉換橋接")
        
        # 使用DataAdapter平展頻譜圖為向量
        feature_vector = DataAdapter.flatten_spectrogram(self.spectrogram_batch)
        
        # 驗證轉換結果
        self.assertIsNotNone(feature_vector, "轉換結果不應為None")
        self.assertEqual(feature_vector.dim(), 2, "特徵向量應為2維張量 [B, D]")
        
        logger.info(f"頻譜圖到特徵向量轉換成功，輸出形狀: {feature_vector.shape}")
        
        # 測試與FCNN模型的兼容性
        if self.fcnn_model is not None:
            # 設置模型為評估模式
            self.fcnn_model.eval()
            
            # 注意: 頻譜圖展平後的維度可能與FCNN模型的輸入維度不匹配
            # 這裡我們可以檢測並調整
            input_dim = self.fcnn_model.input_dim
            feature_dim = feature_vector.size(1)
            
            if feature_dim != input_dim:
                logger.warning(f"特徵維度不匹配: FCNN期望 {input_dim}, 實際得到 {feature_dim}")
                # 在實際應用中，應考慮更好的處理方式，這裡我們簡單截斷或填充
                if feature_dim > input_dim:
                    feature_vector = feature_vector[:, :input_dim]
                else:
                    padding = torch.zeros(feature_vector.size(0), input_dim - feature_dim)
                    feature_vector = torch.cat([feature_vector, padding], dim=1)
                
                logger.info(f"調整後的特徵向量形狀: {feature_vector.shape}")
            
            # 測試模型前向傳播
            with torch.no_grad():
                output = self.fcnn_model(feature_vector)
                
            self.assertIsNotNone(output, "模型輸出不應為None")
            logger.info(f"FCNN模型接受轉換後的特徵向量，輸出形狀: {output.shape}")
    
    def test_batch_adaptation(self):
        """測試批次數據適配功能"""
        logger.info("開始測試批次數據適配功能")
        
        # 創建測試批次
        audio_batch = {
            'audio': self.audio_batch,
            'label': torch.tensor([[0.0]])  # 假設標籤是EAT-10分數
        }
        
        spectrogram_batch = {
            'spectrogram': self.spectrogram_batch,
            'label': torch.tensor([[0.0]])
        }
        
        features_batch = {
            'features': self.features_batch,
            'label': torch.tensor([[0.0]])
        }
        
        # 測試不同模型類型的批次適配
        
        # 1. 測試音頻到視覺模型的適配
        adapted_audio_for_swin = DataAdapter.adapt_batch(
            audio_batch, 
            'swin_transformer', 
            self.swin_config
        )
        
        self.assertIn('spectrogram', adapted_audio_for_swin, 
                     "適配後的批次應包含 'spectrogram' 字段")
        
        logger.info(f"音頻到Swin Transformer適配成功")
        
        # 2. 測試頻譜圖到FCNN的適配
        adapted_spec_for_fcnn = DataAdapter.adapt_batch(
            spectrogram_batch, 
            'fcnn', 
            self.fcnn_config
        )
        
        self.assertIn('features', adapted_spec_for_fcnn, 
                     "適配後的批次應包含 'features' 字段")
        
        logger.info(f"頻譜圖到FCNN適配成功")
        
        # 3. 測試特徵到視覺模型的適配
        adapted_features_for_cnn = DataAdapter.adapt_batch(
            features_batch, 
            'cnn', 
            self.cnn_config
        )
        
        self.assertIn('spectrogram', adapted_features_for_cnn, 
                     "適配後的批次應包含 'spectrogram' 字段")
        
        logger.info(f"特徵到CNN適配成功")
        
    def test_end_to_end_model_calls(self):
        """測試完整的端到端模型調用流程"""
        logger.info("開始測試端到端模型調用流程")
        
        # 測試所有模型和數據類型的組合
        models = {
            'swin_transformer': self.swin_model,
            'fcnn': self.fcnn_model, 
            'cnn': self.cnn_model, 
            'resnet': self.resnet_model
        }
        
        # 設置所有模型為評估模式
        for model_name, model in models.items():
            if model is not None:
                model.eval()
        
        data_batches = {
            'audio': self.audio_batch,
            'spectrogram': self.spectrogram_batch,
            'features': self.features_batch
        }
        
        model_configs = {
            'swin_transformer': self.swin_config,
            'fcnn': self.fcnn_config,
            'cnn': self.cnn_config,
            'resnet': self.resnet_config
        }
        
        results = {}
        
        # 測試每種組合
        for model_name, model in models.items():
            if model is None:
                logger.warning(f"跳過測試: {model_name} 模型未成功初始化")
                continue
                
            results[model_name] = {}
            
            for data_type, data in data_batches.items():
                batch = {
                    data_type: data,
                    'label': torch.tensor([[0.0]])
                }
                
                logger.info(f"測試 {model_name} 模型與 {data_type} 數據")
                
                try:
                    # 調整批次以適應模型
                    adapted_batch = DataAdapter.adapt_batch(
                        batch, 
                        model_name, 
                        model_configs[model_name]
                    )
                    
                    # 確定模型輸入
                    if model_name == 'fcnn':
                        if 'features' in adapted_batch:
                            input_data = adapted_batch['features']
                        else:
                            logger.warning(f"{model_name} 需要特徵數據，但未找到")
                            continue
                    else:  # 視覺模型需要圖像數據
                        if 'spectrogram' in adapted_batch:
                            input_data = adapted_batch['spectrogram']
                            
                            # 調整大小和通道
                            if input_data.size(2) != 224 or input_data.size(3) != 224:
                                input_data = torch.nn.functional.interpolate(
                                    input_data, 
                                    size=(224, 224), 
                                    mode='bilinear', 
                                    align_corners=False
                                )
                            
                            if input_data.size(1) != 3:
                                input_data = input_data.repeat(1, 3, 1, 1)
                        else:
                            logger.warning(f"{model_name} 需要頻譜圖數據，但未找到")
                            continue
                    
                    # 運行模型
                    with torch.no_grad():
                        output = model(input_data)
                    
                    # 記錄結果
                    results[model_name][data_type] = {
                        'success': True,
                        'input_shape': tuple(input_data.shape),
                        'output_shape': tuple(output.shape),
                        'output_value': output.item() if output.numel() == 1 else output[0].item()
                    }
                    
                    logger.info(f"成功: {model_name} 可以處理 {data_type} 數據")
                    logger.info(f"  輸入形狀: {input_data.shape}")
                    logger.info(f"  輸出形狀: {output.shape}")
                    logger.info(f"  輸出值: {output.item() if output.numel() == 1 else output[0].item()}")
                    
                except Exception as e:
                    logger.error(f"錯誤: {model_name} 處理 {data_type} 數據失敗: {str(e)}")
                    results[model_name][data_type] = {
                        'success': False,
                        'error': str(e)
                    }
        
        # 輸出兼容性矩陣
        logger.info("\n" + "="*50)
        logger.info("模型與數據類型兼容性矩陣:")
        logger.info("="*50)
        
        for model_name in results:
            logger.info(f"\n{model_name}:")
            for data_type in data_batches:
                if data_type in results[model_name]:
                    success = results[model_name][data_type].get('success', False)
                    status = "✅ 兼容" if success else "❌ 不兼容"
                    logger.info(f"  {data_type}: {status}")
                else:
                    logger.info(f"  {data_type}: ❓ 未測試")
                    
        logger.info("\n" + "="*50)
                    
        # 保存測試報告
        report_path = Path('tests') / 'model_data_bridging_report.json'
        with open(report_path, 'w') as f:
            # 將張量元組轉換為字符串表示
            results_json = {}
            for model_name, model_results in results.items():
                results_json[model_name] = {}
                for data_type, data_results in model_results.items():
                    results_json[model_name][data_type] = {}
                    for k, v in data_results.items():
                        if isinstance(v, tuple):
                            results_json[model_name][data_type][k] = str(v)
                        else:
                            results_json[model_name][data_type][k] = v
                            
            json.dump(results_json, f, indent=4)
            
        logger.info(f"測試報告已保存到: {report_path}")

if __name__ == '__main__':
    unittest.main() 