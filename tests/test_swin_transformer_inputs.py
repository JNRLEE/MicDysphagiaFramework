"""
Swin Transformer 模型輸入數據測試腳本
用途：分析模型在實際訓練和推理過程中接收的輸入數據，以及輸出預測的特性
這個腳本會載入訓練好的模型，並檢查其對不同輸入的響應，以診斷可能的數據問題
"""

import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import librosa.display
from PIL import Image
from pathlib import Path
import random
from datetime import datetime
import json

# 將專案根目錄加入到路徑中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model_factory import create_model
from utils.data_adapter import DataAdapter
from data.dataset_factory import create_dataset
from utils.data_index_loader import DataIndexLoader
from trainers.trainer_factory import create_trainer

# 設置輸出目錄
OUTPUT_DIR = Path('tests/swin_transformer_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_subdir = OUTPUT_DIR / f"test_{timestamp}"
output_subdir.mkdir(exist_ok=True)

# 模型分析配置
MODEL_CONFIG_PATH = "config/example_classification_drlee.yaml"
MODEL_CHECKPOINT_PATH = "results/indexed_classification_drlee_20250519_012024/models/best_model.pth"

def save_tensor_image(tensor, filepath, title="張量圖像", cmap='viridis'):
    """保存張量為圖像"""
    plt.figure(figsize=(10, 8))
    
    if len(tensor.shape) == 2:  # 單通道
        plt.imshow(tensor, cmap=cmap)
        plt.colorbar()
    elif len(tensor.shape) == 3 and tensor.shape[0] == 3:  # RGB
        plt.imshow(np.transpose(tensor, (1, 2, 0)))
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def save_feature_histograms(features, filepath_prefix, title_prefix="特徵分佈"):
    """保存特徵分佈直方圖"""
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    
    # 保存所有特徵的分佈
    plt.figure(figsize=(12, 6))
    plt.hist(features.flatten(), bins=50)
    plt.title(f"{title_prefix} - 所有值")
    plt.xlabel("特徵值")
    plt.ylabel("頻率")
    plt.tight_layout()
    plt.savefig(f"{filepath_prefix}_all.png")
    plt.close()
    
    # 保存每個通道的分佈（如果是多通道）
    if len(features.shape) == 4:  # [B, C, H, W]
        for c in range(features.shape[1]):
            plt.figure(figsize=(12, 6))
            plt.hist(features[0, c].flatten(), bins=50)
            plt.title(f"{title_prefix} - 通道 {c}")
            plt.xlabel("特徵值")
            plt.ylabel("頻率")
            plt.tight_layout()
            plt.savefig(f"{filepath_prefix}_channel_{c}.png")
            plt.close()

def load_config(config_path):
    """載入YAML配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def prepare_model(config, checkpoint_path=None):
    """準備模型"""
    # 創建模型
    model = create_model(config)
    
    # 載入權重（如果有）
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"載入模型權重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model.eval()  # 設置為評估模式
    return model

def get_sample_batch(config, num_samples=5):
    """從數據集中獲取樣本批次"""
    # 創建數據集和數據加載器
    train_dataset, val_dataset, test_dataset = create_dataset(config)
    
    # 使用測試集（如果有），否則使用驗證集
    dataset = test_dataset if test_dataset is not None else val_dataset
    
    if dataset is None:
        raise ValueError("無法創建數據集，請檢查配置")
    
    # 隨機選擇樣本
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    samples = [dataset[i] for i in indices]
    
    # 將樣本合併為批次
    batch = {}
    
    # 檢查樣本的類型，處理不同的數據格式
    sample_item = samples[0]
    
    # 如果樣本是字典，直接處理
    if isinstance(sample_item, dict):
        for key in sample_item.keys():
            if isinstance(sample_item[key], torch.Tensor):
                batch[key] = torch.stack([s[key] for s in samples])
            elif isinstance(sample_item[key], (str, int, float)):
                batch[key] = [s[key] for s in samples]
    
    # 如果樣本是元組（通常是 (data, label) 的形式）
    elif isinstance(sample_item, tuple):
        # 假設元組的第一個元素是數據，第二個元素是標籤
        if len(sample_item) >= 2:
            # 處理數據
            if isinstance(sample_item[0], torch.Tensor):
                batch['spectrogram'] = torch.stack([s[0] for s in samples])
            
            # 處理標籤
            if isinstance(sample_item[1], torch.Tensor):
                batch['label'] = torch.stack([s[1] for s in samples])
            elif isinstance(sample_item[1], (int, float)):
                batch['label'] = torch.tensor([s[1] for s in samples])
    
    # 如果樣本是張量，假設只有輸入數據
    elif isinstance(sample_item, torch.Tensor):
        batch['spectrogram'] = torch.stack(samples)
    
    # 檢查批次是否包含必要的數據
    if 'spectrogram' not in batch:
        raise ValueError("無法從樣本中提取'spectrogram'數據")
    
    return batch, indices

def analyze_model_input_output(model, batch, output_dir):
    """分析模型的輸入和輸出"""
    # 確保在CPU上運行
    device = torch.device('cpu')
    model = model.to(device)
    
    # 記錄原始批次信息
    original_batch_info = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            original_batch_info[key] = {
                "shape": value.shape,
                "dtype": str(value.dtype),
                "min": float(value.min()),
                "max": float(value.max()),
                "mean": float(value.mean()),
                "std": float(value.std())
            }
        else:
            original_batch_info[key] = str(value)
    
    with open(output_dir / "batch_info.json", 'w') as f:
        json.dump(original_batch_info, f, indent=2)
    
    # 使用 DataAdapter 調整批次數據
    print("使用 DataAdapter 調整批次數據...")
    if 'spectrogram' in batch and len(batch['spectrogram'].shape) == 3:  # [B, C, L]
        # 如果是音頻數據，則轉換為頻譜圖
        if batch['spectrogram'].shape[1] == 1:  # 單通道音頻
            print(f"檢測到音頻數據: {batch['spectrogram'].shape}")
            
            # 使用 DataAdapter 進行處理
            dummy_config = {
                "data": {
                    "preprocessing": {
                        "spectrogram": {
                            "method": "mel_spectrogram",
                            "n_mels": 128,
                            "n_fft": 1024,
                            "hop_length": 512,
                            "log_mel": True
                        }
                    }
                }
            }
            
            # 創建包含音頻的批次
            audio_batch = {"audio": batch['spectrogram']}
            
            # 適配批次
            adapted_batch = DataAdapter.adapt_batch(audio_batch, "swin_transformer", dummy_config)
            
            if 'spectrogram' in adapted_batch:
                batch['spectrogram'] = adapted_batch['spectrogram']
                print(f"已轉換為 spectrogram: {batch['spectrogram'].shape}")
    
    # 檢查並確保頻譜圖大小正確
    if 'spectrogram' in batch:
        input_tensor = batch['spectrogram']
        
        # 確保是3通道 [B, 3, H, W]
        if len(input_tensor.shape) == 3 and input_tensor.shape[1] == 1:  # [B, 1, L]
            # 將音頻轉換為頻譜圖
            print(f"將音頻 {input_tensor.shape} 轉換為頻譜圖...")
            
            # 首先將音頻轉換為梅爾頻譜圖 [B, 1, n_mels, T]
            spectrograms = []
            for i in range(input_tensor.shape[0]):
                audio = input_tensor[i, 0].cpu().numpy()
                mel_spec = librosa.feature.melspectrogram(
                    y=audio, 
                    sr=16000, 
                    n_fft=1024, 
                    hop_length=512, 
                    n_mels=128
                )
                # 轉換為對數尺度
                mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                spectrograms.append(torch.from_numpy(mel_spec).unsqueeze(0))  # [1, n_mels, T]
            
            # 堆疊所有頻譜圖
            spec_tensor = torch.stack(spectrograms)  # [B, 1, n_mels, T]
            
            # 調整大小
            spec_tensor = torch.nn.functional.interpolate(
                spec_tensor,
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            )
            
            # 轉換為3通道
            spec_tensor = spec_tensor.repeat(1, 3, 1, 1)
            
            # 更新批次
            batch['spectrogram'] = spec_tensor
            input_tensor = spec_tensor
            print(f"已轉換為3通道頻譜圖: {input_tensor.shape}")
        
        elif len(input_tensor.shape) == 3:  # [B, F, T] - 特徵序列
            # 將特徵序列調整為 [B, 3, 224, 224]
            print(f"將特徵序列 {input_tensor.shape} 調整為圖像格式...")
            
            # 調整為 [B, 1, sqrt(F), T]
            B, F, T = input_tensor.shape
            H = int(np.sqrt(F))
            W = F // H if F % H == 0 else F // H + 1
            
            # 填充到完整的形狀
            padded = torch.zeros(B, 1, H * W, T, device=input_tensor.device)
            padded[:, 0, :F, :] = input_tensor
            
            # 調整為 [B, 1, H, W*T]
            reshaped = padded.reshape(B, 1, H, W*T)
            
            # 調整大小為 [B, 1, 224, 224]
            resized = torch.nn.functional.interpolate(
                reshaped,
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            )
            
            # 轉換為3通道
            resized = resized.repeat(1, 3, 1, 1)
            
            # 更新批次
            batch['spectrogram'] = resized
            input_tensor = resized
            print(f"已調整為圖像格式: {input_tensor.shape}")
        
        elif len(input_tensor.shape) == 4:  # [B, C, H, W] - 可能已經是圖像格式
            # 確保是3通道
            if input_tensor.shape[1] != 3:
                if input_tensor.shape[1] == 1:
                    # 單通道轉換為3通道
                    input_tensor = input_tensor.repeat(1, 3, 1, 1)
                else:
                    # 保留最多3個通道
                    input_tensor = input_tensor[:, :3, :, :]
                
                # 更新批次
                batch['spectrogram'] = input_tensor
                print(f"已調整通道數: {input_tensor.shape}")
            
            # 確保尺寸為 224x224
            if input_tensor.shape[2] != 224 or input_tensor.shape[3] != 224:
                # 調整大小
                input_tensor = torch.nn.functional.interpolate(
                    input_tensor,
                    size=(224, 224),
                    mode='bilinear',
                    align_corners=False
                )
                
                # 更新批次
                batch['spectrogram'] = input_tensor
                print(f"已調整尺寸: {input_tensor.shape}")
        
        # 保存調整後的輸入張量的統計信息
        input_stats = {
            "shape": input_tensor.shape,
            "dtype": str(input_tensor.dtype),
            "min": float(input_tensor.min()),
            "max": float(input_tensor.max()),
            "mean": float(input_tensor.mean()),
            "std": float(input_tensor.std())
        }
        
        with open(output_dir / "input_stats.json", 'w') as f:
            json.dump(input_stats, f, indent=2)
        
        # 可視化輸入張量的分佈
        save_feature_histograms(input_tensor, str(output_dir / "input_distribution"), "輸入張量分佈")
        
        # 可視化每個樣本的輸入
        for i in range(min(input_tensor.shape[0], 5)):
            sample_dir = output_dir / f"sample_{i}"
            sample_dir.mkdir(exist_ok=True)
            
            # 保存輸入圖像
            save_tensor_image(input_tensor[i].cpu().numpy(), 
                            sample_dir / "input_image.png", 
                            f"輸入圖像 - 樣本 {i}")
    
    # 前向傳播
    with torch.no_grad():
        try:
            # 準備標籤（如果存在）
            label_tensor = None
            if 'label' in batch and isinstance(batch['label'], torch.Tensor):
                label_tensor = batch['label'].to(device)
            
            # 前向傳播
            outputs = model(batch['spectrogram'].to(device))
            
            # 保存輸出統計信息
            output_stats = {}
            
            if isinstance(outputs, dict):
                for key, value in outputs.items():
                    if isinstance(value, torch.Tensor):
                        output_stats[key] = {
                            "shape": value.shape,
                            "min": float(value.min()),
                            "max": float(value.max()),
                            "mean": float(value.mean()),
                            "std": float(value.std())
                        }
            else:
                output_stats["outputs"] = {
                    "shape": outputs.shape,
                    "min": float(outputs.min()),
                    "max": float(outputs.max()),
                    "mean": float(outputs.mean()),
                    "std": float(outputs.std())
                }
            
            with open(output_dir / "output_stats.json", 'w') as f:
                json.dump(output_stats, f, indent=2)
            
            # 可視化輸出分佈
            if isinstance(outputs, dict) and 'logits' in outputs:
                save_feature_histograms(outputs['logits'], 
                                      str(output_dir / "output_logits_distribution"), 
                                      "輸出Logits分佈")
            elif not isinstance(outputs, dict):
                save_feature_histograms(outputs, 
                                      str(output_dir / "output_distribution"), 
                                      "輸出分佈")
            
            # 獲取預測結果
            if isinstance(outputs, dict) and 'logits' in outputs:
                predictions = torch.softmax(outputs['logits'], dim=1)
            else:
                predictions = torch.softmax(outputs, dim=1)
            
            # 保存預測結果
            pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
            pred_probs = predictions.cpu().numpy()
            
            prediction_results = []
            for i in range(len(pred_classes)):
                sample_result = {
                    "predicted_class": int(pred_classes[i]),
                    "prediction_probabilities": pred_probs[i].tolist()
                }
                
                if label_tensor is not None:
                    sample_result["true_label"] = int(label_tensor[i])
                    sample_result["correct"] = (pred_classes[i] == label_tensor[i].item())
                
                prediction_results.append(sample_result)
            
            with open(output_dir / "prediction_results.json", 'w') as f:
                json.dump(prediction_results, f, indent=2)
            
            # 可視化預測概率分佈
            for i in range(min(predictions.shape[0], 5)):
                plt.figure(figsize=(10, 6))
                plt.bar(range(predictions.shape[1]), predictions[i].cpu().numpy())
                plt.xlabel("類別")
                plt.ylabel("概率")
                plt.title(f"樣本 {i} 的預測概率分佈")
                plt.xticks(range(predictions.shape[1]))
                plt.tight_layout()
                plt.savefig(output_dir / f"sample_{i}" / "prediction_probs.png")
                plt.close()
            
            return True, prediction_results
            
        except Exception as e:
            print(f"模型前向傳播錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            
            with open(output_dir / "error.txt", 'w') as f:
                f.write(f"模型前向傳播錯誤: {str(e)}\n")
                f.write(traceback.format_exc())
            
            return False, None

def analyze_model_structure(model, output_dir):
    """分析模型結構，記錄各層的參數和形狀"""
    model_structure = []
    
    # 遍歷模型的命名模塊
    for name, module in model.named_modules():
        if name == '':  # 跳過根模塊
            continue
        
        # 獲取模塊參數數量
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        # 記錄模塊信息
        module_info = {
            "name": name,
            "type": module.__class__.__name__,
            "parameters": params
        }
        
        # 如果模塊有 in_features 或 out_features 屬性（例如線性層）
        if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
            module_info["in_features"] = module.in_features
            module_info["out_features"] = module.out_features
        
        # 如果模塊有 in_channels 或 out_channels 屬性（例如卷積層）
        if hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
            module_info["in_channels"] = module.in_channels
            module_info["out_channels"] = module.out_channels
        
        model_structure.append(module_info)
    
    # 記錄模型總參數數量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 保存模型結構信息
    with open(output_dir / "model_structure.json", 'w') as f:
        json.dump({
            "total_parameters": total_params,
            "modules": model_structure
        }, f, indent=2)

def analyze_weight_distributions(model, output_dir):
    """分析模型權重的分佈"""
    # 獲取所有參數
    weights_dir = output_dir / "weight_distributions"
    weights_dir.mkdir(exist_ok=True)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # 獲取權重值並轉換為 numpy 數組
            weight_np = param.data.cpu().numpy()
            
            # 計算統計信息
            stats = {
                "name": name,
                "shape": list(weight_np.shape),
                "min": float(np.min(weight_np)),
                "max": float(np.max(weight_np)),
                "mean": float(np.mean(weight_np)),
                "std": float(np.std(weight_np))
            }
            
            # 保存統計信息
            with open(weights_dir / f"{name.replace('.', '_')}_stats.json", 'w') as f:
                json.dump(stats, f, indent=2)
            
            # 繪製權重分佈直方圖
            plt.figure(figsize=(10, 6))
            plt.hist(weight_np.flatten(), bins=50)
            plt.title(f"權重分佈 - {name}")
            plt.xlabel("權重值")
            plt.ylabel("頻率")
            plt.tight_layout()
            plt.savefig(weights_dir / f"{name.replace('.', '_')}_distribution.png")
            plt.close()

def main():
    """主函數"""
    print(f"開始 Swin Transformer 模型輸入輸出分析，結果將保存在 {output_subdir}")
    
    # 載入配置
    config = load_config(MODEL_CONFIG_PATH)
    print(f"已載入配置: {MODEL_CONFIG_PATH}")
    
    # 保存配置副本
    with open(output_subdir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # 準備模型
    model = prepare_model(config, MODEL_CHECKPOINT_PATH)
    print(f"已準備模型")
    
    # 分析模型結構
    analyze_model_structure(model, output_subdir)
    print(f"已分析模型結構")
    
    # 分析權重分佈
    analyze_weight_distributions(model, output_subdir)
    print(f"已分析權重分佈")
    
    # 獲取樣本批次
    try:
        batch, indices = get_sample_batch(config)
        print(f"已獲取 {len(indices)} 個樣本")
        
        # 保存樣本索引
        with open(output_subdir / "sample_indices.json", 'w') as f:
            json.dump({"indices": indices}, f)
        
        # 分析模型輸入輸出
        success, prediction_results = analyze_model_input_output(model, batch, output_subdir)
        
        if success:
            print(f"成功分析模型輸入輸出")
            
            # 分析預測結果
            if prediction_results:
                class_counts = {}
                for pred in prediction_results:
                    pred_class = pred["predicted_class"]
                    if pred_class not in class_counts:
                        class_counts[pred_class] = 0
                    class_counts[pred_class] += 1
                
                print(f"預測類別分佈: {class_counts}")
                
                # 檢查是否所有預測都是同一類別
                if len(class_counts) == 1:
                    print(f"警告: 所有樣本都被預測為同一類別 ({list(class_counts.keys())[0]})")
                    
                    # 檢查預測概率是否接近於 1
                    probs = [max(pred["prediction_probabilities"]) for pred in prediction_results]
                    avg_prob = sum(probs) / len(probs)
                    print(f"平均預測概率: {avg_prob:.4f}")
                    
                    if avg_prob > 0.9:
                        print("警告: 預測概率非常高，這可能表明模型對某一類別過度自信")
        else:
            print(f"分析模型輸入輸出時出錯，請查看錯誤日誌")
    
    except Exception as e:
        print(f"執行過程中出錯: {str(e)}")
        import traceback
        traceback.print_exc()
        
        with open(output_subdir / "error.txt", 'w') as f:
            f.write(f"執行過程中出錯: {str(e)}\n")
            f.write(traceback.format_exc())
    
    print(f"分析完成，所有結果保存在 {output_subdir}")

if __name__ == "__main__":
    main() 