"""
全面修復 Swin Transformer 模型只預測單一類別的問題
解決方案包括:
1. 重置輸出層的權重和偏差
2. 應用類別權重以處理不平衡問題
3. 調整預處理步驟以確保正確的輸入格式
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
import yaml
import librosa
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

# 將專案根目錄加入到路徑中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 導入專案模塊
from models.model_factory import create_model
from models.swin_transformer import SwinTransformerModel
from data.dataset_factory import create_dataset
from utils.data_adapter import DataAdapter

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

def load_config(config_path):
    """載入配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model(config, checkpoint_path=None):
    """載入模型"""
    logger.info("建立模型")
    model = create_model(config)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"載入模型權重: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            logger.info("模型載入成功")
        except Exception as e:
            logger.error(f"載入模型時出錯: {str(e)}")
    
    return model

def analyze_model_parameters(model):
    """詳細分析模型參數"""
    logger.info("分析模型參數")
    
    # 分析輸出層參數
    output_params = {}
    for name, param in model.named_parameters():
        if 'head' in name or 'fc' in name or 'output' in name or 'classifier' in name:
            param_np = param.data.cpu().numpy()
            output_params[name] = {
                'shape': param.shape,
                'mean': float(np.mean(param_np)),
                'std': float(np.std(param_np)),
                'min': float(np.min(param_np)),
                'max': float(np.max(param_np)),
                'abs_mean': float(np.mean(np.abs(param_np)))
            }
            logger.info(f"輸出層參數 {name}: 形狀={param.shape}, 均值={output_params[name]['mean']:.6f}, 標準差={output_params[name]['std']:.6f}")
    
    return output_params

def fix_model_weights(model, fix_method="reset_output"):
    """修復模型權重和偏差"""
    logger.info(f"使用 {fix_method} 方法修復模型")
    
    # 深度複製模型
    fixed_model = None
    
    if isinstance(model, SwinTransformerModel):
        # 需要手動複製權重
        logger.info("為 SwinTransformerModel 建立新實例")
        num_classes = model.head[1].out_features
        fixed_model = SwinTransformerModel(num_classes=num_classes)
        
        # 基於修復方法，處理模型複製
        if fix_method == "full_copy":
            # 完整複製所有參數
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in dict(fixed_model.named_parameters()):
                        dict(fixed_model.named_parameters())[name].copy_(param)
            logger.info("完整複製了所有模型參數")
        
        elif fix_method == "reset_output":
            # 複製除輸出層外的所有參數
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in dict(fixed_model.named_parameters()):
                        if 'head' not in name:
                            dict(fixed_model.named_parameters())[name].copy_(param)
            logger.info("複製了非輸出層參數，保持輸出層為隨機初始化")
        
        elif fix_method == "modify_head":
            # 複製所有參數，但修改輸出層權重的分佈
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in dict(fixed_model.named_parameters()):
                        if 'head' in name and 'weight' in name:
                            # 分析原始權重
                            original_weight = param.data.clone()
                            mean = torch.mean(original_weight)
                            std = torch.std(original_weight)
                            
                            # 使用相同的均值和標準差重新初始化權重
                            new_weight = torch.randn_like(original_weight) * std + mean
                            dict(fixed_model.named_parameters())[name].copy_(new_weight)
                            logger.info(f"修改了 {name} 的權重分佈，保持相同的統計特性")
                        elif 'head' in name and 'bias' in name:
                            # 將偏差設置為0
                            dict(fixed_model.named_parameters())[name].zero_()
                            logger.info(f"將 {name} 的偏差設置為零")
                        else:
                            dict(fixed_model.named_parameters())[name].copy_(param)
    else:
        # 普通模型複製
        logger.info("為一般模型建立深度副本")
        import copy
        fixed_model = copy.deepcopy(model)
        
        # 根據修復方法進行額外操作
        if fix_method == "reset_output" or fix_method == "modify_head":
            with torch.no_grad():
                for name, param in fixed_model.named_parameters():
                    if ('head' in name or 'fc' in name or 'output' in name or 'classifier' in name):
                        if 'weight' in name:
                            if fix_method == "reset_output":
                                # 重置權重
                                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                                logger.info(f"重置了 {name} 的權重")
                            else:  # modify_head
                                # 重新縮放權重
                                mean = torch.mean(param.data)
                                std = torch.std(param.data)
                                param.data = (param.data - mean) / (std + 1e-6) * 0.01 + mean
                                logger.info(f"重新縮放了 {name} 的權重")
                        elif 'bias' in name:
                            # 將偏差設置為0
                            param.data.zero_()
                            logger.info(f"將 {name} 的偏差設置為零")
    
    return fixed_model

def add_class_weights_to_model(model, class_weights):
    """在模型中添加類別權重處理不平衡問題"""
    logger.info(f"為模型添加類別權重: {class_weights}")
    
    # 如果是SwinTransformerModel，手動修改輸出層偏差
    if isinstance(model, SwinTransformerModel):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'head' in name and 'bias' in name:
                    # 應用對數空間的類別權重調整
                    bias_adjustment = torch.tensor(np.log(class_weights), dtype=param.dtype)
                    param.data += bias_adjustment
                    logger.info(f"已將類別權重應用到 {name}: {param.data}")
    
    return model

def create_test_spectrogram(audio_path=None):
    """創建用於測試的頻譜圖"""
    logger.info("創建測試頻譜圖輸入")
    
    if audio_path and os.path.exists(audio_path):
        # 從真實音頻文件載入
        logger.info(f"從音頻文件載入: {audio_path}")
        y, sr = librosa.load(audio_path, sr=16000)
    else:
        # 創建合成音頻
        logger.info("創建合成音頻測試數據")
        sr = 16000
        duration = 5  # 秒
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        
        # 創建一個包含多個頻率的信號
        freqs = [300, 800, 1200, 2000]
        y = np.zeros_like(t)
        for freq in freqs:
            y += np.sin(2 * np.pi * freq * t)
        y = y / len(freqs)  # 標準化
    
    # 計算梅爾頻譜圖
    S = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_mels=128,
        n_fft=1024,
        hop_length=512
    )
    
    # 轉換為分貝刻度
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # 轉換為張量，確保是float32類型
    spec_tensor = torch.from_numpy(S_db).unsqueeze(0).float()  # (1, H, W) as float32
    
    # 調整大小為模型需要的尺寸 (通常是224x224)
    spec_tensor = torch.nn.functional.interpolate(
        spec_tensor.unsqueeze(0),  # (1, 1, H, W)
        size=(224, 224),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)  # (1, 224, 224)
    
    # 複製到3通道
    spec_tensor = spec_tensor.repeat(3, 1, 1)  # (3, 224, 224)
    
    logger.info(f"創建了測試頻譜圖，形狀為 {spec_tensor.shape}，類型為 {spec_tensor.dtype}")
    
    return spec_tensor.unsqueeze(0)  # 添加批次維度: (1, 3, 224, 224)

def analyze_real_data(config, num_samples=5):
    """分析真實數據集的特徵分佈"""
    logger.info(f"分析真實數據集的特徵分佈 (樣本數: {num_samples})")
    
    # 創建數據集
    train_dataset, val_dataset, test_dataset = create_dataset(config)
    dataset = val_dataset if val_dataset is not None else train_dataset
    
    if dataset is None:
        logger.error("無法創建數據集")
        return None
    
    # 獲取數據樣本
    samples = []
    labels = []
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        
        # 提取輸入數據和標籤
        if isinstance(sample, tuple) and len(sample) >= 2:
            inputs, label = sample[0], sample[1]
            samples.append(inputs)
            labels.append(label)
        elif isinstance(sample, dict):
            if 'spectrogram' in sample and 'label' in sample:
                samples.append(sample['spectrogram'])
                labels.append(sample['label'])
    
    # 分析樣本特徵
    if len(samples) > 0:
        # 計算特徵統計信息
        sample_means = []
        sample_stds = []
        sample_mins = []
        sample_maxs = []
        
        for s in samples:
            if isinstance(s, torch.Tensor):
                s_np = s.cpu().numpy()
                sample_means.append(float(np.mean(s_np)))
                sample_stds.append(float(np.std(s_np)))
                sample_mins.append(float(np.min(s_np)))
                sample_maxs.append(float(np.max(s_np)))
        
        # 輸出統計信息
        logger.info(f"樣本均值: {np.mean(sample_means):.6f} ± {np.std(sample_means):.6f}")
        logger.info(f"樣本標準差: {np.mean(sample_stds):.6f} ± {np.std(sample_stds):.6f}")
        logger.info(f"樣本最小值: {np.mean(sample_mins):.6f} ± {np.std(sample_mins):.6f}")
        logger.info(f"樣本最大值: {np.mean(sample_maxs):.6f} ± {np.std(sample_maxs):.6f}")
        
        # 計算類別分佈
        class_counts = {}
        for label in labels:
            if isinstance(label, torch.Tensor):
                label = label.item()
            class_counts[int(label)] = class_counts.get(int(label), 0) + 1
        
        logger.info(f"類別分佈: {class_counts}")
        
        return {
            'samples': samples,
            'labels': labels,
            'stats': {
                'means': [float(x) for x in sample_means],
                'stds': [float(x) for x in sample_stds],
                'mins': [float(x) for x in sample_mins],
                'maxs': [float(x) for x in sample_maxs]
            },
            'class_distribution': class_counts
        }
    
    return None

def test_model_predictions(model, test_input, device=None):
    """測試模型在輸入上的預測"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        if isinstance(test_input, torch.Tensor):
            test_input = test_input.to(device)
        
        outputs = model(test_input)
        logits = outputs.cpu().numpy()
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
        logger.info(f"原始輸出 (logits): {logits}")
        logger.info(f"概率: {probs}")
        logger.info(f"預測: {preds}")
        
        return {
            'logits': logits,
            'probs': probs,
            'preds': preds
        }

def test_on_real_data(model, config, device=None, num_samples=10):
    """在真實數據上測試模型"""
    logger.info(f"在真實數據上測試模型 (樣本數: {num_samples})")
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建數據集
    train_dataset, val_dataset, test_dataset = create_dataset(config)
    dataset = val_dataset if val_dataset is not None else train_dataset
    
    if dataset is None:
        logger.error("無法創建數據集")
        return None
    
    model.to(device)
    model.eval()
    
    # 初始化結果
    all_labels = []
    all_preds = []
    all_probs = []
    
    # 測試樣本
    for i in tqdm(range(min(num_samples, len(dataset))), desc="測試樣本"):
        try:
            sample = dataset[i]
            
            # 提取輸入數據和標籤
            if isinstance(sample, tuple) and len(sample) >= 2:
                inputs, label = sample[0], sample[1]
                if isinstance(label, torch.Tensor):
                    label = label.item()
                
                # 處理輸入數據
                if len(inputs.shape) == 2:  # (C, L)
                    # 使用librosa處理音頻
                    audio = inputs[0].cpu().numpy()
                    S = librosa.feature.melspectrogram(
                        y=audio, 
                        sr=16000, 
                        n_mels=128,
                        n_fft=1024,
                        hop_length=512
                    )
                    S_db = librosa.power_to_db(S, ref=np.max)
                    spec_tensor = torch.from_numpy(S_db).unsqueeze(0).float()  # 確保是float32
                    spec_tensor = torch.nn.functional.interpolate(
                        spec_tensor.unsqueeze(0),
                        size=(224, 224),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                    # 複製到3通道
                    inputs = spec_tensor.repeat(3, 1, 1)
                elif not inputs.is_floating_point() or inputs.dtype != torch.float32:
                    # 確保是float32類型
                    inputs = inputs.float()
                
                # 添加批次維度
                inputs = inputs.unsqueeze(0).to(device)
                
                # 預測
                with torch.no_grad():
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                
                # 記錄結果
                all_labels.append(label)
                all_preds.append(preds.item())
                all_probs.append(probs.cpu().numpy()[0])
            
            elif isinstance(sample, dict) and 'spectrogram' in sample and 'label' in sample:
                inputs = sample['spectrogram']
                label = sample['label']
                if isinstance(label, torch.Tensor):
                    label = label.item()
                
                # 確保是float32類型
                if not inputs.is_floating_point() or inputs.dtype != torch.float32:
                    inputs = inputs.float()
                
                # 添加批次維度
                inputs = inputs.unsqueeze(0).to(device)
                
                # 預測
                with torch.no_grad():
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                
                # 記錄結果
                all_labels.append(label)
                all_preds.append(preds.item())
                all_probs.append(probs.cpu().numpy()[0])
        
        except Exception as e:
            logger.error(f"處理樣本 {i} 時出錯: {str(e)}")
    
    # 分析結果
    if len(all_labels) > 0:
        accuracy = np.mean(np.array(all_labels) == np.array(all_preds)) * 100
        logger.info(f"准確率: {accuracy:.2f}%")
        
        # 分析每個類別的準確率
        classes = np.unique(all_labels)
        class_accuracies = {}
        
        for cls in classes:
            mask = np.array(all_labels) == cls
            if np.sum(mask) > 0:
                cls_acc = np.mean(np.array(all_preds)[mask] == cls) * 100
                class_accuracies[int(cls)] = float(cls_acc)
                logger.info(f"類別 {cls} 准確率: {cls_acc:.2f}% (樣本數: {np.sum(mask)})")
        
        # 分析預測分佈
        pred_counts = {}
        for pred in all_preds:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        logger.info(f"預測分佈: {pred_counts}")
        
        # 檢查是否只預測一個類別
        if len(set(all_preds)) == 1:
            logger.warning(f"模型只預測一個類別: {all_preds[0]}")
        
        return {
            'accuracy': float(accuracy),
            'class_accuracies': class_accuracies,
            'pred_distribution': pred_counts,
            'all_labels': [int(l) for l in all_labels],
            'all_preds': [int(p) for p in all_preds],
            'all_probs': [p.tolist() for p in all_probs]
        }
    
    return None

def save_model(model, save_path):
    """保存模型"""
    logger.info(f"保存修復後的模型: {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

def main():
    import math
    import torch.nn as nn
    
    # 建立輸出目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"tests/swin_transformer_fix/comprehensive_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 設置文件日誌
    file_handler = logging.FileHandler(output_dir / "fix.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"開始全面修復 Swin Transformer 模型，結果將保存在 {output_dir}")
    
    # 載入配置
    config_path = "config/example_classification_drlee.yaml"
    logger.info(f"載入配置: {config_path}")
    config = load_config(config_path)
    
    # 載入原始模型
    checkpoint_path = "results/indexed_classification_drlee_20250519_012024/models/best_model.pth"
    original_model = load_model(config, checkpoint_path)
    
    # 分析模型參數
    model_params = analyze_model_parameters(original_model)
    
    # 保存參數分析結果
    with open(output_dir / "model_params_analysis.json", 'w') as f:
        json.dump(model_params, f, indent=2)
    
    # 分析真實數據集
    real_data_stats = analyze_real_data(config, num_samples=10)
    
    if real_data_stats:
        # 保存數據集分析結果
        with open(output_dir / "data_analysis.json", 'w') as f:
            # 將樣本和張量轉換為可序列化格式
            serializable_stats = {
                'stats': real_data_stats['stats'],
                'class_distribution': real_data_stats['class_distribution']
            }
            json.dump(serializable_stats, f, indent=2)
        
        # 從類別分佈計算反向權重
        class_distribution = real_data_stats['class_distribution']
        classes = sorted(class_distribution.keys())
        total_samples = sum(class_distribution.values())
        
        # 計算反向權重: 1 / (樣本比例)
        class_weights = {}
        for cls in classes:
            class_ratio = class_distribution[cls] / total_samples
            class_weights[cls] = 1.0 / (class_ratio + 1e-5)  # 添加小值避免除零
        
        # 標準化權重使其平均為1
        weight_sum = sum(class_weights.values())
        for cls in class_weights:
            class_weights[cls] = class_weights[cls] * len(class_weights) / weight_sum
        
        logger.info(f"計算的類別權重: {class_weights}")
    else:
        # 如果無法獲取真實數據，使用平均權重
        class_weights = {0: 1.0, 1: 1.0, 2: 1.0}
    
    # 創建測試輸入
    test_input = create_test_spectrogram()
    
    # 測試原始模型
    logger.info("\n===== 測試原始模型 =====")
    original_preds = test_model_predictions(original_model, test_input)
    
    # 在真實數據上測試原始模型
    logger.info("\n===== 在真實數據上測試原始模型 =====")
    original_real_results = test_on_real_data(original_model, config, num_samples=20)
    
    if original_real_results:
        with open(output_dir / "original_real_results.json", 'w') as f:
            json.dump(original_real_results, f, indent=2)
    
    # 嘗試不同的修復方法
    fix_methods = ["reset_output", "modify_head"]
    
    for method in fix_methods:
        logger.info(f"\n===== 嘗試修復方法: {method} =====")
        
        # 修復模型權重
        fixed_model = fix_model_weights(original_model, method)
        
        # 應用類別權重
        fixed_model_with_weights = add_class_weights_to_model(fixed_model, [class_weights.get(i, 1.0) for i in range(3)])
        
        # 獲取修復後模型的預測
        logger.info(f"測試修復後模型 ({method}) 預測")
        fixed_preds = test_model_predictions(fixed_model_with_weights, test_input)
        
        # 在真實數據上測試修復後的模型
        logger.info(f"\n===== 在真實數據上測試修復後模型 ({method}) =====")
        fixed_real_results = test_on_real_data(fixed_model_with_weights, config, num_samples=20)
        
        if fixed_real_results:
            with open(output_dir / f"{method}_real_results.json", 'w') as f:
                json.dump(fixed_real_results, f, indent=2)
        
        # 保存修復後的模型
        save_path = output_dir / f"fixed_model_{method}.pth"
        save_model(fixed_model_with_weights, save_path)
    
    logger.info(f"完成所有修復方法測試，結果已保存到 {output_dir}")

if __name__ == "__main__":
    main() 