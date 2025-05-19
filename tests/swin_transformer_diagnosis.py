"""
Swin Transformer 診斷腳本
用途：診斷Swin Transformer模型只預測單一類別的問題
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import random
import json
import yaml
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from sklearn.metrics import confusion_matrix
import librosa

# 將專案根目錄加入到路徑中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 導入專案模塊
from models.model_factory import create_model
from utils.data_adapter import DataAdapter
from data.dataset_factory import create_dataset
from utils.data_index_loader import DataIndexLoader
from trainers.trainer_factory import create_trainer

# 設置輸出目錄
OUTPUT_DIR = Path('tests/swin_transformer_diagnosis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_subdir = OUTPUT_DIR / f"diagnosis_{timestamp}"
output_subdir.mkdir(exist_ok=True)

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(output_subdir / "diagnosis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """載入YAML配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, output_path):
    """保存配置到YAML文件"""
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def prepare_model(config, checkpoint_path=None):
    """準備模型"""
    # 創建模型
    model = create_model(config)
    
    # 載入權重（如果有）
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"載入模型權重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model.eval()  # 設置為評估模式
    return model

def analyze_weight_distributions(model, output_dir):
    """分析模型關鍵層的權重分佈"""
    logger.info("分析模型權重分佈")
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    
    # 特別關注最後的分類層
    head_weights = {}
    
    # 遍歷所有命名參數
    for name, param in model.named_parameters():
        if 'head' in name and param.requires_grad:
            # 獲取權重值
            weight_np = param.data.cpu().numpy()
            
            # 計算統計信息
            stats = {
                "name": name,
                "shape": list(weight_np.shape),
                "min": float(np.min(weight_np)),
                "max": float(np.max(weight_np)),
                "mean": float(np.mean(weight_np)),
                "std": float(np.std(weight_np)),
                "zeros_percentage": float(np.sum(weight_np == 0) / weight_np.size * 100)
            }
            
            # 保存到字典
            head_weights[name] = stats
            
            # 繪製權重分佈直方圖
            plt.figure(figsize=(10, 6))
            plt.hist(weight_np.flatten(), bins=50)
            plt.title(f"權重分佈 - {name}")
            plt.xlabel("權重值")
            plt.ylabel("頻率")
            plt.tight_layout()
            plt.savefig(weights_dir / f"{name.replace('.', '_')}_distribution.png")
            plt.close()
    
    # 保存統計信息
    with open(weights_dir / "head_weights_stats.json", 'w') as f:
        json.dump(head_weights, f, indent=2)
        
    # 分析最後一層的偏差項
    for name, param in model.named_parameters():
        if 'head' in name and 'bias' in name and param.requires_grad:
            bias_np = param.data.cpu().numpy()
            
            # 繪製偏差條形圖
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(bias_np)), bias_np)
            plt.title(f"分類層偏差項 - {name}")
            plt.xlabel("類別索引")
            plt.ylabel("偏差值")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(weights_dir / f"{name.replace('.', '_')}_bias.png")
            plt.close()
            
            # 檢查是否存在偏差問題
            max_bias_idx = np.argmax(bias_np)
            max_bias_val = bias_np[max_bias_idx]
            min_bias_val = np.min(bias_np)
            bias_range = max_bias_val - min_bias_val
            
            logger.info(f"最大偏差值: {max_bias_val} (類別 {max_bias_idx})")
            logger.info(f"最小偏差值: {min_bias_val}")
            logger.info(f"偏差範圍: {bias_range}")
            
            if bias_range > 2.0:
                logger.warning(f"偏差範圍過大 ({bias_range})，可能導致預測偏向特定類別 {max_bias_idx}")
                
            # 保存偏差統計信息
            with open(weights_dir / f"{name.replace('.', '_')}_bias_stats.json", 'w') as f:
                json.dump({
                    "max_bias_value": float(max_bias_val),
                    "max_bias_class": int(max_bias_idx),
                    "min_bias_value": float(min_bias_val),
                    "bias_range": float(bias_range),
                    "values": bias_np.tolist()
                }, f, indent=2)

def analyze_data_balance(config, output_dir):
    """分析數據集類別平衡性"""
    logger.info("分析數據集類別平衡")
    
    # 創建數據集
    train_dataset, val_dataset, test_dataset = create_dataset(config)
    
    # 獲取類別計數
    def count_classes(dataset):
        class_counts = {}
        
        for i in tqdm(range(len(dataset)), desc="計算類別分佈"):
            item = dataset[i]
            
            # 獲取標籤
            if isinstance(item, tuple) and len(item) >= 2:
                label = item[1]
            elif isinstance(item, dict) and 'label' in item:
                label = item['label']
            else:
                continue
            
            # 轉換為整數
            if isinstance(label, torch.Tensor):
                label = label.item()
            
            # 確保label是Python原生型別
            label = int(label)
            
            # 更新計數
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
        
        return class_counts
    
    # 分析訓練集
    if train_dataset:
        logger.info(f"分析訓練集 ({len(train_dataset)} 個樣本)")
        train_counts = count_classes(train_dataset)
        
        # 繪製類別分佈
        plt.figure(figsize=(10, 6))
        classes = sorted(train_counts.keys())
        counts = [train_counts[c] for c in classes]
        plt.bar(classes, counts)
        plt.title("訓練集類別分佈")
        plt.xlabel("類別")
        plt.ylabel("樣本數量")
        plt.xticks(classes)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / "train_class_distribution.png")
        plt.close()
        
        # 保存統計信息，確保所有鍵都是原生型別
        class_counts_native = {int(k): int(v) for k, v in train_counts.items()}
        class_percentages = {int(k): float(count/len(train_dataset)*100) for k, count in train_counts.items()}
        
        with open(output_dir / "train_class_stats.json", 'w') as f:
            json.dump({
                "total_samples": int(len(train_dataset)),
                "class_counts": class_counts_native,
                "class_percentages": class_percentages
            }, f, indent=2)
        
        # 檢查類別不平衡
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        logger.info(f"訓練集類別分佈: {train_counts}")
        logger.info(f"最多的類別: {counts.index(max_count)} ({max_count} 樣本)")
        logger.info(f"最少的類別: {counts.index(min_count)} ({min_count} 樣本)")
        logger.info(f"不平衡比例: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 10:
            logger.warning(f"訓練集嚴重不平衡 (比例 {imbalance_ratio:.2f})，可能需要使用類別權重或重採樣")
    
    # 分析驗證集
    if val_dataset:
        logger.info(f"分析驗證集 ({len(val_dataset)} 個樣本)")
        val_counts = count_classes(val_dataset)
        
        # 繪製類別分佈
        plt.figure(figsize=(10, 6))
        classes = sorted(val_counts.keys())
        counts = [val_counts[c] for c in classes]
        plt.bar(classes, counts)
        plt.title("驗證集類別分佈")
        plt.xlabel("類別")
        plt.ylabel("樣本數量")
        plt.xticks(classes)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / "val_class_distribution.png")
        plt.close()
        
        # 保存統計信息，確保所有鍵都是原生型別
        class_counts_native = {int(k): int(v) for k, v in val_counts.items()}
        class_percentages = {int(k): float(count/len(val_dataset)*100) for k, count in val_counts.items()}
        
        with open(output_dir / "val_class_stats.json", 'w') as f:
            json.dump({
                "total_samples": int(len(val_dataset)),
                "class_counts": class_counts_native,
                "class_percentages": class_percentages
            }, f, indent=2)

def test_model_predictions(model, config, output_dir):
    """測試模型在驗證集上的預測表現"""
    logger.info("測試模型預測")
    
    # 創建數據集
    train_dataset, val_dataset, test_dataset = create_dataset(config)
    dataset = test_dataset if test_dataset is not None else val_dataset
    
    if dataset is None:
        logger.error("無法創建數據集")
        return
    
    # 直接從數據集獲取一些樣本，避免使用DataLoader
    num_samples = min(20, len(dataset))
    logger.info(f"直接從數據集提取 {num_samples} 個樣本進行測試")
    
    # 準備記錄
    all_preds = []
    all_probs = []
    all_labels = []
    logit_stats = []
    
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 運行推理
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="處理樣本"):
            try:
                # 獲取單個樣本
                sample = dataset[i]
                
                # 提取輸入數據和標籤
                if isinstance(sample, tuple) and len(sample) >= 2:
                    inputs, labels = sample[0], sample[1]
                    # 處理音頻輸入 (B, C, L) -> (B, C, H, W)
                    if len(inputs.shape) == 2:  # (C, L)
                        logger.info("將一維音頻轉換為頻譜圖")
                        # 使用librosa先轉換為頻譜圖
                        audio = inputs[0].cpu().numpy()  # 提取單聲道音頻
                        
                        # 計算梅爾頻譜圖
                        S = librosa.feature.melspectrogram(
                            y=audio, 
                            sr=16000, 
                            n_mels=128,
                            n_fft=1024,
                            hop_length=512
                        )
                        
                        # 轉換為分貝刻度
                        S_db = librosa.power_to_db(S, ref=np.max)
                        
                        # 轉換為張量
                        spec_tensor = torch.from_numpy(S_db).unsqueeze(0)  # (1, H, W)
                        
                        # 調整大小
                        spec_tensor = torch.nn.functional.interpolate(
                            spec_tensor.unsqueeze(0),  # (1, 1, H, W)
                            size=(224, 224),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)  # (1, 224, 224)
                        
                        # 複製到3通道
                        spec_tensor = spec_tensor.repeat(3, 1, 1)  # (3, 224, 224)
                        
                        inputs = spec_tensor
                    elif len(inputs.shape) == 3 and inputs.shape[0] == 1:  # (1, L)
                        logger.info("將單通道一維音頻轉換為頻譜圖")
                        # 使用librosa先轉換為頻譜圖
                        audio = inputs[0].cpu().numpy()  # 提取單聲道音頻
                        
                        # 計算梅爾頻譜圖
                        S = librosa.feature.melspectrogram(
                            y=audio, 
                            sr=16000, 
                            n_mels=128,
                            n_fft=1024,
                            hop_length=512
                        )
                        
                        # 轉換為分貝刻度
                        S_db = librosa.power_to_db(S, ref=np.max)
                        
                        # 轉換為張量
                        spec_tensor = torch.from_numpy(S_db).unsqueeze(0)  # (1, H, W)
                        
                        # 調整大小
                        spec_tensor = torch.nn.functional.interpolate(
                            spec_tensor.unsqueeze(0),  # (1, 1, H, W)
                            size=(224, 224),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)  # (1, 224, 224)
                        
                        # 複製到3通道
                        spec_tensor = spec_tensor.repeat(3, 1, 1)  # (3, 224, 224)
                        
                        inputs = spec_tensor
                    
                    # 構建批次
                    batch = {'spectrogram': inputs.unsqueeze(0), 'label': labels.unsqueeze(0) if isinstance(labels, torch.Tensor) else torch.tensor([labels])}
                elif isinstance(sample, dict):
                    batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else torch.tensor([v]) for k, v in sample.items()}
                else:
                    logger.error(f"樣本 {i} 格式無法處理: {type(sample)}")
                    continue
                
                # 適配批次
                adapted_batch = DataAdapter.adapt_batch(batch, "swin_transformer", config)
                
                # 前向傳播
                outputs = model(adapted_batch['spectrogram'].to(device))
                
                # 獲取預測
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                # 移至CPU
                cpu_probs = probs.cpu().numpy()
                cpu_preds = preds.cpu().numpy()
                cpu_labels = batch['label'].cpu().numpy()
                
                # 記錄當前批次的統計信息
                logit_stats.append({
                    "batch_min": float(outputs.min().cpu()),
                    "batch_max": float(outputs.max().cpu()),
                    "batch_mean": float(outputs.mean().cpu()),
                    "batch_std": float(outputs.std().cpu()),
                })
                
                # 記錄結果
                all_preds.append(cpu_preds[0])
                all_probs.append(cpu_probs[0])
                all_labels.append(cpu_labels[0])
            except Exception as e:
                logger.error(f"處理樣本 {i} 時出錯: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                continue
    
    # 檢查是否有足夠的預測
    if len(all_preds) == 0:
        logger.error("沒有成功處理任何樣本，無法分析結果")
        return
    
    # 計算模型評估指標
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 計算整體准確率
    accuracy = np.mean(all_preds == all_labels) * 100
    logger.info(f"整體准確率: {accuracy:.2f}%")
    
    # 計算混淆矩陣
    cm = confusion_matrix(all_labels, all_preds)
    
    # 繪製混淆矩陣
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("混淆矩陣")
    plt.colorbar()
    
    classes = sorted(np.unique(np.concatenate([all_labels, all_preds])))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    plt.ylabel('真實類別')
    plt.xlabel('預測類別')
    
    # 在格子中顯示數值
    if cm.size > 0:  # 確保混淆矩陣不為空
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()
    
    # 檢查是否只預測一個類別
    unique_preds = np.unique(all_preds)
    logger.info(f"預測的唯一類別: {unique_preds}")
    
    if len(unique_preds) == 1:
        logger.warning(f"模型只預測一個類別: {unique_preds[0]}")
        
        # 計算平均預測概率
        mean_prob = np.mean([probs[pred] for probs, pred in zip(all_probs, all_preds)])
        logger.warning(f"模型對類別 {unique_preds[0]} 的平均預測概率: {mean_prob:.4f}")
        
        if mean_prob > 0.9:
            logger.warning(f"模型對該類別的置信度非常高，可能存在權重初始化或訓練問題")
    
    # 分析logit輸出
    logit_summary = {
        "mean_min": np.mean([stats["batch_min"] for stats in logit_stats]),
        "mean_max": np.mean([stats["batch_max"] for stats in logit_stats]),
        "mean_mean": np.mean([stats["batch_mean"] for stats in logit_stats]),
        "mean_std": np.mean([stats["batch_std"] for stats in logit_stats]),
    }
    
    with open(output_dir / "logit_stats.json", 'w') as f:
        json.dump(logit_summary, f, indent=2)
    
    logger.info(f"Logit統計: {logit_summary}")
    
    # 只分析出現在標籤中的類別
    class_probs = {}
    try:
        for cls in np.unique(all_labels):
            cls_int = int(cls)  # 確保類別是整數
            mask = (all_labels == cls)
            
            # 計算平均概率
            avg_true_probs = []
            for j in np.where(mask)[0]:
                avg_true_probs.append(all_probs[j][cls_int])
            
            avg_pred_probs = []
            for j in np.where(mask)[0]:
                pred_cls = int(all_preds[j])
                avg_pred_probs.append(all_probs[j][pred_cls])
            
            class_probs[cls_int] = {
                "samples": int(np.sum(mask)),
                "correct": int(np.sum(all_preds[mask] == cls)),
                "accuracy": float(np.mean(all_preds[mask] == cls) * 100),
                "avg_prob_for_true_class": float(np.mean(avg_true_probs)) if avg_true_probs else 0,
                "avg_prob_for_pred_class": float(np.mean(avg_pred_probs)) if avg_pred_probs else 0
            }
        
        with open(output_dir / "class_prediction_stats.json", 'w') as f:
            json.dump(class_probs, f, indent=2)
    except Exception as e:
        logger.error(f"計算類別概率時出錯: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    return {
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "unique_predictions": [int(p) for p in unique_preds],
        "logit_stats": logit_summary,
        "class_stats": class_probs
    }

def fix_model_bias(model, output_dir):
    """創建修復偏差的模型"""
    logger.info("創建修復偏差的模型副本")
    
    # 創建修復目錄
    fix_dir = output_dir / "fixed_model"
    fix_dir.mkdir(exist_ok=True)
    
    # 檢查模型是否有偏差項
    head_bias = None
    for name, param in model.named_parameters():
        if 'head' in name and 'bias' in name and param.requires_grad:
            head_bias = param
            break
    
    if head_bias is None:
        logger.warning("模型沒有可修復的偏差項")
        return None
    
    # 獲取當前偏差
    bias_np = head_bias.data.cpu().numpy()
    logger.info(f"當前偏差: {bias_np}")
    
    # 創建修復模型
    try:
        # 使用深度複製而不是重新創建
        fixed_model = type(model)()
        fixed_model.load_state_dict(model.state_dict())
        
        # 重置偏差項
        with torch.no_grad():
            for name, param in fixed_model.named_parameters():
                if 'head' in name and 'bias' in name and param.requires_grad:
                    # 方法1: 設置為零
                    param.data.zero_()
                    logger.info("已將偏差項設置為零")
        
        # 保存修復後的模型
        torch.save(fixed_model.state_dict(), fix_dir / "model_fixed_bias.pth")
        logger.info(f"已保存修復後的模型: {fix_dir / 'model_fixed_bias.pth'}")
        
    except Exception as e:
        logger.error(f"創建修復模型時出錯: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
    return fixed_model

def compare_models(original_model, fixed_model, config, output_dir):
    """比較原始模型和修復後的模型"""
    if fixed_model is None:
        logger.warning("沒有修復後的模型可比較")
        return
    
    logger.info("比較原始模型和修復後的模型")
    
    # 創建比較目錄
    compare_dir = output_dir / "model_comparison"
    compare_dir.mkdir(exist_ok=True)
    
    # 測試原始模型
    logger.info("測試原始模型")
    original_stats = test_model_predictions(original_model, config, compare_dir / "original")
    
    # 測試修復後的模型
    logger.info("測試修復後的模型")
    fixed_stats = test_model_predictions(fixed_model, config, compare_dir / "fixed")
    
    # 比較結果
    comparison = {
        "original": {
            "accuracy": original_stats["accuracy"],
            "unique_predictions": original_stats["unique_predictions"]
        },
        "fixed": {
            "accuracy": fixed_stats["accuracy"],
            "unique_predictions": fixed_stats["unique_predictions"]
        },
        "accuracy_change": fixed_stats["accuracy"] - original_stats["accuracy"],
        "prediction_diversity_change": len(fixed_stats["unique_predictions"]) - len(original_stats["unique_predictions"])
    }
    
    with open(compare_dir / "comparison_results.json", 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"比較結果: {comparison}")
    
    return comparison

def analyze_inputs(config, output_dir):
    """分析模型輸入數據"""
    logger.info("分析模型輸入數據")
    
    # 創建數據集
    train_dataset, val_dataset, test_dataset = create_dataset(config)
    dataset = test_dataset if test_dataset is not None else val_dataset
    
    if dataset is None:
        logger.error("無法創建數據集")
        return
    
    # 創建輸入分析目錄
    inputs_dir = output_dir / "input_analysis"
    inputs_dir.mkdir(exist_ok=True)
    
    # 隨機選擇樣本
    indices = random.sample(range(len(dataset)), min(5, len(dataset)))
    samples = [dataset[i] for i in indices]
    
    # 分析樣本
    for i, sample in enumerate(samples):
        sample_dir = inputs_dir / f"sample_{i}"
        sample_dir.mkdir(exist_ok=True)
        
        # 提取數據和標籤
        if isinstance(sample, tuple) and len(sample) >= 2:
            data, label = sample[0], sample[1]
        elif isinstance(sample, dict):
            data = sample.get('spectrogram', sample.get('audio', None))
            label = sample.get('label', None)
        else:
            logger.warning(f"無法解析樣本 {i}")
            continue
        
        # 分析數據
        if data is not None:
            # 獲取統計信息
            if isinstance(data, torch.Tensor):
                stats = {
                    "shape": list(data.shape),
                    "min": float(data.min()),
                    "max": float(data.max()),
                    "mean": float(data.mean()),
                    "std": float(data.std())
                }
                
                with open(sample_dir / "stats.json", 'w') as f:
                    json.dump(stats, f, indent=2)
                
                # 繪製數據
                if len(data.shape) == 3:  # [C, H, W]
                    # 如果是RGB圖像
                    plt.figure(figsize=(10, 8))
                    plt.imshow(data.permute(1, 2, 0).numpy())
                    plt.title(f"樣本 {i} - 標籤 {label}")
                    plt.savefig(sample_dir / "image.png")
                    plt.close()
                    
                    # 分別保存每個通道
                    for c in range(data.shape[0]):
                        plt.figure(figsize=(8, 8))
                        plt.imshow(data[c].numpy(), cmap='viridis')
                        plt.colorbar()
                        plt.title(f"樣本 {i} - 通道 {c}")
                        plt.savefig(sample_dir / f"channel_{c}.png")
                        plt.close()
                
                elif len(data.shape) == 2:  # [H, W] 或 [L, F]
                    plt.figure(figsize=(10, 8))
                    plt.imshow(data.numpy(), cmap='viridis')
                    plt.colorbar()
                    plt.title(f"樣本 {i} - 標籤 {label}")
                    plt.savefig(sample_dir / "spectrogram.png")
                    plt.close()
                
                elif len(data.shape) == 1:  # [L]
                    plt.figure(figsize=(10, 4))
                    plt.plot(data.numpy())
                    plt.title(f"樣本 {i} - 標籤 {label}")
                    plt.savefig(sample_dir / "audio.png")
                    plt.close()

def main():
    """主函數"""
    logger.info(f"開始 Swin Transformer 診斷，結果將保存在 {output_subdir}")
    
    # 指定配置和模型檢查點
    config_path = "config/example_classification_drlee.yaml"
    checkpoint_path = "results/indexed_classification_drlee_20250519_012024/models/best_model.pth"
    
    # 載入配置
    config = load_config(config_path)
    logger.info(f"已載入配置: {config_path}")
    
    # 保存配置副本
    save_config(config, output_subdir / "config.yaml")
    
    # 準備模型
    model = prepare_model(config, checkpoint_path)
    logger.info(f"已準備模型")
    
    # 分析模型權重分佈
    analyze_weight_distributions(model, output_subdir)
    
    # 分析數據集類別平衡性
    analyze_data_balance(config, output_subdir)
    
    # 分析模型輸入
    analyze_inputs(config, output_subdir)
    
    # 測試模型預測
    test_model_predictions(model, config, output_subdir)
    
    # 嘗試修復模型偏差
    fixed_model = fix_model_bias(model, output_subdir)
    
    # 比較原始模型和修復後的模型
    if fixed_model:
        compare_models(model, fixed_model, config, output_subdir)
    
    logger.info(f"診斷完成，所有結果保存在 {output_subdir}")

if __name__ == "__main__":
    main() 