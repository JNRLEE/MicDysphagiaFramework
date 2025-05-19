"""
修復 Swin Transformer 模型特徵量級不平衡問題
此方法通過調整模型內部特徵表示的量級來解決只預測單一類別的問題
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

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
    import yaml
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

def collect_feature_statistics(model, dataloader, device=None, feature_layer='backbone.features.7'):
    """收集特徵層統計信息"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    # 創建鉤子以獲取特徵
    features = []
    labels = []
    
    def hook_fn(module, input, output):
        features.append(output.detach().cpu())
    
    # 註冊鉤子
    hook_handle = None
    for name, module in model.named_modules():
        if name == feature_layer:
            hook_handle = module.register_forward_hook(hook_fn)
            logger.info(f"註冊鉤子到層: {name}")
            break
    
    if hook_handle is None:
        logger.error(f"找不到層: {feature_layer}")
        return None
    
    # 收集特徵
    try:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="收集特徵"):
                # 嘗試從批次中適配輸入數據
                adapted_batch = None
                
                try:
                    # 使用 DataAdapter 適配批次
                    # 這將處理不同的輸入格式並將其轉換為模型期望的格式
                    adapted_batch = DataAdapter.adapt_batch(batch, "swin_transformer", {})
                    
                    if 'spectrogram' in adapted_batch:
                        inputs = adapted_batch['spectrogram'].to(device)
                        model(inputs)
                        
                        # 記錄標籤
                        if 'label' in adapted_batch:
                            label_tensor = adapted_batch['label']
                            if len(label_tensor.shape) == 0:  # 單個標量
                                label_tensor = label_tensor.unsqueeze(0)
                            labels.append(label_tensor.cpu())
                    else:
                        logger.warning(f"適配的批次中沒有 'spectrogram' 鍵")
                        
                except Exception as e:
                    logger.error(f"適配批次時出錯: {str(e)}")
                    continue
                
    except Exception as e:
        logger.error(f"收集特徵時出錯: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # 移除鉤子
        hook_handle.remove()
    
    # 處理收集到的特徵
    if not features:
        logger.error("未收集到任何特徵")
        return None
    
    # 合併特徵和標籤
    all_features = torch.cat(features, dim=0)
    all_labels = torch.cat(labels, dim=0) if labels else None
    
    logger.info(f"收集了 {all_features.shape[0]} 個樣本的特徵")
    logger.info(f"特徵形狀: {all_features.shape}")
    
    # 計算特徵統計信息
    feature_stats = {
        'mean': all_features.mean(dim=0).numpy().tolist(),
        'std': all_features.std(dim=0).numpy().tolist(),
        'min': all_features.min(dim=0)[0].numpy().tolist(),
        'max': all_features.max(dim=0)[0].numpy().tolist(),
    }
    
    # 按類別計算特徵統計信息
    if all_labels is not None:
        class_stats = {}
        for cls in torch.unique(all_labels):
            cls_idx = (all_labels == cls).nonzero().squeeze()
            if cls_idx.numel() > 0:
                cls_features = all_features[cls_idx]
                cls_stats = {
                    'count': cls_idx.numel(),
                    'mean': cls_features.mean(dim=0).numpy().tolist(),
                    'std': cls_features.std(dim=0).numpy().tolist(),
                    'min': cls_features.min(dim=0)[0].numpy().tolist(),
                    'max': cls_features.max(dim=0)[0].numpy().tolist(),
                }
                class_stats[int(cls.item())] = cls_stats
        
        feature_stats['by_class'] = class_stats
    
    return feature_stats, all_features, all_labels

def create_feature_adjusted_model(model, feature_stats, magnitude_scale=0.1):
    """創建特徵量級調整的模型"""
    logger.info(f"創建特徵量級調整的模型，調整因子: {magnitude_scale}")
    
    # 深度複製模型
    fixed_model = None
    
    if isinstance(model, SwinTransformerModel):
        # 需要手動複製權重
        logger.info("為 SwinTransformerModel 建立新實例")
        num_classes = model.head[1].out_features
        fixed_model = SwinTransformerModel(num_classes=num_classes)
        
        # 複製參數
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in dict(fixed_model.named_parameters()):
                    dict(fixed_model.named_parameters())[name].copy_(param)
        
        # 調整輸出層的權重量級
        with torch.no_grad():
            head_weight = fixed_model.head[1].weight
            
            # 調整權重的標準差
            current_std = torch.std(head_weight, dim=1, keepdim=True)
            target_std = current_std * magnitude_scale
            
            # 縮放權重
            scaled_weight = head_weight * (target_std / current_std)
            fixed_model.head[1].weight.copy_(scaled_weight)
            
            logger.info(f"頭部權重縮放前標準差: {current_std.mean().item():.6f}")
            logger.info(f"頭部權重縮放後標準差: {torch.std(fixed_model.head[1].weight, dim=1).mean().item():.6f}")
        
        # 初始化偏置為零
        with torch.no_grad():
            fixed_model.head[1].bias.zero_()
            logger.info("頭部偏置已初始化為零")
    else:
        # 普通模型複製
        logger.info("為一般模型建立深度副本")
        import copy
        fixed_model = copy.deepcopy(model)
        
        # 調整輸出層的權重量級
        for name, param in fixed_model.named_parameters():
            if ('head' in name or 'fc' in name or 'output' in name or 'classifier' in name) and 'weight' in name:
                with torch.no_grad():
                    current_std = torch.std(param, dim=1, keepdim=True)
                    target_std = current_std * magnitude_scale
                    scaled_param = param * (target_std / current_std)
                    param.copy_(scaled_param)
                    logger.info(f"{name} 權重量級已調整，縮放因子: {magnitude_scale}")
            elif ('head' in name or 'fc' in name or 'output' in name or 'classifier' in name) and 'bias' in name:
                with torch.no_grad():
                    param.zero_()
                    logger.info(f"{name} 偏置已初始化為零")
    
    return fixed_model

def create_feature_balanced_model(model, feature_stats, class_weights=None):
    """創建特徵平衡的模型"""
    logger.info("創建特徵平衡的模型")
    
    # 深度複製模型
    fixed_model = None
    
    if isinstance(model, SwinTransformerModel):
        # 需要手動複製權重
        logger.info("為 SwinTransformerModel 建立新實例")
        num_classes = model.head[1].out_features
        fixed_model = SwinTransformerModel(num_classes=num_classes)
        
        # 複製參數
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in dict(fixed_model.named_parameters()):
                    dict(fixed_model.named_parameters())[name].copy_(param)
        
        # 獲取類別特徵統計信息
        if 'by_class' in feature_stats and class_weights is None:
            class_stats = feature_stats['by_class']
            class_means = {}
            class_stds = {}
            
            for cls, stats in class_stats.items():
                class_means[cls] = torch.tensor(stats['mean'])
                class_stds[cls] = torch.tensor(stats['std'])
            
            # 計算標準差的均值作為目標標準差
            std_values = [std.mean().item() for std in class_stds.values()]
            target_std = sum(std_values) / len(std_values)
            
            # 調整輸出層權重以平衡不同類別的特徵差異
            with torch.no_grad():
                head_weight = fixed_model.head[1].weight
                head_bias = fixed_model.head[1].bias
                
                for cls in range(num_classes):
                    if cls in class_means:
                        # 標準化權重
                        cls_std = class_stds[cls].mean().item()
                        scale_factor = target_std / cls_std if cls_std > 0 else 1.0
                        
                        # 縮放該類別的權重
                        head_weight[cls] = head_weight[cls] * scale_factor
                        
                        # 計算該類別的特徵偏移量
                        mean_diff = class_means[cls].mean().item()
                        head_bias[cls] = -mean_diff * scale_factor
            
            logger.info(f"已調整每個類別的權重和偏置")
        
        # 如果提供了類別權重，使用它們來調整偏置
        elif class_weights is not None:
            with torch.no_grad():
                # 在偏置上應用對數空間的類別權重調整
                bias_adjustment = torch.tensor([np.log(class_weights.get(i, 1.0)) for i in range(num_classes)])
                fixed_model.head[1].bias.copy_(bias_adjustment)
                logger.info(f"已將類別權重應用到輸出層偏置: {bias_adjustment}")
    else:
        # 普通模型複製
        logger.info("為一般模型建立深度副本")
        import copy
        fixed_model = copy.deepcopy(model)
        
        # 如果提供了類別權重，使用它們來調整偏置
        if class_weights is not None:
            for name, param in fixed_model.named_parameters():
                if ('head' in name or 'fc' in name or 'output' in name or 'classifier' in name) and 'bias' in name:
                    with torch.no_grad():
                        num_classes = param.shape[0]
                        bias_adjustment = torch.tensor([np.log(class_weights.get(i, 1.0)) for i in range(num_classes)])
                        param.copy_(bias_adjustment)
                        logger.info(f"已將類別權重應用到 {name}: {bias_adjustment}")
    
    return fixed_model

def test_model_on_dataset(model, dataset, device=None, num_samples=20):
    """在數據集上測試模型"""
    logger.info(f"在數據集上測試模型 (樣本數: {num_samples})")
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if dataset is None:
        logger.error("無法創建數據集")
        return None
    
    model.to(device)
    model.eval()
    
    # 初始化結果
    all_labels = []
    all_preds = []
    all_probs = []
    all_logits = []
    
    # 測試樣本
    for i in tqdm(range(min(num_samples, len(dataset))), desc="測試樣本"):
        try:
            sample = dataset[i]
            
            # 提取輸入數據和標籤
            if isinstance(sample, tuple) and len(sample) >= 2:
                inputs, label = sample[0], sample[1]
                if isinstance(label, torch.Tensor):
                    label = label.item()
                
                # 確保是float32類型
                if not inputs.is_floating_point() or inputs.dtype != torch.float32:
                    inputs = inputs.float()
                
                # 確保有三個通道
                if inputs.shape[0] == 1:
                    inputs = inputs.repeat(3, 1, 1)
                
                # 添加批次維度
                inputs = inputs.unsqueeze(0).to(device)
                
                # 前向傳播
                with torch.no_grad():
                    outputs = model(inputs)
                    logits = outputs
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                
                # 記錄結果
                all_labels.append(label)
                all_preds.append(preds.item())
                all_probs.append(probs.cpu().numpy()[0])
                all_logits.append(logits.cpu().numpy()[0])
            
            elif isinstance(sample, dict) and 'spectrogram' in sample and 'label' in sample:
                inputs = sample['spectrogram']
                label = sample['label']
                if isinstance(label, torch.Tensor):
                    label = label.item()
                
                # 確保是float32類型
                if not inputs.is_floating_point() or inputs.dtype != torch.float32:
                    inputs = inputs.float()
                
                # 確保有三個通道
                if inputs.shape[0] == 1:
                    inputs = inputs.repeat(3, 1, 1)
                
                # 添加批次維度
                inputs = inputs.unsqueeze(0).to(device)
                
                # 前向傳播
                with torch.no_grad():
                    outputs = model(inputs)
                    logits = outputs
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                
                # 記錄結果
                all_labels.append(label)
                all_preds.append(preds.item())
                all_probs.append(probs.cpu().numpy()[0])
                all_logits.append(logits.cpu().numpy()[0])
        
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
        
        # 分析 logits 分佈
        logits_array = np.array(all_logits)
        logits_mean = np.mean(logits_array, axis=0)
        logits_std = np.std(logits_array, axis=0)
        logger.info(f"Logits 均值: {logits_mean}")
        logger.info(f"Logits 標準差: {logits_std}")
        
        return {
            'accuracy': float(accuracy),
            'class_accuracies': class_accuracies,
            'pred_distribution': {str(k): v for k, v in pred_counts.items()},
            'all_labels': [int(l) for l in all_labels],
            'all_preds': [int(p) for p in all_preds],
            'all_probs': [p.tolist() for p in all_probs],
            'all_logits': [l.tolist() for l in all_logits],
            'logits_mean': logits_mean.tolist(),
            'logits_std': logits_std.tolist()
        }
    
    return None

def save_model(model, save_path):
    """保存模型"""
    logger.info(f"保存模型: {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

def create_dataloader(dataset, batch_size=16):
    """創建數據加載器"""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

def main():
    # 建立輸出目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"tests/swin_transformer_fix/feature_magnitude_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 設置文件日誌
    file_handler = logging.FileHandler(output_dir / "fix.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"開始修復 Swin Transformer 模型特徵量級問題，結果將保存在 {output_dir}")
    
    # 載入配置
    config_path = "config/example_classification_drlee.yaml"
    logger.info(f"載入配置: {config_path}")
    config = load_config(config_path)
    
    # 載入原始模型
    checkpoint_path = "results/indexed_classification_drlee_20250519_012024/models/best_model.pth"
    original_model = load_model(config, checkpoint_path)
    
    # 創建數據集
    train_dataset, val_dataset, test_dataset = create_dataset(config)
    dataset_for_testing = val_dataset if val_dataset is not None else test_dataset if test_dataset is not None else train_dataset
    
    if dataset_for_testing is None:
        logger.error("無法創建任何數據集")
        return
    
    # 創建數據加載器
    train_dataloader = create_dataloader(train_dataset) if train_dataset is not None else None
    
    # 收集特徵統計信息
    feature_stats = None
    all_features = None
    all_labels = None
    class_weights = {0: 1.0, 1: 3.0, 2: 5.0}  # 默認類別權重
    
    if train_dataloader is not None:
        logger.info("收集特徵統計信息")
        try:
            result = collect_feature_statistics(
                original_model, 
                train_dataloader,
                feature_layer='backbone.features.7'  # 適應 Swin Transformer 架構
            )
            
            if result is not None:
                feature_stats, all_features, all_labels = result
                
                # 保存特徵統計信息
                with open(output_dir / "feature_stats.json", 'w') as f:
                    json.dump(feature_stats, f, indent=2)
                
                # 如果成功收集到了特徵，則計算類別權重
                if all_labels is not None:
                    class_distribution = {}
                    for label in all_labels:
                        cls = int(label.item())
                        class_distribution[cls] = class_distribution.get(cls, 0) + 1
                    
                    total_samples = sum(class_distribution.values())
                    class_weights = {}
                    for cls, count in class_distribution.items():
                        class_ratio = count / total_samples
                        class_weights[cls] = 1.0 / (class_ratio + 1e-5)  # 避免除零
                    
                    # 標準化權重使其平均為1
                    weight_sum = sum(class_weights.values())
                    for cls in class_weights:
                        class_weights[cls] = class_weights[cls] * len(class_weights) / weight_sum
            else:
                logger.warning("無法收集特徵統計信息，將使用默認類別權重")
        except Exception as e:
            logger.error(f"收集特徵統計信息時出錯: {str(e)}")
            logger.warning("將使用默認類別權重")
    else:
        logger.warning("無法創建訓練數據加載器，無法收集特徵統計信息")
        logger.warning("將使用默認類別權重")
    
    logger.info(f"使用的類別權重: {class_weights}")
    
    # 測試原始模型
    logger.info("\n===== 測試原始模型 =====")
    original_results = test_model_on_dataset(original_model, dataset_for_testing, num_samples=20)
    
    if original_results:
        with open(output_dir / "original_results.json", 'w') as f:
            json.dump(original_results, f, indent=2)
    
    # 嘗試不同的特徵量級調整值
    magnitude_scales = [0.01, 0.1, 0.5, 1.0]
    
    for scale in magnitude_scales:
        logger.info(f"\n===== 嘗試特徵量級調整: {scale} =====")
        
        # 創建特徵量級調整的模型
        fixed_model = create_feature_adjusted_model(original_model, feature_stats, magnitude_scale=scale)
        
        # 測試修復後的模型
        scale_results = test_model_on_dataset(fixed_model, dataset_for_testing, num_samples=20)
        
        if scale_results:
            with open(output_dir / f"scale_{scale}_results.json", 'w') as f:
                json.dump(scale_results, f, indent=2)
        
        # 保存修復後的模型
        save_model(fixed_model, output_dir / f"fixed_model_scale_{scale}.pth")
    
    # 創建特徵平衡的模型（使用類別權重）
    logger.info("\n===== 嘗試特徵平衡模型 =====")
    
    # 創建特徵平衡的模型
    balanced_model = create_feature_balanced_model(original_model, feature_stats, class_weights)
    
    # 測試特徵平衡的模型
    balanced_results = test_model_on_dataset(balanced_model, dataset_for_testing, num_samples=20)
    
    if balanced_results:
        with open(output_dir / "balanced_results.json", 'w') as f:
            json.dump(balanced_results, f, indent=2)
    
    # 保存特徵平衡的模型
    save_model(balanced_model, output_dir / "balanced_model.pth")
    
    logger.info(f"完成所有修復方法測試，結果已保存到 {output_dir}")

if __name__ == "__main__":
    main() 