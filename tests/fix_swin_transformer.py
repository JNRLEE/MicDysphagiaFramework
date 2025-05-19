"""
修復 Swin Transformer 模型只預測單一類別的問題
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
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        logger.info("模型載入成功")
    
    return model

def analyze_model_biases(model):
    """分析模型偏差"""
    logger.info("分析模型輸出層偏差")
    
    output_bias = None
    for name, param in model.named_parameters():
        if 'head' in name and 'bias' in name:
            output_bias = param.data.cpu().numpy()
            logger.info(f"輸出層偏差: {name} = {output_bias}")
    
    return output_bias

def fix_model_bias(model, fix_method="zero"):
    """修復模型偏差"""
    logger.info(f"使用 {fix_method} 方法修復模型偏差")
    
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
        
        # 確保副本成功
        logger.info(f"原模型輸出層大小: {model.head[1].weight.shape}")
        logger.info(f"新模型輸出層大小: {fixed_model.head[1].weight.shape}")
    else:
        # 普通模型複製
        logger.info("為一般模型建立深度副本")
        import copy
        fixed_model = copy.deepcopy(model)
    
    # 套用偏差修復
    with torch.no_grad():
        # 尋找所有可能的輸出層偏差參數
        bias_modified = False
        for name, param in fixed_model.named_parameters():
            if ('head' in name or 'fc' in name or 'output' in name or 'classifier' in name) and 'bias' in name:
                logger.info(f"找到輸出層偏差: {name} = {param.data}")
                
                if fix_method == "zero":
                    # 方法1: 將輸出層偏差設置為零
                    original_bias = param.data.clone()
                    param.data.zero_()
                    logger.info(f"將偏差 {name} 從 {original_bias} 修改為 {param.data}")
                elif fix_method == "balance":
                    # 方法2: 平衡所有類別的偏差
                    original_bias = param.data.clone()
                    mean_bias = torch.mean(param.data)
                    param.data.fill_(mean_bias)
                    logger.info(f"將偏差 {name} 從 {original_bias} 平衡為 {param.data}")
                elif fix_method == "invert_max":
                    # 方法3: 反轉偏差最大的類別
                    original_bias = param.data.clone()
                    max_idx = torch.argmax(param.data)
                    min_idx = torch.argmin(param.data)
                    max_val = param.data[max_idx].item()
                    min_val = param.data[min_idx].item()
                    
                    # 反轉最大的偏差
                    param.data[max_idx] = min_val
                    logger.info(f"將最大偏差類別 {max_idx} 從 {max_val} 修改為 {min_val}")
                elif fix_method == "normalize":
                    # 方法4: 標準化偏差
                    original_bias = param.data.clone()
                    mean_bias = torch.mean(param.data)
                    std_bias = torch.std(param.data)
                    param.data = (param.data - mean_bias) / (std_bias + 1e-6)
                    logger.info(f"將偏差 {name} 從 {original_bias} 標準化為 {param.data}")
                
                bias_modified = True
        
        if not bias_modified:
            logger.warning("未找到可修改的輸出層偏差")
    
    return fixed_model

def test_model_predictions(model, test_input, device=None):
    """測試模型在單個輸入上的預測"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        if isinstance(test_input, torch.Tensor):
            test_input = test_input.to(device)
        
        outputs = model(test_input)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        logger.info(f"原始輸出: {outputs.cpu().numpy()}")
        logger.info(f"概率: {probs.cpu().numpy()}")
        logger.info(f"預測: {preds.cpu().numpy()}")
        
        return {
            'outputs': outputs.cpu().numpy(),
            'probs': probs.cpu().numpy(),
            'preds': preds.cpu().numpy()
        }

def save_model(model, save_path):
    """保存模型"""
    logger.info(f"保存修復後的模型: {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

def main():
    # 建立輸出目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"tests/swin_transformer_fix/fix_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 設置文件日誌
    file_handler = logging.FileHandler(output_dir / "fix.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"開始修復 Swin Transformer 模型，結果將保存在 {output_dir}")
    
    # 載入配置
    config_path = "config/example_classification_drlee.yaml"
    logger.info(f"載入配置: {config_path}")
    config = load_config(config_path)
    
    # 載入模型
    checkpoint_path = "results/indexed_classification_drlee_20250519_012024/models/best_model.pth"
    model = load_model(config, checkpoint_path)
    
    # 分析模型偏差
    output_bias = analyze_model_biases(model)
    
    # 創建測試輸入
    logger.info("創建測試輸入")
    input_shape = (1, 3, 224, 224)
    test_input = torch.randn(input_shape)
    
    # 獲取原始模型的預測
    logger.info("測試原始模型預測")
    original_preds = test_model_predictions(model, test_input)
    
    # 嘗試不同的修復方法
    fix_methods = ["zero", "balance", "invert_max", "normalize"]
    
    for method in fix_methods:
        logger.info(f"\n===== 嘗試修復方法: {method} =====")
        
        # 修復模型偏差
        fixed_model = fix_model_bias(model, method)
        
        # 獲取修復後模型的預測
        logger.info(f"測試修復後模型 ({method}) 預測")
        fixed_preds = test_model_predictions(fixed_model, test_input)
        
        # 保存修復後的模型
        save_path = output_dir / f"fixed_model_{method}.pth"
        save_model(fixed_model, save_path)
    
    logger.info(f"完成所有修復方法測試，結果已保存到 {output_dir}")

if __name__ == "__main__":
    main() 