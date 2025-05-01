"""
測試不同損失函數設置對音頻分類模型的影響
此腳本用於順序測試不同損失函數配置並記錄結果

Args:
    config_path: 基本配置文件路徑
    output_dir: 輸出結果目錄
    
Description:
    使用相同的數據集和模型設置，測試不同損失函數的影響
    記錄訓練和驗證損失、準確率等指標
    生成比較圖表
    
References:
    無
"""

import os
import sys
import yaml
import json
import torch
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import copy
from tqdm import tqdm

# 添加項目根目錄到路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 導入必要的模組
from utils.config_loader import load_config
from data.dataset_factory import create_dataloaders
from models.model_factory import create_model
from losses.loss_factory import LossFactory
from trainers.trainer_factory import create_trainer
from utils.data_adapter import adapt_datasets_to_model


# 設置日誌
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """設置隨機種子以確保實驗可重現
    
    Args:
        seed: 隨機種子
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"設置隨機種子: {seed}")


def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='損失函數測試工具')
    
    parser.add_argument('--config', type=str, required=True, help='基本配置文件路徑')
    parser.add_argument('--output_dir', type=str, default='results/loss_tests', help='輸出目錄')
    parser.add_argument('--epochs', type=int, default=10, help='訓練週期數(對每個損失函數)')
    parser.add_argument('--device', type=str, default='auto', help='使用的設備，例如cuda:0或cpu')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    
    return parser.parse_args()


def get_loss_configs() -> List[Dict[str, Any]]:
    """獲取要測試的損失函數配置列表
    
    Returns:
        List[Dict[str, Any]]: 損失函數配置列表
    """
    # 定義不同的損失函數配置
    loss_configs = [
        # 標準損失函數配置
        {
            "name": "MSELoss",
            "config": {
                "type": "MSELoss",
                "parameters": {"reduction": "mean"}
            }
        },
        {
            "name": "CrossEntropyLoss",
            "config": {
                "type": "CrossEntropyLoss",
                "parameters": {"reduction": "mean"}
            }
        },
        {
            "name": "FocalLoss",
            "config": {
                "type": "FocalLoss",
                "parameters": {
                    "alpha": 0.25,
                    "gamma": 2.0,
                    "reduction": "mean"
                }
            }
        },
        
        # 排序損失函數配置
        {
            "name": "PairwiseRanking",
            "config": {
                "type": "PairwiseRankingLoss",
                "parameters": {
                    "margin": 0.3,
                    "sampling_ratio": 0.25,
                    "sampling_strategy": "score_diff",
                    "use_exp": False
                }
            }
        },
        {
            "name": "ListNet",
            "config": {
                "type": "ListwiseRankingLoss",
                "parameters": {
                    "method": "listnet",
                    "temperature": 1.0,
                    "k": 10,
                    "group_size": 8,
                    "stochastic": True
                }
            }
        },
        {
            "name": "LambdaRank",
            "config": {
                "type": "LambdaRankLoss",
                "parameters": {
                    "sigma": 1.0,
                    "k": 10,
                    "sampling_ratio": 0.3
                }
            }
        },
        
        # 組合損失函數
        {
            "name": "MSE+PairwiseRanking",
            "config": {
                "combined": {
                    "mse": {
                        "type": "MSELoss",
                        "parameters": {"reduction": "mean"},
                        "weight": 0.6
                    },
                    "ranking": {
                        "type": "PairwiseRankingLoss",
                        "parameters": {
                            "margin": 0.3,
                            "sampling_ratio": 0.25,
                            "sampling_strategy": "score_diff"
                        },
                        "weight": 0.4
                    }
                }
            }
        }
    ]
    
    return loss_configs


def train_and_evaluate(config_path: str, loss_config: Dict[str, Any], 
                      output_dir: str, epochs: int, device: str, seed: int) -> Dict[str, Any]:
    """使用指定的損失函數訓練和評估模型
    
    Args:
        config_path: 基本配置文件路徑
        loss_config: 損失函數配置
        output_dir: 輸出目錄
        epochs: 訓練週期數
        device: 使用的設備
        seed: 隨機種子
        
    Returns:
        Dict[str, Any]: 訓練和評估結果
    """
    # 設置種子
    set_seed(seed)
    
    # 加載基本配置
    config_loader = load_config(config_path)
    config = config_loader.config
    
    # 更新損失函數配置
    config['training']['loss'] = loss_config['config']
    
    # 更新訓練週期數
    config['training']['epochs'] = epochs
    
    # 更新設備配置
    if device != 'auto':
        config['global']['device'] = device
    
    # 確保輸出目錄存在
    loss_name = loss_config['name']
    experiment_output_dir = os.path.join(output_dir, loss_name)
    os.makedirs(experiment_output_dir, exist_ok=True)
    config['global']['output_dir'] = experiment_output_dir
    
    # 保存更新後的配置
    config_save_path = os.path.join(experiment_output_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"配置已保存到: {config_save_path}")
    
    # 創建數據加載器
    logger.info(f"[{loss_name}] 初始化數據加載器...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # 創建模型
    logger.info(f"[{loss_name}] 創建模型...")
    model = create_model(config)
    
    # 獲取設備
    device_str = config['global'].get('device', 'auto')
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    model.to(device)
    
    # 創建訓練器
    logger.info(f"[{loss_name}] 創建訓練器...")
    trainer = create_trainer(config, model, (train_loader, val_loader, test_loader))
    
    # 訓練模型
    logger.info(f"[{loss_name}] 開始訓練...")
    training_start = datetime.now()
    trainer.train()
    training_end = datetime.now()
    training_time = (training_end - training_start).total_seconds()
    
    # 評估模型
    logger.info(f"[{loss_name}] 評估模型...")
    metrics = trainer.evaluate()
    
    # 收集結果
    results = {
        "loss_name": loss_name,
        "train_loss": trainer.train_losses,
        "val_loss": trainer.val_losses,
        "metrics": metrics,
        "training_time": training_time,
        "timestamp": datetime.now().isoformat()
    }
    
    # 保存結果
    results_path = os.path.join(experiment_output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"結果已保存到: {results_path}")
    
    return results


def visualize_comparison(all_results: List[Dict[str, Any]], output_dir: str) -> None:
    """可視化不同損失函數的比較結果
    
    Args:
        all_results: 所有損失函數的結果列表
        output_dir: 輸出目錄
    """
    plt.figure(figsize=(15, 12))
    
    # 訓練損失曲線
    plt.subplot(2, 2, 1)
    for result in all_results:
        plt.plot(result['train_loss'], label=result['loss_name'])
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    # 驗證損失曲線
    plt.subplot(2, 2, 2)
    for result in all_results:
        plt.plot(result['val_loss'], label=result['loss_name'])
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    # 最終驗證指標比較
    plt.subplot(2, 2, 3)
    metrics = ['accuracy', 'f1_score', 'precision', 'recall']
    loss_names = [result['loss_name'] for result in all_results]
    metric_values = {metric: [] for metric in metrics}
    
    for result in all_results:
        for metric in metrics:
            if metric in result['metrics'].get('val', {}):
                metric_values[metric].append(result['metrics']['val'][metric])
            else:
                metric_values[metric].append(0)
    
    x = np.arange(len(loss_names))
    width = 0.2
    offsets = np.linspace(-0.3, 0.3, len(metrics))
    
    for i, metric in enumerate(metrics):
        plt.bar(x + offsets[i], metric_values[metric], width, label=metric)
    
    plt.xlabel('Loss Function')
    plt.ylabel('Score')
    plt.title('Validation Metrics Comparison')
    plt.xticks(x, loss_names, rotation=45)
    plt.legend()
    plt.grid(True, axis='y')
    
    # 訓練時間比較
    plt.subplot(2, 2, 4)
    training_times = [result['training_time'] for result in all_results]
    plt.bar(loss_names, training_times)
    plt.xlabel('Loss Function')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_functions_comparison.png'))
    plt.close()
    
    logger.info(f"比較圖表已保存到: {os.path.join(output_dir, 'loss_functions_comparison.png')}")


def main():
    """主函數"""
    # 解析命令行參數
    args = parse_args()
    
    # 確保輸出目錄存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 獲取損失函數配置列表
    loss_configs = get_loss_configs()
    logger.info(f"找到 {len(loss_configs)} 個損失函數配置")
    
    # 收集所有結果
    all_results = []
    
    # 對每個損失函數進行訓練和評估
    for i, loss_config in enumerate(loss_configs):
        loss_name = loss_config['name']
        logger.info(f"[{i+1}/{len(loss_configs)}] 測試損失函數: {loss_name}")
        
        # 訓練和評估
        try:
            result = train_and_evaluate(
                config_path=args.config,
                loss_config=loss_config,
                output_dir=args.output_dir,
                epochs=args.epochs,
                device=args.device,
                seed=args.seed
            )
            all_results.append(result)
            logger.info(f"損失函數 {loss_name} 測試完成")
        except Exception as e:
            logger.error(f"損失函數 {loss_name} 測試失敗: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 保存所有結果
    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    logger.info(f"摘要已保存到: {summary_path}")
    
    # 可視化比較結果
    visualize_comparison(all_results, args.output_dir)
    
    logger.info("損失函數測試完成")


if __name__ == "__main__":
    main() 