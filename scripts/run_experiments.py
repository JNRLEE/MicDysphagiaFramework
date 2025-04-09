"""
實驗執行模組：用於執行吞嚥障礙評估模型實驗
功能：
1. 載入配置檔案
2. 設置實驗記錄
3. 初始化數據加載器
4. 初始化模型和損失函數
5. 進行訓練、驗證和測試
6. 保存結果和視覺化圖表
"""

import os
import sys
import yaml
import torch
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

# 添加項目根目錄到路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 導入必要的模組
from utils.config_loader import load_config
from data.dataset_factory import create_dataloaders
from models.model_factory import create_model
from losses.loss_factory import LossFactory
from trainers.trainer_factory import create_trainer
from utils.data_adapter import adapt_datasets_to_model


def setup_logging(log_dir: str, experiment_id: str) -> logging.Logger:
    """設置日誌記錄
    
    Args:
        log_dir: 日誌保存目錄
        experiment_id: 實驗ID
        
    Returns:
        logging.Logger: 配置好的日誌記錄器
        
    Description:
        建立日誌記錄器，配置格式和輸出目的地
        
    References:
        https://docs.python.org/3/library/logging.html
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # 創建日誌記錄器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除現有的處理器
    if logger.handlers:
        logger.handlers.clear()
    
    # 添加控制台處理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # 添加文件處理器
    file_handler = logging.FileHandler(os.path.join(log_dir, f'{experiment_id}.log'))
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # 同時記錄到公共實驗日誌
    exp_log_path = os.path.join(log_dir, 'experiments.log')
    exp_handler = logging.FileHandler(exp_log_path)
    exp_handler.setLevel(logging.INFO)
    exp_format = logging.Formatter('%(asctime)s - %(message)s')
    exp_handler.setFormatter(exp_format)
    
    # 創建過濾器，只記錄元數據到公共日誌
    class MetadataFilter(logging.Filter):
        def filter(self, record):
            return hasattr(record, 'metadata') and record.metadata
    
    exp_handler.addFilter(MetadataFilter())
    logger.addHandler(exp_handler)
    
    return logger


def log_metadata(logger: logging.Logger, config: Dict[str, Any], experiment_id: str) -> None:
    """記錄實驗元數據
    
    Args:
        logger: 日誌記錄器
        config: 配置字典
        experiment_id: 實驗ID
        
    Description:
        將實驗配置和環境信息記錄到日誌中
        
    References:
        None
    """
    metadata = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        metadata["cuda_device_name"] = torch.cuda.get_device_name(0)
    
    # 使用額外的參數來標記這是元數據記錄
    logger.info(f"實驗開始：{experiment_id}", extra={"metadata": True})
    logger.info(f"配置：{json.dumps(config, indent=2)}", extra={"metadata": True})
    logger.info(f"環境：PyTorch {torch.__version__}, "
               f"CUDA {'可用' if torch.cuda.is_available() else '不可用'}", 
               extra={"metadata": True})


def parse_args():
    """解析命令行參數
    
    Returns:
        argparse.Namespace: 解析後的參數
        
    Description:
        解析命令行參數，設置實驗配置
        
    References:
        https://docs.python.org/3/library/argparse.html
    """
    parser = argparse.ArgumentParser(description='吞嚥障礙評估模型訓練')
    
    parser.add_argument('--config', type=str, required=True, help='配置文件路徑')
    parser.add_argument('--default_config', type=str, help='默認配置文件路徑')
    parser.add_argument('--output_dir', type=str, help='輸出目錄')
    parser.add_argument('--device', type=str, help='使用的設備，例如cuda:0或cpu')
    parser.add_argument('--eval_only', action='store_true', help='僅進行評估，不訓練')
    parser.add_argument('--checkpoint', type=str, help='模型檢查點路徑，用於評估或繼續訓練')
    parser.add_argument('--seed', type=int, help='隨機種子')
    
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """設置隨機種子以確保實驗可重現
    
    Args:
        seed: 隨機種子
        
    Description:
        為Python、NumPy和PyTorch設置隨機種子，確保實驗結果可重現
        
    References:
        https://pytorch.org/docs/stable/notes/randomness.html
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, save_path: str, experiment_id: str, epoch: int) -> None:
    """保存模型檢查點
    
    Args:
        model: 要保存的模型
        save_path: 保存目錄
        experiment_id: 實驗ID
        epoch: 當前訓練輪數
        
    Description:
        將模型權重保存到指定路徑
        
    References:
        https://pytorch.org/tutorials/beginner/saving_loading_models.html
    """
    os.makedirs(save_path, exist_ok=True)
    checkpoint_path = os.path.join(save_path, f"{experiment_id}_epoch_{epoch}.pth")
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, checkpoint_path)
    
    logging.info(f"保存模型檢查點到 {checkpoint_path}")


def plot_training_curves(history: Dict[str, List], save_path: str, experiment_id: str) -> None:
    """繪製訓練曲線
    
    Args:
        history: 訓練歷史記錄
        save_path: 保存目錄
        experiment_id: 實驗ID
        
    Description:
        根據訓練歷史繪製損失和評估指標曲線
        
    References:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html
    """
    os.makedirs(save_path, exist_ok=True)
    
    # 繪製損失曲線
    plt.figure(figsize=(12, 5))
    
    # 訓練集和驗證集損失
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # 評估指標
    plt.subplot(1, 2, 2)
    if 'train_accuracy' in history:
        plt.plot(history['train_accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.ylabel('Accuracy')
    elif 'train_mae' in history:
        plt.plot(history['train_mae'], label='Train MAE')
        plt.plot(history['val_mae'], label='Validation MAE')
        plt.ylabel('MAE')
    
    plt.xlabel('Epoch')
    plt.title('Training and Validation Metric')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{experiment_id}_training_curves.png"))
    plt.close()
    
    logging.info(f"保存訓練曲線圖到 {save_path}/{experiment_id}_training_curves.png")


def main():
    """主函數
    
    Description:
        實驗執行的主入口，處理配置加載、模型訓練和評估
        
    References:
        None
    """
    # 解析命令行參數
    args = parse_args()
    
    # 載入配置
    config_loader = load_config(args.config, args.default_config)
    config = config_loader.config
    
    # 更新配置（如果命令行有指定）
    if args.output_dir:
        config['global']['output_dir'] = args.output_dir
    if args.device:
        config['global']['device'] = args.device
    if args.seed:
        config['global']['seed'] = args.seed
        
    # 創建實驗ID
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = config['global'].get('experiment_name', 'experiment')
    experiment_id = f"{experiment_name}_{timestamp}"
    
    # 設置輸出目錄
    output_dir = config['global'].get('output_dir', 'results')
    log_dir = os.path.join(output_dir, 'logs')
    model_dir = os.path.join(output_dir, 'models')
    results_dir = os.path.join(output_dir, 'plots')
    experiment_dir = os.path.join(output_dir, experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 設置日誌
    logger = setup_logging(log_dir, experiment_id)
    
    # 記錄元數據
    log_metadata(logger, config, experiment_id)
    
    # 設置隨機種子
    seed = config['global'].get('seed', 42)
    set_seed(seed)
    logger.info(f"設置隨機種子: {seed}")
    
    # 設置設備
    device_str = config['global'].get('device', 'auto')
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    logger.info(f"使用設備: {device}")
    
    # 創建數據加載器
    logger.info("初始化數據加載器...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(config)
        logger.info(f"數據加載成功：訓練集 {len(train_loader.dataset)} 樣本，"
                  f"驗證集 {len(val_loader.dataset)} 樣本，"
                  f"測試集 {len(test_loader.dataset)} 樣本")
    except Exception as e:
        logger.error(f"數據加載失敗: {str(e)}")
        raise
    
    # 創建模型
    logger.info("初始化模型...")
    try:
        model = create_model(config)
        logger.info(f"模型初始化成功: {type(model).__name__}")
    except Exception as e:
        logger.error(f"模型初始化失敗: {str(e)}")
        raise
    
    # 數據適配（如果需要）
    model_type = config['model'].get('type', 'swin_transformer').lower()
    data_type = config['data'].get('type', 'audio').lower()
    logger.info(f"模型類型: {model_type}, 數據類型: {data_type}")
    
    try:
        train_loader, val_loader, test_loader = adapt_datasets_to_model(
            model_type=model_type,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        )
        logger.info("數據適配成功")
    except Exception as e:
        logger.warning(f"數據適配過程中出現警告: {str(e)}")
    
    # 載入檢查點（如果有）
    if args.checkpoint:
        logger.info(f"載入檢查點: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # 移動模型到指定設備
    model = model.to(device)
    
    # 創建訓練器
    logger.info("初始化訓練器...")
    trainer = create_trainer(config, model, (train_loader, val_loader, test_loader))
    
    # 訓練或評估
    if args.eval_only:
        logger.info("僅評估模式")
        results = trainer.evaluate()
        logger.info(f"評估結果: {results}")
    else:
        # 訓練模型 - 注意這裡不需要調用train方法，因為TrainerFactory已經處理了
        logger.info("開始訓練...")
        epochs = config['training'].get('epochs', 100)
        save_every = config['training'].get('save_every', 10)
        
        # 從trainer獲取訓練結果
        history = {
            'train_loss': trainer.train_losses,
            'val_loss': trainer.val_losses
        }
        
        # 添加其他指標
        if hasattr(trainer, 'metrics'):
            for key, value in trainer.metrics.get('train', {}).items():
                history[f'train_{key}'] = value
            for key, value in trainer.metrics.get('val', {}).items():
                history[f'val_{key}'] = value
        
        # 保存訓練歷史和最終模型
        save_model(model, model_dir, experiment_id, epochs)
        
        # 繪製訓練曲線
        plot_training_curves(history, results_dir, experiment_id)
        
        # 評估最終模型
        logger.info("評估最終模型...")
        results = {'val_loss': trainer.best_val_loss}
        if hasattr(trainer, 'test_results'):
            results.update(trainer.test_results)
        logger.info(f"最終評估結果: {results}")
    
    logger.info(f"實驗 {experiment_id} 完成")
    
    # 保存完整配置和結果
    final_results = {
        "experiment_id": experiment_id,
        "config": config,
        "results": results,
        "completed_at": datetime.now().isoformat()
    }
    
    with open(os.path.join(experiment_dir, "results.json"), "w") as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"結果已保存到 {experiment_dir}/results.json")
    
    return results


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"執行過程中發生錯誤: {str(e)}", exc_info=True)
        sys.exit(1) 