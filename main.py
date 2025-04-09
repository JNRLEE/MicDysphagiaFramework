"""
主程序模組：整合所有組件，實現配置驅動的訓練和評估流程
功能：
1. 解析命令行參數
2. 加載配置文件
3. 創建數據加載器、模型和訓練器
4. 運行訓練和評估
5. 保存結果和可視化
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# 導入模組
from utils.config_loader import load_config
from data.dataset_factory import create_dataloaders
from models.model_factory import create_model
from trainers.trainer_factory import create_trainer
from visualization import visualize_results

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """解析命令行參數
    
    Returns:
        argparse.Namespace: 解析後的參數
    """
    parser = argparse.ArgumentParser(
        description="MicDysphagiaFramework：統一的吞嚥障礙評估框架"
    )
    
    parser.add_argument(
        "--config", type=str, required=True,
        help="配置文件路徑 (YAML 格式)"
    )
    parser.add_argument(
        "--default_config", type=str, default=None,
        help="默認配置文件路徑 (YAML 格式，可選)"
    )
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="啟用調試模式"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="使用的設備 (例如 'cpu', 'cuda', 'cuda:0')"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="隨機種子 (覆蓋配置文件中的設置)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="輸出目錄 (覆蓋配置文件中的設置)"
    )
    parser.add_argument(
        "--eval_only", action="store_true", default=False,
        help="只進行評估，不訓練"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="用於評估的檢查點路徑"
    )
    
    return parser.parse_args()

def setup_logging(config: Dict[str, Any]) -> None:
    """設置日誌記錄
    
    Args:
        config: 配置字典
    """
    global logger
    
    log_config = config.get('global', {}).get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    
    # 設置根日誌記錄器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 創建控制台處理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # 設置格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # 添加處理器
    root_logger.addHandler(console_handler)
    
    # 如果啟用了文件日誌
    if log_config.get('save_to_file', True):
        output_dir = config.get('global', {}).get('output_dir', 'runs/')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(output_dir, f'log_{timestamp}.txt')
        
        # 創建文件處理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        
        # 添加處理器
        root_logger.addHandler(file_handler)
        
        logger.info(f"日誌將保存到: {log_file}")

def set_seed(seed: int) -> None:
    """設置隨機種子以確保可重現性
    
    Args:
        seed: 隨機種子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"已設置隨機種子: {seed}")

def update_config_from_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """根據命令行參數更新配置
    
    Args:
        config: 配置字典
        args: 命令行參數
        
    Returns:
        Dict[str, Any]: 更新後的配置字典
    """
    # 確保 global 部分存在
    if 'global' not in config:
        config['global'] = {}
    
    # 更新調試模式
    if args.debug:
        config['global']['debug'] = True
    
    # 更新設備
    if args.device:
        config['global']['device'] = args.device
    
    # 更新隨機種子
    if args.seed is not None:
        config['global']['seed'] = args.seed
    
    # 更新輸出目錄
    if args.output_dir:
        config['global']['output_dir'] = args.output_dir
    
    # 確保輸出目錄存在
    output_dir = config['global'].get('output_dir', 'runs/')
    os.makedirs(output_dir, exist_ok=True)
    
    # 確保模型配置使用PyTorch
    if 'model' in config:
        if 'framework' in config['model']:
            logger.warning("配置中的'framework'選項已被棄用，將使用PyTorch作為唯一框架")
            config['model']['framework'] = 'pytorch'
    
    return config

def get_device(config: Dict[str, Any]) -> torch.device:
    """根據配置獲取設備
    
    Args:
        config: 配置字典
        
    Returns:
        torch.device: PyTorch 設備
    """
    device_str = config.get('global', {}).get('device', 'auto')
    
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    
    logger.info(f"使用設備: {device}")
    return device

def main() -> None:
    """主函數"""
    # 解析命令行參數
    args = parse_args()
    
    # 加載配置
    config_loader = load_config(args.config, args.default_config)
    config = config_loader.config
    
    # 根據命令行參數更新配置
    config = update_config_from_args(config, args)
    
    # 設置日誌記錄
    setup_logging(config)
    
    # 設置隨機種子
    seed = config.get('global', {}).get('seed', 42)
    set_seed(seed)
    
    # 獲取設備
    device = get_device(config)
    
    # 記錄配置信息
    logger.info(f"實驗名稱: {config.get('global', {}).get('experiment_name', 'unnamed')}")
    logger.info(f"配置文件: {args.config}")
    if args.default_config:
        logger.info(f"默認配置文件: {args.default_config}")
    
    # 保存配置到輸出目錄
    output_dir = config.get('global', {}).get('output_dir', 'runs/')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_save_path = os.path.join(output_dir, f'config_{timestamp}.yaml')
    config_loader.save_config(config_save_path)
    logger.info(f"配置已保存到: {config_save_path}")
    
    # 創建數據加載器
    logger.info("正在創建數據加載器...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(config)
        logger.info(f"數據加載器創建成功: 訓練集 {len(train_loader)} 批次，驗證集 {len(val_loader)} 批次，測試集 {len(test_loader)} 批次")
    except Exception as e:
        logger.error(f"創建數據加載器失敗: {str(e)}")
        raise
    
    # 創建模型
    logger.info("正在創建模型...")
    try:
        model = create_model(config)
        model.to(device)  # 將模型移到適當的設備上
        logger.info(f"模型創建成功: {type(model).__name__}")
    except Exception as e:
        logger.error(f"創建模型失敗: {str(e)}")
        raise
    
    # 創建訓練器
    logger.info("正在創建訓練器...")
    try:
        trainer = create_trainer(config, model, (train_loader, val_loader, test_loader))
        logger.info(f"訓練器創建成功: {type(trainer).__name__}")
    except Exception as e:
        logger.error(f"創建訓練器失敗: {str(e)}")
        raise
    
    # 如果指定了檢查點，加載檢查點
    if args.checkpoint:
        logger.info(f"從檢查點加載模型: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    
    # 訓練或評估
    if args.eval_only:
        logger.info("僅評估模式")
        metrics = trainer.evaluate()
        logger.info(f"評估指標: {metrics}")
    else:
        # 訓練模型
        logger.info("正在訓練模型...")
        try:
            trainer.train()
            logger.info("模型訓練完成")
        except Exception as e:
            logger.error(f"模型訓練失敗: {str(e)}")
            raise
        
        # 評估模型
        logger.info("正在評估模型...")
        try:
            metrics = trainer.evaluate()
            logger.info(f"評估指標: {metrics}")
        except Exception as e:
            logger.error(f"模型評估失敗: {str(e)}")
            raise
    
    # 可視化
    if config.get('visualization', {}).get('enabled', True):
        logger.info("正在生成可視化...")
        try:
            visualize_results.visualize(config, trainer)
            logger.info("可視化生成完成")
        except Exception as e:
            logger.error(f"生成可視化失敗: {str(e)}")
    
    logger.info("程序執行完成")

if __name__ == "__main__":
    # 設置默認日誌級別
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    try:
        main()
    except Exception as e:
        logger.exception(f"程序執行出錯: {str(e)}")
        sys.exit(1) 