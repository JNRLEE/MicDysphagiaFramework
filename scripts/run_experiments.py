"""
實驗執行腳本：執行基於YAML配置的實驗

此腳本負責解析命令行參數，加載並驗證YAML配置文件，
然後設置數據集、模型和訓練器，並執行實驗。
"""

import argparse
import logging
import os
import sys
import yaml
import torch
import json
from datetime import datetime
from typing import Dict, Any

# 確保當前目錄在系統路徑中，以便導入自定義模塊
# 將項目根目錄添加到系統路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)  # 使用insert(0)確保它是最優先檢查的路徑

# 導入必要的模塊
from data import dataset_factory
from models import model_factory
from trainers import trainer_factory
from models.hook_bridge import get_analyzer_callbacks_from_config  # 新增：導入獲取分析器回調的函數
from utils.save_manager import SaveManager  # 新增：導入 SaveManager

# 設置日誌記錄
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='運行吞嚥聲音分析實驗')
    parser.add_argument('--config', required=True, help='YAML 配置文件路徑')
    parser.add_argument('--device', default=None, help='設備 (例如 cuda:0, cpu)')
    parser.add_argument('--output_dir', default=None, help='輸出目錄')
    parser.add_argument('--debug', action='store_true', help='啟用調試模式')
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """加載並驗證 YAML 配置文件
    
    Args:
        config_path: YAML 配置文件路徑
        
    Returns:
        Dict[str, Any]: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # TODO: 添加配置驗證邏輯
    
    return config

def setup_experiment(config: Dict[str, Any], args) -> Dict[str, Any]:
    """設置實驗環境並應用命令行覆蓋
    
    Args:
        config: 從 YAML 文件加載的配置
        args: 命令行參數
        
    Returns:
        Dict[str, Any]: 修改後的配置
    """
    # 應用命令行參數覆蓋
    if args.device:
        config['global']['device'] = args.device
    
    if args.output_dir:
        config['global']['output_dir'] = args.output_dir
    
    if args.debug:
        config['global']['debug'] = True
    
    # 設置輸出目錄
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config['global']['output_dir'] = os.path.join(
        config['global'].get('output_dir', 'results'),
        f"{config['global']['experiment_name']}_{timestamp}"
    )
    
    # 設置設備
    if config['global'].get('device', 'auto') == 'auto':
        config['global']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 設置種子以確保可重現性
    if 'seed' in config['global']:
        seed = config['global']['seed']
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    # 創建實驗目錄並保存配置
    os.makedirs(config['global']['output_dir'], exist_ok=True)
    save_manager = SaveManager(config['global']['output_dir'])
    save_manager.save_config(config)
    
    return config

def run_experiment(config: Dict[str, Any]):
    """執行實驗
    
    Args:
        config: 實驗配置
    """
    logger.info(f"開始實驗: {config['global']['experiment_name']}")
    logger.info(f"使用設備: {config['global']['device']}")
    
    # 創建 SaveManager 實例
    save_manager = SaveManager(config['global']['output_dir'])
    
    # 創建數據集和數據加載器
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = (
        dataset_factory.create_datasets_and_loaders(config)
    )
    logger.info(f"數據集已創建，訓練集大小: {len(train_dataset)}")
    
    # 創建模型
    model = model_factory.create_model(config)
    logger.info(f"模型已創建，類型: {config['model']['type']}")
    
    # 創建訓練器
    trainer = trainer_factory.create_trainer(config, model, (train_loader, val_loader, test_loader))
    logger.info("訓練器已創建")
    
    # 獲取並添加分析器回調
    analyzer_callbacks = get_analyzer_callbacks_from_config(config)
    if analyzer_callbacks:
        trainer.add_callbacks(analyzer_callbacks)
        logger.info(f"已添加 {len(analyzer_callbacks)} 個分析器回調")
        
    # 執行訓練
    results = trainer.train(train_loader, val_loader, test_loader)
    
    # 使用 SaveManager 保存訓練結果
    results_path = save_manager.save_results(results)
    
    logger.info(f"實驗完成，結果已保存到 {results_path}")
    
    return results

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    config = setup_experiment(config, args)
    run_experiment(config) 