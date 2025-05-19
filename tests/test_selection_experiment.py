"""
測試使用selection篩選並完整運行實驗流程的腳本
測試在使用DrLee_Evaluation作為標籤同時以selection:'乾吞嚥1口'進行篩選的情況
"""

import logging
import yaml
import sys
import os
import torch
from torch.utils.data import DataLoader
from utils.config_loader import load_config
from data.dataset_factory import create_dataset
from models.model_factory import create_model
from trainers.trainer_factory import create_trainer

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 準備測試配置
    test_config = {
        "global": {
            "experiment_name": "test_selection_experiment",
            "seed": 42,
            "debug": False,
            "device": "auto",
            "output_dir": "results",
            "logging_level": "INFO"
        },
        "data": {
            "type": "audio",
            "use_index": True,
            "index_path": "data/metadata/data_index.csv",
            "label_field": "DrLee_Evaluation",
            "filter_criteria": {
                "status": "processed",
                "selection": "乾吞嚥1口"  # 使用selection篩選
            },
            "splits": {
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "split_seed": 42,
                "split_by_patient": True
            }
        },
        "model": {
            "name": "fcnn",
            "params": {
                "input_dim": 512,
                "hidden_dims": [256, 128, 64],
                "dropout": 0.3,
                "activation": "relu",
                "batch_norm": True,
                "num_classes": 3,
                "is_classification": True
            }
        },
        "training": {
            "epochs": 3,  # 只運行幾個epoch用於測試
            "batch_size": 32,
            "optimizer": {
                "name": "adam",
                "params": {
                    "lr": 0.001
                }
            },
            "loss": {
                "name": "CrossEntropyLoss"
            }
        }
    }
    
    # 儲存臨時配置檔案
    config_path = 'temp_selection_experiment.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f)
    
    try:
        # 載入配置
        config = load_config(config_path)
        logger.info("成功載入配置")
        
        # 創建資料集
        logger.info("創建資料集...")
        train_set, val_set, test_set = create_dataset(config)
        
        logger.info(f"訓練集大小: {len(train_set)}")
        logger.info(f"驗證集大小: {len(val_set)}")
        logger.info(f"測試集大小: {len(test_set)}")
        
        # 創建數據加載器
        train_loader = DataLoader(train_set, batch_size=config["training"]["batch_size"], shuffle=True)
        val_loader = DataLoader(val_set, batch_size=config["training"]["batch_size"])
        test_loader = DataLoader(test_set, batch_size=config["training"]["batch_size"])
        
        # 創建模型 - 直接使用配置中的model部分
        logger.info("創建模型...")
        model_config = {
            "name": config["model"]["name"],
            "params": config["model"]["params"]
        }
        model = create_model(model_config)
        
        # 創建訓練器
        logger.info("創建訓練器...")
        trainer = create_trainer(config, model)
        
        # 執行訓練
        logger.info("開始訓練...")
        try:
            trainer.train(train_loader, val_loader)
            
            # 執行評估
            logger.info("開始評估...")
            test_metrics = trainer.evaluate(test_loader)
            logger.info(f"測試集評估結果: {test_metrics}")
        except Exception as e:
            logger.error(f"訓練或評估過程中出錯: {str(e)}")
        
    finally:
        # 清理臨時檔案
        if os.path.exists(config_path):
            os.remove(config_path)

if __name__ == "__main__":
    main() 