"""
測試不篩選selection功能的腳本
測試在使用DrLee_Evaluation作為標籤但不使用selection進行篩選的情況
"""

import logging
import yaml
import sys
import os
import torch
from torch.utils.data import Subset
from utils.config_loader import load_config
from data.dataset_factory import create_dataset

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 準備測試配置 - 不使用selection篩選
    test_config = {
        "global": {
            "experiment_name": "test_no_selection_filter",
            "seed": 42,
            "debug": False,
            "device": "auto",
            "output_dir": "outputs",
            "logging_level": "INFO"
        },
        "data": {
            "type": "audio",
            "use_index": True,
            "index_path": "data/metadata/data_index.csv",
            "label_field": "DrLee_Evaluation",
            "filter_criteria": {
                "status": "processed",
                # 不設置selection篩選
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
            "epochs": 2,
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
    config_path = 'temp_test_no_selection.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f)
    
    try:
        # 載入配置
        config = load_config(config_path)
        logger.info("成功載入配置")
        
        # 創建資料集
        logger.info("創建資料集...")
        train_set, val_set, test_set = create_dataset(config)
        
        # 檢查資料集
        logger.info(f"訓練集大小: {len(train_set)}")
        logger.info(f"驗證集大小: {len(val_set)}")
        logger.info(f"測試集大小: {len(test_set)}")
        
        # 獲取原始數據集的屬性
        if isinstance(train_set, Subset) and hasattr(train_set.dataset, 'num_classes'):
            logger.info(f"類別數: {train_set.dataset.num_classes}")
        else:
            logger.info("無法獲取類別數量")
        
        # 檢查第一個樣本
        if len(train_set) > 0:
            sample = train_set[0]
            logger.info(f"樣本類型: {type(sample)}")
            if isinstance(sample, dict):
                logger.info(f"樣本鍵值: {list(sample.keys())}")
                if 'label' in sample:
                    logger.info(f"標籤: {sample['label']}")
            else:
                logger.info(f"樣本不是字典類型: {sample}")
        else:
            logger.warning("訓練集為空，無法檢查樣本")
        
    finally:
        # 清理臨時檔案
        if os.path.exists(config_path):
            os.remove(config_path)

if __name__ == "__main__":
    main() 