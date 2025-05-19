#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
驗證部署的整合流程

此腳本用於驗證索引數據集特性的部署情況，包括：
1. 確認資料結構完整性
2. 確認索引CSV文件的正確性
3. 驗證數據加載流程
4. 測試模型訓練的基本功能
5. 測試TensorBoard日誌記錄

使用方式:
    python scripts/validate_deployment.py

Args:
    --config: 測試使用的配置文件路徑，默認為 config/example_classification_indexed.yaml
    --check_data_only: 如果設為True，僅檢查數據結構而不執行訓練，默認為False
    --epochs: 測試訓練的輪次數，默認為2

Returns:
    無返回值，所有測試結果將輸出到終端和日誌文件
"""

import os
import sys
import argparse
import logging
import pandas as pd
import yaml
import torch
import time
from pathlib import Path

# 添加項目根目錄到系統路徑
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 導入項目模塊
from utils.config_loader import load_config
from utils.data_index_loader import DataIndexLoader
from data.dataset_factory import DatasetFactory
from models.model_factory import ModelFactory
from trainers.trainer_factory import TrainerFactory

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment_validation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def check_directory_structure():
    """檢查數據目錄結構"""
    required_dirs = [
        'data/metadata',
        'data/raw',
        'data/processed/feature_vectors',
        'data/processed/spectrograms'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            logger.error(f"目錄不存在: {dir_path}")
            all_exist = False
        else:
            logger.info(f"目錄存在: {dir_path}")
    
    return all_exist

def validate_index_csv():
    """驗證索引CSV文件的完整性和正確性"""
    index_path = 'data/metadata/data_index.csv'
    if not os.path.exists(index_path):
        logger.error(f"索引CSV文件不存在: {index_path}")
        return False
    
    try:
        # 讀取索引CSV
        df = pd.read_csv(index_path)
        logger.info(f"成功讀取索引CSV，共 {len(df)} 筆資料")
        
        # 檢查必要欄位
        required_columns = ['file_path', 'score', 'patient_id']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"索引CSV缺少必要欄位: {col}")
                return False
            logger.info(f"索引CSV包含欄位: {col}")
        
        # 檢查文件路徑的有效性
        if 'file_path' in df.columns:
            sample_size = min(10, len(df))
            valid_paths = 0
            
            for idx, row in df.sample(sample_size).iterrows():
                file_path = row['file_path']
                if pd.isna(file_path):
                    continue
                    
                if os.path.exists(file_path):
                    valid_paths += 1
                else:
                    logger.warning(f"文件路徑無效: {file_path}")
            
            logger.info(f"抽樣檢查 {sample_size} 個文件路徑，有效路徑數: {valid_paths}")
            if valid_paths == 0:
                logger.error("所有抽樣的文件路徑都無效")
                return False
        
        return True
    
    except Exception as e:
        logger.error(f"驗證索引CSV時發生錯誤: {e}")
        return False

def test_data_loading(config_path):
    """測試數據加載功能"""
    try:
        # 加載配置
        config = load_config(config_path)
        logger.info(f"成功加載配置文件: {config_path}")
        
        # 創建數據集
        dataset_factory = DatasetFactory(config)
        train_dataset, val_dataset, test_dataset = dataset_factory.create_datasets()
        
        logger.info(f"成功創建數據集:")
        logger.info(f"  訓練集: {len(train_dataset)} 項目")
        logger.info(f"  驗證集: {len(val_dataset)} 項目")
        logger.info(f"  測試集: {len(test_dataset)} 項目")
        
        # 測試數據加載
        train_loader, val_loader, test_loader = dataset_factory.create_dataloaders()
        
        # 獲取一個批次的數據
        inputs, targets = next(iter(train_loader))
        logger.info(f"成功加載一個批次的數據:")
        logger.info(f"  輸入形狀: {inputs.shape}")
        logger.info(f"  目標形狀: {targets.shape}")
        
        return True
    
    except Exception as e:
        logger.error(f"測試數據加載時發生錯誤: {e}")
        return False

def test_model_training(config_path, epochs=2):
    """測試模型訓練功能"""
    try:
        # 設置隨機種子以確保可重現性
        torch.manual_seed(42)
        
        # 加載配置
        config = load_config(config_path)
        logger.info(f"成功加載配置文件: {config_path}")
        
        # 設置較少的訓練輪次用於測試
        config['training']['epochs'] = epochs
        
        # 建立輸出目錄
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/deployment_test_{timestamp}"
        config['global']['output_dir'] = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 創建模型
        model_factory = ModelFactory(config)
        model = model_factory.create_model()
        logger.info(f"成功創建模型: {model.__class__.__name__}")
        
        # 創建數據加載器
        dataset_factory = DatasetFactory(config)
        train_loader, val_loader, test_loader = dataset_factory.create_dataloaders()
        
        # 創建訓練器
        trainer_factory = TrainerFactory(config, model)
        trainer = trainer_factory.create_trainer()
        
        # 執行訓練
        logger.info(f"開始測試訓練，輪次數: {epochs}")
        trainer.train(train_loader, val_loader)
        
        # 檢查模型檢查點是否存在
        checkpoint_path = os.path.join(output_dir, "models", "best_model.pth")
        if os.path.exists(checkpoint_path):
            logger.info(f"成功保存模型檢查點: {checkpoint_path}")
        else:
            logger.warning(f"未找到模型檢查點: {checkpoint_path}")
        
        # 檢查TensorBoard日誌是否存在
        tensorboard_dir = os.path.join(output_dir, "tensorboard_logs")
        if os.path.exists(tensorboard_dir) and len(os.listdir(tensorboard_dir)) > 0:
            logger.info(f"成功生成TensorBoard日誌: {tensorboard_dir}")
        else:
            logger.warning(f"未找到TensorBoard日誌或目錄為空: {tensorboard_dir}")
        
        # 執行評估
        logger.info("開始測試評估")
        eval_results = trainer.evaluate(test_loader)
        logger.info(f"評估結果: {eval_results}")
        
        return True
    
    except Exception as e:
        logger.error(f"測試模型訓練時發生錯誤: {e}")
        return False

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='驗證部署的整合流程')
    parser.add_argument('--config', type=str, default='config/example_classification_indexed.yaml',
                        help='測試使用的配置文件路徑')
    parser.add_argument('--check_data_only', action='store_true', 
                        help='如果設為True，僅檢查數據結構而不執行訓練')
    parser.add_argument('--epochs', type=int, default=2,
                        help='測試訓練的輪次數')
    
    args = parser.parse_args()
    
    logger.info("開始驗證部署")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"只檢查數據: {args.check_data_only}")
    logger.info(f"訓練輪次數: {args.epochs}")
    
    # 檢查目錄結構
    logger.info("=====================")
    logger.info("1. 檢查目錄結構")
    dirs_ok = check_directory_structure()
    if not dirs_ok:
        logger.warning("目錄結構檢查未通過")
    else:
        logger.info("目錄結構檢查通過")
    
    # 驗證索引CSV
    logger.info("=====================")
    logger.info("2. 驗證索引CSV")
    csv_ok = validate_index_csv()
    if not csv_ok:
        logger.warning("索引CSV驗證未通過")
    else:
        logger.info("索引CSV驗證通過")
    
    # 測試數據加載
    logger.info("=====================")
    logger.info("3. 測試數據加載")
    data_ok = test_data_loading(args.config)
    if not data_ok:
        logger.warning("數據加載測試未通過")
    else:
        logger.info("數據加載測試通過")
    
    # 如果只檢查數據，則在此結束
    if args.check_data_only:
        logger.info("=====================")
        logger.info("僅執行數據檢查，跳過模型訓練測試")
        logger.info("驗證部署完成")
        
        if dirs_ok and csv_ok and data_ok:
            logger.info("數據結構驗證全部通過")
            return 0
        else:
            logger.warning("數據結構驗證未全部通過，請檢查日誌")
            return 1
    
    # 測試模型訓練
    logger.info("=====================")
    logger.info("4. 測試模型訓練")
    training_ok = test_model_training(args.config, args.epochs)
    if not training_ok:
        logger.warning("模型訓練測試未通過")
    else:
        logger.info("模型訓練測試通過")
    
    # 總結
    logger.info("=====================")
    logger.info("部署驗證摘要:")
    logger.info(f"1. 目錄結構檢查: {'通過' if dirs_ok else '未通過'}")
    logger.info(f"2. 索引CSV驗證: {'通過' if csv_ok else '未通過'}")
    logger.info(f"3. 數據加載測試: {'通過' if data_ok else '未通過'}")
    logger.info(f"4. 模型訓練測試: {'通過' if training_ok else '未通過'}")
    
    if dirs_ok and csv_ok and data_ok and training_ok:
        logger.info("驗證全部通過，部署成功")
        return 0
    else:
        logger.warning("驗證未全部通過，請檢查日誌")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 