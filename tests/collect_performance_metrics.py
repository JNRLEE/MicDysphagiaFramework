#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
收集性能指標腳本

此腳本用於收集不同配置下模型的性能指標，包括：
1. 訓練時間
2. 內存使用情況
3. 數據加載時間
4. 模型準確率
5. 推理時間

使用方式:
    python scripts/collect_performance_metrics.py --config config/example_classification_indexed.yaml

Args:
    --config: 要測試的配置文件路徑，默認為 config/example_classification_indexed.yaml
    --compare_with: 比較對象的配置文件路徑，用於比較索引模式與傳統模式
    --epochs: 訓練輪次數，默認為3
    --output: 輸出結果的JSON文件路徑，默認為 tests/benchmark_results.json
    --include_memory: 是否包含內存使用情況的測量，默認為True

Returns:
    生成一個JSON文件，包含所有收集的性能指標
"""

import os
import sys
import json
import time
import argparse
import logging
import torch
import psutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# 添加項目根目錄到系統路徑
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 導入項目模塊
from utils.config_loader import load_config
from data.dataset_factory import DatasetFactory
from models.model_factory import ModelFactory
from trainers.trainer_factory import TrainerFactory

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('performance_metrics.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class MetricsCollector:
    """性能指標收集器類"""
    
    def __init__(self, config_path, epochs=3, include_memory=True):
        """
        初始化指標收集器
        
        Args:
            config_path: 配置文件路徑
            epochs: 訓練輪次數
            include_memory: 是否收集內存使用情況
        """
        self.config_path = config_path
        self.epochs = epochs
        self.include_memory = include_memory
        self.results = {
            "config_file": config_path,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "environment": self._get_environment_info(),
            "metrics": {}
        }
        
        # 加載配置
        self.config = load_config(config_path)
        self.config['training']['epochs'] = epochs
        
        # 設置隨機種子
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 建立輸出目錄
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/performance_test_{timestamp}"
        self.config['global']['output_dir'] = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def _get_environment_info(self):
        """獲取環境信息"""
        return {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "device": self.config.get('global', {}).get('device', 'auto')
        }
    
    def _get_memory_usage(self):
        """獲取當前內存使用情況"""
        if not self.include_memory:
            return {"current_memory_mb": "N/A"}
            
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            "current_memory_mb": memory_info.rss / (1024 * 1024)
        }
    
    def collect_data_loading_metrics(self):
        """收集數據加載性能指標"""
        logger.info("開始收集數據加載性能指標")
        metrics = {}
        
        try:
            # 記錄開始時間
            start_time = time.time()
            
            # 記錄初始內存
            initial_memory = self._get_memory_usage()
            
            # 創建數據集
            dataset_factory = DatasetFactory(self.config)
            train_dataset, val_dataset, test_dataset = dataset_factory.create_datasets()
            
            # 記錄數據集創建時間和內存
            dataset_created_time = time.time()
            dataset_memory = self._get_memory_usage()
            
            # 創建數據加載器
            train_loader, val_loader, test_loader = dataset_factory.create_dataloaders()
            
            # 測試數據加載速度
            data_load_times = []
            for i, (inputs, targets) in enumerate(train_loader):
                if i >= 5:  # 只測試前5個批次
                    break
                batch_start = time.time()
                # 模擬數據處理
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                batch_end = time.time()
                data_load_times.append(batch_end - batch_start)
            
            # 記錄結束時間和內存
            end_time = time.time()
            final_memory = self._get_memory_usage()
            
            # 記錄指標
            metrics["dataset_sizes"] = {
                "train": len(train_dataset),
                "val": len(val_dataset),
                "test": len(test_dataset)
            }
            metrics["dataset_creation_time"] = dataset_created_time - start_time
            metrics["dataloader_creation_time"] = end_time - dataset_created_time
            metrics["total_data_prep_time"] = end_time - start_time
            metrics["batch_load_times"] = {
                "mean": np.mean(data_load_times),
                "min": np.min(data_load_times),
                "max": np.max(data_load_times)
            }
            
            if self.include_memory:
                metrics["memory_usage"] = {
                    "initial": initial_memory["current_memory_mb"],
                    "after_dataset_creation": dataset_memory["current_memory_mb"],
                    "final": final_memory["current_memory_mb"],
                    "total_increase": final_memory["current_memory_mb"] - initial_memory["current_memory_mb"]
                }
            
            logger.info("數據加載性能指標收集完成")
            
        except Exception as e:
            logger.error(f"收集數據加載指標時發生錯誤: {e}")
            metrics["error"] = str(e)
        
        self.results["metrics"]["data_loading"] = metrics
        return metrics
    
    def collect_training_metrics(self):
        """收集訓練性能指標"""
        logger.info("開始收集訓練性能指標")
        metrics = {}
        
        try:
            # 記錄初始內存
            initial_memory = self._get_memory_usage()
            
            # 創建模型
            model_factory = ModelFactory(self.config)
            model = model_factory.create_model()
            
            # 創建數據加載器
            dataset_factory = DatasetFactory(self.config)
            train_loader, val_loader, test_loader = dataset_factory.create_dataloaders()
            
            # 創建訓練器
            trainer_factory = TrainerFactory(self.config, model)
            trainer = trainer_factory.create_trainer()
            
            # 記錄開始時間
            start_time = time.time()
            
            # 執行訓練
            trainer.train(train_loader, val_loader)
            
            # 記錄結束時間
            end_time = time.time()
            
            # 記錄訓練後內存
            final_memory = self._get_memory_usage()
            
            # 執行評估
            eval_results = trainer.evaluate(test_loader)
            
            # 記錄推理時間
            inference_start = time.time()
            # 測量推理速度
            with torch.no_grad():
                for i, (inputs, _) in enumerate(test_loader):
                    if i >= 10:  # 只測試前10個批次
                        break
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                    outputs = model(inputs)
            inference_end = time.time()
            
            # 記錄指標
            metrics["training_time"] = end_time - start_time
            metrics["training_time_per_epoch"] = (end_time - start_time) / self.epochs
            metrics["inference_time"] = inference_end - inference_start
            metrics["eval_results"] = eval_results
            
            if self.include_memory:
                metrics["memory_usage"] = {
                    "initial": initial_memory["current_memory_mb"],
                    "after_training": final_memory["current_memory_mb"],
                    "total_increase": final_memory["current_memory_mb"] - initial_memory["current_memory_mb"]
                }
            
            # 提取TensorBoard日誌中的損失曲線數據
            metrics["tensorboard_logs"] = "available" if os.path.exists(os.path.join(self.config['global']['output_dir'], "tensorboard_logs")) else "unavailable"
            
            logger.info("訓練性能指標收集完成")
            
        except Exception as e:
            logger.error(f"收集訓練指標時發生錯誤: {e}")
            metrics["error"] = str(e)
        
        self.results["metrics"]["training"] = metrics
        return metrics
    
    def collect_all_metrics(self):
        """收集所有性能指標"""
        logger.info(f"開始收集配置 {self.config_path} 的所有性能指標")
        
        self.collect_data_loading_metrics()
        self.collect_training_metrics()
        
        logger.info("所有性能指標收集完成")
        return self.results


def compare_configs(config1_results, config2_results):
    """比較兩個配置的性能指標"""
    comparison = {
        "configs": {
            "config1": config1_results["config_file"],
            "config2": config2_results["config_file"]
        },
        "data_loading": {},
        "training": {}
    }
    
    # 比較數據加載指標
    data1 = config1_results["metrics"].get("data_loading", {})
    data2 = config2_results["metrics"].get("data_loading", {})
    
    if data1 and data2:
        comparison["data_loading"] = {
            "total_data_prep_time_diff": data1.get("total_data_prep_time", 0) - data2.get("total_data_prep_time", 0),
            "batch_load_times_mean_diff": data1.get("batch_load_times", {}).get("mean", 0) - data2.get("batch_load_times", {}).get("mean", 0)
        }
    
    # 比較訓練指標
    train1 = config1_results["metrics"].get("training", {})
    train2 = config2_results["metrics"].get("training", {})
    
    if train1 and train2:
        comparison["training"] = {
            "training_time_diff": train1.get("training_time", 0) - train2.get("training_time", 0),
            "inference_time_diff": train1.get("inference_time", 0) - train2.get("inference_time", 0)
        }
        
        # 比較評估結果
        eval1 = train1.get("eval_results", {})
        eval2 = train2.get("eval_results", {})
        
        if eval1 and eval2:
            comparison["training"]["eval_results_diff"] = {
                key: eval1.get(key, 0) - eval2.get(key, 0) for key in set(eval1.keys()).intersection(set(eval2.keys()))
            }
    
    return comparison


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='收集模型性能指標')
    parser.add_argument('--config', type=str, default='config/example_classification_indexed.yaml',
                        help='配置文件路徑')
    parser.add_argument('--compare_with', type=str, default=None,
                        help='比較對象的配置文件路徑')
    parser.add_argument('--epochs', type=int, default=3,
                        help='訓練輪次數')
    parser.add_argument('--output', type=str, default='tests/benchmark_results.json',
                        help='輸出結果的JSON文件路徑')
    parser.add_argument('--no-memory', dest='include_memory', action='store_false',
                        help='不包含內存使用情況的測量')
    
    args = parser.parse_args()
    
    logger.info("開始收集性能指標")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"訓練輪次數: {args.epochs}")
    logger.info(f"包含內存測量: {args.include_memory}")
    
    # 收集主配置的性能指標
    collector = MetricsCollector(args.config, args.epochs, args.include_memory)
    main_results = collector.collect_all_metrics()
    
    # 如果提供了比較對象，收集比較對象的性能指標並進行比較
    if args.compare_with:
        logger.info(f"與配置 {args.compare_with} 進行比較")
        compare_collector = MetricsCollector(args.compare_with, args.epochs, args.include_memory)
        compare_results = compare_collector.collect_all_metrics()
        
        # 比較兩個配置
        comparison = compare_configs(main_results, compare_results)
        
        # 保存結果
        final_results = {
            "main": main_results,
            "compare": compare_results,
            "comparison": comparison
        }
    else:
        final_results = main_results
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 保存結果為JSON
    with open(args.output, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"性能指標已保存到 {args.output}")
    
    # 輸出簡要結果
    if "metrics" in main_results and "training" in main_results["metrics"]:
        train_metrics = main_results["metrics"]["training"]
        logger.info("=====================================")
        logger.info("性能測試結果摘要:")
        logger.info(f"總訓練時間: {train_metrics.get('training_time', 'N/A'):.2f} 秒")
        logger.info(f"每輪訓練時間: {train_metrics.get('training_time_per_epoch', 'N/A'):.2f} 秒")
        
        eval_results = train_metrics.get("eval_results", {})
        for metric, value in eval_results.items():
            logger.info(f"評估指標 - {metric}: {value}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 