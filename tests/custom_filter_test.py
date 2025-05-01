#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
測試數據過濾邏輯並顯示過濾後的樣本詳細信息
此測試腳本將使用與正式系統相同的過濾邏輯，顯示每一筆過濾後數據的完整路徑
"""

import os
import sys
import logging
import json
from pathlib import Path
from collections import defaultdict
import torch
from pprint import pprint

# 添加項目根目錄到路徑以便導入模塊
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_loader import load_config
from data.audio_dataset import AudioDataset
from utils.custom_classification_loader import CustomClassificationLoader

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

def run_filter_test(config_path):
    """
    運行數據過濾測試，顯示過濾後的樣本詳細信息
    
    Args:
        config_path (str): 配置文件路徑
        
    Returns:
        None
    """
    # 載入配置
    logger.info(f"載入配置文件: {config_path}")
    config = load_config(config_path)
    
    # 獲取音頻目錄
    wav_dir = config['data']['source']['wav_dir']
    logger.info(f"音頻目錄: {wav_dir}")
    
    # 創建數據集實例（使用與正式系統相同的過濾邏輯）
    dataset = AudioDataset(root_dir=wav_dir, config=config)
    
    # 顯示過濾結果
    logger.info(f"過濾後的樣本總數: {len(dataset.samples)}")
    
    # 按患者ID分組
    samples_by_patient = defaultdict(list)
    for idx, sample in enumerate(dataset.samples):
        patient_id = sample['patient_id']
        samples_by_patient[patient_id].append({
            'idx': idx,
            'path': sample['audio_path'],
            'score': sample['score'].item(),
            'selection': sample['selection'],
            'selection_type': sample['selection_type'],
            'label': sample['label'].item() if isinstance(sample['label'], torch.Tensor) else sample['label']
        })
    
    # 顯示每個患者的樣本信息
    logger.info(f"過濾後的患者總數: {len(samples_by_patient)}")
    logger.info("=" * 80)
    logger.info("每個患者的樣本詳細信息：")
    
    for patient_id, samples in samples_by_patient.items():
        logger.info("-" * 80)
        logger.info(f"患者ID: {patient_id} (共 {len(samples)} 個樣本)")
        for i, sample in enumerate(samples, 1):
            logger.info(f"  樣本 {i}:")
            logger.info(f"    路徑: {sample['path']}")
            logger.info(f"    分數: {sample['score']}")
            logger.info(f"    選擇: {sample['selection']} (類型: {sample['selection_type']})")
            
            # 嘗試獲取自定義分類信息
            if dataset.custom_classifier and dataset.custom_classifier.enabled:
                class_name = dataset.custom_classifier.get_class(patient_id)
                if class_name:
                    logger.info(f"    自定義分類: {class_name} (索引: {sample['label']})")
            else:
                logger.info(f"    標籤: {sample['label']}")
    
    # 分析所有樣本的目錄路徑
    all_dirs = set()
    for sample in dataset.samples:
        path = Path(sample['audio_path'])
        all_dirs.add(str(path.parent))
    
    logger.info("=" * 80)
    logger.info(f"樣本來自 {len(all_dirs)} 個不同的目錄:")
    for dir_path in sorted(all_dirs):
        logger.info(f"  {dir_path}")
    
    # 執行患者ID拆分，查看分割結果
    train_indices, val_indices, test_indices = dataset.split_by_patient(
        train_ratio=config['data']['splits']['train_ratio'],
        val_ratio=config['data']['splits']['val_ratio'],
        test_ratio=config['data']['splits']['test_ratio'],
        seed=config['data']['splits']['split_seed']
    )
    
    # 顯示拆分結果中每個集合的患者ID
    train_patient_ids = set(dataset.samples[idx]['patient_id'] for idx in train_indices)
    val_patient_ids = set(dataset.samples[idx]['patient_id'] for idx in val_indices)
    test_patient_ids = set(dataset.samples[idx]['patient_id'] for idx in test_indices)
    
    logger.info("=" * 80)
    logger.info("數據集拆分結果:")
    logger.info(f"訓練集: {len(train_indices)} 樣本，{len(train_patient_ids)} 位患者")
    logger.info(f"  患者IDs: {sorted(train_patient_ids)}")
    logger.info(f"驗證集: {len(val_indices)} 樣本，{len(val_patient_ids)} 位患者")
    logger.info(f"  患者IDs: {sorted(val_patient_ids)}")
    logger.info(f"測試集: {len(test_indices)} 樣本，{len(test_patient_ids)} 位患者")
    logger.info(f"  患者IDs: {sorted(test_patient_ids)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="測試數據過濾邏輯並顯示過濾後的樣本詳細信息")
    parser.add_argument("--config", required=True, help="配置文件路徑")
    args = parser.parse_args()
    
    run_filter_test(args.config) 