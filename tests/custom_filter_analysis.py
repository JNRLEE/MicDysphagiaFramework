#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分析數據過濾邏輯並測試不同的過濾條件
此腳本允許用戶修改過濾條件，測試不同的分類配置，以便找出合適的過濾設定
"""

import os
import sys
import logging
import json
import copy
from pathlib import Path
from collections import defaultdict, Counter
import torch
from pprint import pprint

# 添加項目根目錄到路徑以便導入模塊
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_loader import load_config
from data.audio_dataset import AudioDataset
from utils.custom_classification_loader import CustomClassificationLoader, SELECTION_TYPES

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


def count_samples_by_action_type(dataset, config):
    """
    按動作類型計算樣本數量
    
    Args:
        dataset (AudioDataset): 數據集實例
        config (dict): 配置字典
        
    Returns:
        dict: 每種動作類型的樣本計數
    """
    # 載入患者動作類型
    data_dir = Path(config['data']['source']['wav_dir'])
    classifier = CustomClassificationLoader(config)
    classifier._load_data_action_types()
    
    # 計算每種動作類型的樣本數
    action_type_counts = Counter()
    action_samples = defaultdict(list)
    
    for sample in dataset.samples:
        patient_id = sample['patient_id']
        action_type = classifier.get_patient_action_type(patient_id)
        if action_type:
            action_type_counts[action_type] += 1
            action_samples[action_type].append(patient_id)
    
    # 計算每種動作類型的患者數
    action_patients = {action: len(set(patients)) for action, patients in action_samples.items()}
    
    return action_type_counts, action_patients


def analyze_patient_distribution(config_path, modify_filters=False):
    """
    分析患者分布並測試不同的過濾設定
    
    Args:
        config_path (str): 配置文件路徑
        modify_filters (bool): 是否允許修改過濾設定
        
    Returns:
        None
    """
    # 載入配置
    logger.info(f"載入配置文件: {config_path}")
    config = load_config(config_path)
    
    # 獲取音頻目錄
    wav_dir = config['data']['source']['wav_dir']
    logger.info(f"音頻目錄: {wav_dir}")
    
    # 存儲原始配置
    original_config = copy.deepcopy(config)
    
    # 如果需要修改過濾設定
    if modify_filters:
        print("\n當前的動作類型過濾設定:")
        class_config = config['data']['filtering']['custom_classification'].get('class_config', {})
        for action, value in class_config.items():
            print(f"  {action}: {'啟用' if value == 1 else '禁用'}")
        
        print("\n您想要修改過濾設定嗎? (y/n)")
        choice = input().strip().lower()
        
        if choice == 'y':
            # 顯示可用的動作類型
            print("\n可用的動作類型:")
            for action in SELECTION_TYPES.keys():
                print(f"  {action}: {SELECTION_TYPES[action]}")
            
            # 修改過濾設定
            new_class_config = {}
            for action in SELECTION_TYPES.keys():
                current_value = class_config.get(action, 0)
                print(f"\n啟用 {action} 嗎? (目前: {'啟用' if current_value == 1 else '禁用'}) (y/n)")
                choice = input().strip().lower()
                new_class_config[action] = 1 if choice == 'y' else 0
            
            # 更新配置
            config['data']['filtering']['custom_classification']['class_config'] = new_class_config
            logger.info("已更新動作類型過濾設定")
    
    # 創建數據集實例
    logger.info("使用當前配置載入數據集...")
    dataset = AudioDataset(root_dir=wav_dir, config=config)
    
    # 顯示過濾結果
    logger.info(f"過濾後的樣本總數: {len(dataset.samples)}")
    
    # 按患者ID分組
    samples_by_patient = defaultdict(list)
    for sample in dataset.samples:
        patient_id = sample['patient_id']
        samples_by_patient[patient_id].append(sample)
    
    # 顯示每個患者的樣本數量
    logger.info(f"過濾後的患者總數: {len(samples_by_patient)}")
    
    # 統計每個患者的樣本數量分布
    sample_counts = [len(samples) for samples in samples_by_patient.values()]
    sample_count_distribution = Counter(sample_counts)
    
    logger.info("=" * 80)
    logger.info("患者樣本數量分布:")
    for count, num_patients in sorted(sample_count_distribution.items()):
        logger.info(f"  {count} 個樣本: {num_patients} 個患者")
    
    # 計算最大、最小和平均樣本數
    if sample_counts:
        logger.info(f"  最小樣本數: {min(sample_counts)}")
        logger.info(f"  最大樣本數: {max(sample_counts)}")
        logger.info(f"  平均樣本數: {sum(sample_counts) / len(sample_counts):.2f}")
    
    # 按動作類型統計樣本
    action_counts, action_patients = count_samples_by_action_type(dataset, config)
    
    logger.info("=" * 80)
    logger.info("各動作類型的樣本分布:")
    for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        num_patients = action_patients.get(action, 0)
        logger.info(f"  {action}: {count} 個樣本, {num_patients} 個患者")
    
    # 分析自定義分類分布
    class_distribution = Counter()
    class_patients = defaultdict(set)
    
    for patient_id, samples in samples_by_patient.items():
        if dataset.custom_classifier and dataset.custom_classifier.enabled:
            class_name = dataset.custom_classifier.get_class(patient_id)
            if class_name:
                class_distribution[class_name] += len(samples)
                class_patients[class_name].add(patient_id)
    
    logger.info("=" * 80)
    logger.info("自定義分類分布:")
    for class_name, count in sorted(class_distribution.items(), key=lambda x: x[1], reverse=True):
        num_patients = len(class_patients[class_name])
        logger.info(f"  {class_name}: {count} 個樣本, {num_patients} 個患者")
    
    # 執行患者ID拆分，查看分割結果
    train_indices, val_indices, test_indices = dataset.split_by_patient(
        train_ratio=config['data']['splits']['train_ratio'],
        val_ratio=config['data']['splits']['val_ratio'],
        test_ratio=config['data']['splits']['test_ratio'],
        seed=config['data']['splits']['split_seed']
    )
    
    # 分析每個集合中的自定義分類分布
    train_classes = Counter()
    val_classes = Counter()
    test_classes = Counter()
    
    for idx in train_indices:
        patient_id = dataset.samples[idx]['patient_id']
        if dataset.custom_classifier and dataset.custom_classifier.enabled:
            class_name = dataset.custom_classifier.get_class(patient_id)
            if class_name:
                train_classes[class_name] += 1
    
    for idx in val_indices:
        patient_id = dataset.samples[idx]['patient_id']
        if dataset.custom_classifier and dataset.custom_classifier.enabled:
            class_name = dataset.custom_classifier.get_class(patient_id)
            if class_name:
                val_classes[class_name] += 1
    
    for idx in test_indices:
        patient_id = dataset.samples[idx]['patient_id']
        if dataset.custom_classifier and dataset.custom_classifier.enabled:
            class_name = dataset.custom_classifier.get_class(patient_id)
            if class_name:
                test_classes[class_name] += 1
    
    logger.info("=" * 80)
    logger.info("數據集拆分結果:")
    logger.info(f"訓練集: {len(train_indices)} 樣本")
    logger.info("  分類分布:")
    for class_name, count in sorted(train_classes.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"    {class_name}: {count} 個樣本")
    
    logger.info(f"驗證集: {len(val_indices)} 樣本")
    logger.info("  分類分布:")
    for class_name, count in sorted(val_classes.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"    {class_name}: {count} 個樣本")
    
    logger.info(f"測試集: {len(test_indices)} 樣本")
    logger.info("  分類分布:")
    for class_name, count in sorted(test_classes.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"    {class_name}: {count} 個樣本")
    
    # 如果修改了配置，詢問是否要保存
    if modify_filters and config != original_config:
        print("\n您想要保存修改後的配置嗎? (y/n)")
        choice = input().strip().lower()
        
        if choice == 'y':
            output_path = Path(config_path).with_suffix('.modified.yaml')
            with open(output_path, 'w', encoding='utf-8') as f:
                import yaml
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"已保存修改後的配置到: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="分析數據過濾邏輯並測試不同的過濾條件")
    parser.add_argument("--config", required=True, help="配置文件路徑")
    parser.add_argument("--modify", action="store_true", help="允許修改過濾設定")
    args = parser.parse_args()
    
    analyze_patient_distribution(args.config, args.modify) 