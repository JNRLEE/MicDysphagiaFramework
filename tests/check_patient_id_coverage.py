#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
檢查數據目錄中的所有患者ID是否都有在Excel中定義，並分析各動作類型的樣本分佈
"""

import os
import sys
import logging
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import torch

# 添加項目根目錄到路徑以便導入模塊
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_loader import load_config
from utils.custom_classification_loader import CustomClassificationLoader, SELECTION_TYPES

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

def check_patient_id_coverage(config_path):
    """
    檢查數據目錄中的所有患者ID是否都有在Excel中定義
    
    Args:
        config_path (str): 配置文件路徑
        
    Returns:
        None
    """
    # 載入配置
    logger.info(f"載入配置文件: {config_path}")
    config = load_config(config_path)
    
    # 獲取音頻目錄和Excel路徑
    wav_dir = config['data']['source']['wav_dir']
    excel_path = config['data']['filtering']['custom_classification']['excel_path']
    patient_id_column = config['data']['filtering']['custom_classification']['patient_id_column']
    class_column = config['data']['filtering']['custom_classification']['class_column']
    
    logger.info(f"音頻目錄: {wav_dir}")
    logger.info(f"Excel路徑: {excel_path}")
    
    # 讀取Excel數據
    logger.info("正在讀取Excel數據...")
    try:
        df = pd.read_excel(excel_path)
        
        # 如果使用Excel列名(A, B, C...)而非直接欄位名稱
        if _is_excel_column(patient_id_column):
            patient_id_col_idx = _excel_column_to_index(patient_id_column)
            patient_id_col = df.columns[patient_id_col_idx]
        else:
            patient_id_col = patient_id_column
        
        if _is_excel_column(class_column):
            class_col_idx = _excel_column_to_index(class_column)
            class_col = df.columns[class_col_idx]
        else:
            class_col = class_column
        
        # 獲取所有Excel中定義的患者ID和分類
        excel_patient_ids = set()
        patient_id_to_class = {}
        
        for _, row in df.iterrows():
            patient_id = str(row[patient_id_col]).strip()
            class_value = str(row[class_col]).strip()
            
            # 跳過空值
            if not patient_id or pd.isna(patient_id):
                continue
            
            excel_patient_ids.add(patient_id)
            
            if not pd.isna(class_value) and class_value != 'nan':
                patient_id_to_class[patient_id] = class_value
        
        logger.info(f"Excel中定義的患者ID總數: {len(excel_patient_ids)}")
        logger.info(f"Excel中有分類的患者ID總數: {len(patient_id_to_class)}")
        
    except Exception as e:
        logger.error(f"讀取Excel文件時發生錯誤: {str(e)}")
        excel_patient_ids = set()
        patient_id_to_class = {}
    
    # 載入患者動作類型
    classifier = CustomClassificationLoader(config)
    classifier._load_data_action_types()
    
    # 獲取數據目錄中的所有患者ID
    data_dir = Path(wav_dir)
    patient_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    data_patient_ids = set([d.name for d in patient_dirs])
    
    logger.info(f"數據目錄中的患者目錄總數: {len(patient_dirs)}")
    
    # 統計每個目錄的音頻文件數量
    patient_file_counts = {}
    patient_action_types = defaultdict(Counter)
    total_files = 0
    
    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        audio_files = list(patient_dir.glob("*.wav"))
        patient_file_counts[patient_id] = len(audio_files)
        total_files += len(audio_files)
        
        # 統計每個患者的動作類型分布
        action_type = classifier.get_patient_action_type(patient_id)
        if action_type:
            patient_action_types[patient_id][action_type] += len(audio_files)
    
    logger.info(f"數據目錄中的音頻文件總數: {total_files}")
    
    # 檢查患者ID覆蓋率
    patients_only_in_data = data_patient_ids - excel_patient_ids
    patients_only_in_excel = excel_patient_ids - data_patient_ids
    patients_in_both = data_patient_ids.intersection(excel_patient_ids)
    
    logger.info("=" * 80)
    logger.info("患者ID覆蓋分析:")
    logger.info(f"數據目錄中的患者總數: {len(data_patient_ids)}")
    logger.info(f"Excel中定義的患者總數: {len(excel_patient_ids)}")
    logger.info(f"患者ID重疊數量: {len(patients_in_both)}")
    logger.info(f"僅在數據目錄中存在的患者數量: {len(patients_only_in_data)}")
    if patients_only_in_data:
        logger.info(f"僅在數據目錄中存在的患者ID(前10個): {sorted(list(patients_only_in_data))[:10]}")
    logger.info(f"僅在Excel中存在的患者數量: {len(patients_only_in_excel)}")
    if patients_only_in_excel:
        logger.info(f"僅在Excel中存在的患者ID(前10個): {sorted(list(patients_only_in_excel))[:10]}")
    
    # 分析Excel中有分類的患者覆蓋率
    patients_with_class = set(patient_id_to_class.keys())
    patients_with_class_in_data = patients_with_class.intersection(data_patient_ids)
    
    logger.info("=" * 80)
    logger.info("患者分類覆蓋分析:")
    logger.info(f"Excel中有分類的患者總數: {len(patients_with_class)}")
    logger.info(f"Excel中有分類且在數據目錄中存在的患者數量: {len(patients_with_class_in_data)}")
    logger.info(f"缺失率: {(len(patients_with_class) - len(patients_with_class_in_data)) / len(patients_with_class):.2%}")
    
    # 分析每種動作類型的數量
    action_type_counts = Counter()
    for patient_id, counts in patient_action_types.items():
        for action_type, count in counts.items():
            action_type_counts[action_type] += count
    
    logger.info("=" * 80)
    logger.info("各動作類型的樣本分布:")
    for action, count in sorted(action_type_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {action}: {count} 個樣本")
    
    # 分析每個患者的所有動作類型
    patients_with_multiple_actions = 0
    for patient_id, counts in patient_action_types.items():
        if len(counts) > 1:
            patients_with_multiple_actions += 1
    
    logger.info(f"擁有多種動作類型的患者數量: {patients_with_multiple_actions}")
    
    # 測試按照分類和動作類型過濾的結果
    class_config = config['data']['filtering']['custom_classification'].get('class_config', {})
    filtered_actions = set([action for action, value in class_config.items() if value == 1])
    
    logger.info("=" * 80)
    logger.info("數據過濾模擬:")
    logger.info(f"保留的動作類型: {filtered_actions}")
    
    # 統計每個分類類別中，符合動作類型過濾條件的患者和樣本數
    class_counts = Counter()
    class_patient_counts = defaultdict(set)
    
    for patient_id, action_counts in patient_action_types.items():
        # 檢查是否有符合過濾條件的動作類型
        has_filtered_action = any(action in filtered_actions for action in action_counts.keys())
        
        # 獲取患者的分類類別
        class_name = patient_id_to_class.get(patient_id)
        
        if class_name and has_filtered_action:
            # 計算符合動作類型的樣本數
            action_count = sum(count for action, count in action_counts.items() if action in filtered_actions)
            class_counts[class_name] += action_count
            class_patient_counts[class_name].add(patient_id)
    
    logger.info("按分類和動作類型過濾後的結果:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        num_patients = len(class_patient_counts[class_name])
        logger.info(f"  {class_name}: {count} 個樣本, {num_patients} 個患者")
        
    # 詳細分析所有經過過濾後的患者
    filtered_patient_details = []
    
    for patient_id, action_counts in patient_action_types.items():
        # 檢查是否有符合過濾條件的動作類型
        has_filtered_action = any(action in filtered_actions for action in action_counts.keys())
        
        # 獲取患者的分類類別
        class_name = patient_id_to_class.get(patient_id)
        
        if class_name and has_filtered_action:
            # 計算符合動作類型的樣本數
            action_count = sum(count for action, count in action_counts.items() if action in filtered_actions)
            total_count = sum(action_counts.values())
            
            filtered_patient_details.append({
                'patient_id': patient_id,
                'class': class_name,
                'filtered_action_count': action_count,
                'total_action_count': total_count,
                'action_types': dict(action_counts)
            })
    
    logger.info("=" * 80)
    logger.info(f"符合過濾條件的患者詳細信息(共 {len(filtered_patient_details)} 個):")
    
    for i, patient in enumerate(sorted(filtered_patient_details, key=lambda x: x['patient_id']), 1):
        logger.info(f"[{i}] 患者ID: {patient['patient_id']}")
        logger.info(f"    分類: {patient['class']}")
        logger.info(f"    符合過濾條件的樣本數: {patient['filtered_action_count']}")
        logger.info(f"    總樣本數: {patient['total_action_count']}")
        logger.info(f"    動作類型分布: {patient['action_types']}")

def _is_excel_column(col):
    """檢查是否為Excel列名格式 (A, B, C...)"""
    return col.upper() in [chr(65 + i) for i in range(26)] + [f"A{chr(65 + i)}" for i in range(26)]

def _excel_column_to_index(col):
    """將Excel列名轉換為索引"""
    col = col.upper()
    if len(col) == 1:
        return ord(col) - ord('A')
    else:
        return 26 + (ord(col[1]) - ord('A'))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="檢查患者ID覆蓋率和動作類型分佈")
    parser.add_argument("--config", required=True, help="配置文件路徑")
    args = parser.parse_args()
    
    check_patient_id_coverage(args.config) 