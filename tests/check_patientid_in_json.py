#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
正確檢查info.json中的PatientID與Excel對應關係
此程式會從每個資料目錄的info.json文件中讀取正確的PatientID，而非使用目錄名稱
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

def read_patient_id_from_info_json(info_file_path):
    """
    從info.json讀取PatientID
    
    Args:
        info_file_path (Path): info.json文件路徑
        
    Returns:
        str: PatientID，若無法讀取則返回None
    """
    try:
        with open(info_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('patientID')
    except Exception as e:
        logger.debug(f"讀取 {info_file_path} 出錯: {str(e)}")
        return None

def check_patient_id_json(config_path):
    """
    從info.json讀取PatientID並分析過濾邏輯
    
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
    
    # 獲取動作類型過濾設定
    class_config = config['data']['filtering']['custom_classification'].get('class_config', {})
    enabled_actions = [action for action, value in class_config.items() if value == 1]
    logger.info(f"已啟用的動作類型: {enabled_actions}")
    
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
    
    # 載入自定義分類處理器，用於獲取動作類型
    classifier = CustomClassificationLoader(config)
    classifier._load_data_action_types()
    
    # 從info.json中讀取PatientID和動作類型
    data_dir = Path(wav_dir)
    logger.info(f"掃描 {data_dir} 中的info.json文件...")
    
    # 用於保存每個樣本的PatientID和動作類型
    json_patient_ids = set()
    directory_to_patient_id = {}  # 目錄名到PatientID的映射
    patient_id_to_actions = defaultdict(set)  # PatientID到動作類型的映射
    patient_id_directory_counts = defaultdict(int)  # 每個PatientID的目錄數量
    patient_samples = defaultdict(list)  # 每個PatientID的樣本列表
    patient_action_type_counts = defaultdict(Counter)  # 每個PatientID的動作類型計數
    
    # 遍歷音頻目錄的所有子目錄
    sample_count = 0
    for patient_dir in data_dir.iterdir():
        if not patient_dir.is_dir():
            continue
        
        # 查找info.json文件
        info_files = [f for f in patient_dir.glob("*info.json") 
                    if not f.name.endswith("WavTokenizer_tokens_info.json")]
        
        if not info_files:
            logger.warning(f"在 {patient_dir} 找不到info.json文件")
            continue
        
        for info_file in info_files:
            try:
                # 讀取info.json內容
                with open(info_file, 'r', encoding='utf-8') as f:
                    info_data = json.load(f)
                
                # 提取PatientID和動作類型信息
                patient_id = info_data.get('patientID')
                selection = info_data.get('selection', '')
                
                if not patient_id:
                    logger.warning(f"{info_file} 中找不到patientID")
                    continue
                
                # 記錄PatientID
                json_patient_ids.add(patient_id)
                directory_to_patient_id[patient_dir.name] = patient_id
                patient_id_directory_counts[patient_id] += 1
                
                # 記錄樣本信息
                audio_files = list(patient_dir.glob("*.wav"))
                for audio_file in audio_files:
                    sample_count += 1
                    sample_info = {
                        'dir': patient_dir.name,
                        'file': audio_file.name,
                        'path': str(audio_file),
                        'patient_id': patient_id,
                        'selection': selection
                    }
                    patient_samples[patient_id].append(sample_info)
                
                # 確定動作類型
                action_type = None
                for std_type, selections in SELECTION_TYPES.items():
                    if any(s in selection for s in selections):
                        action_type = std_type
                        break
                
                if action_type:
                    patient_id_to_actions[patient_id].add(action_type)
                    patient_action_type_counts[patient_id][action_type] += len(audio_files)
                
            except Exception as e:
                logger.warning(f"處理 {info_file} 時出錯: {str(e)}")
    
    logger.info(f"從info.json中讀取到的PatientID總數: {len(json_patient_ids)}")
    logger.info(f"樣本總數: {sample_count}")
    
    # 檢查PatientID的匹配
    patient_ids_in_both = json_patient_ids.intersection(excel_patient_ids)
    patient_ids_only_in_json = json_patient_ids - excel_patient_ids
    patient_ids_only_in_excel = excel_patient_ids - json_patient_ids
    
    logger.info("=" * 80)
    logger.info("PatientID匹配分析:")
    logger.info(f"info.json中的PatientID總數: {len(json_patient_ids)}")
    logger.info(f"Excel中的PatientID總數: {len(excel_patient_ids)}")
    logger.info(f"PatientID重疊數量: {len(patient_ids_in_both)}")
    logger.info(f"僅在info.json中存在的PatientID數量: {len(patient_ids_only_in_json)}")
    if patient_ids_only_in_json:
        logger.info(f"僅在info.json中存在的PatientID(前10個): {sorted(list(patient_ids_only_in_json))[:10]}")
    logger.info(f"僅在Excel中存在的PatientID數量: {len(patient_ids_only_in_excel)}")
    if patient_ids_only_in_excel:
        logger.info(f"僅在Excel中存在的PatientID(前10個): {sorted(list(patient_ids_only_in_excel))[:10]}")
    
    # 分析每個PatientID的目錄數量
    multi_dir_patients = [pid for pid, count in patient_id_directory_counts.items() if count > 1]
    logger.info(f"對應多個目錄的PatientID數量: {len(multi_dir_patients)}")
    if multi_dir_patients:
        for pid in sorted(multi_dir_patients)[:5]:  # 只顯示前5個
            logger.info(f"  PatientID {pid} 對應 {patient_id_directory_counts[pid]} 個目錄")
    
    # 分析動作類型
    logger.info("=" * 80)
    logger.info("動作類型分析:")
    
    # 統計每種動作類型的總樣本數
    action_type_total_counts = Counter()
    for pid, counts in patient_action_type_counts.items():
        for action, count in counts.items():
            action_type_total_counts[action] += count
    
    for action, count in sorted(action_type_total_counts.items(), key=lambda x: x[1], reverse=True):
        pid_count = len([pid for pid, actions in patient_id_to_actions.items() if action in actions])
        logger.info(f"  {action}: {count} 個樣本, {pid_count} 個PatientID")
    
    # 多動作類型分析
    patients_with_multiple_actions = [pid for pid, actions in patient_id_to_actions.items() if len(actions) > 1]
    logger.info(f"擁有多種動作類型的PatientID數量: {len(patients_with_multiple_actions)}")
    
    # 模擬過濾邏輯：Excel中有分類且有啟用的動作類型
    patients_with_class = set(patient_id_to_class.keys())
    patients_with_enabled_actions = set(pid for pid, actions in patient_id_to_actions.items() 
                                     if any(action in enabled_actions for action in actions))
    
    # 應該保留的PatientID：在Excel中有分類且有啟用動作類型
    should_keep_patients = patients_with_class.intersection(patients_with_enabled_actions)
    
    logger.info("=" * 80)
    logger.info("過濾邏輯模擬:")
    logger.info(f"Excel中有分類的PatientID數量: {len(patients_with_class)}")
    logger.info(f"有啟用動作類型的PatientID數量: {len(patients_with_enabled_actions)}")
    logger.info(f"符合兩者條件的PatientID數量: {len(should_keep_patients)}")
    
    # 詳細分析每個符合條件的PatientID
    if should_keep_patients:
        class_distribution = Counter()
        sample_counts = []
        
        logger.info("符合條件的PatientID詳情:")
        for i, pid in enumerate(sorted(should_keep_patients), 1):
            class_name = patient_id_to_class.get(pid)
            actions = patient_id_to_actions.get(pid, set())
            enabled_acts = actions.intersection(set(enabled_actions))
            
            # 計算符合條件的樣本數
            enabled_sample_count = sum(patient_action_type_counts[pid][act] for act in enabled_acts)
            total_sample_count = sum(patient_action_type_counts[pid].values())
            
            logger.info(f"[{i}] PatientID: {pid}")
            logger.info(f"    分類: {class_name}")
            logger.info(f"    動作類型: {actions} (已啟用: {enabled_acts})")
            logger.info(f"    符合條件的樣本數: {enabled_sample_count}/{total_sample_count}")
            
            class_distribution[class_name] += enabled_sample_count
            sample_counts.append(enabled_sample_count)
        
        # 分類分布統計
        logger.info("=" * 80)
        logger.info("分類分布:")
        for class_name, count in sorted(class_distribution.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {class_name}: {count} 個樣本")
        
        # 樣本數量統計
        if sample_counts:
            logger.info("樣本數量統計:")
            logger.info(f"  最小樣本數: {min(sample_counts)}")
            logger.info(f"  最大樣本數: {max(sample_counts)}")
            logger.info(f"  平均樣本數: {sum(sample_counts) / len(sample_counts):.2f}")
    
    # 模擬數據集拆分
    train_ratio = config['data']['splits']['train_ratio']
    val_ratio = config['data']['splits']['val_ratio']
    test_ratio = config['data']['splits']['test_ratio']
    split_seed = config['data']['splits']['split_seed']
    
    if should_keep_patients:
        import random
        random.seed(split_seed)
        
        # 將患者ID打亂
        patient_ids_list = list(should_keep_patients)
        random.shuffle(patient_ids_list)
        
        # 計算每個集合的大小
        train_size = int(len(patient_ids_list) * train_ratio)
        val_size = int(len(patient_ids_list) * val_ratio)
        
        # 拆分患者ID
        train_patient_ids = patient_ids_list[:train_size]
        val_patient_ids = patient_ids_list[train_size:train_size + val_size]
        test_patient_ids = patient_ids_list[train_size + val_size:]
        
        # 計算每個集合的樣本數
        def count_enabled_samples(pid_list):
            count = 0
            for pid in pid_list:
                actions = patient_id_to_actions.get(pid, set())
                enabled_acts = actions.intersection(set(enabled_actions))
                count += sum(patient_action_type_counts[pid][act] for act in enabled_acts)
            return count
        
        train_samples = count_enabled_samples(train_patient_ids)
        val_samples = count_enabled_samples(val_patient_ids)
        test_samples = count_enabled_samples(test_patient_ids)
        
        logger.info("=" * 80)
        logger.info("數據集拆分模擬:")
        logger.info(f"訓練集: {train_samples} 樣本, {len(train_patient_ids)} 個PatientID")
        logger.info(f"驗證集: {val_samples} 樣本, {len(val_patient_ids)} 個PatientID")
        logger.info(f"測試集: {test_samples} 樣本, {len(test_patient_ids)} 個PatientID")

def _is_excel_column(col):
    """檢查是否為Excel列名格式 (A, B, C...)"""
    if not isinstance(col, str):
        return False
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
    
    parser = argparse.ArgumentParser(description="從info.json讀取PatientID並分析過濾邏輯")
    parser.add_argument("--config", required=True, help="配置文件路徑")
    args = parser.parse_args()
    
    check_patient_id_json(args.config) 