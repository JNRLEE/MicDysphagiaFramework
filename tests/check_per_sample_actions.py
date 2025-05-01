#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
檢查資料目錄中每個樣本的動作類型，驗證數據過濾邏輯
主要確認實際載入的樣本是否僅包含設定的動作類型(例如Jelly)
"""

import os
import sys
import logging
import json
from pathlib import Path
from collections import defaultdict, Counter
import torch

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

def check_sample_actions(config_path):
    """
    檢查每個樣本的動作類型，驗證過濾邏輯是否正確
    
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
    
    # 獲取設定的動作類型過濾設定
    class_config = config['data']['filtering']['custom_classification'].get('class_config', {})
    enabled_actions = [action for action, value in class_config.items() if value == 1]
    logger.info(f"已啟用的動作類型: {enabled_actions}")
    
    # 創建數據集實例
    logger.info("載入數據集...")
    dataset = AudioDataset(root_dir=wav_dir, config=config)
    
    # 顯示過濾結果
    logger.info(f"過濾後的樣本總數: {len(dataset.samples)}")
    
    # 單獨創建一個CustomClassificationLoader實例用於查詢動作類型
    classifier = CustomClassificationLoader(config)
    classifier._load_data_action_types()
    
    # 按患者ID分組，同時記錄每個樣本的動作類型
    samples_by_patient = defaultdict(list)
    sample_action_types = {}
    unique_actions = set()
    
    for idx, sample in enumerate(dataset.samples):
        patient_id = sample['patient_id']
        file_path = sample['audio_path']
        
        # 從文件名或info.json確定動作類型
        action_type = sample.get('selection_type')
        if not action_type:
            action_type = classifier.get_patient_action_type(patient_id)
        
        sample_action_types[idx] = action_type
        if action_type:
            unique_actions.add(action_type)
        
        samples_by_patient[patient_id].append({
            'idx': idx,
            'path': file_path,
            'action_type': action_type
        })
    
    # 統計每種動作類型的樣本數
    action_counts = Counter(sample_action_types.values())
    
    logger.info("=" * 80)
    logger.info("載入的樣本動作類型分布:")
    for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {action}: {count} 個樣本")
    
    # 檢查是否有非啟用動作類型的樣本
    unexpected_actions = unique_actions - set(enabled_actions)
    if unexpected_actions:
        logger.warning(f"警告：發現 {len(unexpected_actions)} 種非啟用的動作類型樣本：{unexpected_actions}")
        
        # 詳細列出這些非預期的樣本
        logger.warning("非預期動作類型的樣本詳情:")
        for action in unexpected_actions:
            samples_with_action = [idx for idx, act in sample_action_types.items() if act == action]
            logger.warning(f"  動作類型 '{action}' 的樣本數: {len(samples_with_action)}")
            if samples_with_action:
                logger.warning(f"  樣本示例:")
                for idx in samples_with_action[:3]:  # 只顯示前3個
                    logger.warning(f"    {dataset.samples[idx]['audio_path']}")
    else:
        logger.info("驗證通過：所有載入的樣本動作類型都在啟用列表中")
    
    # 檢查每個患者的動作類型分布
    logger.info("=" * 80)
    logger.info("患者動作類型分布:")
    
    patients_with_multiple_actions = 0
    for patient_id, samples in samples_by_patient.items():
        patient_actions = set(sample['action_type'] for sample in samples if sample['action_type'])
        if len(patient_actions) > 1:
            patients_with_multiple_actions += 1
            logger.info(f"患者 {patient_id} 有多種動作類型: {patient_actions}")
    
    if patients_with_multiple_actions:
        logger.warning(f"有 {patients_with_multiple_actions} 個患者包含多種動作類型的樣本")
    else:
        logger.info("每個患者只有一種動作類型的樣本")
    
    # 進行更深入的分析：是每個患者只保留一種動作類型，還是保留所有符合條件的動作類型
    # 首先獲取每個患者的所有動作類型
    all_patient_actions = {}
    for patient_dir in Path(wav_dir).iterdir():
        if not patient_dir.is_dir():
            continue
        
        patient_id = patient_dir.name
        actions = set()
        
        # 從info.json文件中獲取動作類型
        info_files = [f for f in patient_dir.glob("*info.json") 
                   if not f.name.endswith("WavTokenizer_tokens_info.json")]
        
        for info_file in info_files:
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                selection = data.get('selection', '')
                # 根據selection判斷動作類型
                for std_type, selections in SELECTION_TYPES.items():
                    if any(s in selection for s in selections):
                        actions.add(std_type)
                        break
            except Exception as e:
                logger.debug(f"讀取 {info_file} 出錯: {str(e)}")
        
        if actions:
            all_patient_actions[patient_id] = actions
    
    # 分析每個有多種動作類型的患者，檢查是否所有啟用的動作類型都被包含了
    logger.info("=" * 80)
    logger.info("多動作類型患者的詳細分析:")
    
    patients_with_filtered_actions = 0
    for patient_id, all_actions in all_patient_actions.items():
        if len(all_actions) > 1:  # 患者有多種動作類型
            # 檢查該患者是否出現在已過濾的數據集中
            if patient_id in samples_by_patient:
                # 獲取該患者在過濾後數據集中的動作類型
                filtered_actions = set(sample['action_type'] for sample in samples_by_patient[patient_id] if sample['action_type'])
                
                # 比較所有動作類型和過濾後的動作類型
                missing_actions = all_actions - filtered_actions
                if missing_actions:
                    # 只報告缺失的啟用動作類型
                    missing_enabled = set(action for action in missing_actions if action in enabled_actions)
                    if missing_enabled:
                        logger.info(f"患者 {patient_id} 缺少已啟用的動作類型: {missing_enabled}")
                        logger.info(f"  全部動作類型: {all_actions}")
                        logger.info(f"  過濾後保留的動作類型: {filtered_actions}")
                
                patients_with_filtered_actions += 1
    
    logger.info(f"有 {patients_with_filtered_actions} 個多動作類型患者出現在過濾後的數據集中")
    
    # 輸出總結
    logger.info("=" * 80)
    logger.info("總結:")
    if unexpected_actions:
        logger.warning("【過濾邏輯異常】發現非啟用動作類型的樣本，請檢查過濾實現")
    else:
        logger.info("【過濾邏輯正常】所有樣本都屬於啟用的動作類型")
    
    if patients_with_multiple_actions:
        logger.warning("【注意】部分患者擁有多種動作類型，資料集可能包含來自同一患者的不同動作數據")
    else:
        logger.info("【注意】每個患者僅有一種動作類型的資料")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="檢查每個樣本的動作類型，驗證數據過濾邏輯")
    parser.add_argument("--config", required=True, help="配置文件路徑")
    args = parser.parse_args()
    
    check_sample_actions(args.config) 