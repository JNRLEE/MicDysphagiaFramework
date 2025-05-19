#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
此腳本用於更新data_index.csv中的檔案路徑，確保file_path、wav_path、features_path和codes_path指向正確的相對路徑。
腳本會根據session_id找到對應的資料夾，並更新相關的路徑欄位。
"""

import os
import pandas as pd
import glob
from tqdm import tqdm
import logging
from datetime import datetime

# 設定日誌
log_file = f"path_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def find_session_folder(session_id, base_dir):
    """
    根據session_id找到對應的資料夾路徑
    
    Args:
        session_id: 資料集中的會話ID
        base_dir: 資料的基本路徑
        
    Returns:
        str: 找到的資料夾相對路徑，如果沒找到則返回None
    """
    # 嘗試直接查詢完全匹配的資料夾
    direct_match = glob.glob(f"{base_dir}/*{session_id}*")
    if direct_match:
        return direct_match[0]
    
    # 如果沒有直接匹配，嘗試查詢所有資料夾並檢查session_id是否是名稱的一部分
    all_folders = glob.glob(f"{base_dir}/*")
    for folder in all_folders:
        folder_name = os.path.basename(folder)
        if session_id in folder_name:
            return folder
    
    return None

def find_wav_file(folder_path):
    """
    在資料夾中找到WAV檔案
    
    Args:
        folder_path: 資料夾路徑
        
    Returns:
        str: WAV檔案的相對路徑，如果沒找到則返回None
    """
    wav_files = glob.glob(f"{folder_path}/*.wav")
    if wav_files:
        return wav_files[0]
    return None

def find_features_file(folder_path, session_id):
    """
    在資料夾中找到features檔案
    
    Args:
        folder_path: 資料夾路徑
        session_id: 會話ID，用於匹配檔案名
        
    Returns:
        str: features檔案的相對路徑，如果沒找到則返回None
    """
    # 取得資料夾名稱
    folder_name = os.path.basename(folder_path)
    
    # 先嘗試使用資料夾名稱查詢
    features_files = glob.glob(f"{folder_path}/*{folder_name}*_features.npy")
    if features_files:
        return features_files[0]
    
    # 嘗試使用session_id查詢
    features_files = glob.glob(f"{folder_path}/*{session_id}*_features.npy")
    if features_files:
        return features_files[0]
    
    # 如果沒有找到，查詢任何包含_features.npy的檔案
    features_files = glob.glob(f"{folder_path}/*_features.npy")
    if features_files:
        return features_files[0]
    
    return None

def find_codes_file(folder_path, session_id):
    """
    在資料夾中找到codes檔案
    
    Args:
        folder_path: 資料夾路徑
        session_id: 會話ID，用於匹配檔案名
        
    Returns:
        str: codes檔案的相對路徑，如果沒找到則返回None
    """
    # 取得資料夾名稱
    folder_name = os.path.basename(folder_path)
    
    # 先嘗試使用資料夾名稱查詢
    codes_files = glob.glob(f"{folder_path}/*{folder_name}*_codes.npy")
    if codes_files:
        return codes_files[0]
    
    # 嘗試使用session_id查詢
    codes_files = glob.glob(f"{folder_path}/*{session_id}*_codes.npy")
    if codes_files:
        return codes_files[0]
    
    # 如果沒有找到，查詢任何包含_codes.npy的檔案
    codes_files = glob.glob(f"{folder_path}/*_codes.npy")
    if codes_files:
        return codes_files[0]
    
    return None

def update_csv_paths():
    """
    更新CSV檔案中的路徑
    """
    # 檔案路徑
    csv_path = "data/metadata/data_index.csv"
    base_dir = "data/Processed(Cut)"
    
    # 檢查檔案是否存在
    if not os.path.exists(csv_path):
        logger.error(f"檔案不存在: {csv_path}")
        return
    
    # 讀取CSV檔案
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"成功讀取CSV檔案，共 {len(df)} 筆資料")
    except Exception as e:
        logger.error(f"讀取CSV檔案時發生錯誤: {e}")
        return
    
    # 建立一個集合來檢查重複的session_id
    processed_sessions = set()
    
    # 計算更新的項目數
    updates_count = 0
    not_found_count = 0
    
    # 處理每一行
    for index, row in tqdm(df.iterrows(), total=len(df), desc="處理資料"):
        session_id = row.get('session_id')
        if not session_id or pd.isna(session_id):
            logger.warning(f"第 {index} 行沒有session_id")
            continue
        
        # 檢查是否已處理過此session_id
        if session_id in processed_sessions:
            continue
        
        processed_sessions.add(session_id)
        
        # 找到對應的資料夾
        folder_path = find_session_folder(session_id, base_dir)
        if not folder_path:
            logger.warning(f"找不到session_id為 {session_id} 的資料夾")
            not_found_count += 1
            continue
        
        # 更新file_path
        df.loc[df['session_id'] == session_id, 'file_path'] = folder_path
        
        # 找到並更新wav_path
        wav_file = find_wav_file(folder_path)
        if wav_file:
            df.loc[df['session_id'] == session_id, 'wav_path'] = wav_file
        else:
            logger.warning(f"在資料夾 {folder_path} 中找不到WAV檔案")
        
        # 找到並更新features_path
        features_file = find_features_file(folder_path, session_id)
        if features_file:
            df.loc[df['session_id'] == session_id, 'features_path'] = features_file
        else:
            logger.warning(f"在資料夾 {folder_path} 中找不到features檔案")
        
        # 找到並更新codes_path
        codes_file = find_codes_file(folder_path, session_id)
        if codes_file:
            df.loc[df['session_id'] == session_id, 'codes_path'] = codes_file
        else:
            logger.warning(f"在資料夾 {folder_path} 中找不到codes檔案")
        
        updates_count += 1
    
    # 儲存更新後的CSV檔案
    backup_path = f"{csv_path}.backup"
    df.to_csv(backup_path, index=False)
    logger.info(f"已將原始檔案備份至 {backup_path}")
    
    df.to_csv(csv_path, index=False)
    logger.info(f"已更新CSV檔案")
    logger.info(f"總共處理 {len(processed_sessions)} 個唯一session_id")
    logger.info(f"成功更新 {updates_count} 個session_id的路徑資訊")
    logger.info(f"無法找到對應資料夾的session_id數量: {not_found_count}")

if __name__ == "__main__":
    logger.info("開始更新檔案路徑...")
    update_csv_paths()
    logger.info("更新完成！") 