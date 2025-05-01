# -*- coding: utf-8 -*-
"""
此程式用於分析原始音訊檔與重建音訊檔之間的差異，並計算相關指標
計算項目包含：MSE、MAE、RMSE、SI-SDR等loss function
"""

import os
import json
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import glob

def load_audio(file_path: str) -> Tuple[np.ndarray, float]:
    """
    載入音訊檔案並返回波形數據和採樣率
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None, None

def si_sdr(reference: np.ndarray, estimation: np.ndarray) -> float:
    """
    計算Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
    
    參數:
    - reference: 原始音訊信號（ground truth）
    - estimation: 重建音訊信號
    
    返回:
    - si_sdr: SI-SDR值（越高越好）
    """
    # 確保信號長度相同
    min_len = min(len(reference), len(estimation))
    reference = reference[:min_len]
    estimation = estimation[:min_len]
    
    # 移除直流分量（零均值化）
    reference = reference - np.mean(reference)
    estimation = estimation - np.mean(estimation)
    
    # 計算最佳縮放因子
    alpha = np.dot(estimation, reference) / np.dot(reference, reference)
    
    # 計算縮放後的參考信號
    scaled_reference = alpha * reference
    
    # 計算噪聲
    noise = estimation - scaled_reference
    
    # 計算SI-SDR
    si_sdr_value = 10 * np.log10(
        np.sum(scaled_reference ** 2) / (np.sum(noise ** 2) + 1e-10)
    )
    
    return float(si_sdr_value)

def calculate_loss_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    """
    計算各種loss metrics
    """
    # 確保兩個音訊長度相同
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]
    
    # 計算各種loss
    mse = np.mean((original - reconstructed) ** 2)
    mae = np.mean(np.abs(original - reconstructed))
    rmse = np.sqrt(mse)
    
    # 計算相關係數
    correlation = np.corrcoef(original, reconstructed)[0, 1]
    
    # 計算SI-SDR
    si_sdr_value = si_sdr(original, reconstructed)
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'Correlation': correlation,
        'SI_SDR': si_sdr_value
    }

def load_info_json(folder_path: str) -> Optional[Dict]:
    """
    載入info.json檔案
    尋找非WavTokenizer的info.json檔案
    """
    try:
        # 使用glob找到所有info.json檔案
        info_files = glob.glob(os.path.join(folder_path, "*info.json"))
        
        # 過濾掉WavTokenizer相關的檔案
        info_files = [f for f in info_files if "WavTokenizer" not in f]
        
        if not info_files:
            print(f"No valid info.json found in {folder_path}")
            return None
            
        # 使用找到的第一個符合條件的檔案
        with open(info_files[0], 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading info.json from {folder_path}: {str(e)}")
        return None

def analyze_folder(folder_path: str) -> Optional[Dict]:
    """
    分析單一資料夾中的音訊檔案
    """
    # 檢查必要檔案是否存在
    original_path = os.path.join(folder_path, "Probe0_RX_IN_TDM4CH0.wav")
    reconstructed_path = os.path.join(folder_path, "Probe0_RX_IN_TDM4CH0_reconstructed.wav")
    
    if not (os.path.exists(original_path) and os.path.exists(reconstructed_path)):
        return None
    
    # 載入音訊檔案
    original, sr1 = load_audio(original_path)
    reconstructed, sr2 = load_audio(reconstructed_path)
    
    if original is None or reconstructed is None:
        return None
    
    # 載入info.json
    info_data = load_info_json(folder_path)
    if info_data is None:
        return None
    
    # 計算loss metrics
    loss_metrics = calculate_loss_metrics(original, reconstructed)
    
    # 組合結果
    folder_name = os.path.basename(folder_path)
    result = {
        'folder_name': folder_name,
        'patient_id': info_data.get('PatientID', ''),
        'selection': info_data.get('selection', ''),
        'score': info_data.get('score', ''),
        **loss_metrics
    }
    
    return result

def main():
    """
    主程式：分析所有資料夾並產生CSV報告
    """
    # 設定基礎路徑
    base_path = os.path.dirname(os.path.abspath(__file__))
    results = []
    
    # 取得所有資料夾
    folders = [f for f in os.listdir(base_path) 
              if os.path.isdir(os.path.join(base_path, f)) 
              and (f.startswith('N') or f.startswith('P'))]
    
    # 分析每個資料夾
    for folder in tqdm(folders, desc="分析資料夾"):
        folder_path = os.path.join(base_path, folder)
        result = analyze_folder(folder_path)
        if result:
            results.append(result)
    
    # 建立DataFrame並儲存為CSV
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(base_path, 'wav_analysis_results.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"分析完成，結果已儲存至：{csv_path}")
        
        # 顯示統計摘要
        print("\n統計摘要：")
        print(df.describe())
        
        # 特別顯示SI-SDR的統計
        print("\nSI-SDR統計（越高越好）：")
        print(df.groupby('selection')['SI_SDR'].agg(['mean', 'std', 'min', 'max']))
    else:
        print("沒有找到可分析的資料")

if __name__ == "__main__":
    main() 