"""
配置測試模組：測試配置檔案的加載和驗證
功能：
1. 測試配置檔案能否正確加載
2. 驗證配置項是否完整
3. 檢查路徑是否存在
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

# 添加項目根目錄到路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 導入配置加載器
from utils.config_loader import load_config

def parse_args():
    """解析命令行參數
    
    Returns:
        argparse.Namespace: 解析後的參數
        
    Description:
        解析命令行參數
        
    References:
        https://docs.python.org/3/library/argparse.html
    """
    parser = argparse.ArgumentParser(description='測試配置檔案加載')
    parser.add_argument('--config', type=str, required=True, help='配置文件路徑')
    return parser.parse_args()

def check_paths(config):
    """檢查配置中的路徑是否存在
    
    Args:
        config: 配置字典
        
    Returns:
        bool: 所有路徑是否存在
        
    Description:
        檢查配置中指定的路徑是否存在
        
    References:
        None
    """
    all_paths_exist = True
    
    # 檢查數據源路徑
    data_config = config.get('data', {})
    source_config = data_config.get('source', {})
    
    if 'wav_dir' in source_config:
        wav_dir = source_config['wav_dir']
        if not os.path.exists(wav_dir):
            print(f"警告: 找不到音頻目錄 {wav_dir}")
            all_paths_exist = False
    
    if 'spectrogram_dir' in source_config:
        spectrogram_dir = source_config['spectrogram_dir']
        if not os.path.exists(spectrogram_dir):
            print(f"警告: 找不到頻譜圖目錄 {spectrogram_dir}")
            # 嘗試創建目錄
            os.makedirs(spectrogram_dir, exist_ok=True)
            print(f"已創建頻譜圖目錄 {spectrogram_dir}")
    
    if 'feature_dir' in source_config:
        feature_dir = source_config['feature_dir']
        if not os.path.exists(feature_dir):
            print(f"警告: 找不到特徵目錄 {feature_dir}")
            all_paths_exist = False
    
    # 檢查輸出目錄
    output_dir = config.get('global', {}).get('output_dir')
    if output_dir and not os.path.exists(output_dir):
        print(f"警告: 找不到輸出目錄 {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"已創建輸出目錄 {output_dir}")
    
    return all_paths_exist

def main():
    """主函數
    
    Description:
        測試配置檔案加載的主函數
        
    References:
        None
    """
    args = parse_args()
    
    print(f"測試配置檔案: {args.config}")
    
    try:
        # 加載配置
        config_loader = load_config(args.config)
        config = config_loader.config
        
        print("配置加載成功!")
        
        # 檢查基本配置項
        if 'global' not in config:
            print("警告: 配置中缺少 'global' 部分")
        if 'data' not in config:
            print("錯誤: 配置中缺少 'data' 部分")
            return False
        if 'model' not in config:
            print("錯誤: 配置中缺少 'model' 部分")
            return False
        if 'training' not in config:
            print("錯誤: 配置中缺少 'training' 部分")
            return False
        
        # 檢查路徑
        paths_exist = check_paths(config)
        
        # 打印主要配置
        print("\n配置摘要:")
        print(f"實驗名稱: {config.get('global', {}).get('experiment_name', 'unknown')}")
        print(f"數據類型: {config.get('data', {}).get('type', 'unknown')}")
        print(f"模型類型: {config.get('model', {}).get('type', 'unknown')}")
        loss_config = config.get('training', {}).get('loss', {})
        if 'combined' in loss_config:
            loss_types = [details.get('type') for name, details in loss_config['combined'].items()]
            print(f"損失函數: 組合 - {', '.join(loss_types)}")
        else:
            print(f"損失函數: {loss_config.get('type', 'unknown')}")
        
        print(f"\n配置測試結果: {'成功' if paths_exist else '部分路徑不存在，但已創建必要目錄'}")
        return paths_exist
        
    except Exception as e:
        print(f"配置加載失敗: {str(e)}")
        return False

if __name__ == "__main__":
    main() 