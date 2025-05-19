#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
測試 DataAdapter 對於元組批次的兼容性問題
"""

import sys
import os
import logging
from pathlib import Path

# 將項目根目錄添加到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

try:
    # 顯式相對導入
    sys.path.insert(0, str(project_root / "utils"))
    from data_adapter import DataAdapter, AdapterDataLoader
except ImportError as e:
    print(f"無法導入 DataAdapter: {e}")
    print(f"Python 路徑: {sys.path}")
    print("嘗試列出 utils 目錄中的文件:")
    utils_path = project_root / "utils"
    if utils_path.exists():
        for file in utils_path.iterdir():
            print(f"  {file.name}")
    sys.exit(1)

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# 創建一個簡單的數據集類，模擬索引數據集的行為
class SimpleDataset(Dataset):
    """簡單數據集，返回元組 (data, label)"""
    
    def __init__(self, size=100):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 創建一個隨機音頻和標籤
        data = torch.randn(1, 16000)  # 模擬1秒16kHz的音頻
        label = torch.tensor(idx % 2, dtype=torch.float32)  # 簡單的二分類標籤
        return data, label

def test_data_adapter():
    """測試 DataAdapter 對元組批次的處理"""
    # 創建簡單數據集
    dataset = SimpleDataset(size=10)
    
    # 創建數據加載器
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )
    
    # 配置
    config = {
        'data': {
            'preprocessing': {
                'audio': {
                    'sample_rate': 16000,
                    'duration': 1.0,
                    'normalize': True
                },
                'spectrogram': {
                    'n_fft': 1024,
                    'hop_length': 512,
                    'n_mels': 128
                }
            }
        },
        'model': {
            'parameters': {
                'input_dim': 1000
            }
        }
    }
    
    # 模型類型
    model_type = 'cnn'
    
    # 測試獲取一個批次
    print("原始數據加載器:")
    for batch_idx, batch in enumerate(dataloader):
        print(f"批次 {batch_idx + 1} 類型: {type(batch)}")
        print(f"批次 {batch_idx + 1} 長度: {len(batch)}")
        for i, item in enumerate(batch):
            print(f"  項目 {i + 1} 類型: {type(item)}")
            print(f"  項目 {i + 1} 形狀: {item.shape if hasattr(item, 'shape') else 'N/A'}")
        break
    
    # 測試 adapt_batch 方法
    print("\n測試 DataAdapter.adapt_batch 方法:")
    for batch_idx, batch in enumerate(dataloader):
        print(f"原始批次類型: {type(batch)}")
        
        # 將元組轉換為字典
        batch_dict = {'audio': batch[0], 'label': batch[1]}
        print(f"批次字典鍵: {batch_dict.keys()}")
        
        try:
            adapted_batch = DataAdapter.adapt_batch(batch_dict, model_type, config)
            print(f"適配後的批次鍵: {adapted_batch.keys()}")
        except Exception as e:
            print(f"適配批次時發生錯誤: {e}")
        break
    
    # 測試 AdapterDataLoader
    print("\n測試 AdapterDataLoader 封裝:")
    try:
        # 這個將會失敗，因為 AdapterDataLoader.__next__ 期望 batch 是字典
        adapted_loader = AdapterDataLoader(dataloader, model_type, config)
        for adapted_batch_idx, adapted_batch in enumerate(adapted_loader):
            print(f"適配後的批次類型: {type(adapted_batch)}")
            break
    except Exception as e:
        print(f"使用 AdapterDataLoader 時發生錯誤: {e}")
        print("錯誤原因：AdapterDataLoader.__next__ 期望批次是字典，但數據集返回的是元組")
    
    # 提出解決方案
    print("\n建議解決方案:")
    print("1. 修改 utils/data_adapter.py 中的 AdapterDataLoader.__next__ 方法，以支持元組批次")
    print("2. 修改所有數據集的 __getitem__ 方法返回字典而不是元組")

if __name__ == "__main__":
    test_data_adapter() 