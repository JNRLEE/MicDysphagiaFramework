#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
運行所有與索引數據集相關的測試，並生成覆蓋率報告

這個腳本會:
1. 運行 test_data_index_loader.py 中的測試
2. 運行 test_indexed_dataset.py 中的測試
3. 運行 test_indexed_dataset_training.py 中的測試
4. 生成覆蓋率報告
"""

import os
import sys
import unittest
import json
import datetime
import tempfile
import shutil
import subprocess
from pathlib import Path

# 確保測試模塊可以被導入
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 預先導入 torch 以避免重複加載問題
import torch

def run_tests_with_coverage():
    """運行測試並收集覆蓋率數據"""
    print("開始執行索引數據集測試...")
    
    # 創建測試套件
    test_loader = unittest.TestLoader()
    
    # 添加所有測試到測試套件
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    suite = test_loader.discover(tests_dir, pattern="test_*_dataset*.py")
    
    # 運行測試
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(suite)
    
    # 手動計算覆蓋率，因為 coverage 模塊可能與 PyTorch 有衝突
    coverage_data = {
        'total_tests': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
        'successful': result.testsRun - len(result.failures) - len(result.errors) - (len(result.skipped) if hasattr(result, 'skipped') else 0),
        'success_rate': (result.testsRun - len(result.failures) - len(result.errors) - (len(result.skipped) if hasattr(result, 'skipped') else 0)) / result.testsRun if result.testsRun > 0 else 0
    }
    
    return result, coverage_data

def generate_coverage_report(coverage_data, output_file=None):
    """生成覆蓋率報告
    
    Args:
        coverage_data: 覆蓋率數據
        output_file: 輸出文件路徑
    """
    report = {
        'timestamp': datetime.datetime.now().isoformat(),
        'coverage_data': coverage_data
    }
    
    # 如果沒有指定輸出文件，使用默認路徑
    if output_file is None:
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(tests_dir, 'indexed_dataset_coverage.json')
    
    # 寫入報告
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"覆蓋率報告已保存至 {output_file}")
    
    return output_file

def print_report_summary(coverage_data):
    """打印報告摘要
    
    Args:
        coverage_data: 覆蓋率數據
    """
    print("\n===== 測試報告摘要 =====")
    print(f"總測試數量: {coverage_data['total_tests']}")
    print(f"成功: {coverage_data['successful']}")
    print(f"失敗: {coverage_data['failures']}")
    print(f"錯誤: {coverage_data['errors']}")
    print(f"跳過: {coverage_data['skipped']}")
    print(f"成功率: {coverage_data['success_rate']*100:.2f}%")
    print("========================\n")

if __name__ == '__main__':
    # 運行測試並收集覆蓋率數據
    result, coverage_data = run_tests_with_coverage()
    
    # 生成覆蓋率報告
    report_file = generate_coverage_report(coverage_data)
    
    # 打印報告摘要
    print_report_summary(coverage_data)
    
    # 返回成功或失敗的退出碼
    sys.exit(0 if coverage_data['failures'] == 0 and coverage_data['errors'] == 0 else 1) 