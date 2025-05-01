"""
運行數據類型兼容性測試

功能：
1. 運行數據類型兼容性測試腳本
2. 輸出詳細測試結果
3. 生成測試報告

Description:
    運行數據類型兼容性測試，檢查標籤是否正確轉換為float32類型。

References:
    None
"""

import os
import sys
import unittest
import json
from datetime import datetime
from pathlib import Path

# 添加項目根目錄到路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from test_data_type_compatibility import TestDataTypeCompatibility

def run_tests():
    """運行所有測試並生成報告"""
    # 設置輸出目錄
    output_dir = Path('tests/results')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 創建測試套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDataTypeCompatibility)
    
    # 創建報告文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = output_dir / f'data_type_compatibility_report_{timestamp}.txt'
    
    # 運行測試並捕獲輸出
    with open(report_file, 'w') as f:
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        result = runner.run(suite)
    
    # 打印簡要結果
    print(f"\n= 數據類型兼容性測試結果 =")
    print(f"運行測試: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失敗: {len(result.failures)}")
    print(f"錯誤: {len(result.errors)}")
    print(f"報告已保存到: {report_file}")
    
    # 創建測試結果摘要
    summary = {
        'timestamp': timestamp,
        'tests_run': result.testsRun,
        'successful': result.testsRun - len(result.failures) - len(result.errors),
        'failures': len(result.failures),
        'errors': len(result.errors),
        'report_file': str(report_file)
    }
    
    # 保存摘要
    summary_file = output_dir / f'data_type_compatibility_summary_{timestamp}.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1) 