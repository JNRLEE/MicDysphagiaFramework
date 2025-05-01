"""
運行自定義分類測試腳本
功能：
1. 設置測試環境
2. 運行自定義分類測試
3. 輸出測試報告

Description:
    此腳本用於運行自定義分類功能的測試，驗證Excel檔案讀取與分類邏輯正確性。

References:
    None
"""

import os
import sys
import logging
import unittest
from datetime import datetime

# 設置日誌
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加項目根目錄到Python路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 確保tests目錄存在
tests_dir = os.path.join(project_root, 'tests')
if not os.path.exists(tests_dir):
    os.makedirs(tests_dir)

# 匯入測試模組
from tests.test_custom_classification import TestCustomClassification

def run_tests():
    """運行自定義分類測試"""
    logger.info("開始運行自定義分類測試...")
    
    # 記錄測試時間
    test_start_time = datetime.now()
    
    # 創建測試套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestCustomClassification)
    
    # 創建測試結果檔案
    result_file = os.path.join(tests_dir, f"custom_classification_test_report_{test_start_time.strftime('%Y%m%d_%H%M%S')}.txt")
    
    # 運行測試並將結果輸出到檔案
    with open(result_file, 'w', encoding='utf-8') as f:
        # 寫入測試標題
        f.write("=== 自定義分類功能測試報告 ===\n")
        f.write(f"測試時間: {test_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 運行測試
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        test_result = runner.run(test_suite)
        
        # 寫入測試摘要
        f.write("\n=== 測試摘要 ===\n")
        f.write(f"運行測試數: {test_result.testsRun}\n")
        f.write(f"成功: {test_result.testsRun - len(test_result.errors) - len(test_result.failures)}\n")
        f.write(f"失敗: {len(test_result.failures)}\n")
        f.write(f"錯誤: {len(test_result.errors)}\n")
    
    # 輸出測試結果摘要
    logger.info(f"測試完成，報告已保存至: {result_file}")
    logger.info(f"運行測試數: {test_result.testsRun}")
    logger.info(f"成功: {test_result.testsRun - len(test_result.errors) - len(test_result.failures)}")
    
    if test_result.failures:
        logger.error(f"失敗: {len(test_result.failures)}")
        for failure in test_result.failures:
            logger.error(f"- {failure[0]}")
    
    if test_result.errors:
        logger.error(f"錯誤: {len(test_result.errors)}")
        for error in test_result.errors:
            logger.error(f"- {error[0]}")
    
    # 在實驗日誌中記錄測試結果
    try:
        log_file = os.path.join(project_root, 'experiments.log')
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - custom_classification_test - ")
            f.write(f"Tests: {test_result.testsRun}, ")
            f.write(f"Success: {test_result.testsRun - len(test_result.errors) - len(test_result.failures)}, ")
            f.write(f"Failures: {len(test_result.failures)}, ")
            f.write(f"Errors: {len(test_result.errors)}\n")
    except Exception as e:
        logger.warning(f"無法寫入實驗日誌: {str(e)}")
    
    # 返回是否所有測試都通過
    return len(test_result.failures) == 0 and len(test_result.errors) == 0

if __name__ == "__main__":
    """
    Description: 運行自定義分類功能測試的入口點
    Args: None
    Returns: None
    References: 無
    """
    import torch
    
    # 設置隨機種子以確保結果可重現
    torch.manual_seed(42)
    
    # 運行測試
    success = run_tests()
    
    # 設置退出碼
    sys.exit(0 if success else 1) 