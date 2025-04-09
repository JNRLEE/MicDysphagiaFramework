"""
執行模型數據橋接測試腳本
測試日期: 2024/04/09

功能說明:
此腳本簡化了模型數據橋接測試的執行流程，可以直接從命令行執行。
它將運行test_model_data_bridging.py中的所有測試，並生成測試報告。

使用方法:
python tests/run_model_data_test.py
"""

import unittest
import sys
import os
import logging
from pathlib import Path

# 設置日誌
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 將專案根目錄添加到系統路徑
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

if __name__ == '__main__':
    logger.info("開始執行模型數據橋接測試")
    
    # 載入測試模組
    from tests.test_model_data_bridging import TestModelDataBridging
    
    # 創建測試套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestModelDataBridging)
    
    # 運行測試
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_result = test_runner.run(test_suite)
    
    # 輸出測試結果
    logger.info(f"測試完成！成功: {test_result.testsRun - len(test_result.errors) - len(test_result.failures)}, 失敗: {len(test_result.failures)}, 錯誤: {len(test_result.errors)}")
    
    # 檢查是否生成報告
    report_path = Path('tests') / 'model_data_bridging_report.json'
    if report_path.exists():
        logger.info(f"測試報告已生成: {report_path}")
    else:
        logger.warning("測試報告未生成") 