"""
測試老闆自定義分類功能
測試日期：2023-04-30
測試目的：驗證使用老闆Excel檔案進行自定義分類的功能是否正確工作
"""

import os
import sys
import logging
from pathlib import Path

# 添加項目根目錄到路徑
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 設置日誌
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from utils.custom_classification_loader import CustomClassificationLoader
from utils.config_loader import load_config

def test_boss_custom_classification():
    """測試老闆自定義分類功能
    
    1. 驗證是否能正確讀取Excel檔案
    2. 驗證是否能正確過濾掉不在Excel中的患者
    """
    logger.info("=== 開始測試老闆自定義分類功能 ===")
    
    # 加載配置檔案
    config_path = os.path.join(project_root, "config", "boss_custom_classification.yaml")
    if not os.path.exists(config_path):
        logger.error(f"找不到配置檔案: {config_path}")
        return False
    
    logger.info(f"正在讀取配置檔案: {config_path}")
    config = load_config(config_path)
    
    # 初始化自定義分類加載器
    logger.info("初始化自定義分類加載器")
    custom_classifier = CustomClassificationLoader(config)
    
    # 檢查是否成功啟用自定義分類
    if not custom_classifier.enabled:
        logger.error("自定義分類未啟用，可能是Excel檔案路徑不正確或格式錯誤")
        return False
    
    # 檢查是否成功讀取分類數據
    if not custom_classifier.patient_id_to_class:
        logger.error("未能從Excel檔案中讀取任何分類數據")
        return False
    
    # 輸出分類結果統計
    total_patients = len(custom_classifier.patient_id_to_class)
    total_classes = custom_classifier.get_total_classes()
    logger.info(f"成功讀取 {total_patients} 位患者的分類數據")
    logger.info(f"共有 {total_classes} 種分類: {custom_classifier.get_all_classes()}")
    
    # 測試幾個患者的分類結果
    test_patients = [
        "P001", "P002", "N001", "隨機ID", 
        # 添加更多測試ID...
    ]
    
    for patient_id in test_patients:
        class_name = custom_classifier.get_class(patient_id)
        class_idx = custom_classifier.get_class_index(patient_id)
        
        if class_name:
            logger.info(f"患者 {patient_id} 的分類: {class_name} (索引: {class_idx})")
        else:
            logger.info(f"患者 {patient_id} 不在分類名單中")
    
    logger.info("=== 測試完成 ===")
    return True

if __name__ == "__main__":
    test_boss_custom_classification() 