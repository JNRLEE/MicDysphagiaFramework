"""
測試 SpectrogramDataset._collect_samples 方法
測試日期：2023-04-28
測試目的：驗證修改後的 SpectrogramDataset._collect_samples 方法能夠正確獲取和使用配置變量
"""

import os
import sys
import torch
import logging
import unittest
from pathlib import Path
import tempfile
import json
import shutil

# 將專案根目錄加入 sys.path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 設置日誌
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from data.spectrogram_dataset import SpectrogramDataset
    from utils.patient_info_loader import load_patient_info
except ImportError as e:
    logger.error(f"導入錯誤: {e}")
    raise

class TestSpectrogramDataset(unittest.TestCase):
    """
    測試 SpectrogramDataset 的樣本收集功能，尤其是 _collect_samples 方法。
    
    Args:
        unittest.TestCase: 繼承 unittest.TestCase
    """
    
    def setUp(self):
        """
        設置測試環境，創建臨時目錄和必要的配置。
        """
        self.test_dir = tempfile.mkdtemp()
        logger.info(f"測試目錄創建於: {self.test_dir}")
        
        # 創建4個患者目錄
        self.patients = ["P001", "P002", "P003", "N001"]
        self.patient_dirs = []
        for patient in self.patients:
            self.patient_dirs.append(self.create_patient_dir(patient))
            
        # 分類任務配置
        self.classification_config = {
            "task_type": "classification",
            "dataset": {
                "type": "spectrogram",
                "data_dir": self.test_dir,
                "split_ratio": 0.8,
                "random_seed": 42,
                "class_mapping": {
                    "P": 1,  # 陽性樣本
                    "N": 0   # 陰性樣本
                }
            }
        }
        
        # 回歸任務配置
        self.regression_config = {
            "task_type": "regression",
            "dataset": {
                "type": "spectrogram",
                "data_dir": self.test_dir,
                "split_ratio": 0.8,
                "random_seed": 42,
                "target_score": "EAT10"
            }
        }
        
    def tearDown(self):
        """
        清理測試環境，刪除臨時目錄。
        """
        shutil.rmtree(self.test_dir)
        logger.info(f"測試目錄已清理: {self.test_dir}")
        
    def create_patient_dir(self, patient_id):
        """
        創建患者目錄並新增測試文件。
        
        Args:
            patient_id: 患者ID
            
        Returns:
            Path: 患者目錄路徑
        """
        patient_dir = Path(self.test_dir) / patient_id
        patient_dir.mkdir(exist_ok=True)
        
        # 創建info.json
        info = {
            "patientID": patient_id,
            "score": 5 if patient_id.startswith("P") else 0,
            "selection": "Liquid" if patient_id.startswith("P") else "Normal",
            "EAT10": 15 if patient_id.startswith("P") else 0
        }
        
        info_file = patient_dir / "patient_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f)
            
        logger.info(f"已創建患者 {patient_id} 目錄: {patient_dir}")
        logger.info(f"info.json 文件存在: {info_file.exists()}")
        
        # 檢查info.json是否可以被patient_info_loader正確讀取
        patient_info = load_patient_info(patient_dir)
        logger.info(f"使用patient_info_loader讀取結果: {patient_info}")
        
        # 創建一個假的頻譜圖檔案
        spectrogram_file = patient_dir / f"{patient_id}_spectrogram_{1}.pt"
        dummy_spectrogram = torch.randn(1, 128, 32)
        torch.save(dummy_spectrogram, spectrogram_file)
        
        return patient_dir
        
    def test_collect_samples_classification(self):
        """
        測試分類任務下的樣本收集。
        """
        logger.info("測試分類任務的樣本收集")
        dataset = SpectrogramDataset(self.classification_config, "train")
        logger.info(f"分類數據集信息: {len(dataset.samples)} 頻譜圖樣本已加載")
        self.assertGreater(len(dataset.samples), 0, "應該收集到至少一個樣本")
        
    def test_collect_samples_regression(self):
        """
        測試回歸任務下的樣本收集。
        """
        logger.info("測試回歸任務的樣本收集")
        dataset = SpectrogramDataset(self.regression_config, "train")
        logger.info(f"回歸數據集信息: {len(dataset.samples)} 頻譜圖樣本已加載")
        self.assertGreater(len(dataset.samples), 0, "應該收集到至少一個樣本")
        
if __name__ == "__main__":
    unittest.main() 