"""
測試患者信息讀取模組：對utils/patient_info_loader.py進行單元測試
功能：
1. 測試正確讀取患者info.json文件
2. 測試處理不存在的目錄
3. 測試處理沒有info.json的目錄
4. 測試處理畸形的info.json文件
"""

import unittest
import logging
import os
import json
import tempfile
import shutil
import sys
from pathlib import Path

# 將項目根目錄添加到 Python 路徑中
sys.path.insert(0, str(Path(__file__).parent.parent))

# 導入要測試的模組
from utils.patient_info_loader import load_patient_info, list_patient_dirs

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestPatientInfoLoader(unittest.TestCase):
    """
    Description: 測試患者信息讀取模組的功能。
    
    Tests:
        1. 測試正確讀取標準格式的info.json
        2. 測試處理不存在的目錄
        3. 測試處理沒有info.json的目錄
        4. 測試處理畸形的info.json文件
        5. 測試從目錄名中提取患者ID
    
    References:
        無
    """
    
    def setUp(self):
        """
        Description: 建立測試環境，創建臨時目錄和測試文件。
        """
        # 創建臨時測試目錄
        self.test_dir = Path(tempfile.mkdtemp())
        
        # 創建標準患者目錄和info.json
        self.standard_patient_dir = self.test_dir / "P001_標準患者"
        self.standard_patient_dir.mkdir(exist_ok=True)
        
        # 建立標準info.json
        standard_info = {
            "patientID": "P001",
            "score": 15,
            "selection": "乾吞嚥"
        }
        
        with open(self.standard_patient_dir / "standard_info.json", 'w', encoding='utf-8') as f:
            json.dump(standard_info, f, ensure_ascii=False)
        
        # 創建備用患者ID格式的目錄和info.json
        self.alt_patient_dir = self.test_dir / "P002_備用格式"
        self.alt_patient_dir.mkdir(exist_ok=True)
        
        # 建立使用patient_id而非patientID的info.json
        alt_info = {
            "patient_id": "P002",
            "score": 20,
            "selection": "吞水10ml"
        }
        
        with open(self.alt_patient_dir / "alt_info.json", 'w', encoding='utf-8') as f:
            json.dump(alt_info, f, ensure_ascii=False)
        
        # 創建沒有ID的info.json目錄
        self.no_id_patient_dir = self.test_dir / "P003_無ID"
        self.no_id_patient_dir.mkdir(exist_ok=True)
        
        # 建立沒有患者ID的info.json
        no_id_info = {
            "score": 25,
            "selection": "果凍"
        }
        
        with open(self.no_id_patient_dir / "no_id_info.json", 'w', encoding='utf-8') as f:
            json.dump(no_id_info, f, ensure_ascii=False)
        
        # 創建空目錄（沒有info.json的目錄）
        self.empty_dir = self.test_dir / "empty_dir"
        self.empty_dir.mkdir(exist_ok=True)
        
        # 創建有畸形info.json的目錄
        self.malformed_dir = self.test_dir / "malformed_dir"
        self.malformed_dir.mkdir(exist_ok=True)
        
        # 寫入畸形的JSON
        with open(self.malformed_dir / "malformed_info.json", 'w', encoding='utf-8') as f:
            f.write("{這不是有效的JSON}")
        
        # 創建有WavTokenizer文件的目錄
        self.wav_tokenizer_dir = self.test_dir / "wav_tokenizer_dir"
        self.wav_tokenizer_dir.mkdir(exist_ok=True)
        
        # 寫入WavTokenizer文件和正常info.json
        wavtokenizer_info = {"type": "wavtokenizer"}
        with open(self.wav_tokenizer_dir / "WavTokenizer_tokens_info.json", 'w', encoding='utf-8') as f:
            json.dump(wavtokenizer_info, f)
        
        normal_info = {"patientID": "WT001", "score": 30, "selection": "吞水"}
        with open(self.wav_tokenizer_dir / "normal_info.json", 'w', encoding='utf-8') as f:
            json.dump(normal_info, f)
            
        logger.info(f"測試環境建立完成，臨時目錄: {self.test_dir}")
    
    def tearDown(self):
        """
        Description: 清理測試環境，刪除臨時目錄和文件。
        """
        # 刪除臨時測試目錄
        shutil.rmtree(self.test_dir)
        logger.info(f"測試環境清理完成，已刪除臨時目錄: {self.test_dir}")
    
    def test_standard_info(self):
        """
        Description: 測試讀取標準格式的info.json。
        """
        info = load_patient_info(self.standard_patient_dir)
        self.assertIsNotNone(info)
        self.assertEqual(info['patient_id'], "P001")
        self.assertEqual(info['score'], 15)
        self.assertEqual(info['selection'], "乾吞嚥")
    
    def test_alternative_format(self):
        """
        Description: 測試讀取使用patient_id而非patientID的info.json。
        """
        info = load_patient_info(self.alt_patient_dir)
        self.assertIsNotNone(info)
        self.assertEqual(info['patient_id'], "P002")
        self.assertEqual(info['score'], 20)
        self.assertEqual(info['selection'], "吞水10ml")
    
    def test_extract_id_from_dirname(self):
        """
        Description: 測試從目錄名中提取患者ID。
        """
        info = load_patient_info(self.no_id_patient_dir)
        self.assertIsNotNone(info)
        self.assertEqual(info['patient_id'], "P003")
        self.assertEqual(info['score'], 25)
        self.assertEqual(info['selection'], "果凍")
    
    def test_nonexistent_dir(self):
        """
        Description: 測試處理不存在的目錄。
        """
        info = load_patient_info(self.test_dir / "not_exist")
        self.assertIsNone(info)
    
    def test_empty_dir(self):
        """
        Description: 測試處理沒有info.json的目錄。
        """
        info = load_patient_info(self.empty_dir)
        self.assertIsNone(info)
    
    def test_malformed_json(self):
        """
        Description: 測試處理畸形的info.json文件。
        """
        info = load_patient_info(self.malformed_dir)
        self.assertIsNone(info)
    
    def test_exclude_wavtokenizer(self):
        """
        Description: 測試排除WavTokenizer_tokens_info.json文件。
        """
        info = load_patient_info(self.wav_tokenizer_dir)
        self.assertIsNotNone(info)
        self.assertEqual(info['patient_id'], "WT001")
    
    def test_list_patient_dirs(self):
        """
        Description: 測試列出患者目錄功能。
        """
        dirs = list_patient_dirs(self.test_dir)
        self.assertEqual(len(dirs), 6)  # 應該能找到6個目錄
        
        # 測試無效輸入
        invalid_dirs = list_patient_dirs(self.test_dir / "not_exist")
        self.assertEqual(len(invalid_dirs), 0)
        

# 直接運行測試
if __name__ == "__main__":
    """
    Description: 運行患者信息讀取模組的所有單元測試。
    Args: None
    Returns: None
    References: 無
    """
    unittest.main() 