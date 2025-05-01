"""
患者信息讀取模組：統一處理患者info.json文件的讀取與解析
功能：
1. 自動搜尋患者資料夾中的info.json文件
2. 排除WavTokenizer相關文件
3. 標準化讀取患者ID、分數和動作選擇
4. 提供錯誤處理和日誌記錄
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

def load_patient_info(patient_dir: Path) -> Optional[Dict[str, Any]]:
    """
    從患者資料夾中讀取患者信息
    
    Description: 
        從指定患者資料夾自動尋找並讀取info.json，排除WavTokenizer_tokens_info.json，
        回傳標準化的患者資訊，包含patient_id、score、selection等欄位。
        如果找不到有效的info.json或解析失敗，回傳None。
    
    Args:
        patient_dir (Path): 患者資料夾的路徑
        
    Returns:
        Optional[Dict[str, Any]]: 包含患者資訊的字典，若找不到或解析失敗則回傳None
        
    References:
        無
    """
    if not isinstance(patient_dir, Path):
        patient_dir = Path(patient_dir)
    
    if not patient_dir.is_dir():
        logger.warning(f"{patient_dir} 不是有效的目錄")
        return None
    
    # 查找info.json文件（排除WavTokenizer_tokens_info.json）
    info_files = list(patient_dir.glob("*_info.json"))
    info_files = [f for f in info_files if "WavTokenizer" not in f.name]
    
    if not info_files:
        logger.warning(f"在 {patient_dir} 中未找到info.json文件")
        return None
    
    # 讀取info.json文件
    try:
        with open(info_files[0], 'r', encoding='utf-8') as f:
            info = json.load(f)
        
        # 提取患者信息
        patient_id = info.get('patientID', '')
        if not patient_id:
            patient_id = info.get('patient_id', '')
        
        # 如果仍然沒有patient_id，嘗試從目錄名提取
        if not patient_id:
            try:
                patient_id = str(patient_dir.name).split('_')[0]
            except:
                logger.warning(f"無法從目錄名 {patient_dir.name} 提取患者ID")
                patient_id = ''
        
        # 獲取分數
        score = info.get('score', -1)
        
        # 獲取選擇
        selection = info.get('selection', '')
        
        # 標準化輸出字典
        result = {
            'patient_id': patient_id,
            'score': score,
            'selection': selection,
            'info_path': str(info_files[0]),
            'raw_info': info  # 保留原始信息，以防需要其他欄位
        }
        
        return result
        
    except Exception as e:
        logger.error(f"讀取info.json文件 {info_files[0]} 時發生錯誤: {str(e)}")
        return None

def list_patient_dirs(root_dir: Path) -> List[Path]:
    """
    列出根目錄中的所有患者目錄
    
    Description:
        從根目錄搜尋所有子目錄，視為患者目錄。
        這是一個輔助函數，用於快速獲取患者目錄列表。
    
    Args:
        root_dir (Path): 根目錄路徑
        
    Returns:
        List[Path]: 患者目錄路徑列表
        
    References:
        無
    """
    if not isinstance(root_dir, Path):
        root_dir = Path(root_dir)
    
    if not root_dir.is_dir():
        logger.error(f"{root_dir} 不是有效的目錄")
        return []
    
    # 獲取所有患者目錄
    patient_dirs = [d for d in root_dir.iterdir() if d.is_dir()]
    
    return patient_dirs

# 中文註解：這是patient_info_loader.py的Minimal Executable Unit，測試讀取功能和對無效輸入的異常處理
if __name__ == "__main__":
    """
    Description: Minimal Executable Unit for patient_info_loader.py，測試讀取功能和對無效輸入的異常處理。
    Args: None
    Returns: None
    References: 無
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 測試無效路徑
    info = load_patient_info("./not_exist_dir")
    print(f"讀取無效路徑: {info}")  # 應該顯示None
    
    # 如果有真實數據，可以取消下面的註釋進行測試
    # from pathlib import Path
    # real_dir = Path("./data/sample_patient_dir")
    # if real_dir.exists():
    #     real_info = load_patient_info(real_dir)
    #     print(f"讀取真實數據: {real_info}") 