"""
常量定義：用於數據處理和標籤映射的常量

功能：
1. 定義標準化的動作類型映射
2. 提供標籤索引映射
3. 集中管理不變的常量

Description:
    維護系統中使用的固定常量，確保代碼中的常量引用一致性。

References:
    None
"""

# 實驗類型映射字典（不可修改）
SELECTION_TYPES = {
    'NoMovement': ["無動作", "無吞嚥"],
    'DrySwallow': ["乾吞嚥1口", "乾吞嚥2口", "乾吞嚥3口", "乾吞嚥"],
    'Cracker': ["餅乾1塊", "餅乾2塊", "餅乾"],
    'Jelly': ["吞果凍", "果凍"],
    'WaterDrinking': ["吞水10ml", "吞水20ml", "喝水", "吞水"]
}

# 標準類別標籤映射 (十分類)
CLASS_LABELS = [
    'Normal-NoMovement',
    'Normal-DrySwallow',
    'Normal-Cracker',
    'Normal-Jelly',
    'Normal-WaterDrinking',
    'Patient-NoMovement',
    'Patient-DrySwallow',
    'Patient-Cracker',
    'Patient-Jelly',
    'Patient-WaterDrinking'
]

# 標籤到索引的映射
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(CLASS_LABELS)}

# 索引到標籤的映射
INDEX_TO_LABEL = {idx: label for idx, label in enumerate(CLASS_LABELS)}

# 中文註解：這是constants.py的Minimal Executable Unit，檢查常量字典與映射能正確查詢與轉換，並測試查詢不存在key時的行為
if __name__ == "__main__":
    """
    Description: Minimal Executable Unit for constants.py，檢查常量字典與映射能正確查詢與轉換，並測試查詢不存在key時的行為。
    Args: None
    Returns: None
    References: 無
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    try:
        print("SELECTION_TYPES['Jelly']: ", SELECTION_TYPES['Jelly'])
        print("LABEL_TO_INDEX['Normal-DrySwallow']: ", LABEL_TO_INDEX['Normal-DrySwallow'])
        print("INDEX_TO_LABEL[3]: ", INDEX_TO_LABEL[3])
        print("常量查詢測試成功")
    except Exception as e:
        print(f"常量查詢測試失敗: {e}")
    # 測試不存在key
    try:
        print(LABEL_TO_INDEX['NotExist'])
    except Exception as e:
        print(f"查詢不存在key時的報錯（預期行為）: {e}") 