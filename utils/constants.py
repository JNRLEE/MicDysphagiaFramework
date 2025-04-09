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