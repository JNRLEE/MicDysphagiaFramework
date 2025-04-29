"""
自定義分類載入模組：處理Excel檔案中的自定義分類邏輯
功能：
1. 讀取Excel檔案中的分類方案
2. 提供患者ID到分類的映射
3. 支援多種Excel格式與欄位設定
4. 提供適當的錯誤處理和日誌
5. 支援動作類型過濾功能
6. 支援從數據JSON文件讀取動作類型
7. 支援排除特定分類類別

Description:
    用於讀取老闆提供的Excel檔案中的自定義分類邏輯，並提供映射功能。
    支援根據配置檔案指定的欄位名稱讀取相關資訊。
    可以根據class_config過濾不需要的動作類型。
    能夠從數據的JSON文件中讀取動作類型信息。
    可以通過exclude_classes排除不需要的分類類別。

References:
    None
"""

import pandas as pd
import logging
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set

logger = logging.getLogger(__name__)

# 動作類型與檔案命名對應表
SELECTION_TYPES = {
    "NoMovement": ["無動作", "無聲音", "無動", "無"],
    "DrySwallow": ["乾吞嚥", "乾吞", "口水", "吞口水"],
    "Cracker": ["餅乾", "薄餅", "cracker"],
    "Jelly": ["果凍", "布丁", "jelly"],
    "WaterDrinking": ["水", "吞水", "water", "drinking"]
}

class CustomClassificationLoader:
    """自定義分類載入器，用於處理Excel中的分類邏輯
    
    Description:
        從Excel檔案讀取自定義分類邏輯，並提供患者ID到分類的映射。
        支援根據配置檔案指定的欄位名稱讀取相關資訊。
        可以根據class_config過濾不需要的動作類型。
        能夠從數據的JSON文件中讀取動作類型信息。
        可以通過exclude_classes排除不需要的分類類別。
    
    Args:
        None
        
    Returns:
        None
        
    References:
        None
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化自定義分類載入器
        
        Args:
            config (Dict[str, Any]): 配置字典，包含custom_classification設定
        """
        self.config = config
        self.custom_config = config.get('data', {}).get('filtering', {}).get('custom_classification', {})
        self.enabled = self.custom_config.get('enabled', False)
        self.excel_path = self.custom_config.get('excel_path', '')
        self.patient_id_column = self.custom_config.get('patient_id_column', 'A')
        self.class_column = self.custom_config.get('class_column', 'P')
        
        # 新增：動作類型過濾設定
        self.class_config = self.custom_config.get('class_config', {})
        
        # 如果class_config為空，嘗試從data.filtering.class_config獲取
        if not self.class_config:
            self.class_config = config.get('data', {}).get('filtering', {}).get('class_config', {})
        
        # 新增：分類類別排除設定
        self.exclude_classes = self.custom_config.get('exclude_classes', [])
        
        self.filtered_actions = self._get_filtered_actions()
        
        # 數據目錄，用於讀取JSON文件
        self.data_dir = config.get('data', {}).get('source', {}).get('wav_dir', '')
        self.data_dir = Path(self.data_dir) if self.data_dir else None
        
        # 從數據文件讀取的動作類型映射
        self.data_action_types = {}
        if self.data_dir and self.data_dir.exists():
            self._load_data_action_types()
        
        self.patient_id_to_class = {}
        self.class_to_index = {}
        self.class_names = []
        
        if self.enabled and self.excel_path:
            self._load_excel_data()
    
    def _load_data_action_types(self) -> None:
        """從數據目錄加載動作類型信息
        
        Description:
            遍歷數據目錄，讀取每個患者子目錄中的info.json文件，
            提取患者ID和選擇類型(selection)，建立映射。
            
        Returns:
            None
        """
        logger.info(f"正在從數據目錄 {self.data_dir} 讀取動作類型信息...")
        
        # 獲取所有患者目錄
        patient_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        count = 0
        for patient_dir in patient_dirs:
            # 查找info.json文件
            # 排除WavTokenizer_tokens_info.json文件
            info_files = [f for f in patient_dir.glob("*info.json") 
                        if not f.name.endswith("WavTokenizer_tokens_info.json")]
            
            if not info_files:
                logger.debug(f"在 {patient_dir} 找不到info.json文件")
                continue
            
            # 讀取info.json文件
            for info_file in info_files:
                try:
                    with open(info_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 提取patientID和selection
                    patient_id = data.get('patientID')
                    selection = data.get('selection')
                    
                    if patient_id and selection:
                        # 映射選擇類型到標準動作類型
                        selection_type = None
                        for std_type, selections in SELECTION_TYPES.items():
                            if any(s in selection for s in selections):
                                selection_type = std_type
                                break
                        
                        if selection_type:
                            self.data_action_types[patient_id] = selection_type
                            count += 1
                except Exception as e:
                    logger.warning(f"讀取 {info_file} 時發生錯誤: {str(e)}")
        
        logger.info(f"從數據目錄成功讀取 {count} 個患者的動作類型信息")
    
    def _get_filtered_actions(self) -> Set[str]:
        """獲取需要過濾的動作類型
        
        Description:
            根據class_config配置，返回需要保留的動作類型（值為1的動作）。
            
        Returns:
            Set[str]: 需要保留的動作類型集合
        """
        filtered_actions = set()
        for action, value in self.class_config.items():
            if value == 1:
                filtered_actions.add(action)
        return filtered_actions
    
    def is_action_allowed(self, action_type: str) -> bool:
        """檢查動作類型是否應該被保留
        
        Description:
            根據class_config配置，判斷指定的動作類型是否應該被保留。
            如果class_config為空（未設定），則所有動作類型都被保留。
            
        Args:
            action_type (str): 動作類型名稱
            
        Returns:
            bool: 如果動作類型應該被保留，返回True；否則返回False
        """
        # 如果沒有設定class_config，則保留所有動作類型
        if not self.class_config:
            return True
        
        # 檢查動作類型是否在保留列表中
        return action_type in self.filtered_actions
    
    def get_patient_action_type(self, patient_id: str) -> Optional[str]:
        """獲取患者的動作類型
        
        Description:
            根據患者ID獲取其對應的動作類型。
            
        Args:
            patient_id (str): 患者ID
            
        Returns:
            Optional[str]: 動作類型名稱，若找不到則返回None
        """
        return self.data_action_types.get(patient_id)
    
    def is_patient_action_allowed(self, patient_id: str) -> bool:
        """檢查患者的動作類型是否應該被保留
        
        Description:
            根據患者ID獲取其動作類型，並判斷該動作類型是否應該被保留。
            
        Args:
            patient_id (str): 患者ID
            
        Returns:
            bool: 如果患者的動作類型應該被保留，返回True；否則返回False
        """
        action_type = self.get_patient_action_type(patient_id)
        if action_type is None:
            # 如果找不到患者的動作類型，默認保留
            return True
        
        return self.is_action_allowed(action_type)
    
    def is_class_excluded(self, class_value: str) -> bool:
        """檢查分類類別是否應該被排除
        
        Description:
            根據exclude_classes配置，判斷指定的分類類別是否應該被排除。
            
        Args:
            class_value (str): 分類類別名稱
            
        Returns:
            bool: 如果分類類別應該被排除，返回True；否則返回False
        """
        return class_value in self.exclude_classes
    
    def _load_excel_data(self) -> None:
        """從Excel檔案載入分類數據
        
        讀取Excel檔案中的分類邏輯，建立患者ID到分類的映射。
        如果設定了class_config，則只保留配置中值為1的動作類型。
        如果設定了exclude_classes，則排除指定的分類類別。
        """
        try:
            excel_path = Path(self.excel_path)
            if not excel_path.exists():
                logger.error(f"找不到Excel檔案: {self.excel_path}")
                self.enabled = False
                return
            
            # 讀取Excel檔案
            df = pd.read_excel(excel_path)
            
            # 檢查列是否存在
            if self.patient_id_column not in df.columns and not self._is_excel_column(self.patient_id_column):
                logger.error(f"找不到患者ID欄位: {self.patient_id_column}")
                self.enabled = False
                return
            
            if self.class_column not in df.columns and not self._is_excel_column(self.class_column):
                logger.error(f"找不到分類欄位: {self.class_column}")
                self.enabled = False
                return
            
            # 如果使用Excel列名(A, B, C...)而非直接欄位名稱
            if self._is_excel_column(self.patient_id_column):
                patient_id_col_idx = self._excel_column_to_index(self.patient_id_column)
                patient_id_col = df.columns[patient_id_col_idx]
            else:
                patient_id_col = self.patient_id_column
            
            if self._is_excel_column(self.class_column):
                class_col_idx = self._excel_column_to_index(self.class_column)
                class_col = df.columns[class_col_idx]
            else:
                class_col = self.class_column
            
            # 建立患者ID到分類的映射
            for _, row in df.iterrows():
                patient_id = str(row[patient_id_col]).strip()
                class_value = str(row[class_col]).strip()
                
                # 跳過空值
                if not patient_id or not class_value or pd.isna(patient_id) or pd.isna(class_value):
                    continue
                
                # 根據患者ID檢查其動作類型是否應該被保留
                if not self.is_patient_action_allowed(patient_id):
                    logger.debug(f"患者 {patient_id} 的動作類型被過濾掉")
                    continue
                
                # 檢查分類類別是否應該被排除
                if self.is_class_excluded(class_value):
                    logger.debug(f"分類類別 {class_value} 被排除")
                    continue
                
                self.patient_id_to_class[patient_id] = class_value
                
                # 紀錄所有唯一的分類名稱
                if class_value not in self.class_names:
                    self.class_names.append(class_value)
            
            # 建立分類名稱到索引的映射
            self.class_to_index = {class_name: idx for idx, class_name in enumerate(self.class_names)}
            
            logger.info(f"成功載入自定義分類數據，共 {len(self.patient_id_to_class)} 個患者")
            logger.info(f"分類類別: {self.class_names}")
            
            # 新增：記錄過濾設定
            if self.class_config:
                logger.info(f"根據class_config進行過濾，保留的動作類型: {self.filtered_actions}")
            
            # 新增：記錄排除的分類類別
            if self.exclude_classes:
                logger.info(f"排除的分類類別: {self.exclude_classes}")
            
            # 新增：記錄從數據文件讀取的動作類型信息
            logger.info(f"從數據文件讀取的動作類型數量: {len(self.data_action_types)}")
            
        except Exception as e:
            logger.error(f"載入Excel檔案時發生錯誤: {str(e)}")
            self.enabled = False
    
    def _is_excel_column(self, col: str) -> bool:
        """檢查是否為Excel列名格式 (A, B, C...)
        
        Args:
            col (str): 列名
            
        Returns:
            bool: 是否為Excel列名格式
        """
        return col.upper() in [chr(65 + i) for i in range(26)] + [f"A{chr(65 + i)}" for i in range(26)]
    
    def _excel_column_to_index(self, col: str) -> int:
        """將Excel列名轉換為索引
        
        Args:
            col (str): Excel列名 (A, B, C...)
            
        Returns:
            int: 對應的索引
        """
        col = col.upper()
        if len(col) == 1:
            return ord(col) - ord('A')
        else:
            return 26 + (ord(col[1]) - ord('A'))
    
    def get_class(self, patient_id: str) -> Optional[str]:
        """根據患者ID獲取分類
        
        Args:
            patient_id (str): 患者ID
            
        Returns:
            Optional[str]: 分類名稱，若找不到則返回None
        """
        if not self.enabled:
            return None
        
        return self.patient_id_to_class.get(patient_id)
    
    def get_class_index(self, patient_id: str) -> Optional[int]:
        """根據患者ID獲取分類索引
        
        Args:
            patient_id (str): 患者ID
            
        Returns:
            Optional[int]: 分類索引，若找不到則返回None
        """
        if not self.enabled:
            return None
        
        class_name = self.get_class(patient_id)
        if class_name is None:
            return None
        
        return self.class_to_index.get(class_name)
    
    def get_all_classes(self) -> List[str]:
        """獲取所有分類名稱
        
        Returns:
            List[str]: 分類名稱列表
        """
        return self.class_names.copy()
    
    def get_total_classes(self) -> int:
        """獲取分類總數
        
        Returns:
            int: 分類總數
        """
        return len(self.class_names)

# 中文註解：這是custom_classification_loader.py的Minimal Executable Unit，測試載入功能和對無效輸入的異常處理
if __name__ == "__main__":
    """
    Description: Minimal Executable Unit for custom_classification_loader.py，測試載入功能和對無效輸入的異常處理。
    Args: None
    Returns: None
    References: 無
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 測試無效配置
    config = {
        'data': {
            'source': {
                'wav_dir': './tests/dataloader_test/dataset_test'  # 測試數據目錄
            },
            'filtering': {
                'custom_classification': {
                    'enabled': True,
                    'excel_path': './not_exist.xlsx',
                    'patient_id_column': 'A',
                    'class_column': 'P',
                    'class_config': {
                        'NoMovement': 1,
                        'DrySwallow': 1,
                        'Cracker': 1,
                        'Jelly': 1,
                        'WaterDrinking': 1,
                        'OtherAction': 0  # 這個會被過濾掉
                    },
                    'exclude_classes': ['nan', '正常']  # 測試排除分類類別
                }
            }
        }
    }
    
    loader = CustomClassificationLoader(config)
    print(f"啟用狀態: {loader.enabled}")  # 應該顯示False，因為文件不存在
    print(f"保留的動作類型: {loader.filtered_actions}")  # 應該顯示保留的動作類型
    print(f"排除的分類類別: {loader.exclude_classes}")  # 應該顯示排除的分類類別
    print(f"從數據讀取的動作類型數量: {len(loader.data_action_types)}")  # 應該顯示從數據讀取的動作類型數量
    
    # 測試動作類型過濾
    print(f"NoMovement允許: {loader.is_action_allowed('NoMovement')}")  # 應該顯示True
    print(f"OtherAction允許: {loader.is_action_allowed('OtherAction')}")  # 應該顯示False
    
    # 測試分類類別排除
    print(f"nan類別排除: {loader.is_class_excluded('nan')}")  # 應該顯示True
    print(f"正常類別排除: {loader.is_class_excluded('正常')}")  # 應該顯示True
    print(f"其他類別排除: {loader.is_class_excluded('其他')}")  # 應該顯示False
    
    # 測試患者動作類型過濾
    for patient_id, action_type in list(loader.data_action_types.items())[:5]:
        print(f"患者 {patient_id} 動作類型: {action_type}, 允許: {loader.is_patient_action_allowed(patient_id)}")
    
    # 如果有真實數據，可以取消下面的註釋進行測試
    # real_config = {
    #     'data': {
    #         'source': {
    #             'wav_dir': './tests/dataloader_test/dataset_test'  # 測試數據目錄
    #         },
    #         'filtering': {
    #             'custom_classification': {
    #                 'enabled': True,
    #                 'excel_path': './test_classification.xlsx',
    #                 'patient_id_column': 'A',
    #                 'class_column': 'P',
    #                 'class_config': {
    #                     'NoMovement': 1,
    #                     'DrySwallow': 1,
    #                     'Cracker': 1,
    #                     'Jelly': 1,
    #                     'WaterDrinking': 1
    #                 },
    #                 'exclude_classes': ['nan', '正常']  # 排除的分類類別
    #             }
    #         }
    #     }
    # }
    # real_loader = CustomClassificationLoader(real_config)
    # print(f"分類類別: {real_loader.get_all_classes()}")
    # print(f"患者P001的分類: {real_loader.get_class('P001')}") 