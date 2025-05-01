"""
測試 DataAdapter.convert_audio_to_spectrogram 方法對列表輸入的處理
測試日期：2023-04-28
測試目的：驗證修改後的 convert_audio_to_spectrogram 方法能夠正確處理列表輸入
"""

import os
import sys
import torch
import logging

# 添加項目根目錄到路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from utils.data_adapter import DataAdapter
    
    # 設置日誌
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    def test_convert_audio_to_spectrogram_list():
        """測試 convert_audio_to_spectrogram 方法對列表輸入的處理"""
        logger.info("=== 測試 convert_audio_to_spectrogram 方法對列表輸入的處理 ===")
        
        # 創建一個簡單的音頻張量列表
        audio_list = [torch.randn(16000) for _ in range(3)]
        
        try:
            # 調用方法
            specs = DataAdapter.convert_audio_to_spectrogram(audio_list)
            
            # 檢查結果
            logger.info(f"成功將音頻列表轉換為頻譜圖，形狀: {specs.shape}")
            assert specs.dim() == 4, f"頻譜圖應為4維，實際為 {specs.dim()} 維"
            assert specs.size(0) == 3, f"批次大小應為3，實際為 {specs.size(0)}"
            
            logger.info("測試通過！✅")
            return True
        except Exception as e:
            logger.error(f"測試失敗: {str(e)}")
            return False
    
    if __name__ == "__main__":
        test_convert_audio_to_spectrogram_list()
except ImportError as e:
    print(f"導入出錯: {str(e)}") 