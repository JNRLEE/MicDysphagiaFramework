#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
依據 selection 欄位將資料分類並存為 CSV 檔案。

此腳本讀取 data/metadata/data_index.csv 檔案，並根據 selection 欄位將資料分類：
- 乾吞嚥1口
- 無動作

將符合條件的資料儲存為獨立的 CSV 檔案。
"""

import os
import pandas as pd
import logging
from datetime import datetime

# 設定記錄
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 設定檔案路徑
DATA_INDEX_PATH = 'data/metadata/data_index.csv'
OUTPUT_DIR = 'data/metadata/selection_groups'

# 定義要篩選的選項
SELECTIONS = ["乾吞嚥1口", "無動作"]

def main():
    """執行資料分類並儲存為 CSV 檔案的主函數。"""
    logger.info("開始執行資料分類...")

    # 檢查資料索引檔案是否存在
    if not os.path.exists(DATA_INDEX_PATH):
        logger.error(f"找不到資料索引檔案: {DATA_INDEX_PATH}")
        return

    # 確保輸出目錄存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # 讀取資料索引檔案
        logger.info(f"讀取資料索引檔案: {DATA_INDEX_PATH}")
        df = pd.read_csv(DATA_INDEX_PATH)
        
        # 確認 selection 欄位存在
        if 'selection' not in df.columns:
            logger.error("資料索引檔案中沒有 'selection' 欄位")
            return
            
        # 記錄原始資料總數
        total_records = len(df)
        logger.info(f"原始資料總數: {total_records} 筆")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 篩選符合條件的資料
        filtered_df = df[df['selection'].isin(SELECTIONS)]
        
        # 計算篩選後的資料筆數
        filtered_count = len(filtered_df)
        
        if filtered_count == 0:
            logger.warning("沒有符合條件的資料")
            return
            
        # 建立輸出檔案名稱
        output_filename = f"dry_swallow1_no_action_{timestamp}.csv"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # 儲存篩選後的資料
        filtered_df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"已將 {filtered_count} 筆資料儲存至: {output_path}")
        
        # 統計各選項的分布
        selection_counts = filtered_df['selection'].value_counts()
        for selection, count in selection_counts.items():
            logger.info(f"    - {selection}: {count} 筆")
        
        logger.info("資料分類完成")
        
    except Exception as e:
        logger.error(f"處理資料時發生錯誤: {str(e)}")

if __name__ == "__main__":
    main()