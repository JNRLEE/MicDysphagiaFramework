#!/usr/bin/env python
"""
老闆自定義分類測試腳本
功能：
1. 測試從Excel檔案讀取自定義分類
2. 測試數據過濾邏輯
3. 測試模型是否能夠正確處理自定義分類
4. 儲存測試結果

Description:
    此腳本模擬整個訓練流程，但只進行一個小批次的訓練，主要用於測試自定義分類功能。

Args:
    None

Returns:
    None

References:
    None
"""

import os
import sys
import json
import logging
import torch
import datetime
from pathlib import Path

# 添加項目根目錄到路徑
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 設置日誌
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from utils.config_loader import load_config
from utils.custom_classification_loader import CustomClassificationLoader
from data.dataset_factory import DatasetFactory
from models.model_factory import create_model
from torch.utils.data import DataLoader
from data.audio_dataset import AudioDataset

def test_boss_classification():
    """測試老闆自定義分類功能的完整流程
    
    1. 加載配置檔案
    2. 初始化自定義分類加載器
    3. 創建數據集並檢查過濾邏輯
    4. 創建模型並檢查輸出類別數
    5. 儲存測試結果
    """
    # 記錄測試開始時間
    start_time = datetime.datetime.now()
    
    # 創建測試結果目錄
    test_output_dir = os.path.join(project_root, "tests", "output", "boss_classification_test")
    os.makedirs(test_output_dir, exist_ok=True)
    
    # 加載配置檔案
    config_path = os.path.join(project_root, "config", "boss_custom_classification.yaml")
    if not os.path.exists(config_path):
        logger.error(f"找不到配置檔案: {config_path}")
        return
    
    logger.info(f"正在讀取配置檔案: {config_path}")
    config = load_config(config_path)
    
    # 初始化自定義分類加載器
    logger.info("初始化自定義分類加載器")
    custom_classifier = CustomClassificationLoader(config)
    
    # 檢查是否成功啟用自定義分類
    if not custom_classifier.enabled:
        logger.error("自定義分類未啟用，可能是Excel檔案路徑不正確或格式錯誤")
        return
    
    # 輸出分類結果統計
    total_patients = len(custom_classifier.patient_id_to_class)
    total_classes = custom_classifier.get_total_classes()
    logger.info(f"成功讀取 {total_patients} 位患者的分類數據")
    logger.info(f"共有 {total_classes} 種分類: {custom_classifier.get_all_classes()}")
    
    # 創建數據集
    logger.info("創建數據集")
    try:
        # 獲取數據路徑
        data_path = config.get('data', {}).get('source', {}).get('wav_dir', '')
        if not data_path:
            logger.error("配置中未指定數據路徑")
            return
        
        # 直接使用 AudioDataset 創建數據集，而不是使用 DatasetFactory
        train_dataset = AudioDataset(
            root_dir=data_path,
            config=config,
            is_train=True
        )
        
        # 檢查數據集是否為空
        if len(train_dataset) == 0:
            logger.warning("數據集為空，請檢查數據路徑和過濾配置")
            return
        
        logger.info(f"成功創建數據集，共有 {len(train_dataset)} 個樣本")
        logger.info(f"數據集樣本數: {len(train_dataset.samples)}")
        
        # 查看數據集分類統計
        class_counts = {}
        for sample in train_dataset.samples:
            label = sample['label'].item()
            # 轉換為整數
            if isinstance(label, float):
                label = int(label)
            
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1
        
        logger.info(f"數據集分類統計: {class_counts}")
        
        # 獲取數據樣本
        if len(train_dataset.samples) > 0:
            logger.info("獲取數據樣本")
            sample = train_dataset.samples[0]
            logger.info(f"樣本示例:\n{json.dumps({k: str(v) for k, v in sample.items() if k != 'label'}, indent=2, ensure_ascii=False)}")
        
    except Exception as e:
        logger.error(f"創建數據集時發生錯誤: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # 創建模型
    logger.info("創建模型")
    try:
        # 更新配置中的類別數
        if custom_classifier.enabled and custom_classifier.get_total_classes() > 0:
            config['model']['parameters']['num_classes'] = custom_classifier.get_total_classes()
        
        # 創建模型
        model = create_model(config)
        
        # 檢查模型輸出維度
        if hasattr(model, 'num_classes'):
            logger.info(f"模型輸出類別數: {model.num_classes}")
        elif hasattr(model, 'head') and hasattr(model.head, 'out_features'):
            logger.info(f"模型輸出維度: {model.head.out_features}")
        else:
            logger.info("無法確定模型輸出維度")
        
        # 創建一個隨機輸入進行前向傳播
        if config['data']['type'] == 'audio':
            dummy_input = torch.randn(1, config['data']['preprocessing']['audio'].get('duration', 5) * config['data']['preprocessing']['audio'].get('sr', 16000))
        elif config['data']['type'] == 'spectrogram':
            dummy_input = torch.randn(1, 3, 224, 224)
        else:
            dummy_input = torch.randn(1, 1024)
        
        # 前向傳播
        model.eval()
        with torch.no_grad():
            try:
                outputs = model(dummy_input)
                logger.info(f"模型前向傳播成功，輸出形狀: {outputs.shape}")
            except Exception as e:
                logger.error(f"模型前向傳播失敗: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
    except Exception as e:
        logger.error(f"創建模型時發生錯誤: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # 儲存測試結果
    test_results = {
        "test_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": config_path,
        "custom_classification_enabled": custom_classifier.enabled,
        "total_patients": total_patients,
        "total_classes": total_classes,
        "class_names": custom_classifier.get_all_classes(),
        "dataset_size": len(train_dataset.samples) if 'train_dataset' in locals() else 0,
        "class_counts": class_counts if 'class_counts' in locals() else {},
        "model_type": config['model']['type'],
        "model_num_classes": config['model']['parameters']['num_classes']
    }
    
    with open(os.path.join(test_output_dir, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"測試結果已儲存至: {os.path.join(test_output_dir, 'test_results.json')}")
    logger.info("測試完成")

if __name__ == "__main__":
    test_boss_classification() 