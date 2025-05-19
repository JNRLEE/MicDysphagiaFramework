"""
實驗執行腳本：執行基於YAML配置的實驗

此腳本負責解析命令行參數，加載並驗證YAML配置文件，
然後設置數據集、模型和訓練器，並執行實驗。
"""

import argparse
import logging
import os
import sys
import yaml
import torch
import json
from datetime import datetime
from typing import Dict, Any

# 確保當前目錄在系統路徑中，以便導入自定義模塊
# 將項目根目錄添加到系統路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)  # 使用insert(0)確保它是最優先檢查的路徑

# 導入必要的模塊
from data import dataset_factory
from models import model_factory
from trainers import trainer_factory
from models.hook_bridge import get_analyzer_callbacks_from_config  # 新增：導入獲取分析器回調的函數
from utils.save_manager import SaveManager  # 新增：導入 SaveManager

# 設置日誌記錄
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='運行吞嚥聲音分析實驗')
    parser.add_argument('--config', required=True, help='YAML 配置文件路徑')
    parser.add_argument('--device', default=None, help='設備 (例如 cuda:0, cpu)')
    parser.add_argument('--output_dir', default=None, help='輸出目錄')
    parser.add_argument('--debug', action='store_true', help='啟用調試模式')
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """加載並驗證 YAML 配置文件
    
    Args:
        config_path: YAML 配置文件路徑
        
    Returns:
        Dict[str, Any]: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # TODO: 添加配置驗證邏輯
    
    return config

def setup_experiment(config: Dict[str, Any], args) -> Dict[str, Any]:
    """設置實驗環境並應用命令行覆蓋
    
    Args:
        config: 從 YAML 文件加載的配置
        args: 命令行參數
        
    Returns:
        Dict[str, Any]: 修改後的配置
    """
    # 應用命令行參數覆蓋
    if args.device:
        config['global']['device'] = args.device
    
    if args.output_dir:
        config['global']['output_dir'] = args.output_dir
    
    if args.debug:
        config['global']['debug'] = True
    
    # 設置輸出目錄
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config['global']['output_dir'] = os.path.join(
        config['global'].get('output_dir', 'results'),
        f"{config['global']['experiment_name']}_{timestamp}"
    )
    
    # 設置設備
    if config['global'].get('device', 'auto') == 'auto':
        config['global']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 設置種子以確保可重現性
    if 'seed' in config['global']:
        seed = config['global']['seed']
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    # 創建實驗目錄並保存配置
    os.makedirs(config['global']['output_dir'], exist_ok=True)
    save_manager = SaveManager(config['global']['output_dir'])
    save_manager.save_config(config)
    
    return config

def run_experiment(config: Dict[str, Any]):
    """執行實驗
    
    Args:
        config: 實驗配置
    """
    logger.info(f"開始實驗: {config['global']['experiment_name']}")
    logger.info(f"使用設備: {config['global']['device']}")
    
    # 創建 SaveManager 實例
    save_manager = SaveManager(config['global']['output_dir'])
    
    # 創建數據集和數據加載器
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = (
        dataset_factory.create_datasets_and_loaders(config)
    )
    logger.info(f"數據集已創建，訓練集大小: {len(train_dataset) if train_dataset else 0}, 驗證集大小: {len(val_dataset) if val_dataset else 0}, 測試集大小: {len(test_dataset) if test_dataset else 0}")
    
    # 保存資料集詳細資訊
    logger.info("保存資料集詳細資訊...")
    datasets_to_save = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
    for name, ds in datasets_to_save.items():
        if ds is not None:
            file_paths = [] # 重新初始化 file_paths
            try:
                # 從不同類型的資料集中獲取檔案路徑
                if hasattr(ds, 'samples') and isinstance(ds.samples, list):
                    for sample in ds.samples:
                        if isinstance(sample, dict):
                            for key in ['file_path', 'path', 'audio_path', 'feature_path', 'image_path', 'spectrogram_path']:
                                if key in sample and sample[key]:
                                    file_paths.append(sample[key])
                                    break
                elif hasattr(ds, 'dataset') and hasattr(ds, 'indices'): # 處理 Subset
                    original_dataset = ds.dataset
                    indices = ds.indices
                    if hasattr(original_dataset, 'samples') and isinstance(original_dataset.samples, list):
                        for idx in indices:
                            if idx < len(original_dataset.samples):
                                sample = original_dataset.samples[idx]
                                if isinstance(sample, dict):
                                    for key in ['file_path', 'path', 'audio_path', 'feature_path', 'image_path', 'spectrogram_path']:
                                        if key in sample and sample[key]:
                                            file_paths.append(sample[key])
                                            break
                save_manager.save_dataset_info(name, ds, config, file_paths)
            except Exception as e:
                logger.error(f"保存 {name} 資料集資訊時出錯: {e}", exc_info=True)
        else:
            logger.info(f"資料集 '{name}' 未提供，跳過保存其資訊。")

    # 保存資料集綜合統計資訊 (僅當所有數據集都存在時)
    if train_dataset and val_dataset and test_dataset:
        try:
            save_manager.save_datasets_statistics(train_dataset, val_dataset, test_dataset, config)
            logger.info("資料集綜合統計資訊保存完成")
        except Exception as e:
            logger.error(f"保存資料集綜合統計資訊時出錯: {e}", exc_info=True)
    else:
        logger.warning("一個或多個數據集未提供，跳過保存資料集綜合統計資訊。")
    
    # 從數據集中獲取類別數量，並更新模型配置
    num_classes_source = train_dataset if train_dataset else (val_dataset if val_dataset else test_dataset)
    if num_classes_source and hasattr(num_classes_source, 'num_classes') and num_classes_source.num_classes is not None:
        num_classes = num_classes_source.num_classes
        logger.info(f"從數據集 ('{num_classes_source.__class__.__name__}') 獲取類別數量: {num_classes}")
        
        if 'model' in config and 'parameters' in config['model']:
            config['model']['parameters']['num_classes'] = num_classes
            logger.info(f"更新模型配置的類別數量為: {num_classes}")
    else:
        logger.warning("無法從任何數據集獲取類別數量，模型配置中的 num_classes 將保持不變或依賴默認值。")
    
    model = model_factory.create_model(config)
    logger.info(f"模型已創建，類型: {config['model']['type']}")
    
    trainer = trainer_factory.create_trainer(config, model, (train_loader, val_loader, test_loader))
    logger.info("訓練器已創建")
    
    save_every = config.get('training', {}).get('save_every', 0)
    if 'hooks' in config and 'activation_capture' in config['hooks'] and 'save_frequency' in config['hooks']['activation_capture']:
        logger.info(f"覆蓋hooks.activation_capture.save_frequency={config['hooks']['activation_capture']['save_frequency']}為training.save_every={save_every}")
        config['hooks']['activation_capture']['save_frequency'] = save_every
    
    analyzer_callbacks = get_analyzer_callbacks_from_config(config)
    if analyzer_callbacks:
        trainer.add_callbacks(analyzer_callbacks)
        logger.info(f"已添加 {len(analyzer_callbacks)} 個分析器回調")
    
    # 檢查是否已經有 EvaluationResultsHook 的實例 (通過比較類名字符串)
    # 確保在比較 cb.dataset_name 之前檢查 cb 是否有 dataset_name 屬性
    has_eval_hook_test = any(
        cb.__class__.__name__ == "EvaluationResultsHook" and 
        hasattr(cb, 'dataset_name') and 
        cb.dataset_name == 'test' 
        for cb in trainer.callbacks
    )
    has_eval_hook_val = any(
        cb.__class__.__name__ == "EvaluationResultsHook" and 
        hasattr(cb, 'dataset_name') and 
        cb.dataset_name == 'val' 
        for cb in trainer.callbacks
    )

    # if not has_eval_hook_test:
    #     logger.info("未找到測試集評估結果捕獲回調，手動添加 EvaluationResultsHook(dataset_name='test')")
    #     from models.hook_bridge import EvaluationResultsHook # 導入仍然需要，以便創建新實例
    #     trainer.add_callback(EvaluationResultsHook(save_manager, 'test'))
    # if not has_eval_hook_val:
    #     logger.info("未找到驗證集評估結果捕獲回調，手動添加 EvaluationResultsHook(dataset_name='val')")
    #     from models.hook_bridge import EvaluationResultsHook # 再次導入以確保作用域
    #     trainer.add_callback(EvaluationResultsHook(save_manager, 'val'))

    trainer.set_eval_epoch_tracking(True)
    logger.info("啟用評估epoch追蹤，以支持特定epoch的特徵向量捕獲和其他功能")
    
    train_begin_logs = {'config': config}
    for callback in trainer.callbacks:
        if hasattr(callback, 'on_train_begin'):
            callback.on_train_begin(trainer.model, train_begin_logs)
    
    results = trainer.train(train_loader, val_loader, test_loader)
    
    # 保存最終實驗結果到 results/results.json
    final_results_dir = os.path.join(config['global']['output_dir'], 'results')
    os.makedirs(final_results_dir, exist_ok=True)
    results_json_path = os.path.join(final_results_dir, 'results.json')
    try:
        # 創建一個可序列化的結果副本
        serializable_results = {}
        for k, v in results.items():
            if isinstance(v, (list, dict, str, int, float, bool)):
                serializable_results[k] = v
            elif isinstance(v, torch.Tensor):
                serializable_results[k] = v.tolist() # 將Tensor轉為列表
            else:
                serializable_results[k] = str(v) # 其他類型轉為字符串
        
        with open(results_json_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        logger.info(f"最終實驗結果已保存到: {results_json_path}")
    except Exception as e:
        logger.error(f"保存最終實驗結果到 {results_json_path} 時出錯: {e}", exc_info=True)
    
    # 複製 evaluation_results_test.pt (如果存在) 到根目錄的 test_predictions.pt
    # EvaluationResultsHook 已經處理了這個邏輯，此處不再需要重複
    # logger.info(f"實驗完成，結果摘要已保存到 {results_json_path}")
    
    return results

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    config = setup_experiment(config, args)
    run_experiment(config) 