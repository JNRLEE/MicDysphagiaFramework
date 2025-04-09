"""
測試不同損失函數在訓練過程中的具體行為
此腳本專門用於檢查損失函數的適配性和可能的錯誤

Args:
    config_path: 基本配置文件路徑
    loss_type: 要測試的損失函數類型
    
Description:
    在訓練或評估過程的關鍵點插入檢查數據，確認不同損失函數能正確處理特定的數據結構
    驗證梯度計算是否正確進行
    記錄並分析可能的輸入/輸出不匹配問題
    
References:
    無
"""

import os
import sys
import torch
import logging
import argparse
from typing import Dict, Any, Tuple, Optional
import traceback

# 添加項目根目錄到路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 導入必要的模組
from utils.config_loader import load_config
from data.dataset_factory import create_dataloaders
from models.model_factory import create_model
from losses.loss_factory import LossFactory
from trainers.pytorch_trainer import PyTorchTrainer

# 設置日誌
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='損失函數行為測試工具')
    
    parser.add_argument('--config', type=str, required=True, help='基本配置文件路徑')
    parser.add_argument('--loss_type', type=str, required=True, 
                        help='要測試的損失函數類型，例如MSELoss, CrossEntropyLoss, PairwiseRankingLoss等')
    parser.add_argument('--params', type=str, default='{}', 
                        help='損失函數參數，JSON格式，例如{"margin": 0.3}')
    parser.add_argument('--output_dir', type=str, default='results/loss_debug', help='輸出目錄')
    parser.add_argument('--device', type=str, default='auto', help='使用的設備，例如cuda:0或cpu')
    
    return parser.parse_args()


def get_single_batch(data_loader) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """從數據加載器中提取一個批次並返回
    
    Args:
        data_loader: 數據加載器
        
    Returns:
        Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: 批次數據和相關信息
    """
    # 提取一個批次
    for batch in data_loader:
        # 返回數據和日誌信息
        batch_info = {
            'batch_size': len(batch['label']) if 'label' in batch else (
                          len(batch['score']) if 'score' in batch else 'unknown'),
            'keys': list(batch.keys()),
            'device': next(iter(batch.values())).device if len(batch) > 0 else 'unknown',
            'dtypes': {k: v.dtype if isinstance(v, torch.Tensor) else type(v) for k, v in batch.items()}
        }
        return batch, batch_info


class LossTester:
    """損失函數測試類"""
    
    def __init__(self, config: Dict[str, Any], loss_type: str, loss_params: Dict[str, Any]):
        """初始化損失函數測試器
        
        Args:
            config: 配置字典
            loss_type: 損失函數類型
            loss_params: 損失函數參數
        """
        self.config = config
        self.loss_type = loss_type
        self.loss_params = loss_params
        
        # 設置設備
        device_str = config.get('global', {}).get('device', 'auto')
        if device_str == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_str)
        
        logger.info(f"使用設備: {self.device}")
        
        # 創建損失函數
        self.loss_config = {
            'type': self.loss_type,
            'parameters': self.loss_params
        }
        
        try:
            self.loss_fn = LossFactory.get_loss(self.loss_config)
            logger.info(f"成功創建損失函數: {self.loss_type}")
        except Exception as e:
            logger.error(f"創建損失函數時出錯: {str(e)}")
            raise
    
    def test_loss_with_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """使用數據批次測試損失函數
        
        Args:
            batch: 數據批次
            
        Returns:
            Dict[str, Any]: 測試結果
        """
        # 移動到設備上
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # 創建模型
        model = create_model(self.config)
        model.to(self.device)
        
        # 驗證任務類型
        is_classification = self.config.get('model', {}).get('parameters', {}).get('is_classification', True)
        logger.info(f"任務類型: {'分類' if is_classification else '回歸'}")
        
        # 獲取標籤
        if 'label' in batch:
            targets = batch['label']
            logger.info(f"使用'label'作為目標，形狀: {targets.shape}, 類型: {targets.dtype}")
        elif 'score' in batch:
            targets = batch['score']
            logger.info(f"使用'score'作為目標，形狀: {targets.shape}, 類型: {targets.dtype}")
        else:
            raise ValueError("批次中找不到'label'或'score'字段")
        
        # 確保標籤合適的數據類型
        if is_classification:
            targets = targets.long()
        
        # 打印標籤信息
        logger.info(f"標籤值範圍: [{targets.min().item()}, {targets.max().item()}]")
        logger.info(f"標籤唯一值: {torch.unique(targets).tolist()}")
        
        # 獲取輸入數據
        if 'audio' in batch:
            inputs = batch['audio']
            input_type = 'audio'
        elif 'image' in batch:
            inputs = batch['image']
            input_type = 'image'
        elif 'features' in batch:
            inputs = batch['features']
            input_type = 'features'
        else:
            raise ValueError("批次中找不到有效的輸入數據")
        
        logger.info(f"輸入類型: {input_type}, 形狀: {inputs.shape}")
        
        # 前向傳播
        try:
            with torch.no_grad():
                outputs = model(inputs)
            logger.info(f"模型輸出形狀: {outputs.shape}")
        except Exception as e:
            logger.error(f"模型前向傳播時出錯: {str(e)}")
            traceback.print_exc()
            raise
        
        # 計算損失 - 無梯度
        try:
            with torch.no_grad():
                loss_no_grad = self.loss_fn(outputs, targets)
            logger.info(f"無梯度損失計算結果: {loss_no_grad.item()}")
        except Exception as e:
            logger.error(f"無梯度損失計算時出錯: {str(e)}")
            logger.error(f"輸出形狀: {outputs.shape}, 標籤形狀: {targets.shape}")
            traceback.print_exc()
            return {"success": False, "error": str(e), "stage": "loss_no_grad"}
        
        # 測試梯度計算
        try:
            # 重設模型以用於梯度計算
            model.zero_grad()
            
            # 啟用梯度計算的前向傳播
            outputs = model(inputs)
            
            # 計算損失並反向傳播
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            
            # 檢查梯度是否成功計算
            grad_norms = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norms.append((name, param.grad.norm().item()))
            
            if grad_norms:
                logger.info(f"反向傳播成功，梯度計算正常")
                for name, norm in grad_norms[:5]:  # 只顯示前5個
                    logger.info(f"  - {name}: 梯度範數 = {norm}")
                if len(grad_norms) > 5:
                    logger.info(f"  - 還有 {len(grad_norms) - 5} 個參數有梯度...")
            else:
                logger.warning("反向傳播完成，但沒有參數具有梯度")
            
            return {
                "success": True, 
                "loss_value": loss.item(),
                "has_gradients": len(grad_norms) > 0,
                "output_shape": list(outputs.shape),
                "target_shape": list(targets.shape)
            }
            
        except Exception as e:
            logger.error(f"梯度計算時出錯: {str(e)}")
            traceback.print_exc()
            return {"success": False, "error": str(e), "stage": "gradient_computation"}


def main():
    """主函數"""
    # 解析命令行參數
    args = parse_args()
    
    # 加載配置
    config_loader = load_config(args.config)
    config = config_loader.config
    
    # 解析損失函數參數
    import json
    try:
        loss_params = json.loads(args.params)
    except json.JSONDecodeError:
        logger.error(f"無法解析損失函數參數: {args.params}")
        loss_params = {}
    
    # 創建數據加載器
    logger.info("創建數據加載器...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # 獲取一個批次用於測試
    logger.info("提取測試批次...")
    try:
        batch, batch_info = get_single_batch(train_loader)
        logger.info(f"成功提取批次: {batch_info}")
    except Exception as e:
        logger.error(f"提取批次時出錯: {str(e)}")
        traceback.print_exc()
        return
    
    # 測試損失函數
    logger.info(f"測試損失函數: {args.loss_type}")
    tester = LossTester(config, args.loss_type, loss_params)
    
    # 執行測試
    result = tester.test_loss_with_batch(batch)
    
    # 打印結果
    if result.get("success", False):
        logger.info("損失函數測試成功!")
        logger.info(f"損失值: {result.get('loss_value')}")
        logger.info(f"是否有梯度: {result.get('has_gradients')}")
        logger.info(f"輸出形狀: {result.get('output_shape')}")
        logger.info(f"標籤形狀: {result.get('target_shape')}")
    else:
        logger.error(f"損失函數測試失敗: {result.get('error')}")
        logger.error(f"失敗階段: {result.get('stage')}")
    
    logger.info("測試完成")


if __name__ == "__main__":
    main() 