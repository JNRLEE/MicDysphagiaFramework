"""
回調接口模塊：定義與SBP_analyzer交互的回調接口

這個模塊定義了標準的回調接口，用於在模型訓練過程中的關鍵點插入分析和監控功能。
通過這個接口，SBP_analyzer可以在訓練過程中收集數據、監控模型和可視化結果。
"""

from typing import Dict, Any, List, Callable, Optional, Union
import torch
import torch.nn as nn

class CallbackInterface:
    """定義回調接口，用於集成 SBP_analyzer 的回調功能"""
    
    def on_train_begin(self, model: nn.Module, logs: Dict[str, Any] = None) -> None:
        """訓練開始時調用
        
        Args:
            model: 正在訓練的模型
            logs: 包含各種訓練相關資訊的字典
        """
        pass
        
    def on_epoch_begin(self, epoch: int, model: nn.Module, logs: Dict[str, Any] = None) -> None:
        """每個 epoch 開始時調用
        
        Args:
            epoch: 當前epoch索引
            model: 正在訓練的模型
            logs: 包含各種訓練相關資訊的字典
        """
        pass
    
    def on_batch_begin(self, batch: int, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, logs: Dict[str, Any] = None) -> None:
        """每個批次開始時調用
        
        Args:
            batch: 當前批次索引
            model: 正在訓練的模型
            inputs: 輸入數據批次
            targets: 目標數據批次
            logs: 包含各種訓練相關資訊的字典
        """
        pass
    
    def on_batch_end(self, batch: int, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, 
                    outputs: torch.Tensor, loss: torch.Tensor, logs: Dict[str, Any] = None) -> None:
        """每個批次結束時調用，可獲取損失和梯度等信息
        
        Args:
            batch: 當前批次索引
            model: 正在訓練的模型
            inputs: 輸入數據批次
            targets: 目標數據批次
            outputs: 模型輸出結果
            loss: 計算的損失值
            logs: 包含各種訓練相關資訊的字典
        """
        pass
    
    def on_epoch_end(self, epoch: int, model: nn.Module, train_logs: Dict[str, Any], 
                   val_logs: Dict[str, Any], logs: Dict[str, Any] = None) -> None:
        """每個 epoch 結束時調用，可獲取評估指標等信息
        
        Args:
            epoch: 當前epoch索引
            model: 正在訓練的模型
            train_logs: 訓練集上的指標和結果
            val_logs: 驗證集上的指標和結果
            logs: 包含各種訓練相關資訊的字典
        """
        pass
    
    def on_train_end(self, model: nn.Module, history: Dict[str, List], logs: Dict[str, Any] = None) -> None:
        """訓練結束時調用
        
        Args:
            model: 訓練完成的模型
            history: 訓練歷史記錄，包含損失和指標
            logs: 包含各種訓練相關資訊的字典
        """
        pass

    def on_evaluation_begin(self, model: nn.Module, logs: Dict[str, Any] = None) -> None:
        """評估開始時調用
        
        Args:
            model: 要評估的模型
            logs: 包含各種評估相關資訊的字典
        """
        pass

    def on_evaluation_end(self, model: nn.Module, results: Dict[str, Any], logs: Dict[str, Any] = None) -> None:
        """評估結束時調用
        
        Args:
            model: 評估的模型
            results: 評估結果
            logs: 包含各種評估相關資訊的字典
        """
        pass 

# 中文註解：這是callback_interface.py的Minimal Executable Unit，檢查CallbackInterface能否被正確繼承與調用，並測試未實作抽象方法時的報錯
if __name__ == "__main__":
    """
    Description: Minimal Executable Unit for callback_interface.py，檢查CallbackInterface能否被正確繼承與調用，並測試未實作抽象方法時的報錯。
    Args: None
    Returns: None
    References: 無
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    try:
        class MyCallback(CallbackInterface):
            def on_train_begin(self):
                print("訓練開始")
            def on_train_end(self):
                print("訓練結束")
        cb = MyCallback()
        cb.on_train_begin()
        cb.on_train_end()
        print("CallbackInterface繼承測試成功")
    except Exception as e:
        print(f"CallbackInterface繼承測試失敗: {e}")
    # 測試未實作抽象方法
    try:
        class BadCallback(CallbackInterface):
            pass
        bad_cb = BadCallback()
    except Exception as e:
        print(f"CallbackInterface未實作抽象方法時的報錯（預期行為）: {e}") 