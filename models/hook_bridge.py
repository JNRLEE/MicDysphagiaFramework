"""
模型鉤子橋接模塊：提供與外部分析工具交互的簡化橋接接口

該模塊實現了數據抽取和存儲功能，允許 MicDysphagiaFramework 在不依賴外部工具的情況下
捕獲中間層激活值和梯度，並將其保存在結構化的目錄中，以便後續使用 SBP_analyzer 進行離線分析。
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable, Union, Set, Tuple
import importlib
import logging
import warnings
import os
import numpy as np
from datetime import datetime
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import json

# 導入回調接口和存檔管理器
from utils.callback_interface import CallbackInterface
from utils.save_manager import SaveManager

logger = logging.getLogger(__name__)

# 設置更詳細的日誌級別，方便調試
logger.setLevel(logging.DEBUG)

def is_sbp_analyzer_available() -> bool:
    """檢查 SBP_analyzer 是否可用
    
    Returns:
        bool: 如果 SBP_analyzer 已安裝且可用，則返回 True
    """
    try:
        importlib.import_module('sbp_analyzer')
        return True
    except ImportError:
        return False

class SimpleActivationHook:
    """簡易激活值鉤子，用於捕獲並存儲模型中間層的激活值
    
    無需依賴外部庫，直接存儲數據到指定目錄
    """
    
    def __init__(self, model: nn.Module, layer_names: Optional[List[str]] = None, save_manager: Optional[SaveManager] = None):
        """初始化激活值鉤子
        
        Args:
            model: 要監控的 PyTorch 模型
            layer_names: 要監控的層名稱列表，如果為 None 則嘗試監控所有命名層
            save_manager: 存檔管理器，如果為 None 則不保存到文件
        """
        self.model = model
        self.save_manager = save_manager
        self.activations = {}
        self.hooks = []
        
        # 如果沒有指定層名稱，就嘗試獲取所有命名層
        if layer_names is None:
            layer_names = []
            for name, _ in model.named_modules():
                if name:  # 排除空名稱
                    layer_names.append(name)
        
        # 註冊鉤子
        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(
                    lambda mod, inp, out, name=name: self._hook_fn(name, out)
                )
                self.hooks.append(hook)
        
        logger.info(f"已註冊 {len(self.hooks)} 個激活值鉤子，監控層: {layer_names}")
    
    def _hook_fn(self, name: str, output):
        """鉤子回調函數，保存激活值
        
        Args:
            name: 層名稱
            output: 層輸出
        """
        # 確保輸出是 Tensor
        if isinstance(output, tuple):
            output = output[0]
        
        # 存儲激活值的副本
        self.activations[name] = output.detach().cpu()
        logger.debug(f"捕獲層 '{name}' 的激活值，形狀: {output.shape}")
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """獲取當前保存的激活值
        
        Returns:
            Dict[str, torch.Tensor]: 層名到激活值的映射
        """
        return self.activations
    
    def save_activations(self, epoch: int = 0, batch_idx: Optional[int] = None):
        """保存激活值到文件
        
        Args:
            epoch: 當前 epoch
            batch_idx: 當前批次索引，如果為 None 則使用 'all' 表示聚合結果
        """
        if not self.save_manager:
            logger.warning("未設置存檔管理器，無法保存激活值")
            return
        
        # 使用 SaveManager 保存激活值
        self.save_manager.save_activations(self.activations, epoch, batch_idx)
    
    def clear_activations(self):
        """清除保存的激活值"""
        self.activations = {}
    
    def remove_hooks(self):
        """移除所有已註冊的鉤子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        logger.info("已移除所有激活值鉤子")
    
    def __del__(self):
        """析構時移除鉤子"""
        self.remove_hooks()

class SimpleGradientHook:
    """簡易梯度鉤子，用於捕獲並存儲模型參數的梯度
    
    無需依賴外部庫，直接存儲數據到指定目錄
    """
    
    def __init__(self, model: nn.Module, param_names: Optional[List[str]] = None, save_manager: Optional[SaveManager] = None):
        """初始化梯度鉤子
    
    Args:
        model: 要監控的 PyTorch 模型
            param_names: 要監控的參數名稱列表，如果為 None 則監控所有參數
            save_manager: 存檔管理器，如果為 None 則不保存到文件
        """
        self.model = model
        self.save_manager = save_manager
        self.gradients = {}
        self.hooks = []
        
        # 如果沒有指定參數名稱，就監控所有參數
        if param_names is None:
            param_names = []
            for name, _ in model.named_parameters():
                if name:  # 排除空名稱
                    param_names.append(name)
        
        # 註冊鉤子
        for name, param in model.named_parameters():
            if name in param_names and param.requires_grad:
                hook = param.register_hook(
                    lambda grad, name=name: self._hook_fn(name, grad)
                )
                self.hooks.append((name, hook))
        
        logger.info(f"已註冊 {len(self.hooks)} 個梯度鉤子，監控參數: {param_names}")
    
    def _hook_fn(self, name: str, grad: torch.Tensor):
        """鉤子回調函數，保存梯度
        
        Args:
            name: 參數名稱
            grad: 梯度
        """
        self.gradients[name] = grad.detach().cpu().clone()
        logger.debug(f"捕獲參數 '{name}' 的梯度，形狀: {grad.shape}")
        return grad  # 不改變梯度
    
    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """獲取當前保存的梯度
        
        Returns:
            Dict[str, torch.Tensor]: 參數名到梯度的映射
        """
        return self.gradients
    
    def get_gradient_statistics(self) -> Dict[str, Dict[str, float]]:
        """獲取梯度的統計數據
        
    Returns:
            Dict[str, Dict[str, float]]: 參數名到統計數據的映射
        """
        stats = {}
        for name, grad in self.gradients.items():
            if grad is not None:
                grad_np = grad.numpy()
                stats[name] = {
                    'mean': float(np.mean(grad_np)),
                    'std': float(np.std(grad_np)),
                    'min': float(np.min(grad_np)),
                    'max': float(np.max(grad_np)),
                    'norm': float(torch.norm(grad).item())
                }
        return stats
    
    def save_gradients(self, epoch: int = 0, batch_idx: Optional[int] = None):
        """保存梯度到文件
        
        Args:
            epoch: 當前 epoch
            batch_idx: 當前批次索引，如果為 None 則使用 'all' 表示聚合結果
        """
        if not self.save_manager:
            logger.warning("未設置存檔管理器，無法保存梯度")
            return
        
        # 使用 SaveManager 保存梯度
        self.save_manager.save_gradients(self.gradients, epoch, batch_idx)
    
    def clear_gradients(self):
        """清除保存的梯度"""
        self.gradients = {}
    
    def remove_hooks(self):
        """移除所有已註冊的鉤子"""
        for _, hook in self.hooks:
            hook.remove()
        self.hooks = []
        logger.info("已移除所有梯度鉤子")
    
    def __del__(self):
        """析構時移除鉤子"""
        self.remove_hooks()

class SimpleModelHookManager:
    """簡易模型鉤子管理器，同時管理激活值和梯度鉤子
    
    無需依賴外部庫，直接存儲數據到指定目錄
    """
    
    def __init__(self, model: nn.Module,
                 monitored_layers: Optional[List[str]] = None,
                 monitored_params: Optional[List[str]] = None,
                 output_dir: Optional[str] = None,
                 save_manager: Optional[SaveManager] = None):
        """初始化模型鉤子管理器
        
        Args:
            model: 要監控的 PyTorch 模型
            monitored_layers: 要監控激活值的層名稱列表
            monitored_params: 要監控梯度的參數名稱列表
            output_dir: 輸出目錄（如果提供save_manager則不使用）
            save_manager: 已存在的SaveManager實例，如果提供則優先使用
        """
        self.model = model
        self.output_dir = output_dir
        
        # 如果提供了save_manager，直接使用；否則根據output_dir創建
        if save_manager:
            self.save_manager = save_manager
            if output_dir:
                logger.info(f"使用提供的SaveManager，忽略output_dir參數: {output_dir}")
        elif output_dir:
            self.save_manager = SaveManager(output_dir)
            logger.info(f"創建新的SaveManager，輸出目錄: {output_dir}")
        else:
            self.save_manager = None
            logger.warning("未提供save_manager或output_dir，將無法保存數據")
        
        # 創建激活值和梯度鉤子
        self.activation_hook = SimpleActivationHook(model, monitored_layers, self.save_manager)
        self.gradient_hook = SimpleGradientHook(model, monitored_params, self.save_manager)
        
        # 記錄當前批次數據
        self.current_epoch = 0
        self.current_batch = 0
        self.inputs = None
        self.outputs = None
        self.targets = None
        self.loss = None
        
        logger.info(f"模型鉤子管理器已初始化")
    
    def update_batch_data(self, 
                         inputs: Optional[torch.Tensor] = None,
                         outputs: Optional[torch.Tensor] = None,
                         targets: Optional[torch.Tensor] = None,
                         loss: Optional[torch.Tensor] = None,
                         epoch: int = None,
                         batch_idx: int = None):
        """更新當前批次數據
        
        Args:
            inputs: 輸入數據
            outputs: 模型輸出
            targets: 目標數據
            loss: 損失值
            epoch: 當前 epoch
            batch_idx: 當前批次索引
        """
        if inputs is not None:
            self.inputs = inputs.detach().cpu() if isinstance(inputs, torch.Tensor) else inputs
        if outputs is not None:
            self.outputs = outputs.detach().cpu() if isinstance(outputs, torch.Tensor) else outputs
        if targets is not None:
            self.targets = targets.detach().cpu() if isinstance(targets, torch.Tensor) else targets
        if loss is not None:
            self.loss = loss.detach().cpu() if isinstance(loss, torch.Tensor) else loss
        if epoch is not None:
            self.current_epoch = epoch
        if batch_idx is not None:
            self.current_batch = batch_idx
            
        logger.debug(f"更新批次數據: epoch={self.current_epoch}, batch={self.current_batch}, loss={self.loss}")
    
    def save_current_data(self):
        """保存當前數據到文件"""
        if not self.save_manager:
            logger.warning("未設置存檔管理器，無法保存數據")
            return
        
        # 保存激活值和梯度
        self.activation_hook.save_activations(self.current_epoch, self.current_batch)
        self.gradient_hook.save_gradients(self.current_epoch, self.current_batch)
        
        # 保存批次數據
        batch_data = {
            'epoch': self.current_epoch,
            'batch': self.current_batch,
            'timestamp': datetime.now().isoformat()
        }
        
        # 只保存非 None 的數據
        if self.loss is not None:
            batch_data['loss'] = self.loss
        
        # 輸入/輸出/目標可能很大，只保存統計信息或小樣本
        if self.outputs is not None and isinstance(self.outputs, torch.Tensor):
            batch_data['outputs_stats'] = {
                'shape': list(self.outputs.shape),
                'mean': float(torch.mean(self.outputs).item()),
                'std': float(torch.std(self.outputs).item()),
                'min': float(torch.min(self.outputs).item()),
                'max': float(torch.max(self.outputs).item()),
            }
            # 保存小樣本（最多 10 個樣本）
            max_samples = min(10, self.outputs.shape[0])
            batch_data['outputs_sample'] = self.outputs[:max_samples].numpy().tolist()
        
        if self.targets is not None and isinstance(self.targets, torch.Tensor):
            batch_data['targets_stats'] = {
                'shape': list(self.targets.shape),
                'mean': float(torch.mean(self.targets).item()) if self.targets.dtype.is_floating_point else None,
                'std': float(torch.std(self.targets).item()) if self.targets.dtype.is_floating_point else None,
                'min': float(torch.min(self.targets).item()),
                'max': float(torch.max(self.targets).item()),
            }
            # 保存小樣本（最多 10 個樣本）
            max_samples = min(10, self.targets.shape[0])
            batch_data['targets_sample'] = self.targets[:max_samples].numpy().tolist()
        
        # 使用 SaveManager 保存批次數據
        self.save_manager.save_batch_data(batch_data, self.current_epoch, self.current_batch)
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """獲取層激活值"""
        return self.activation_hook.get_activations()
    
    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """獲取參數梯度"""
        return self.gradient_hook.get_gradients()
    
    def get_gradient_statistics(self) -> Dict[str, Dict[str, float]]:
        """獲取梯度統計數據"""
        return self.gradient_hook.get_gradient_statistics()
    
    def clear(self):
        """清除所有保存的數據"""
        self.activation_hook.clear_activations()
        self.gradient_hook.clear_gradients()
        self.inputs = None
        self.outputs = None
        self.targets = None
        self.loss = None
    
    def remove_hooks(self):
        """移除所有已註冊的鉤子"""
        self.activation_hook.remove_hooks()
        self.gradient_hook.remove_hooks()
    
    def __del__(self):
        """析構時移除鉤子"""
        self.remove_hooks()

# 以下是用於與 trainers 模組銜接的函數 ---------------------

class SimpleModelAnalyticsCallback(CallbackInterface):
    """簡易模型分析回調，用於在訓練過程中收集並保存模型數據
    
    該類實現了 CallbackInterface 接口定義的所有方法，能夠在訓練過程中
    捕獲中間層激活值和梯度，支持離線分析和可視化。無需依賴外部庫，直接存儲數據到指定目錄。
    """
    
    def __init__(self, model: nn.Module = None,
                output_dir: str = 'results',
                monitored_layers: Optional[List[str]] = None,
                monitored_params: Optional[List[str]] = None,
                save_frequency: int = 1):
        """初始化模型分析回調
        
        Args:
            model: 要監控的 PyTorch 模型 (可在 on_train_begin 中設置)
            output_dir: 輸出目錄
            monitored_layers: 要監控激活值的層名稱列表
            monitored_params: 要監控梯度的參數名稱列表
            save_frequency: 保存頻率（每 N 個 epoch）
        """
        self.model = model
        self.output_dir = output_dir
        self.monitored_layers = monitored_layers
        self.monitored_params = monitored_params
        self.save_frequency = save_frequency
        self.hook_manager = None
        self.save_manager = None
        self.current_epoch = 0
        
        logger.info(f"初始化模型分析回調，輸出目錄: {output_dir}, 保存頻率: {save_frequency} epoch")
        if monitored_layers:
            logger.info(f"監控層: {monitored_layers}")
        if monitored_params:
            logger.info(f"監控參數: {monitored_params}")
    
    def on_train_begin(self, model: nn.Module, logs: Dict[str, Any] = None) -> None:
        """訓練開始時的回調
        
        Args:
            model: 訓練的模型
            logs: 日誌字典
        """
        if self.model is None:
            self.model = model
        
        # 如果輸出目錄是相對路徑，嘗試從日誌中獲取基本輸出目錄
        if logs and 'tensorboard_writer' in logs:
            tb_writer = logs['tensorboard_writer']
            if hasattr(tb_writer, 'log_dir'):
                # 使用 TensorBoard 日誌目錄所在的父目錄（即實驗根目錄）
                # 注意：TensorBoard 日誌目錄通常是 實驗根目錄/tensorboard_logs
                parent_dir = os.path.dirname(os.path.dirname(tb_writer.log_dir))
                self.output_dir = parent_dir
                logger.info(f"從TensorBoard日誌獲取輸出目錄 (實驗根目錄): {self.output_dir}")
        
        # 創建存檔管理器
        self.save_manager = SaveManager(self.output_dir)
        
        # 創建鉤子管理器，不再傳遞output_dir參數，而是傳遞save_manager
        self.hook_manager = SimpleModelHookManager(
            self.model,
            self.monitored_layers,
            self.monitored_params,
            save_manager=self.save_manager
        )
        logger.info(f"模型分析回調已初始化，輸出目錄: {self.output_dir}")
    
    def on_epoch_begin(self, epoch: int, model: nn.Module, logs: Dict[str, Any] = None) -> None:
        """epoch 開始時的回調
        
        Args:
            epoch: 當前 epoch 編號
            model: 訓練的模型
            logs: 日誌字典
        """
        self.current_epoch = epoch
        if self.hook_manager:
            self.hook_manager.current_epoch = epoch
            logger.info(f"開始 epoch {epoch}")
    
    def on_batch_begin(self, batch: int, model: nn.Module, inputs: torch.Tensor = None, 
                     targets: torch.Tensor = None, logs: Dict[str, Any] = None) -> None:
        """批次開始時的回調
        
        Args:
            batch: 當前批次編號
            model: 訓練的模型
            inputs: 輸入數據
            targets: 目標數據
            logs: 日誌字典
        """
        if self.hook_manager:
            logger.debug(f"開始批次 {batch} 處理 in epoch {self.current_epoch}")
    
    def on_batch_end(self, batch: int, model: nn.Module, inputs: torch.Tensor = None, 
                   targets: torch.Tensor = None, outputs: torch.Tensor = None, 
                   loss: torch.Tensor = None, logs: Dict[str, Any] = None) -> None:
        """批次結束時的回調
        
        Args:
            batch: 當前批次編號
            model: 訓練的模型
            inputs: 輸入數據
            targets: 目標數據
            outputs: 模型輸出
            loss: 損失值
            logs: 日誌字典
        """
        if self.hook_manager:
            # 更新當前批次數據
            self.hook_manager.update_batch_data(
                inputs=inputs,
                outputs=outputs,
                targets=targets,
                loss=loss,
                epoch=self.current_epoch,
                batch_idx=batch
            )
            
            # 每個 epoch 的最後一個批次或指定間隔保存數據
            save_this_batch = False
            if logs and 'is_last_batch' in logs and logs['is_last_batch']:
                save_this_batch = True
                logger.info(f"Epoch {self.current_epoch}, 批次 {batch} 是最後一個批次，保存數據")
            elif batch % 100 == 0:  # 每 100 個批次保存一次
                save_this_batch = True
                logger.info(f"Epoch {self.current_epoch}, 批次 {batch} 是 100 的倍數，保存數據")
            elif batch == 0:  # 確保至少保存第一個批次
                save_this_batch = True
                logger.info(f"Epoch {self.current_epoch}, 批次 {batch} 是第一個批次，保存數據")
            
            if save_this_batch:
                logger.info(f"保存 epoch {self.current_epoch}, 批次 {batch} 的數據")
                self.hook_manager.save_current_data()
                self.hook_manager.clear()  # 清除已保存的數據
    
    def on_epoch_end(self, epoch: int, model: nn.Module, train_logs: Dict[str, Any] = None, 
                   val_logs: Dict[str, Any] = None, logs: Dict[str, Any] = None) -> None:
        """epoch 結束時的回調
        
        Args:
            epoch: 當前 epoch 編號
            model: 訓練的模型
            train_logs: 訓練日誌
            val_logs: 驗證日誌
            logs: 綜合日誌
        """
        logger.info(f"結束 epoch {epoch}")
        
        # 保存驗證集預測結果 (如果存在)
        validation_outputs = None
        validation_targets = None
        validation_predictions = None
        
        # 直接從 val_logs 中查找
        if val_logs:
            if 'outputs' in val_logs and 'targets' in val_logs:
                validation_outputs = val_logs['outputs']
                validation_targets = val_logs['targets'] 
                validation_predictions = val_logs.get('predictions')
            # 從 val_logs['metrics'] 中查找 (PyTorchTrainer 中使用的嵌套結構)
            elif 'metrics' in val_logs and isinstance(val_logs['metrics'], dict):
                metrics = val_logs['metrics']
                if 'outputs' in metrics and 'targets' in metrics:
                    validation_outputs = metrics['outputs']
                    validation_targets = metrics['targets']
                    validation_predictions = metrics.get('predictions')
        
        if validation_outputs is not None and validation_targets is not None:
            try:
                validation_results = {
                    'outputs': validation_outputs.detach().cpu() if isinstance(validation_outputs, torch.Tensor) else validation_outputs,
                    'targets': validation_targets.detach().cpu() if isinstance(validation_targets, torch.Tensor) else validation_targets,
                }
                
                if validation_predictions is not None:
                    validation_results['predictions'] = validation_predictions.detach().cpu() if isinstance(validation_predictions, torch.Tensor) else validation_predictions
                
                # 保存到標準位置
                hooks_dir = self.save_manager.get_path('hooks', '')
                os.makedirs(hooks_dir, exist_ok=True)
                
                save_path = os.path.join(hooks_dir, f'epoch_{epoch}_validation_predictions.pt')
                torch.save(validation_results, save_path)
                logger.info(f"Epoch {epoch} 驗證集預測結果已保存到: {save_path}")
            except Exception as e:
                logger.error(f"保存 Epoch {epoch} 驗證集預測結果時出錯: {e}")
                logger.error(f"錯誤詳情: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        # 如果達到保存頻率，保存 epoch 級別的摘要數據
        if (epoch + 1) % self.save_frequency == 0 and self.save_manager:
            self.save_manager.save_epoch_summary(epoch, train_logs, val_logs)
    
    def on_train_end(self, model: nn.Module, history: Dict[str, List] = None, 
                   logs: Dict[str, Any] = None) -> None:
        """訓練結束時的回調
        
        Args:
            model: 訓練的模型
            history: 訓練歷史
            logs: 日誌字典
        """
        try:
            if self.save_manager and history:
                # 嘗試保存訓練摘要
                try:
                    self.save_manager.save_training_summary(history, self.current_epoch + 1)
                except Exception as e:
                    logger.error(f"保存訓練摘要時出錯: {e}")
                    
                    # 備用方案：保存為 PT 檔案而非 JSON
                    try:
                        save_path = self.save_manager.get_path('results', 'training_summary.pt')
                        torch.save({
                            'total_epochs': self.current_epoch + 1,
                            'timestamp': datetime.now().isoformat()
                        }, save_path)
                        logger.info(f"訓練摘要已保存為 PT 檔案: {save_path}")
                    except Exception as e2:
                        logger.error(f"保存備用訓練摘要時也出錯: {e2}")
        except Exception as e:
            logger.error(f"on_train_end 出錯: {e}")
        finally:        
            if self.hook_manager:
                # 移除鉤子
                self.hook_manager.remove_hooks()
                self.hook_manager = None

class EvaluationResultsHook(CallbackInterface):
    """評估結果鉤子：在評估階段結束時收集並保存所有測試樣本的真實標籤和模型預測
    
    這個鉤子實現了在測試/評估階段結束後，將所有測試樣本的真實標籤和模型預測標籤保存到標準位置的功能。
    """
    
    def __init__(self, save_manager: SaveManager, dataset_name: str = 'test'):
        """初始化評估結果鉤子
        
        Args:
            save_manager: 存檔管理器實例
            dataset_name: 數據集名稱，預設為 'test'
            
        Returns:
            None
            
        Description:
            創建一個評估結果鉤子，用於收集評估階段的預測結果
            
        References:
            無
        """
        self.save_manager = save_manager
        self.dataset_name = dataset_name
        self.all_targets = []
        self.all_predictions = []
        self.all_probabilities = []
        self.metrics = {}
        self.batch_count = 0
    
    def on_evaluation_begin(self, model: nn.Module, logs: Dict[str, Any] = None) -> None:
        """評估開始時的處理
        
        Args:
            model: 被評估的模型
            logs: 附加日誌信息
            
        Returns:
            None
            
        Description:
            在評估開始時清空之前的結果
            
        References:
            無
        """
        self.all_targets = []
        self.all_predictions = []
        self.all_probabilities = []
        self.metrics = {}
        self.batch_count = 0
        logger.info(f"開始收集 {self.dataset_name} 數據集的評估結果")
    
    def on_batch_end(self, batch: int, model: nn.Module, inputs: torch.Tensor = None, 
                   targets: torch.Tensor = None, outputs: torch.Tensor = None, 
                   loss: torch.Tensor = None, logs: Dict[str, Any] = None) -> None:
        """評估批次結束時的處理
        
        Args:
            batch: 批次索引
            model: 被評估的模型
            inputs: 輸入張量
            targets: 目標張量
            outputs: 輸出張量
            loss: 損失張量
            logs: 附加日誌信息
            
        Returns:
            None
            
        Description:
            在每個評估批次結束時累積目標和預測結果
            
        References:
            無
        """
        # 確保在評估階段才收集數據
        if logs and (logs.get('phase') == 'validation' or logs.get('phase') == 'test' or 
                     logs.get('phase') == 'val' or logs.get('phase') == self.dataset_name):
            # 檢查 targets 是否為 None
            if targets is not None:
                self.all_targets.append(targets.detach().cpu())
            # 如果 targets 是 None，嘗試從日誌或其他來源獲取，例如 logs.get('actual_targets')
            elif logs and 'actual_targets' in logs:
                actual_targets = logs['actual_targets']
                if isinstance(actual_targets, torch.Tensor):
                    self.all_targets.append(actual_targets.detach().cpu())
            
            if outputs is not None:
                # 處理分類任務（保存概率和預測類別）
                if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                    # 多類別分類
                    self.all_probabilities.append(outputs.detach().cpu())
                    predictions = torch.argmax(outputs, dim=1)
                    self.all_predictions.append(predictions.detach().cpu())
                else:
                    # 回歸或二元分類
                    self.all_predictions.append(outputs.detach().cpu())
            
            self.batch_count += 1
    
    def on_evaluation_end(self, model: nn.Module, results: Dict[str, Any] = None, 
                        logs: Dict[str, Any] = None) -> None:
        """評估結束時的處理
        
        Args:
            model: 被評估的模型
            results: 評估結果
            logs: 附加日誌信息
            
        Returns:
            None
            
        Description:
            在評估結束時合併所有批次的數據並保存結果
            
        References:
            無
        """
        logger.info(f"評估完成，收集了 {self.batch_count} 個批次的數據")
        
        # 從logs中取得結果（如果results為空）
        if not results and logs and 'metrics' in logs:
            results = logs.get('metrics', {})
            
        if self.all_targets and self.all_predictions:
            # 合併所有批次的數據
            targets = torch.cat(self.all_targets, dim=0)
            predictions = torch.cat(self.all_predictions, dim=0)
            
            # 合併概率（如果有）
            probabilities = None
            if self.all_probabilities:
                probabilities = torch.cat(self.all_probabilities, dim=0)
            
            # 準備要保存的結果
            eval_results = {
                'targets': targets,
                'predictions': predictions,
                'timestamp': datetime.now().isoformat()
            }
            
            # 添加概率（如果有）
            if probabilities is not None:
                eval_results['probabilities'] = probabilities
            
            # 添加指標（如果有）
            if results:
                eval_results['metrics'] = results
            
            # 保存結果到 hooks 目錄 (舊路徑)
            hooks_dir = self.save_manager.get_path('hooks', '')
            os.makedirs(hooks_dir, exist_ok=True)
            
            # 路徑命名方式1: evaluation_results_{dataset_name}.pt
            save_path1 = os.path.join(hooks_dir, f'evaluation_results_{self.dataset_name}.pt')
            torch.save(eval_results, save_path1)
            
            # 路徑命名方式2: {dataset_name}_results.pt
            save_path2 = os.path.join(hooks_dir, f'{self.dataset_name}_results.pt')
            torch.save(eval_results, save_path2)
            
            # 保存到根目錄，以便直接找到
            root_save_path = os.path.join(self.save_manager.experiment_dir, f'test_predictions.pt')
            torch.save(eval_results, root_save_path)
            
            # 同時保存到 results 目錄，以符合 framework_data_structure.md 的規範
            results_dir = self.save_manager.get_path('results', '')
            os.makedirs(results_dir, exist_ok=True)
            results_save_path = os.path.join(results_dir, f'evaluation_results_{self.dataset_name}.pt')
            torch.save(eval_results, results_save_path)
            
            logger.info(f"評估結果已保存到以下路徑:")
            logger.info(f"- {save_path1}")
            logger.info(f"- {save_path2}")
            logger.info(f"- {root_save_path}")
            logger.info(f"- {results_save_path}")
            logger.info(f"預測結果形狀: targets={targets.shape}, predictions={predictions.shape}")
            
            if probabilities is not None:
                logger.info(f"概率形狀: probabilities={probabilities.shape}")

class ActivationCaptureHook(CallbackInterface):
    """目標層激活值捕獲鉤子：在測試集評估過程中，捕獲並保存特定目標層的激活值
    
    這個鉤子實現了在測試集評估過程中，捕獲並保存特定目標層的激活值，用於後續餘弦相似度分析等。
    """
    
    def __init__(self, model: nn.Module, layer_names: List[str], 
                save_manager: SaveManager, dataset_name: str = 'test',
                target_epochs: Optional[Set[int]] = None,
                save_frequency: int = None,
                save_first_last: bool = True,
                compute_similarity: bool = True,
                compute_tsne: bool = True,
                include_sample_ids: bool = True,
                generate_random_labels: bool = False):
        """初始化激活值捕獲鉤子
        
        Args:
            model: 要監控的模型
            layer_names: 要捕獲激活值的層名稱列表
            save_manager: 存檔管理器實例
            dataset_name: 數據集名稱，預設為 'test'
            target_epochs: 指定要捕獲激活值的epoch集合，如果為None則使用save_frequency
            save_frequency: 每間隔多少個epoch保存一次特徵向量，優先於target_epochs
            save_first_last: 是否一定要保存第一個和最後一個epoch
            compute_similarity: 是否計算餘弦相似度
            compute_tsne: 是否計算t-SNE嵌入
            include_sample_ids: 是否包含樣本ID
            generate_random_labels: 當無法獲取真實標籤時是否生成隨機標籤
            
        Returns:
            None
            
        Description:
            創建一個激活值捕獲鉤子，用於收集測試集特定層的激活值
            
        References:
            無
        """
        self.model = model
        self.layer_names = layer_names
        self.save_manager = save_manager
        self.dataset_name = dataset_name
        self.target_epochs = target_epochs
        self.save_frequency = save_frequency
        self.save_first_last = save_first_last
        self.compute_similarity = compute_similarity
        self.compute_tsne = compute_tsne
        self.include_sample_ids = include_sample_ids
        self.generate_random_labels = generate_random_labels
        self.activation_hook = None
        self.all_activations = {layer_name: [] for layer_name in layer_names}
        self.all_targets = []
        self.all_sample_ids = []
        self.batch_count = 0
        self.current_epoch = None  # 追蹤當前epoch
        self.total_epochs = None   # 追蹤總epoch數
        self.captured_epochs = set()  # 新增：記錄已經捕獲的epochs
        
        logger.info(f"初始化激活值捕獲鉤子，監控層: {layer_names}")
        logger.info(f"捕獲模式: dataset={dataset_name}, 已捕獲的epochs={self.captured_epochs}")
        if target_epochs:
            logger.info(f"僅捕獲指定epoch的激活值: {sorted(target_epochs)}")
        elif save_frequency:
            logger.info(f"每 {save_frequency} 個epoch捕獲一次激活值")
        if generate_random_labels:
            logger.info("啟用隨機標籤生成，當無法獲取真實標籤時將生成隨機標籤")
    
    def _find_module_by_name(self, name: str) -> Optional[nn.Module]:
        """通過名稱查找模型中的模塊
        
        Args:
            name: 模塊名稱
            
        Returns:
            Optional[nn.Module]: 找到的模塊，如果未找到則為 None
            
        Description:
            通過名稱遞歸查找模型中的模塊
            
        References:
            無
        """
        if name == '':
            return self.model
        
        for n, m in self.model.named_modules():
            if n == name:
                return m
        
        logger.warning(f"未找到名稱為 '{name}' 的模塊")
        return None
    
    def on_train_begin(self, model: nn.Module, logs: Dict[str, Any] = None) -> None:
        """訓練開始時的處理
        
        Args:
            model: 被訓練的模型
            logs: 附加日誌信息
            
        Returns:
            None
            
        Description:
            在訓練開始時獲取總epoch數
            
        References:
            無
        """
        if logs and 'config' in logs:
            config = logs['config']
            self.total_epochs = config.get('training', {}).get('epochs', None)
            
            # 如果沒有設定target_epochs但有save_frequency，則生成target_epochs
            if self.target_epochs is None and self.save_frequency is not None and self.total_epochs is not None:
                # 修改：改用列表推導生成目標epochs
                epochs_to_save = list(range(0, self.total_epochs, self.save_frequency))
                
                # 如果需要保存第一個和最後一個epoch
                if self.save_first_last:
                    if 0 not in epochs_to_save:
                        epochs_to_save.append(0)  # 第一個epoch
                    if (self.total_epochs - 1) not in epochs_to_save:
                        epochs_to_save.append(self.total_epochs - 1)  # 最後一個epoch
                
                self.target_epochs = set(epochs_to_save)
                logger.info(f"基於save_frequency={self.save_frequency}，生成target_epochs: {sorted(self.target_epochs)}")
    
    def _should_process_epoch(self, epoch: int) -> bool:
        """判斷是否應該在當前epoch處理激活值
        
        Args:
            epoch: 當前epoch
            
        Returns:
            bool: 是否應該處理
            
        Description:
            根據target_epochs和save_frequency判斷是否處理當前epoch
            
        References:
            無
        """
        # 如果已經處理過這個epoch，直接返回False避免重複處理
        if epoch in self.captured_epochs:
            logger.info(f"跳過epoch {epoch}的激活值處理，已經處理過")
            return False
            
        # 優先使用target_epochs
        if self.target_epochs is not None:
            should_process = epoch in self.target_epochs
            if should_process:
                logger.info(f"epoch {epoch} 在目標epoch列表中，將進行特徵向量處理")
            else:
                logger.debug(f"跳過epoch {epoch}的激活值處理，不在目標epoch列表中")
            return should_process
            
        # 如果沒有設置target_epochs，則使用save_frequency
        elif self.save_frequency is not None:
            # 使用save_frequency決定是否處理
            should_process = epoch % self.save_frequency == 0
            # 對於第一個和最後一個epoch，如果設置了save_first_last，也進行處理
            if self.save_first_last and (epoch == 0 or 
                                       (self.total_epochs is not None and epoch == self.total_epochs - 1)):
                should_process = True
                
            if should_process:
                logger.info(f"根據設定，將處理epoch {epoch}的特徵向量")
            else:
                logger.debug(f"跳過epoch {epoch}的激活值處理，不符合設定條件")
            return should_process
            
        # 如果既沒有target_epochs也沒有save_frequency，處理所有epoch
        return True
    
    def on_evaluation_begin(self, model: nn.Module, logs: Dict[str, Any] = None) -> None:
        """評估開始時的處理
        
        Args:
            model: 被評估的模型
            logs: 附加日誌信息
            
        Returns:
            None
            
        Description:
            在評估開始時註冊激活值鉤子，並清空之前的結果
            
        References:
            無
        """
        self.model = model
        self.all_activations = {layer_name: [] for layer_name in self.layer_names}
        self.all_targets = []
        self.all_sample_ids = []
        self.batch_count = 0
        
        # 獲取當前epoch
        if logs and 'epoch' in logs:
            self.current_epoch = logs['epoch']
        else:
            logger.warning("未能從日誌中獲取當前epoch信息，特徵向量捕獲可能不準確")
            # 如果無法獲取epoch，保守處理：假設應該處理
            self.current_epoch = 0
        
        # 檢查當前epoch是否已被處理過
        if self.current_epoch in self.captured_epochs:
            logger.info(f"評估開始：epoch {self.current_epoch} 已被處理過，跳過此次處理")
            return
            
        # 修改：使用抽取出的方法判斷是否應該捕獲
        should_capture = self._should_process_epoch(self.current_epoch)
                
        if not should_capture:
            return
        
        # 創建激活值鉤子
        self.hooks = []
        for layer_name in self.layer_names:
            module = self._find_module_by_name(layer_name)
            if module is not None:
                hook = module.register_forward_hook(
                    lambda mod, inp, out, name=layer_name: self._activation_hook(name, out)
                )
                self.hooks.append(hook)
                logger.info(f"已為層 '{layer_name}' 註冊激活值鉤子")
            else:
                logger.warning(f"無法為層 '{layer_name}' 註冊激活值鉤子：未找到該層")
        
        logger.info(f"開始收集 {self.dataset_name} 數據集的層激活值 (epoch {self.current_epoch})")
    
    def _activation_hook(self, layer_name: str, output):
        """激活值鉤子回調函數
        
        Args:
            layer_name: 層名稱
            output: 層輸出
            
        Returns:
            None
            
        Description:
            保存層的激活值
            
        References:
            無
        """
        # 確保輸出是 Tensor
        if isinstance(output, tuple):
            output = output[0]
        
        # 檢查是否是評估模式
        if not self.model.training:
            # 儲存當前激活值（不保存所有批次的激活值，因為內存可能不夠）
            self.all_activations[layer_name].append(output.detach().cpu())
    
    def on_batch_end(self, batch: int, model: nn.Module, inputs: torch.Tensor = None, 
                   targets: torch.Tensor = None, outputs: torch.Tensor = None, 
                   loss: torch.Tensor = None, logs: Dict[str, Any] = None) -> None:
        """評估批次結束時的處理
        
        Args:
            batch: 批次索引
            model: 被評估的模型
            inputs: 輸入張量
            targets: 目標張量
            outputs: 輸出張量
            loss: 損失張量
            logs: 附加日誌信息
            
        Returns:
            None
            
        Description:
            在評估批次結束時收集目標標籤
            
        References:
            無
        """
        # 檢查是否應該在當前epoch捕獲
        if not hasattr(self, 'hooks') or not self.hooks:
            return
            
        # 收集目標標籤
        if targets is not None:
            self.all_targets.append(targets.detach().cpu())
            
        # 如果logs中包含樣本ID，也收集它們
        if logs and 'sample_ids' in logs and self.include_sample_ids:
            sample_ids = logs['sample_ids']
            self.all_sample_ids.extend(sample_ids)
            
        self.batch_count += 1
    
    def on_evaluation_end(self, model: nn.Module, results: Dict[str, Any] = None, 
                        logs: Dict[str, Any] = None) -> None:
        """評估結束時的處理
        
        Args:
            model: 被評估的模型
            results: 評估結果
            logs: 附加日誌信息
            
        Returns:
            None
            
        Description:
            在評估結束時合併所有批次的數據並保存結果
            
        References:
            無
        """
        # 檢查當前epoch是否已被處理過
        if self.current_epoch in self.captured_epochs:
            logger.info(f"評估結束：epoch {self.current_epoch} 已被處理過，跳過此次處理")
            return
            
        # 檢查是否應該在當前epoch保存激活值
        # 注意：這裡的邏輯需要與on_evaluation_begin保持一致
        # 修改：使用抽取出的方法判斷是否應該保存
        should_save = self._should_process_epoch(self.current_epoch)
                
        if not should_save:
            return
        
        # 確保有註冊過激活值鉤子
        if not hasattr(self, 'hooks') or not self.hooks:
            logger.warning(f"epoch {self.current_epoch}沒有註冊激活值鉤子，無法保存特徵向量")
            return
        
        logger.info(f"評估完成，收集了 {self.batch_count} 個批次的層激活值")
        
        # 移除鉤子
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # 修改：將特徵向量處理邏輯提取到單獨的方法中
        self._process_features(self.current_epoch)
    
    def _process_features(self, epoch: int) -> None:
        """處理並保存特徵向量
        
        Args:
            epoch: 要處理的epoch
            
        Returns:
            None
            
        Description:
            處理並保存指定epoch的特徵向量
            
        References:
            無
        """
        # 為每一層保存激活值
        for layer_name, activations in self.all_activations.items():
            if not activations:
                logger.warning(f"層 '{layer_name}' 沒有收集到激活值")
                continue
            
            # 合併所有批次的激活值
            try:
                all_activations = torch.cat(activations, dim=0)
                
                # 合併所有批次的目標
                targets = None
                if self.all_targets:
                    try:
                        targets = torch.cat(self.all_targets, dim=0)
                        
                        # 確保目標數量與激活值數量匹配
                        if len(targets) != len(all_activations):
                            logger.warning(f"目標數量 ({len(targets)}) 與激活值數量 ({len(all_activations)}) 不匹配，將截取匹配長度的部分")
                            min_length = min(len(targets), len(all_activations))
                            targets = targets[:min_length]
                            all_activations = all_activations[:min_length]
                            logger.info(f"已將目標和激活值截取到相同長度: {min_length}")
                    except Exception as e:
                        logger.error(f"合併目標時出錯: {e}")
                        targets = None
                
                # 如果沒有目標但啟用了隨機標籤生成，則生成隨機標籤
                if targets is None and self.generate_random_labels:
                    num_samples = all_activations.shape[0]
                    # 生成4個類別的隨機標籤 (可根據需求調整類別數)
                    targets = torch.randint(0, 4, (num_samples,))
                    logger.warning(f"已生成 {num_samples} 個隨機標籤，用於特徵向量分析")
                
                # 展平高維特徵向量為2D格式 [batch_size, feature_dim]
                original_shape = all_activations.shape
                if all_activations.ndim > 2:
                    # 保留第一維(樣本數)，將其餘維度展平
                    all_activations = all_activations.reshape(original_shape[0], -1)
                    logger.info(f"已將特徵向量從形狀 {original_shape} 展平為 {all_activations.shape}")
                
                # 準備要保存的結果
                activation_data = {
                    'layer_name': layer_name,
                    'activations': all_activations,
                    'timestamp': datetime.now().isoformat(),
                    'epoch': epoch
                }
                
                # 添加目標和樣本 ID（如果有）
                if targets is not None:
                    activation_data['targets'] = targets
                
                if self.all_sample_ids and self.include_sample_ids:
                    # 確保樣本ID數量與激活值數量匹配
                    if len(self.all_sample_ids) == len(all_activations):
                        activation_data['sample_ids'] = self.all_sample_ids
                    else:
                        logger.warning(f"樣本ID數量 ({len(self.all_sample_ids)}) 與激活值數量 ({len(all_activations)}) 不匹配，將跳過樣本ID")
                
                # 創建feature_vectors目錄
                feature_dir = self.save_manager.get_path('feature_vectors', '')
                os.makedirs(feature_dir, exist_ok=True)
                
                # 創建epoch特定目錄
                epoch_dir = os.path.join(feature_dir, f'epoch_{epoch}')
                os.makedirs(epoch_dir, exist_ok=True)
                
                # 保存特徵向量
                feature_path = os.path.join(epoch_dir, f'layer_{layer_name.replace(".", "_")}_features.pt')
                torch.save(activation_data, feature_path)
                logger.info(f"層 '{layer_name}' 的特徵向量已保存到: {feature_path}")
                
                # 計算並保存餘弦相似度矩陣（如果啟用且有目標）
                if self.compute_similarity and all_activations.shape[0] > 1 and targets is not None:
                    try:
                        similarity_data = self._compute_cosine_similarity(all_activations, targets)
                        similarity_path = os.path.join(epoch_dir, f'layer_{layer_name.replace(".", "_")}_cosine_similarity.pt')
                        torch.save(similarity_data, similarity_path)
                        logger.info(f"層 '{layer_name}' 的餘弦相似度矩陣已保存到: {similarity_path}")
                    except Exception as e:
                        logger.error(f"計算層 '{layer_name}' 的餘弦相似度時出錯: {e}")
                
                # 計算並保存t-SNE嵌入（如果啟用且樣本足夠）
                if self.compute_tsne and all_activations.shape[0] > 5:  # 至少需要幾個樣本
                    try:
                        tsne_data = self._compute_tsne(all_activations, targets)
                        tsne_path = os.path.join(epoch_dir, f'layer_{layer_name.replace(".", "_")}_tsne.pt')
                        torch.save(tsne_data, tsne_path)
                        logger.info(f"層 '{layer_name}' 的t-SNE座標已保存到: {tsne_path}")
                    except Exception as e:
                        logger.error(f"計算層 '{layer_name}' 的t-SNE座標時出錯: {e}")
                
                # 添加到特徵分析摘要
                self._update_feature_analysis_summary(layer_name, all_activations, targets, epoch)
                
            except Exception as e:
                logger.error(f"保存層 '{layer_name}' 的激活值時出錯: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # 標記這個epoch已經處理過，避免重複處理
        self.captured_epochs.add(epoch)
        logger.info(f"已完成epoch {epoch}的特徵向量保存")
    
    def _compute_cosine_similarity(self, features: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """計算特徵向量間的餘弦相似度
        
        Args:
            features: 特徵向量，形狀為 [樣本數, 特徵維度]
            targets: 目標標籤，形狀為 [樣本數]
            
        Returns:
            Dict[str, Any]: 包含相似度矩陣和相關統計資訊的字典
            
        Description:
            計算特徵向量間的餘弦相似度，同時計算同類樣本和不同類樣本之間的平均相似度
            
        References:
            無
        """
        try:
            # 將特徵張量轉換為numpy數組
            features_np = features.cpu().numpy()
            
            # 計算完整的餘弦相似度矩陣
            similarity_matrix = cosine_similarity(features_np)
            
            # 準備結果字典
            result = {
                'similarity_matrix': torch.tensor(similarity_matrix),
                'timestamp': datetime.now().isoformat(),
                'num_samples': features.shape[0]
            }
            
            # 如果提供了目標標籤，計算類別內和類別間的平均相似度
            if targets is not None and len(targets) == len(features):
                try:
                    targets_np = targets.cpu().numpy()
                    
                    # 檢查目標標籤是否有效
                    unique_targets = np.unique(targets_np)
                    if len(unique_targets) < 2:
                        logger.warning(f"目標標籤只有一個類別 {unique_targets}，無法計算類別間相似度")
                        return result
                    
                    # 類別內相似度
                    intra_class_similarities = []
                    # 類別間相似度
                    inter_class_similarities = []
                    # 每個類別的質心
                    centroids = {}
                    
                    # 計算每個類別的質心
                    for cls in unique_targets:
                        cls_indices = np.where(targets_np == cls)[0]
                        if len(cls_indices) > 0:
                            cls_features = features_np[cls_indices]
                            centroid = np.mean(cls_features, axis=0, keepdims=True)
                            centroids[cls] = centroid
                    
                    # 計算類別內相似度（同一類別樣本之間）
                    for cls in unique_targets:
                        cls_indices = np.where(targets_np == cls)[0]
                        if len(cls_indices) > 1:  # 至少需要2個樣本
                            cls_similarity = similarity_matrix[np.ix_(cls_indices, cls_indices)]
                            # 排除對角線元素 (自己和自己的相似度)
                            mask = ~np.eye(cls_similarity.shape[0], dtype=bool)
                            intra_class_similarities.append(float(cls_similarity[mask].mean()))
                    
                    # 計算類別間相似度（不同類別樣本之間）
                    for i, cls1 in enumerate(unique_targets):
                        for cls2 in unique_targets[i+1:]:
                            cls1_indices = np.where(targets_np == cls1)[0]
                            cls2_indices = np.where(targets_np == cls2)[0]
                            if len(cls1_indices) > 0 and len(cls2_indices) > 0:
                                cross_similarity = similarity_matrix[np.ix_(cls1_indices, cls2_indices)]
                                inter_class_similarities.append(float(cross_similarity.mean()))
                    
                    # 計算樣本與其類別質心的相似度
                    centroid_similarities = []
                    for i, sample in enumerate(features_np):
                        cls = targets_np[i]
                        if cls in centroids:
                            centroid_sim = cosine_similarity([sample], centroids[cls])[0][0]
                            centroid_similarities.append(float(centroid_sim))
                    
                    # 添加到結果字典
                    result['classes'] = unique_targets.tolist()
                    result['intra_class_avg_similarity'] = np.mean(intra_class_similarities) if intra_class_similarities else None
                    result['inter_class_avg_similarity'] = np.mean(inter_class_similarities) if inter_class_similarities else None
                    result['sample_to_centroid_avg_similarity'] = np.mean(centroid_similarities) if centroid_similarities else None
                    
                    # 每個類別的內部相似度 (類到其統計)
                    class_specific_similarities = {}
                    for cls in unique_targets:
                        cls_indices = np.where(targets_np == cls)[0]
                        if len(cls_indices) > 1:  # 至少需要2個樣本
                            cls_similarity = similarity_matrix[np.ix_(cls_indices, cls_indices)]
                            mask = ~np.eye(cls_similarity.shape[0], dtype=bool)
                            cls_mean_sim = float(cls_similarity[mask].mean())
                            class_specific_similarities[int(cls)] = cls_mean_sim
                    
                    if class_specific_similarities:
                        result['intra_class_similarities_by_class'] = class_specific_similarities
                except Exception as e:
                    logger.error(f"計算類別相關相似度時出錯: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            return result
            
        except Exception as e:
            logger.error(f"計算餘弦相似度時出現未預期的錯誤: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 返回一個基本的結果
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'num_samples': features.shape[0] if hasattr(features, 'shape') else 0
            }
    
    def _compute_tsne(self, features: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """使用t-SNE計算特徵向量的二維嵌入
        
        Args:
            features: 特徵向量，形狀為 [樣本數, 特徵維度]
            targets: 目標標籤，形狀為 [樣本數]
            
        Returns:
            Dict[str, Any]: 包含t-SNE座標和相關統計資訊的字典
            
        Description:
            使用t-SNE計算特徵向量的二維嵌入
            
        References:
            無
        """
        try:
            # 將特徵張量轉換為numpy數組
            features_np = features.cpu().numpy()
            
            # 先使用PCA降維，以加速t-SNE計算
            # 樣本數和特徵數
            n_samples, n_features = features_np.shape
            
            # 修改：根據樣本數調整PCA組件數
            # 選擇 min(樣本數-1, 特徵數, 50) 作為組件數
            n_components = min(n_samples - 1, n_features, 50)
            if n_components < 2:
                n_components = 2  # 確保至少有兩個組件
                
            logger.info(f"t-SNE計算：使用PCA降維到{n_components}個組件 (樣本數={n_samples}, 特徵數={n_features})")
            
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components)
            features_np = pca.fit_transform(features_np)
            
            # 使用t-SNE進一步降至2維
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42)
            tsne_result = tsne.fit_transform(features_np)
            
            # 準備結果字典
            result = {
                'tsne_coordinates': torch.tensor(tsne_result),
                'explained_variance_ratio': torch.tensor(pca.explained_variance_ratio_),
                'timestamp': datetime.now().isoformat(),
                'num_samples': features.shape[0]
            }
            
            # 如果提供了目標標籤，添加到結果中
            if targets is not None:
                result['targets'] = targets
            
            return result
        
        except Exception as e:
            logger.error(f"計算t-SNE時出現錯誤: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 返回基本結果
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'num_samples': features.shape[0] if hasattr(features, 'shape') else 0
            }
    
    def _update_feature_analysis_summary(self, layer_name: str, features: torch.Tensor, 
                                        targets: Optional[torch.Tensor], epoch: int) -> None:
        """更新特徵分析摘要
        
        Args:
            layer_name: 層名稱
            features: 特徵向量
            targets: 目標標籤
            epoch: 當前epoch
            
        Returns:
            None
            
        Description:
            更新並保存特徵分析摘要，包含不同epoch特徵向量的變化情況
            
        References:
            無
        """
        summary_path = self.save_manager.get_path('feature_vectors', 'feature_analysis.json')
        
        # 初始化或加載摘要
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
            except Exception:
                summary = {}
        else:
            summary = {}
        
        # 確保層條目存在
        if layer_name not in summary:
            summary[layer_name] = {}
        
        # 添加當前epoch的摘要
        layer_summary = {
            'epoch': epoch,
            'num_samples': features.shape[0],
            'feature_dim': features.shape[1],
            'timestamp': datetime.now().isoformat(),
        }
        
        # 計算特徵統計量
        layer_summary['feature_norm_mean'] = float(torch.norm(features, dim=1).mean().item())
        layer_summary['feature_std'] = float(features.std().item())
        
        # 如果提供了目標標籤，添加類別統計
        if targets is not None:
            unique_targets = torch.unique(targets).cpu().tolist()
            layer_summary['num_classes'] = len(unique_targets)
            layer_summary['class_distribution'] = {int(t): int((targets == t).sum().item()) for t in unique_targets}
        
        # 更新摘要
        summary[layer_name][str(epoch)] = layer_summary
        
        # 保存摘要
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

def create_analyzer_callback(
    output_dir: str = 'results',
    monitored_layers: Optional[List[str]] = None,
    monitored_params: Optional[List[str]] = None,
    save_frequency: int = 1,
    capture_evaluation_results: bool = False
) -> Any:
    """創建分析器回調
    
    Args:
        output_dir: 輸出目錄
        monitored_layers: 要監控的層名稱列表
        monitored_params: 要監控的參數名稱列表
        save_frequency: 保存頻率（每 N 個 epoch 保存一次）
        capture_evaluation_results: 是否捕獲評估結果
        
    Returns:
        Any: 分析器回調
        
    Description:
        創建合適的分析器回調，根據環境選擇 SBP Analyzer 或簡易模型分析器
        
    References:
        無
    """
    # 檢查 SBP_analyzer 是否可用
    if is_sbp_analyzer_available():
        try:
            from sbp_analyzer.callbacks import ModelAnalyticsCallback
            return ModelAnalyticsCallback(
                output_dir=output_dir,
                monitored_layers=monitored_layers,
                monitored_params=monitored_params,
                save_frequency=save_frequency
            )
        except ImportError as e:
            logger.warning(f"無法導入 SBP_analyzer.callbacks.ModelAnalyticsCallback: {e}")
            logger.warning("將使用內置的簡易模型分析器")
    
    # 回退到內置的簡易分析器
    analytics_callback = SimpleModelAnalyticsCallback(
        output_dir=output_dir,
        monitored_layers=monitored_layers,
        monitored_params=monitored_params,
        save_frequency=save_frequency
    )
    
    # 如果需要捕獲評估結果，創建存檔管理器並添加評估結果鉤子
    if capture_evaluation_results:
        from utils.save_manager import SaveManager
        save_manager = SaveManager(base_dir=output_dir, create_subdirs=True)
        return [analytics_callback, EvaluationResultsHook(save_manager)]
    
    return analytics_callback

def get_analyzer_callbacks_from_config(config: Dict[str, Any]) -> List[Any]:
    """從配置中獲取分析器回調
    
    Args:
        config: 配置字典
        
    Returns:
        List[Any]: 分析器回調列表
        
    Description:
        根據配置創建所有必要的分析器回調
        
    References:
        無
    """
    callbacks = []
    output_dir = config.get('global', {}).get('output_dir', 'results')
    
    # 從training部分獲取save_every作為默認保存頻率
    default_save_frequency = config.get('training', {}).get('save_every', 1)
    
    # 檢查是否配置了 hook 部分
    hooks_config = config.get('hooks', {})
    
    # 模型分析回調
    if hooks_config.get('model_analytics', {}).get('enabled', False):
        analytics_config = hooks_config.get('model_analytics', {})
        monitored_layers = analytics_config.get('monitored_layers', [])
        monitored_params = analytics_config.get('monitored_params', [])
        # 優先使用training.save_every
        save_frequency = analytics_config.get('save_frequency', default_save_frequency)
        logger.info(f"模型分析回調將每 {save_frequency} 個epoch保存一次結果")
        
        analytics_callback = create_analyzer_callback(
            output_dir=output_dir,
            monitored_layers=monitored_layers,
            monitored_params=monitored_params,
            save_frequency=save_frequency
        )
        
        if isinstance(analytics_callback, list):
            callbacks.extend(analytics_callback)
        else:
            callbacks.append(analytics_callback)
    
    # 評估結果捕獲
    if hooks_config.get('evaluation_capture', {}).get('enabled', False):
        datasets = hooks_config.get('evaluation_capture', {}).get('datasets', ['test'])
        
        # 創建存檔管理器
        from utils.save_manager import SaveManager
        save_manager = SaveManager(base_dir=output_dir, create_subdirs=True)
        
        # 為每個數據集創建評估結果鉤子
        for dataset_name in datasets:
            callbacks.append(EvaluationResultsHook(save_manager, dataset_name))
    
    # 激活值捕獲
    if hooks_config.get('activation_capture', {}).get('enabled', False):
        activation_config = hooks_config.get('activation_capture', {})
        target_layers = activation_config.get('target_layers', [])
        datasets = activation_config.get('datasets', ['test'])
        
        # 獲取目標epoch列表（如果指定）
        target_epochs = activation_config.get('target_epochs', None)
        if target_epochs:
            target_epochs = set(target_epochs)
        
        # 修改：優先使用training.save_every作為保存頻率
        # 嘗試獲取保存頻率，如果沒有顯式定義則嘗試使用訓練的save_every
        save_frequency = activation_config.get('save_frequency', None)
        if save_frequency is None:
            save_frequency = default_save_frequency
            logger.info(f"未設定特徵向量捕獲的save_frequency，將使用training.save_every={save_frequency}")
        else:
            # 如果同時定義了save_frequency和training.save_every，優先使用training.save_every
            if default_save_frequency != save_frequency:
                logger.warning(f"發現衝突的保存頻率設定：hooks.activation_capture.save_frequency={save_frequency}，"
                             f"training.save_every={default_save_frequency}")
                logger.info(f"為確保一致性，將使用training.save_every={default_save_frequency}作為保存頻率")
                save_frequency = default_save_frequency
        
        # 是否保存第一個和最後一個epoch
        save_first_last = activation_config.get('save_first_last', True)
        
        # 是否計算相似度和t-SNE
        compute_similarity = activation_config.get('compute_similarity', True)
        compute_tsne = activation_config.get('compute_tsne', True)
        
        # 新增參數：是否包含樣本ID和是否生成隨機標籤
        include_sample_ids = activation_config.get('include_sample_ids', True)
        generate_random_labels = activation_config.get('generate_random_labels', False)
        
        if target_layers:
            # 創建存檔管理器（如果還未創建）
            if 'save_manager' not in locals():
                from utils.save_manager import SaveManager
                save_manager = SaveManager(base_dir=output_dir, create_subdirs=True)
            
            # 直接創建 ActivationCaptureHook 實例
            for dataset_name in datasets:
                capture_hook = ActivationCaptureHook(
                    model=None,  # 在 on_evaluation_begin 回調中設置模型
                    layer_names=target_layers,
                    save_manager=save_manager,
                    dataset_name=dataset_name,
                    target_epochs=target_epochs,
                    save_frequency=save_frequency,
                    save_first_last=save_first_last,
                    compute_similarity=compute_similarity,
                    compute_tsne=compute_tsne,
                    include_sample_ids=include_sample_ids,
                    generate_random_labels=generate_random_labels
                )
                callbacks.append(capture_hook)
                logger.info(f"已添加激活值捕獲鉤子，目標層: {target_layers}, 數據集: {dataset_name}")
                if target_epochs:
                    logger.info(f"激活值捕獲將僅在 epochs {sorted(target_epochs)} 進行")
                elif save_frequency:
                    logger.info(f"激活值捕獲將每 {save_frequency} 個 epochs 進行一次")
                    if save_first_last:
                        logger.info(f"另外也會捕獲第一個和最後一個 epoch")
                if generate_random_labels:
                    logger.info(f"已啟用隨機標籤生成，當無法獲取真實標籤時將生成隨機標籤")
    
    return callbacks

# 中文註解：這是hook_bridge.py的Minimal Executable Unit，檢查SimpleActivationHook與SimpleGradientHook能否正確初始化與基本功能，並測試錯誤參數時的優雅報錯
if __name__ == "__main__":
    """
    Description: Minimal Executable Unit for hook_bridge.py，檢查SimpleActivationHook與SimpleGradientHook能否正確初始化與基本功能，並測試錯誤參數時的優雅報錯。
    Args: None
    Returns: None
    References: 無
    """
    import torch
    import torch.nn as nn
    import logging
    logging.basicConfig(level=logging.INFO)
    # 不要再import自己
    # from models.hook_bridge import SimpleActivationHook, SimpleGradientHook

    # 測試SimpleActivationHook
    try:
        class DummyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 2, 3)
                self.relu = nn.ReLU()
            def forward(self, x):
                return self.relu(self.conv(x))
        model = DummyNet()
        hook = SimpleActivationHook(model, layer_names=["conv", "relu"])
        dummy_input = torch.randn(1, 1, 8, 8)
        _ = model(dummy_input)
        acts = hook.get_activations()
        print(f"SimpleActivationHook測試成功，捕獲層: {list(acts.keys())}")
    except Exception as e:
        print(f"SimpleActivationHook遇到錯誤（預期行為）: {e}")

    # 測試SimpleGradientHook
    try:
        model = DummyNet()
        hook = SimpleGradientHook(model, param_names=["conv.weight"])
        dummy_input = torch.randn(1, 1, 8, 8)
        output = model(dummy_input)
        loss = output.sum()
        loss.backward()
        grads = hook.get_gradients()
        print(f"SimpleGradientHook測試成功，捕獲參數: {list(grads.keys())}")
    except Exception as e:
        print(f"SimpleGradientHook遇到錯誤（預期行為）: {e}")

    # 測試錯誤參數
    try:
        model = DummyNet()
        hook = SimpleActivationHook(model, layer_names=["not_exist_layer"])
        dummy_input = torch.randn(1, 1, 8, 8)
        _ = model(dummy_input)
        acts = hook.get_activations()
        print(f"SimpleActivationHook錯誤參數測試，捕獲層: {list(acts.keys())}")
    except Exception as e:
        print(f"SimpleActivationHook遇到錯誤（預期行為）: {e}") 