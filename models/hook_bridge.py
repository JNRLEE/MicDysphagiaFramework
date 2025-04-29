"""
模型鉤子橋接模塊：提供與外部分析工具交互的簡化橋接接口

該模塊實現了數據抽取和存儲功能，允許 MicDysphagiaFramework 在不依賴外部工具的情況下
捕獲中間層激活值和梯度，並將其保存在結構化的目錄中，以便後續使用 SBP_analyzer 進行離線分析。
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable, Union
import importlib
import logging
import warnings
import os
import numpy as np
from datetime import datetime

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
        if self.save_manager and history:
            # 保存訓練摘要
            self.save_manager.save_training_summary(history, self.current_epoch + 1)
            
        if self.hook_manager:
            # 移除鉤子
            self.hook_manager.remove_hooks()
            self.hook_manager = None
    
    def on_evaluation_begin(self, model: nn.Module, logs: Dict[str, Any] = None) -> None:
        """評估開始時的回調
        
        Args:
            model: 評估的模型
            logs: 日誌字典
        """
        logger.info(f"開始評估")
    
    def on_evaluation_end(self, model: nn.Module, results: Dict[str, Any] = None, 
                        logs: Dict[str, Any] = None) -> None:
        """評估結束時的回調
        
        Args:
            model: 評估的模型
            results: 評估結果
            logs: 日誌字典
        """
        logger.info(f"評估結束，結果: {results}")
        
        # 保存評估結果
        if results and self.save_manager:
            mode = logs.get('mode') if logs else None
            self.save_manager.save_evaluation_results(results, mode)

def create_analyzer_callback(
    output_dir: str = 'results',
                            monitored_layers: Optional[List[str]] = None,
                            monitored_params: Optional[List[str]] = None,
    save_frequency: int = 1
) -> Any:
    """創建模型分析回調
    
    Args:
        output_dir: 輸出目錄
        monitored_layers: 要監控的層名稱列表
        monitored_params: 要監控的參數名稱列表
        save_frequency: 保存頻率（每 N 個 epoch）
        
    Returns:
        模型分析回調實例
    """
    # 輸出調試信息
    logger.info(f"創建分析器回調: output_dir={output_dir}, save_frequency={save_frequency}")
    if monitored_layers:
        logger.info(f"監控層: {monitored_layers}")
    if monitored_params:
        logger.info(f"監控參數: {monitored_params}")
    
    # 優先使用 SBP_analyzer 的實現（如果可用）
    if is_sbp_analyzer_available():
        try:
            from sbp_analyzer import ModelAnalyticsCallback
            logger.info("使用 SBP_analyzer 的 ModelAnalyticsCallback")
            return ModelAnalyticsCallback(
                output_dir=output_dir,
                monitored_layers=monitored_layers,
                monitored_params=monitored_params,
                save_frequency=save_frequency
            )
        except ImportError:
            logger.warning("無法導入 SBP_analyzer 的 ModelAnalyticsCallback，將使用內置簡易版本")
    
    # 回退到簡易版本
    logger.info("使用內置的 SimpleModelAnalyticsCallback")
    return SimpleModelAnalyticsCallback(
        output_dir=output_dir,
        monitored_layers=monitored_layers,
        monitored_params=monitored_params,
        save_frequency=save_frequency
    )

def get_analyzer_callbacks_from_config(config: Dict[str, Any]) -> List[Any]:
    """從配置中獲取分析器回調列表
    
    Args:
        config: 實驗配置字典
        
    Returns:
        List[Any]: 回調實例列表
    """
    callbacks = []
    
    # 檢查配置中是否包含分析配置部分
    if 'analysis' not in config or not config['analysis'].get('enabled', False):
        logger.info("配置中未啟用分析功能，不添加分析回調")
        return callbacks
    
    analysis_config = config['analysis']
    logger.info(f"從配置中讀取分析設置: {analysis_config}")
    
    # 獲取模型分析回調
    if analysis_config.get('model_analytics', {}).get('enabled', False):
        model_analytics_config = analysis_config.get('model_analytics', {})
        
        # 獲取輸出目錄，優先使用分析配置中的，否則使用全局配置中的
        output_dir = analysis_config.get('output_dir', None)
        if output_dir is None:
            output_dir = config.get('global', {}).get('output_dir', 'results')
            
        logger.info(f"創建模型分析回調，輸出目錄: {output_dir}")
        
        callback = create_analyzer_callback(
            output_dir=output_dir,
            monitored_layers=model_analytics_config.get('monitored_layers'),
            monitored_params=model_analytics_config.get('monitored_params'),
            save_frequency=model_analytics_config.get('save_frequency', 1)
        )
        if callback:
            callbacks.append(callback)
            logger.info("已添加模型分析回調")
    
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