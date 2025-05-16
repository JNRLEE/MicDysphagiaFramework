import logging
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, List, Optional, Union, Callable
import time
import json
from datetime import datetime
import numpy as np
from tqdm import tqdm
import re

# 引入數據適配器
from utils.data_adapter import adapt_datasets_to_model, DataAdapter
# 引入 TensorBoard 的 SummaryWriter
from torch.utils.tensorboard import SummaryWriter
# 引入回調接口
from utils.callback_interface import CallbackInterface
from losses.loss_factory import LossFactory
# 引入存檔管理器
from utils.save_manager import SaveManager

# 導入模型結構信息工具類
from models.model_structure import ModelStructureInfo

logger = logging.getLogger(__name__)

class PyTorchTrainer:
    """PyTorch 訓練器實現"""

    def __init__(self, config: Dict[str, Any], model: nn.Module):
        """初始化 PyTorch 訓練器
        
        Args:
            config: 配置字典，包含訓練參數
            model: PyTorch 模型實例
        """
        self.config = config
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_gpu', True) else 'cpu')
        self.model.to(self.device)
        
        # 設置訓練參數
        self.epochs = config.get('training', {}).get('epochs', 100)
        
        # 處理早期停止配置
        early_stopping_config = config.get('training', {}).get('early_stopping', {})
        if isinstance(early_stopping_config, dict):
            self.early_stopping_enabled = early_stopping_config.get('enabled', False)
            self.early_stopping_patience = early_stopping_config.get('patience', 10)
            self.early_stopping_min_delta = early_stopping_config.get('min_delta', 0.001)
        else:
            # 處理早期版本的配置，其中 early_stopping 可能只是一個整數
            self.early_stopping_enabled = bool(early_stopping_config)
            self.early_stopping_patience = early_stopping_config if isinstance(early_stopping_config, int) else 10
            self.early_stopping_min_delta = 0.001
        
        self.patience = self.early_stopping_patience
        self.best_val_loss = float('inf')
        self.best_state_dict = None
        
        # 獲取優化器
        lr = config.get('training', {}).get('learning_rate', 0.001)
        weight_decay = config.get('training', {}).get('weight_decay', 0.0)
        
        # 使用模型自定義的優化器配置（如果存在）
        if hasattr(model, 'configure_optimizers'):
            self.optimizer = model.configure_optimizers(lr, weight_decay)
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # 獲取損失函數
        self.criterion = self._get_loss_function()
        
        # 記錄訓練過程
        self.train_losses = []
        self.val_losses = []
        self.metrics = {'train': {}, 'val': {}}
        
        # 設置訓練目錄
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_type = config.get('model', {}).get('type', 'model')
        experiment_name = config.get('global', {}).get('experiment_name', model_type)
        
        # 獲取輸出目錄，避免重複的 results 目錄
        base_output_dir = config.get('global', {}).get('output_dir', 'results')
        
        self.output_dir = os.path.join(
            base_output_dir,
            f"{experiment_name}_{timestamp}"
        )
        # 修正以避免在 run_experiments.py 之後再次加入時間戳記
        # 檢查路徑中是否已經包含時間戳記格式的字串 (yyyymmdd_HHMMSS)
        if re.search(r'\d{8}_\d{6}', base_output_dir):
            # 如果路徑已經包含時間戳記，就直接使用路徑而不添加新的
            self.output_dir = base_output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 創建 SaveManager 實例
        self.save_manager = SaveManager(self.output_dir)
        
        # 初始化 TensorBoard Writer
        log_dir = self.save_manager.get_path('tensorboard', '')
        self.writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard logs 將保存至 {log_dir}")

        logger.info(f"訓練輸出將保存至 {self.output_dir}")
        
        # 保存配置
        self.save_manager.save_config(config)
            
        # 新增：初始化回調列表
        self.callbacks = []

    def add_callback(self, callback: CallbackInterface) -> None:
        """添加回調函數到訓練器
        
        Args:
            callback: 實現 CallbackInterface 的回調實例
        """
        self.callbacks.append(callback)
        logger.info(f"添加回調: {callback.__class__.__name__}")
        
    def add_callbacks(self, callbacks: List[CallbackInterface]) -> None:
        """批量添加多個回調函數
        
        Args:
            callbacks: 回調實例列表
        """
        for callback in callbacks:
            self.add_callback(callback)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """使用提供的數據加載器訓練模型
        
        Args:
            train_loader: 訓練數據加載器
            val_loader: 驗證數據加載器
            test_loader: 測試數據加載器，可選
            
        Returns:
            Dict[str, Any]: 訓練結果，包含損失、指標和最佳模型路徑
        """
        logger.info(f"開始訓練，模型類型: {self.config.get('model', {}).get('type', 'unknown')}")
        logger.info(f"設備: {self.device}")
        
        # 檢查模型和數據兼容性，必要時調整數據加載器
        model_type = self.config.get('model', {}).get('type', '')
        train_loader, val_loader, test_loader = adapt_datasets_to_model(
            model_type, 
            self.config, 
            train_loader, 
            val_loader, 
            test_loader
        )
        
        start_time = time.time()
        
        # 新增：調用訓練開始回調
        logs = {'tensorboard_writer': self.writer}
        for callback in self.callbacks:
            callback.on_train_begin(self.model, logs)
        
        for epoch in range(self.epochs):
            # 新增：調用 epoch 開始回調
            epoch_logs = {'epoch': epoch, 'tensorboard_writer': self.writer}
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch, self.model, epoch_logs)
            
            # 訓練一個 epoch
            train_loss, train_metrics = self._train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            # 驗證一個 epoch
            val_loss, val_metrics = self._validate_epoch(val_loader, epoch)
            self.val_losses.append(val_loss)

            # --- TensorBoard 記錄 ---
            # 記錄損失
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            # 記錄學習率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning_rate', current_lr, epoch)
            # 記錄訓練指標
            for metric_name, metric_value in train_metrics.items():
                # 僅當指標值是標量時才記錄到 TensorBoard
                if not isinstance(metric_value, torch.Tensor) or metric_value.numel() == 1:
                    # 確保值是標量
                    if isinstance(metric_value, torch.Tensor):
                        metric_value = metric_value.item()
                    self.writer.add_scalar(f'Metrics/train/{metric_name}', metric_value, epoch)
                else:
                    logger.warning(f"跳過記錄非標量指標 '{metric_name}' 到 TensorBoard (形狀: {metric_value.shape if isinstance(metric_value, torch.Tensor) else type(metric_value)})")
            
            # 記錄驗證指標
            for metric_name, metric_value in val_metrics.items():
                # 僅當指標值是標量時才記錄到 TensorBoard，並跳過我們添加的特殊鍵
                if metric_name not in ['outputs', 'targets', 'predictions'] and (not isinstance(metric_value, torch.Tensor) or metric_value.numel() == 1):
                    # 確保值是標量
                    if isinstance(metric_value, torch.Tensor):
                        metric_value = metric_value.item()
                    self.writer.add_scalar(f'Metrics/val/{metric_name}', metric_value, epoch)
                elif metric_name in ['outputs', 'targets', 'predictions']:
                    # 這些鍵是為了 Hook 添加的，不需要寫入 TensorBoard
                    pass
                else:
                    logger.warning(f"跳過記錄非標量指標 '{metric_name}' 到 TensorBoard (形狀: {metric_value.shape if isinstance(metric_value, torch.Tensor) else type(metric_value)})")
            # ------------------------
            
            # 更新指標記錄 (用於最終返回的 history)
            for metric_name, metric_value in train_metrics.items():
                if metric_name not in self.metrics['train']:
                    self.metrics['train'][metric_name] = []
                self.metrics['train'][metric_name].append(metric_value)
            
            for metric_name, metric_value in val_metrics.items():
                if metric_name not in self.metrics['val']:
                    self.metrics['val'][metric_name] = []
                self.metrics['val'][metric_name].append(metric_value)
            
            # 檢查是否為最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
                self.patience = self.early_stopping_patience
                
                # 保存最佳模型
                best_model_path = self._save_best_model(epoch, val_loss)
                
                logger.info(f"Epoch {epoch+1}/{self.epochs} - 保存新的最佳模型: {best_model_path}")
            elif self.early_stopping_enabled:  # 只有在啟用早期停止時才減少耐心值
                self.patience -= 1
                if self.patience <= 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs} - 提前停止訓練！")
                    break
            
            # 檢查是否需要定期保存模型
            save_every = self.config.get('training', {}).get('save_every', 0)
            if save_every > 0 and (epoch + 1) % save_every == 0:
                checkpoint_path = self._save_checkpoint(epoch, val_loss)
                logger.info(f"Epoch {epoch+1}/{self.epochs} - 已保存檢查點: {checkpoint_path}")

                # ========== 新增：定期紀錄梯度分布與GNS ========== #
                # 1. 收集所有參數的梯度
                gradients = {}
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        gradients[name] = param.grad.detach().cpu()
                self.save_manager.save_gradients(gradients, epoch)

                # 2. 收集本epoch所有batch的梯度（需在 _train_epoch 收集）
                if hasattr(self, '_epoch_batch_grads') and self._epoch_batch_grads:
                    # 將所有batch的梯度攤平成一維，組成一個list
                    grads_list = [g.view(-1) for g in self._epoch_batch_grads]
                    grads_tensor = torch.stack(grads_list)
                    # GNS計算：Var/MeanNorm^2
                    grad_mean = grads_tensor.mean(dim=0)
                    grad_var = grads_tensor.var(dim=0, unbiased=False)
                    mean_norm_sq = grad_mean.norm().item() ** 2
                    total_var = grad_var.sum().item()
                    gns = total_var / (mean_norm_sq + 1e-12)
                    gns_stats = {
                        'gns': gns,
                        'total_var': total_var,
                        'mean_norm_sq': mean_norm_sq,
                        'epoch': epoch,
                        'timestamp': datetime.now().isoformat(),
                        'reference': 'https://arxiv.org/abs/2006.08536'
                    }
                    self.save_manager.save_gns_stats(gns_stats, epoch)
                    # metadata logging
                    self._log_experiment_metadata({
                        'event': 'save_gns',
                        'epoch': epoch,
                        'gns': gns,
                        'total_var': total_var,
                        'mean_norm_sq': mean_norm_sq,
                        'timestamp': gns_stats['timestamp']
                    })
                    # 清空暫存
                    self._epoch_batch_grads = []
                # ========== END ========== #
            
            # 新增：調用 epoch 結束回調
            train_logs = {'loss': train_loss, 'metrics': train_metrics}
            val_logs = {'loss': val_loss, 'metrics': val_metrics}
            combined_logs = {
                'epoch': epoch, 
                'learning_rate': current_lr,
                'tensorboard_writer': self.writer
            }
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, self.model, train_logs, val_logs, combined_logs)
            
            logger.info(f"Epoch {epoch+1}/{self.epochs} - 訓練損失: {train_loss:.4f}, 驗證損失: {val_loss:.4f}")
        
        # 訓練結束
        elapsed_time = time.time() - start_time
        logger.info(f"訓練完成，耗時: {elapsed_time:.2f} 秒")
        
        # 新增：調用訓練結束回調
        history = {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'metrics': self.metrics,
            'best_val_loss': self.best_val_loss,
            'training_time': elapsed_time
        }
        final_logs = {'tensorboard_writer': self.writer}
        for callback in self.callbacks:
            callback.on_train_end(self.model, history, final_logs)
        
        # 關閉 TensorBoard writer
        self.writer.close()
        
        # 保存訓練歷史
        history_dict = {
            'train_loss': [float(loss) for loss in self.train_losses],
            'val_loss': [float(loss) for loss in self.val_losses],
            'best_val_loss': float(self.best_val_loss),
            'training_time': float(elapsed_time)
        }
        
        # 添加指標
        for metric_type in ['train', 'val']:
            for metric_name, values in self.metrics[metric_type].items():
                # 跳過我們添加的非標量指標
                if metric_name in ['outputs', 'targets', 'predictions']:
                    continue
                
                # 嘗試將指標值轉換為標量
                try:
                    history_dict[f'{metric_type}_{metric_name}'] = []
                    for val in values:
                        if isinstance(val, torch.Tensor):
                            if val.numel() == 1:  # 只有單元素張量才能轉換為標量
                                history_dict[f'{metric_type}_{metric_name}'].append(float(val.item()))
                            else:
                                # 對於多元素張量，使用平均值作為標量表示
                                logger.warning(f"指標 '{metric_name}' 是多元素張量，使用平均值作為標量表示")
                                history_dict[f'{metric_type}_{metric_name}'].append(float(val.mean().item()))
                        else:
                            history_dict[f'{metric_type}_{metric_name}'].append(float(val))
                except Exception as e:
                    logger.warning(f"添加指標 '{metric_name}' 到訓練歷史時出錯: {e}, 值類型: {type(values[0] if values else None)}")
        
        # 保存訓練歷史到文件
        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=4)
        
        # 如果最佳模型存在，載入它
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        
        # 返回訓練結果
        results = {
            'best_val_loss': float(self.best_val_loss),
            'history': history_dict,
            'best_model_path': self._save_best_model(self.epochs - 1, self.best_val_loss),
            'training_time': float(elapsed_time)
        }
        
        # 如果有測試集，進行評估
        if test_loader is not None:
            logger.info("在測試集上評估最佳模型...")
            
            # 新增：調用評估開始回調
            eval_logs = {'mode': 'test', 'tensorboard_writer': self.writer}
            for callback in self.callbacks:
                callback.on_evaluation_begin(self.model, eval_logs)
                
            test_loss, test_metrics = self._validate_epoch(test_loader, self.epochs - 1, is_test=True)
            
            # 記錄測試結果
            results['test_loss'] = float(test_loss)
            results['test_metrics'] = {k: float(v) for k, v in test_metrics.items()}
            
            # 保存測試預測結果
            if hasattr(self, 'test_predictions') and self.test_predictions is not None:
                pred_path = os.path.join(self.output_dir, 'test_predictions.pt')
                torch.save(self.test_predictions, pred_path)
                logger.info(f"測試預測已保存至 {pred_path}")
            
            # 輸出測試結果
            logger.info(f"測試損失: {test_loss:.4f}")
            for metric_name, metric_value in test_metrics.items():
                logger.info(f"測試 {metric_name}: {metric_value:.4f}")
                
            # 新增：調用評估結束回調
            eval_results = {'loss': test_loss, 'metrics': test_metrics}
            for callback in self.callbacks:
                callback.on_evaluation_end(self.model, eval_results, eval_logs)
        
        return results

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, Dict[str, float]]:
        """訓練一個 epoch
        
        Args:
            train_loader: 訓練數據加載器
            epoch: 當前 epoch 編號
            
        Returns:
            Tuple[float, Dict[str, float]]: 訓練損失和指標字典
        """
        self.model.train()
        total_loss = 0.0
        all_outputs = []
        all_targets = []
        # ========== 新增：本epoch所有batch梯度暫存 ========== #
        self._epoch_batch_grads = []
        # ========== END ========== #
        # 使用 tqdm 顯示進度條
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 準備批次數據
            batch = self._prepare_batch(batch)
            inputs = batch.get('inputs')
            targets = batch.get('targets')
            
            # 新增：調用批次開始回調
            batch_logs = {
                'batch_idx': batch_idx, 
                'epoch': epoch, 
                'phase': 'train',
                'tensorboard_writer': self.writer
            }
            for callback in self.callbacks:
                callback.on_batch_begin(batch_idx, self.model, inputs, targets, batch_logs)
                
            # 前向傳播
            outputs = self._forward_pass(batch)
            
            # 計算損失
            loss = self._compute_loss(outputs, batch)
            
            # 反向傳播
            self.optimizer.zero_grad()
            loss.backward()
            # ========== 新增：收集本batch梯度 ========== #
            grads = []
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grads.append(param.grad.detach().cpu().view(-1))
            if grads:
                self._epoch_batch_grads.append(torch.cat(grads))
            # ========== END ========== #
            
            # 優化步驟
            self.optimizer.step()
            
            # 更新總損失
            total_loss += loss.item()
            
            # 更新進度條
            progress_bar.set_postfix(loss=loss.item())
            
            # 收集預測和目標值，用於計算指標
            predictions = self._get_predictions(outputs)
            if predictions is not None and targets is not None:
                all_outputs.append(predictions.detach().cpu())
                all_targets.append(targets.detach().cpu())
                
            # 新增：調用批次結束回調
            batch_end_logs = {
                'batch_idx': batch_idx, 
                'epoch': epoch, 
                'loss': loss.item(),
                'phase': 'train',
                'tensorboard_writer': self.writer
            }
            for callback in self.callbacks:
                callback.on_batch_end(batch_idx, self.model, inputs, targets, outputs, loss, batch_end_logs)
        
        # 計算平均損失
        avg_loss = total_loss / len(train_loader)
        
        # 計算全局指標（僅當有輸出和目標時）
        metrics = {}
        if all_outputs and all_targets:
            try:
                # 將所有批次的輸出和目標連接起來
                combined_outputs = torch.cat(all_outputs, dim=0)
                combined_targets = torch.cat(all_targets, dim=0)
                
                # 計算指標
                metrics = self._compute_metrics(combined_outputs, {'targets': combined_targets})
            except Exception as e:
                logger.warning(f"計算訓練指標時出錯: {str(e)}")
        
        return avg_loss, metrics

    def _validate_epoch(self, val_loader: DataLoader, epoch: int, is_test: bool = False) -> Tuple[float, Dict[str, float]]:
        """驗證或測試一個 epoch
        
        Args:
            val_loader: 驗證數據加載器
            epoch: 當前 epoch 編號
            is_test: 是否為測試模式（而非驗證）
            
        Returns:
            Tuple[float, Dict[str, float]]: 驗證/測試損失和指標
        """
        self.model.eval()
        total_loss = 0.0
        all_outputs = []
        all_targets = []
        all_predictions = []
        all_raw_inputs = []  # 保存原始輸入，用於後續分析

        phase = "Test" if is_test else "Val"
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [{phase}]")
        
        # 儲存預測結果和真實值，用於後續分析
        all_batch_outputs = []
        all_batch_targets = []
        
        # 跟蹤批次數量
        batch_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                # 準備批次數據
                batch = self._prepare_batch(batch)
                inputs = batch.get('inputs')
                # 正確地獲取目標值，檢查 'label', 'score', 或 'target'
                actual_targets = batch.get('label', batch.get('score', batch.get('target', None)))
                
                # 新增：調用批次開始回調（驗證/測試階段）
                batch_logs = {
                    'batch_idx': batch_idx, 
                    'epoch': epoch,
                    'phase': 'test' if is_test else 'val',
                    'tensorboard_writer': self.writer
                }
                for callback in self.callbacks:
                    callback.on_batch_begin(batch_idx, self.model, inputs, actual_targets, batch_logs) # 傳遞 actual_targets
                
                # 前向傳播
                outputs = self._forward_pass(batch)
                
                # 計算損失 (內部會查找 'label', 'score', 'target')
                loss = self._compute_loss(outputs, batch)
                
                # 更新總損失
                total_loss += loss.item()
                batch_count += 1
                
                # 獲取預測結果
                predictions = self._get_predictions(outputs)
                
                # 使用 actual_targets 進行條件判斷和數據收集
                if predictions is not None and actual_targets is not None:
                    all_outputs.append(outputs.detach().cpu())
                    all_targets.append(actual_targets.detach().cpu()) # 使用 actual_targets
                    all_predictions.append(predictions.detach().cpu())
                    if inputs is not None:
                         # 確保 inputs 也是 Tensor 才進行 detach 和 cpu 操作
                         if isinstance(inputs, torch.Tensor):
                             all_raw_inputs.append(inputs.detach().cpu())
                         else:
                              # 如果 inputs 不是 Tensor (例如路徑列表)，直接附加或進行其他處理
                              # all_raw_inputs.append(inputs) # 根據需要調整
                              logger.debug(f"Batch inputs are not a tensor in {phase} phase, type: {type(inputs)}")
                else:
                     # 添加調試信息，了解為何數據未被收集
                     if predictions is None:
                          logger.debug(f"Skipping metrics accumulation in {phase}: predictions are None. Batch keys: {list(batch.keys())}")
                     if actual_targets is None:
                          logger.debug(f"Skipping metrics accumulation in {phase}: targets (label/score/target) are None. Batch keys: {list(batch.keys())}")

                
                # 更新進度條
                progress_bar.set_postfix({'loss': loss.item()})
                
                # 新增：調用批次結束回調（驗證/測試階段）
                batch_end_logs = {
                    'batch_idx': batch_idx, 
                    'epoch': epoch,
                    'loss': loss.item(),
                    'phase': 'test' if is_test else 'val',
                    'tensorboard_writer': self.writer
                }
                for callback in self.callbacks:
                    # 確保傳遞正確的 targets 給回調
                    callback.on_batch_end(batch_idx, self.model, inputs, actual_targets, outputs, loss, batch_end_logs)
        
        # 計算平均損失，添加保護以防止除以零錯誤
        if batch_count > 0:
            avg_loss = total_loss / batch_count
        else:
            logger.warning(f"{phase} 數據加載器為空，無法計算損失，返回零損失")
            avg_loss = 0.0
        
        # 計算全局指標
        metrics = {}
        if len(all_outputs) > 0 and len(all_targets) > 0:
            try:
                # 準備合併所有批次的數據
                combined_outputs = torch.cat(all_outputs, dim=0)
                combined_targets = torch.cat(all_targets, dim=0)
                combined_predictions = torch.cat(all_predictions, dim=0)
                
                # 計算指標 (修改這裡，傳遞包含 'targets' 的字典)
                metrics = self._compute_metrics(combined_predictions, {'targets': combined_targets})
                
                # 如果是測試模式，保存預測結果和真實值
                if is_test:
                    self.test_predictions = {
                        'outputs': combined_outputs,
                        'targets': combined_targets,
                        'predictions': combined_predictions
                    }
                    
                    # 如果有保存原始輸入，也加入到結果中
                    if all_raw_inputs:
                        try:
                            combined_inputs = torch.cat(all_raw_inputs, dim=0)
                            self.test_predictions['inputs'] = combined_inputs
                        except Exception as e:
                            logger.warning(f"合併測試輸入數據時出錯: {str(e)}")
                else:
                    # 為了在 epoch 結束回調中使用，將驗證集預測結果添加到指標字典中
                    metrics['outputs'] = combined_outputs
                    metrics['targets'] = combined_targets
                    metrics['predictions'] = combined_predictions
            except Exception as e:
                logger.warning(f"計算{phase.lower()}指標時出錯: {str(e)}")
        else:
            logger.warning(f"{phase} 數據集中沒有足夠的樣本進行評估，指標計算已跳過")
        
        return avg_loss, metrics

    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """準備批次數據：將數據移至適當的設備
        
        Args:
            batch: 批次數據字典
            
        Returns:
            Dict[str, Any]: 處理後的批次數據
        """
        prepared_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared_batch[key] = value.to(self.device)
            else:
                prepared_batch[key] = value
        return prepared_batch

    def _forward_pass(self, batch: Dict[str, Any]) -> torch.Tensor:
        """執行前向傳播
        
        Args:
            batch: 批次數據字典
            
        Returns:
            torch.Tensor: 模型輸出
        """
        # 優先選擇數據鍵的順序：spectrogram > image > features > audio > input
        if 'spectrogram' in batch:
            return self.model(batch['spectrogram'])
        elif 'image' in batch:
            return self.model(batch['image'])
        elif 'features' in batch:
            return self.model(batch['features'])
        elif 'audio' in batch:
            return self.model(batch['audio'])
        elif 'input' in batch:
            return self.model(batch['input'])
        else:
            # 記錄錯誤並顯示批次中可用的鍵
            available_keys = list(batch.keys())
            error_msg = f"批次數據中沒有可用的輸入鍵。可用的鍵有: {available_keys}"
            logger.error(error_msg)
            raise KeyError(error_msg)

    def _compute_loss(self, outputs: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """計算損失
        
        Args:
            outputs: 模型輸出
            batch: 數據批次
            
        Returns:
            loss: 計算得到的損失值
        """
        # 獲取標籤
        if isinstance(batch, dict):
            targets = batch.get('label', None)
            if targets is None:
                targets = batch.get('score', None)
                if targets is None:
                     targets = batch.get('target', None) # 添加對 'target' 的檢查
                     if targets is None:
                        raise ValueError("批次中找不到 'label', 'score' 或 'target' 鍵")
        else:
             # 假設 batch 是 (input, target) 形式的元組
             if len(batch) >= 2:
                targets = batch[1]
             else:
                raise ValueError("無法從非字典批次中獲取標籤")
        
        # 確保標籤在正確的設備上
        if isinstance(targets, torch.Tensor):
             targets = targets.to(self.device)
        else:
             # 如果標籤不是張量，嘗試轉換 (例如列表或numpy數組)
             try:
                 targets = torch.tensor(targets, device=self.device)
             except Exception as e:
                 raise TypeError(f"無法將標籤轉換為張量: {e}, 原始類型: {type(targets)}")

        # 調試信息 (減少冗餘輸出)
        # logger.debug("=== 損失計算調試信息 ===")
        # logger.debug(f"輸出形狀: {outputs.shape}")
        # logger.debug(f"標籤形狀: {targets.shape}")
        # logger.debug(f"標籤類型: {targets.dtype}")
        # unique_labels = torch.unique(targets)
        # logger.debug(f"唯一標籤值: {unique_labels}")
        
        # 獲取任務類型和損失函數類型
        is_classification = self.config.get('model', {}).get('parameters', {}).get('is_classification', True)
        loss_type = self.config.get('training', {}).get('loss', {}).get('type', '')
        
        # logger.debug(f"任務類型: {'分類' if is_classification else '回歸'}")
        # logger.debug(f"損失函數類型: {loss_type}")
        
        original_targets_dtype = targets.dtype
        original_targets_shape = targets.shape

        # 對標籤進行適當的轉換
        if is_classification:
            # 分類任務標籤處理
            if 'Rank' in loss_type or 'rank' in loss_type:
                targets = targets.float()
                if targets.dim() == 1 and outputs.dim() > 1 and outputs.size(1) != 1: # 僅在多輸出且標籤為1D時擴展
                    targets = targets.unsqueeze(1)
                    # logger.debug(f"為排序損失調整標籤形狀: {targets.shape}")
            elif loss_type == 'FocalLoss' or loss_type == 'CrossEntropyLoss' or loss_type == 'NLLLoss':
                 # 確保標籤是長整型，即使它們原本是浮點數 (例如 [0., 1., 0.])
                 # 但如果標籤值不是整數，轉換會出錯，需要額外處理
                 if not torch.all(targets == targets.long()):
                      logger.warning(f"分類任務標籤包含非整數值，損失計算可能出錯。標籤: {targets.unique()}")
                 # CrossEntropyLoss/NLLLoss/FocalLoss 需要 Long 類型的標籤
                 targets = targets.long()
            elif loss_type == 'BCEWithLogitsLoss':
                 # BCEWithLogitsLoss 需要 Float 類型的標籤
                 targets = targets.float()
                 # 如果輸出和目標維度不匹配 (常見於多標籤分類)
                 if outputs.shape != targets.shape:
                      if outputs.dim() > 1 and targets.dim() == 1:
                           # 嘗試將標籤 one-hot 編碼，如果適用
                           try:
                                num_classes = outputs.size(1)
                                targets_one_hot = torch.nn.functional.one_hot(targets.long(), num_classes=num_classes).float()
                                if outputs.shape == targets_one_hot.shape:
                                     targets = targets_one_hot
                                     # logger.debug(f"為 BCEWithLogitsLoss 將標籤 one-hot 編碼: {targets.shape}")
                                else:
                                     logger.warning(f"嘗試 one-hot 編碼後形狀仍不匹配: 輸出 {outputs.shape}, one-hot 標籤 {targets_one_hot.shape}")
                           except Exception as e:
                                logger.warning(f"嘗試 one-hot 編碼失敗: {e}")
                      # 如果輸出是 [N, 1] 而標籤是 [N]，調整標籤
                      elif outputs.dim() == 2 and outputs.size(1) == 1 and targets.dim() == 1:
                           targets = targets.unsqueeze(1)
                           # logger.debug(f"為 BCEWithLogitsLoss 調整標籤形狀: {targets.shape}")

            else:
                 # 其他分類損失，假設需要 Long
                 targets = targets.long()

            # 驗證標籤範圍 (僅對需要類別索引的損失)
            if loss_type in ['CrossEntropyLoss', 'NLLLoss', 'FocalLoss'] and outputs.dim() > 1 and outputs.size(1) > 1:
                if targets.numel() > 0: # 確保 targets 不為空
                    max_label = torch.max(targets).item()
                    min_label = torch.min(targets).item()
                    num_classes = outputs.size(1)
                    # logger.debug(f"標籤範圍: [{min_label}, {max_label}]")
                    # logger.debug(f"模型輸出類別數: {num_classes}")
                    if max_label >= num_classes or min_label < 0:
                        logger.error(f"標籤值超出有效範圍 [0, {num_classes-1}]。檢測到範圍 [{min_label}, {max_label}]")
                        # 可以選擇拋出錯誤或進行裁剪/警告
                        # targets = torch.clamp(targets, 0, num_classes - 1)
                        # logger.warning("標籤值已被裁剪到有效範圍")
                        raise ValueError(f"標籤值 {max_label} 或 {min_label} 超出模型輸出類別數 {num_classes} 的有效範圍 [0, {num_classes-1}]")
                else:
                     logger.warning("標籤張量為空，無法檢查範圍")
        else:
            # 回歸任務通常需要浮點型標籤
            targets = targets.float()
            
            # 如果輸出和目標維度不匹配
            if outputs.shape != targets.shape:
                # logger.debug(f"回歸任務中形狀不匹配: 輸出 {outputs.shape}, 標籤 {targets.shape}")
                # 情況 1: 輸出是 [N, 1], 標籤是 [N] -> 調整標籤
                if outputs.dim() == 2 and outputs.size(1) == 1 and targets.dim() == 1:
                    targets = targets.unsqueeze(1)
                    # logger.debug(f"調整回歸標籤形狀: {targets.shape}")
                # 情況 2: 輸出是 [N], 標籤是 [N, 1] -> 調整輸出 (通常不建議修改模型輸出) 或標籤
                elif outputs.dim() == 1 and targets.dim() == 2 and targets.size(1) == 1:
                     targets = targets.squeeze(1)
                     # logger.debug(f"調整回歸標籤形狀: {targets.shape}")
                # 情況 3: 其他不匹配，可能需要檢查模型或數據加載器
                else:
                     logger.warning(f"無法自動解決的回歸任務形狀不匹配: 輸出 {outputs.shape}, 標籤 {targets.shape}")

        # 特殊處理某些具有特定輸入要求的損失函數
        if 'BCE' in loss_type and 'Logits' not in loss_type:
            # BCE需要0-1範圍的輸出，應用sigmoid
            # 但通常建議使用 BCEWithLogitsLoss 以提高數值穩定性
            logger.warning("檢測到使用 BCELoss。推薦使用 BCEWithLogitsLoss 以獲得更好的數值穩定性。")
            outputs = torch.sigmoid(outputs)
            # logger.debug(f"為BCE損失應用sigmoid到輸出")
        
        # 使用criterion計算損失
        try:
            # logger.debug(f"最終計算損失前: 輸出形狀 {outputs.shape}, 標籤形狀 {targets.shape}, 標籤類型 {targets.dtype}")
            loss = self.criterion(outputs, targets)
            # logger.debug(f"損失計算成功: {loss.item()}")
            return loss
        except Exception as e:
            logger.error(f"計算損失時出錯: {str(e)}")
            logger.error(f"原始標籤形狀: {original_targets_shape}, 原始標籤類型: {original_targets_dtype}")
            logger.error(f"嘗試計算時: 輸出形狀: {outputs.shape}, 標籤形狀: {targets.shape}, 標籤類型: {targets.dtype}")
            logger.error(f"損失函數類型: {type(self.criterion)}")
            
            # 嘗試更多自動調整 (謹慎使用)
            # 例如，如果標籤是 float 而損失需要 long
            if isinstance(e, RuntimeError) and "expected scalar type Long" in str(e) and targets.dtype == torch.float:
                 try:
                     logger.warning("檢測到類型不匹配 (需要 Long)，嘗試將標籤轉換為 Long")
                     targets_adjusted = targets.long()
                     loss = self.criterion(outputs, targets_adjusted)
                     logger.info(f"調整標籤類型後損失計算成功")
                     return loss
                 except Exception as inner_e:
                      logger.error(f"調整標籤類型後計算失敗: {inner_e}")
                      raise e # 重新拋出原始錯誤

            # 如果是形狀不匹配
            if isinstance(e, (ValueError, RuntimeError)) and ("shape" in str(e) or "size" in str(e)):
                 logger.error("檢測到形狀不匹配，請檢查模型輸出維度和數據加載器。")
                 # 可以嘗試 squeeze/unsqueeze，但可能掩蓋根本問題
                 # if outputs.dim() == targets.dim() + 1 and outputs.size(-1) == 1:
                 #     outputs_squeezed = outputs.squeeze(-1)
                 #     if outputs_squeezed.shape == targets.shape: ...
                 # elif targets.dim() == outputs.dim() + 1 and targets.size(-1) == 1:
                 #     targets_squeezed = targets.squeeze(-1)
                 #     if outputs.shape == targets_squeezed.shape: ...
            
            raise e # 重新拋出原始錯誤

    def _compute_metrics(self, outputs, batch):
        """計算評估指標
        
        Args:
            outputs: 模型輸出
            batch: 數據批次
            
        Returns:
            dict: 包含各項指標的字典
        """
        # # 這個註解是為了說明這個程式碼的功能
        metrics = {}
        try:
            # 獲取標籤
            if isinstance(batch, dict):
                targets = batch.get('label', None)
                if targets is None:
                    targets = batch.get('score', None)
                    if targets is None:
                        targets = batch.get('target', None) # 添加對 'target' 的檢查
                        if targets is None:
                            # 嘗試獲取 'targets' 鍵 (這可能是在評估階段由 _validate_epoch 傳遞的)
                            targets = batch.get('targets', None)
                            if targets is None:
                                logger.warning("在指標計算中找不到 'label', 'score', 'target' 或 'targets' 鍵，跳過指標計算。")
                                logger.debug(f"可用的鍵: {list(batch.keys() if isinstance(batch, dict) else [])}")
                                return {} # 返回空字典
            else:
                # 假設 batch 是 (input, target) 形式的元組
                if len(batch) >= 2:
                    targets = batch[1]
                else:
                     logger.warning("無法從非字典批次中獲取標籤，跳過指標計算。")
                     return {} # 返回空字典

            # 確保標籤在 CPU 上並且是 Tensor (因為 sklearn 需要 numpy)
            if isinstance(targets, torch.Tensor):
                targets = targets.cpu()
            else:
                try:
                    targets = torch.tensor(targets).cpu()
                except Exception as e:
                    logger.warning(f"無法將指標計算的標籤轉換為張量: {e}, 原始類型: {type(targets)}。跳過指標計算。")
                    return {}

            # 確保輸出在 CPU 上
            outputs = outputs.detach().cpu()

            # 獲取任務類型和損失函數類型
            is_classification = self.config.get('model', {}).get('parameters', {}).get('is_classification', True)
            loss_type = self.config.get('training', {}).get('loss', {}).get('type', '')
            
            # 針對特定損失函數類型的處理 (排序指標)
            if 'Rank' in loss_type or 'rank' in loss_type:
                return self._compute_ranking_metrics(outputs, targets) # targets 已經在 CPU 上
            
            # 獲取預測結果 (在 CPU 上)
            if is_classification:
                # 對於分類任務
                if outputs.dim() > 1 and outputs.size(1) > 1: # 多分類
                    predictions = torch.argmax(outputs, dim=1)
                else: # 二分類或單輸出分類
                     # 使用 sigmoid 獲取概率，然後閾值化
                     # 如果損失是 BCEWithLogitsLoss，輸出是 logits，需要 sigmoid
                     # 如果損失是 BCELoss，輸出已經是概率 (理論上)
                     if 'Logits' in loss_type or outputs.min() < 0 or outputs.max() > 1: # 假設是 logits
                          outputs_prob = torch.sigmoid(outputs)
                     else: # 假設已經是概率
                          outputs_prob = outputs
                     
                     # 如果輸出是 [N, 1]，壓縮維度
                     if outputs_prob.dim() == 2 and outputs_prob.size(1) == 1:
                          outputs_prob = outputs_prob.squeeze(1)

                     predictions = (outputs_prob > 0.5).long() # 使用 .long() 匹配標籤類型
                
                # 確保標籤是 LongTensor
                targets = targets.long()

            else:
                # 回歸任務，直接使用輸出作為預測
                predictions = outputs
                if predictions.dim() > 1 and predictions.size(1) == 1:
                    predictions = predictions.squeeze(1)
                 # 確保標籤是 FloatTensor
                targets = targets.float()

            # ----- 指標計算 -----
            # 確保預測和標籤形狀兼容
            if predictions.shape != targets.shape:
                 # 嘗試壓縮單一維度
                 if predictions.numel() == targets.numel():
                      logger.warning(f"指標計算中形狀不匹配: 預測 {predictions.shape}, 標籤 {targets.shape}。嘗試壓縮維度。")
                      predictions = predictions.view(-1)
                      targets = targets.view(-1)
                 else:
                      logger.error(f"指標計算中形狀不匹配且元素數量不同: 預測 {predictions.shape}, 標籤 {targets.shape}。跳過指標計算。")
                      return {} # 無法計算指標

            # 轉換為 numpy 數組
            y_true = targets.numpy()
            y_pred = predictions.numpy()

            if is_classification:
                # 準確率
                try:
                    from sklearn.metrics import accuracy_score
                    metrics['accuracy'] = accuracy_score(y_true, y_pred)
                except Exception as e:
                    logger.warning(f"計算準確率時出錯: {str(e)}. y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
                    metrics['accuracy'] = np.nan # 使用 nan 表示計算失敗

                # 其他分類指標 (使用 zero_division=0 避免警告/錯誤)
                common_args = {'y_true': y_true, 'y_pred': y_pred, 'average': 'macro', 'zero_division': 0}
                try:
                    from sklearn.metrics import f1_score
                    metrics['f1_score'] = f1_score(**common_args)
                except Exception as e:
                    logger.warning(f"計算F1分數時出錯: {str(e)}")
                    metrics['f1_score'] = np.nan
                
                try:
                    from sklearn.metrics import precision_score
                    metrics['precision'] = precision_score(**common_args)
                except Exception as e:
                    logger.warning(f"計算精確度時出錯: {str(e)}")
                    metrics['precision'] = np.nan
                
                try:
                    from sklearn.metrics import recall_score
                    metrics['recall'] = recall_score(**common_args)
                except Exception as e:
                    logger.warning(f"計算召回率時出錯: {str(e)}")
                    metrics['recall'] = np.nan
            else:
                # 回歸任務
                common_args_reg = {'y_true': y_true, 'y_pred': y_pred}
                try:
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, max_error
                    metrics['mse'] = mean_squared_error(**common_args_reg)
                    metrics['rmse'] = np.sqrt(metrics['mse'])
                    metrics['mae'] = mean_absolute_error(**common_args_reg)
                    # R2 可能因模型非常差而為負
                    metrics['r2'] = r2_score(**common_args_reg)
                    metrics['explained_variance'] = explained_variance_score(**common_args_reg)
                    metrics['max_error'] = max_error(**common_args_reg)
                    
                    # MAPE (避免除以零)
                    epsilon = 1e-10
                    non_zero_indices = np.abs(y_true) > epsilon
                    if np.any(non_zero_indices):
                        mape = np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100
                        metrics['mape'] = mape
                    else:
                        metrics['mape'] = np.nan # 如果所有真實值都接近於零

                except Exception as e:
                    logger.warning(f"計算回歸指標時出錯: {str(e)}. y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
                    metrics['mse'] = metrics['rmse'] = metrics['mae'] = metrics['r2'] = metrics['explained_variance'] = metrics['max_error'] = metrics['mape'] = np.nan

        except Exception as e:
            logger.error(f"計算指標時發生未預期錯誤: {str(e)}")
            # 返回包含 NaN 的指標，表示計算失敗
            metrics = {k: np.nan for k in metrics} # 將已有的指標設為 NaN
            # 可以考慮添加一個錯誤標記
            metrics['metric_error'] = 1.0 

        # 過濾掉 NaN 值，因為 TensorBoard 不接受 NaN
        final_metrics = {k: v for k, v in metrics.items() if not np.isnan(v)}
        if len(final_metrics) < len(metrics):
             nan_keys = [k for k, v in metrics.items() if np.isnan(v)]
             logger.warning(f"指標計算結果包含 NaN，已過濾: {nan_keys}")

        return final_metrics

    def _compute_ranking_metrics(self, outputs, targets):
        """計算排序指標
        
        Args:
            outputs: 模型輸出 (在 CPU 上)
            targets: 標籤數據 (在 CPU 上)
            
        Returns:
            dict: 包含排序指標的字典
        """
        # # 這個註解是為了說明這個程式碼的功能
        metrics = {}
        
        # 確保輸入是 numpy 數組
        try:
            outputs_np = outputs.numpy()
            targets_np = targets.numpy()
        except Exception as e:
             logger.error(f"轉換排序指標輸入為 numpy 時出錯: {e}")
             return {} # 無法計算

        # 確保輸入是一維的
        if outputs_np.ndim > 1:
            outputs_np = outputs_np.squeeze()
        if targets_np.ndim > 1:
            targets_np = targets_np.squeeze()

        # 檢查維度是否仍然不匹配或不是一維
        if outputs_np.ndim != 1 or targets_np.ndim != 1 or outputs_np.shape != targets_np.shape:
            logger.warning(f"計算排序指標的輸入維度不正確或不匹配: 輸出 {outputs_np.shape}, 標籤 {targets_np.shape}。跳過計算。")
            return {}

        # 檢查是否有足夠的數據點 (至少需要 2 個點來計算相關性)
        if len(outputs_np) < 2:
             logger.warning(f"數據點不足 ({len(outputs_np)}) 無法計算排序指標。")
             return {}

        try:
            # Spearman 相關性
            from scipy.stats import spearmanr
            # 檢查是否有 NaN 或 Inf
            if np.any(np.isnan(outputs_np)) or np.any(np.isnan(targets_np)) or \
               np.any(np.isinf(outputs_np)) or np.any(np.isinf(targets_np)):
                 logger.warning("輸入包含 NaN 或 Inf，無法計算 Spearman 相關性。")
                 metrics['spearman'] = np.nan
            # 檢查方差是否為零 (如果所有值都相同，相關性無定義)
            elif np.var(outputs_np) == 0 or np.var(targets_np) == 0:
                 logger.warning("輸入方差為零，Spearman 相關性無定義。")
                 metrics['spearman'] = np.nan # 或 0 或 1 取決於定義
            else:
                 corr, p_value = spearmanr(outputs_np, targets_np)
                 metrics['spearman'] = corr if not np.isnan(corr) else np.nan # 確保結果不是 NaN

        except ImportError:
            logger.warning("無法導入 scipy.stats，跳過 Spearman 相關性計算。")
            metrics['spearman'] = np.nan
        except Exception as e:
            logger.warning(f"計算 Spearman 相關性時出錯: {e}")
            metrics['spearman'] = np.nan

        try:
            # Kendall's Tau
            from scipy.stats import kendalltau
            if np.any(np.isnan(outputs_np)) or np.any(np.isnan(targets_np)) or \
               np.any(np.isinf(outputs_np)) or np.any(np.isinf(targets_np)):
                 logger.warning("輸入包含 NaN 或 Inf，無法計算 Kendall Tau。")
                 metrics['kendall_tau'] = np.nan
            elif np.var(outputs_np) == 0 or np.var(targets_np) == 0:
                 logger.warning("輸入方差為零，Kendall Tau 可能無定義或為零。")
                 # kendalltau 對常量數組通常返回 (nan, nan) 或 (0, 1)
                 tau, _ = kendalltau(outputs_np, targets_np)
                 metrics['kendall_tau'] = tau if not np.isnan(tau) else np.nan
            else:
                 tau, _ = kendalltau(outputs_np, targets_np)
                 metrics['kendall_tau'] = tau if not np.isnan(tau) else np.nan

        except ImportError:
            logger.warning("無法導入 scipy.stats，跳過 Kendall Tau 計算。")
            metrics['kendall_tau'] = np.nan
        except Exception as e:
            logger.warning(f"計算 Kendall Tau 時出錯: {e}")
            metrics['kendall_tau'] = np.nan

        # MRR (通常用於信息檢索或推薦，這裡可能不太適用，除非有明確的排名任務)
        # 如果需要，可以根據具體任務邏輯實現

        # 過濾 NaN
        final_metrics = {k: v for k, v in metrics.items() if not np.isnan(v)}
        if len(final_metrics) < len(metrics):
             nan_keys = [k for k, v in metrics.items() if np.isnan(v)]
             logger.warning(f"排序指標計算結果包含 NaN，已過濾: {nan_keys}")

        return final_metrics

    def _get_predictions(self, outputs: torch.Tensor) -> torch.Tensor:
        """獲取預測結果
        
        Args:
            outputs: 模型輸出
            
        Returns:
            torch.Tensor: 預測結果
        """
        # 根據任務類型獲取預測
        is_classification = self.config.get('model', {}).get('parameters', {}).get('is_classification', True)
        
        if is_classification:
            # 分類任務
            if outputs.dim() > 1 and outputs.size(1) > 1:
                # 多分類，輸出softmax
                return torch.softmax(outputs, dim=1)
            else:
                # 二分類，輸出sigmoid
                return torch.sigmoid(outputs)
        else:
            # 回歸任務，直接返回輸出
            return outputs

    def _get_loss_function(self) -> Callable:
        """獲取損失函數
        
        Returns:
            Callable: 損失函數
            
        Description:
            根據配置創建損失函數
        """
        # 獲取損失函數類型和參數
        loss_config = self.config.get('training', {}).get('loss', {})
        
        # 使用LossFactory創建損失函數
        try:
            # 嘗試使用LossFactory創建損失函數
            criterion = LossFactory.create_from_config(loss_config)
            logger.info(f"使用LossFactory創建損失函數成功: {loss_config.get('type', '未指定類型')}")
            return criterion
        except Exception as e:
            logger.warning(f"使用LossFactory創建損失函數失敗: {str(e)}")
            logger.warning("回退到內置損失函數創建方法")
            
            # 如果LossFactory失敗，回退到內置方法
            loss_type = loss_config.get('type', 'MSELoss')
            loss_params = loss_config.get('parameters', {})
            
            logger.info(f"使用內置損失函數: {loss_type}，參數: {loss_params}")
            
            # 根據損失函數類型創建損失函數實例
            if loss_type == 'MSELoss':
                return nn.MSELoss(**loss_params)
            elif loss_type == 'CrossEntropyLoss':
                return nn.CrossEntropyLoss(**loss_params)
            elif loss_type == 'L1Loss':
                return nn.L1Loss(**loss_params)
            elif loss_type == 'SmoothL1Loss':
                return nn.SmoothL1Loss(**loss_params)
            elif loss_type == 'BCELoss':
                return nn.BCELoss(**loss_params)
            elif loss_type == 'BCEWithLogitsLoss':
                return nn.BCEWithLogitsLoss(**loss_params)
            elif loss_type == 'NLLLoss':
                return nn.NLLLoss(**loss_params)
            elif loss_type == 'KLDivLoss':
                return nn.KLDivLoss(**loss_params)
            else:
                logger.warning(f"未知的損失函數類型: {loss_type}，使用默認的MSELoss")
                return nn.MSELoss()

    def evaluate(self, test_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """評估模型性能
        
        Args:
            test_loader: 測試數據加載器，如果不提供則使用訓練時的測試加載器
            
        Returns:
            Dict[str, float]: 評估結果，包含損失和指標
            
        Description:
            在測試集上評估模型性能
        """
        logger.info("開始評估模型...")
        
        if test_loader is None:
            logger.warning("未提供測試加載器，無法執行評估")
            return {'error': 'No test loader provided'}
        
        # 切換到評估模式
        self.model.eval()
        
        total_loss = 0.0
        all_metrics = {}
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                # 準備批次數據
                batch = self._prepare_batch(batch)
                
                # 前向傳播
                outputs = self._forward_pass(batch)
                
                # 計算損失
                loss = self._compute_loss(outputs, batch)
                total_loss += loss.item()
                
                # 計算指標
                metrics = self._compute_metrics(outputs, batch)
                
                for metric_name, metric_value in metrics.items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(metric_value)
        
        # 計算平均損失和指標
        avg_loss = total_loss / len(test_loader)
        avg_metrics = {name: sum(values) / len(values) for name, values in all_metrics.items()}
        
        logger.info(f"評估完成 - 損失: {avg_loss:.4f}")
        for name, value in avg_metrics.items():
            logger.info(f"{name}: {value:.4f}")
        
        result = {'loss': avg_loss}
        result.update(avg_metrics)
        
        return result 

    def _save_checkpoint(self, epoch: int, val_loss: float) -> str:
        """保存檢查點
        
        Args:
            epoch: 當前 epoch
            val_loss: 驗證損失
            
        Returns:
            str: 保存的檢查點路徑
        """
        # 使用 SaveManager 保存檢查點
        return self.save_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            loss=val_loss,
            filename='checkpoint.pth',
            additional_data={
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'metrics': self.metrics
            }
        )
    
    def _save_best_model(self, epoch: int, val_loss: float) -> str:
        """保存最佳模型
        
        Args:
            epoch: 當前 epoch
            val_loss: 驗證損失
            
        Returns:
            str: 保存的模型路徑
        """
        # 使用 SaveManager 保存最佳模型
        save_path = self.save_manager.save_model(
            model=self.model,
            filename='best_model.pth',
            additional_data={
                'epoch': epoch,
                'val_loss': val_loss,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'metrics': self.metrics
            }
        )
        
        # 保存模型結構信息
        self._save_model_structure()
        
        return save_path
        
    def _save_model_structure(self) -> str:
        """保存模型結構信息
        
        Returns:
            str: 保存的結構信息路徑
        """
        # 獲取模型結構信息
        model_structure = ModelStructureInfo(self.model)
        structure_dict = model_structure.get_structure_dict()
        
        # 保存模型結構信息到 JSON 文件
        structure_path = os.path.join(self.output_dir, 'model_structure.json')
        with open(structure_path, 'w') as f:
            json.dump(structure_dict, f, indent=2)
            
        logger.info(f"模型結構信息已保存到 {structure_path}")
        return structure_path 

    def _log_experiment_metadata(self, meta: Dict[str, Any]):
        """將metadata寫入 experiments.log
        Args:
            meta: metadata 字典
        Returns:
            None
        Description:
            將metadata以JSONL格式寫入 results/{exp_id}/logs/experiments.log
        References:
            無
        """
        # 中文註解：確保logs目錄存在
        logs_dir = self.save_manager.get_path('logs', '')
        os.makedirs(logs_dir, exist_ok=True)
        log_path = os.path.join(logs_dir, 'experiments.log')
        with open(log_path, 'a') as f:
            f.write(json.dumps(meta, ensure_ascii=False) + '\n') 

# 中文註解：這是pytorch_trainer.py的Minimal Executable Unit，檢查PyTorchTrainer能否正確初始化與執行train流程，並測試錯誤配置時的優雅報錯
if __name__ == "__main__":
    """
    Description: Minimal Executable Unit for pytorch_trainer.py，檢查PyTorchTrainer能否正確初始化與執行train流程，並測試錯誤配置時的優雅報錯。
    Args: None
    Returns: None
    References: 無
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import logging
    logging.basicConfig(level=logging.INFO)
    # 不要再import自己
    # from trainers.pytorch_trainer import PyTorchTrainer

    # Dummy model
    class DummyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)
        def forward(self, x):
            return self.fc(x)

    # Dummy data
    x = torch.randn(20, 4)
    y = torch.randint(0, 2, (20,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=4)

    # 正確配置
    config = {
        "model": {"type": "fcnn", "parameters": {"is_classification": True}},
        "training": {"epochs": 2, "learning_rate": 0.01, "loss": {"type": "CrossEntropyLoss", "parameters": {}}},
        "data": {"type": "feature"},
        "global": {"experiment_name": "dummy_exp", "output_dir": "results"}
    }

    try:
        model = DummyNet()
        trainer = PyTorchTrainer(config, model)
        # 需要包裝dataloader為dict格式
        def dict_loader(dl):
            for xb, yb in dl:
                yield {"features": xb, "label": yb}
        train_loader = dict_loader(loader)
        val_loader = dict_loader(loader)
        trainer.train(train_loader, val_loader)
        print("PyTorchTrainer測試成功")
    except Exception as e:
        print(f"PyTorchTrainer遇到錯誤（預期行為）: {e}")

    # 錯誤配置測試
    try:
        bad_config = {"model": {}, "training": {}, "data": {}, "global": {}}
        model = DummyNet()
        trainer = PyTorchTrainer(bad_config, model)
        train_loader = dict_loader(loader)
        val_loader = dict_loader(loader)
        trainer.train(train_loader, val_loader)
    except Exception as e:
        print(f"PyTorchTrainer遇到錯誤配置時的報錯（預期行為）: {e}") 