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

# 引入數據適配器
from utils.data_adapter import adapt_datasets_to_model, DataAdapter

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
        self.output_dir = os.path.join(
            config.get('output_dir', 'outputs'),
            f"{config.get('model', {}).get('type', 'model')}_{timestamp}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"訓練輸出將保存至 {self.output_dir}")
        
        # 保存配置
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

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
        
        for epoch in range(self.epochs):
            # 訓練一個 epoch
            train_loss, train_metrics = self._train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            # 驗證一個 epoch
            val_loss, val_metrics = self._validate_epoch(val_loader, epoch)
            self.val_losses.append(val_loss)
            
            # 更新指標記錄
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
                best_model_path = os.path.join(self.output_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.best_state_dict,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config
                }, best_model_path)
                
                logger.info(f"Epoch {epoch+1}/{self.epochs} - 保存新的最佳模型: {best_model_path}")
            elif self.early_stopping_enabled:  # 只有在啟用早期停止時才減少耐心值
                self.patience -= 1
                if self.patience <= 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs} - 提前停止訓練！")
                    break
            
            # 保存當前 epoch 的模型
            current_model_path = os.path.join(self.output_dir, f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': {k: v.cpu() for k, v in self.model.state_dict().items()},
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': self.config
            }, current_model_path)
            
            logger.info(f"Epoch {epoch+1}/{self.epochs} - 訓練損失: {train_loss:.4f}, 驗證損失: {val_loss:.4f}")
        
        # 訓練結束
        elapsed_time = time.time() - start_time
        logger.info(f"訓練完成，耗時: {elapsed_time:.2f} 秒")
        
        # 保存訓練歷史
        history = {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'metrics': self.metrics,
            'best_val_loss': self.best_val_loss,
            'training_time': elapsed_time
        }
        
        with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=4)
        
        # 加載最佳模型進行評估
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        
        # 如果提供了測試數據，則進行測試評估
        test_results = {}
        if test_loader is not None:
            logger.info("在測試集上評估最佳模型...")
            test_loss, test_metrics = self._validate_epoch(test_loader, -1, is_test=True)
            test_results = {
                'test_loss': test_loss,
                'test_metrics': test_metrics
            }
            logger.info(f"測試損失: {test_loss:.4f}")
            for metric_name, metric_value in test_metrics.items():
                logger.info(f"測試 {metric_name}: {metric_value:.4f}")
            
            # 保存測試結果
            with open(os.path.join(self.output_dir, 'test_results.json'), 'w') as f:
                json.dump(test_results, f, indent=4)
        
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'metrics': self.metrics,
            'best_model_path': os.path.join(self.output_dir, 'best_model.pth'),
            'test_results': test_results
        }

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, Dict[str, float]]:
        """訓練一個 epoch
        
        Args:
            train_loader: 訓練數據加載器
            epoch: 當前 epoch 索引
            
        Returns:
            Tuple[float, Dict[str, float]]: 平均訓練損失和指標
        """
        self.model.train()
        total_loss = 0.0
        batch_metrics = {}
        
        # 創建進度條
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 提取數據並移至設備
            batch = self._prepare_batch(batch)
            
            # 檢查並調整數據格式
            model_type = self.config.get('model', {}).get('type', '')
            batch = DataAdapter.adapt_batch(batch, model_type, self.config)
            
            # 清除梯度
            self.optimizer.zero_grad()
            
            # 前向傳播
            outputs = self._forward_pass(batch)
            
            # 計算損失
            loss = self._compute_loss(outputs, batch)
            
            # 反向傳播和優化
            loss.backward()
            self.optimizer.step()
            
            # 更新累計損失
            total_loss += loss.item()
            
            # 計算和更新指標
            metrics = self._compute_metrics(outputs, batch)
            for metric_name, metric_value in metrics.items():
                if metric_name not in batch_metrics:
                    batch_metrics[metric_name] = []
                batch_metrics[metric_name].append(metric_value)
            
            # 更新進度條訊息
            progress_bar.set_postfix({'loss': loss.item()})
        
        # 計算平均損失和指標
        avg_loss = total_loss / len(train_loader)
        avg_metrics = {name: np.mean(values) for name, values in batch_metrics.items()}
        
        return avg_loss, avg_metrics

    def _validate_epoch(self, val_loader: DataLoader, epoch: int, is_test: bool = False) -> Tuple[float, Dict[str, float]]:
        """驗證一個 epoch
        
        Args:
            val_loader: 驗證數據加載器
            epoch: 當前 epoch 索引
            is_test: 是否為測試評估
            
        Returns:
            Tuple[float, Dict[str, float]]: 平均驗證損失和指標
        """
        self.model.eval()
        total_loss = 0.0
        batch_metrics = {}
        
        # 預測和標籤收集器
        all_predictions = []
        all_targets = []
        
        # 創建進度條
        desc = f"Epoch {epoch+1}/{self.epochs} [{'Test' if is_test else 'Val'}]"
        progress_bar = tqdm(val_loader, desc=desc)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                # 提取數據並移至設備
                batch = self._prepare_batch(batch)
                
                # 檢查並調整數據格式
                model_type = self.config.get('model', {}).get('type', '')
                batch = DataAdapter.adapt_batch(batch, model_type, self.config)
                
                # 前向傳播
                outputs = self._forward_pass(batch)
                
                # 計算損失
                loss = self._compute_loss(outputs, batch)
                
                # 更新累計損失
                total_loss += loss.item()
                
                # 計算和更新指標
                metrics = self._compute_metrics(outputs, batch)
                for metric_name, metric_value in metrics.items():
                    if metric_name not in batch_metrics:
                        batch_metrics[metric_name] = []
                    batch_metrics[metric_name].append(metric_value)
                
                # 收集預測和標籤
                predictions = self._get_predictions(outputs)
                
                # 獲取標籤
                if 'score' in batch:
                    targets = batch['score']
                elif 'label' in batch:
                    targets = batch['label']
                elif 'target' in batch:
                    targets = batch['target']
                else:
                    targets = None
                    logger.warning(f"無法找到標籤，跳過收集預測。批次鍵: {list(batch.keys())}")
                
                if predictions is not None and targets is not None:
                    all_predictions.append(predictions.cpu())
                    all_targets.append(targets.cpu())
                
                # 更新進度條訊息
                progress_bar.set_postfix({'loss': loss.item()})
        
        # 計算平均損失和指標
        avg_loss = total_loss / len(val_loader)
        avg_metrics = {name: np.mean(values) for name, values in batch_metrics.items()}
        
        # 如果是測試評估，保存預測結果
        if is_test and all_predictions and all_targets:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            # 保存預測結果
            predictions_path = os.path.join(self.output_dir, 'test_predictions.pt')
            torch.save({
                'predictions': all_predictions,
                'targets': all_targets
            }, predictions_path)
            
            logger.info(f"測試預測已保存至 {predictions_path}")
        
        return avg_loss, avg_metrics

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
                raise ValueError("批次中找不到'label'鍵")
        else:
            targets = batch[1]
        
        # 調試信息
        logger.info("=== 損失計算調試信息 ===")
        logger.info(f"輸出形狀: {outputs.shape}")
        logger.info(f"標籤形狀: {targets.shape}")
        logger.info(f"標籤值: {targets}")
        logger.info(f"標籤類型: {targets.dtype}")
        unique_labels = torch.unique(targets)
        logger.info(f"唯一標籤值: {unique_labels}")
        
        # 確保標籤是長整型
        if self.config.get('model', {}).get('parameters', {}).get('is_classification', True):
            targets = targets.long()
            # 驗證標籤範圍
            max_label = torch.max(targets).item()
            min_label = torch.min(targets).item()
            num_classes = outputs.size(1)
            logger.info(f"標籤範圍: [{min_label}, {max_label}]")
            logger.info(f"模型輸出類別數: {num_classes}")
            
            if max_label >= num_classes:
                raise ValueError(f"標籤值 {max_label} 超出模型輸出類別數 {num_classes}")
        else:
            # 回歸任務不需要驗證標籤範圍
            logger.info("回歸任務，不驗證標籤範圍")
        
        return self.criterion(outputs, targets)

    def _compute_metrics(self, outputs, batch):
        """計算評估指標
        
        Args:
            outputs: 模型輸出
            batch: 數據批次
            
        Returns:
            dict: 包含各項指標的字典
        """
        # 獲取標籤
        if isinstance(batch, dict):
            targets = batch.get('label', None)
            if targets is None:
                raise ValueError("批次中找不到'label'鍵")
        else:
            targets = batch[1]
        
        # 確保標籤是長整型
        targets = targets.long()
        
        # 獲取預測結果
        is_classification = self.config.get('model', {}).get('parameters', {}).get('is_classification', True)
        
        if is_classification:
            predictions = torch.argmax(outputs, dim=1)
        else:
            # 回歸任務，直接使用輸出作為預測
            predictions = outputs
        
        # 計算指標
        metrics = {}
        
        if is_classification:
            # 準確率
            correct = (predictions == targets).float()
            accuracy = torch.mean(correct)
            metrics['accuracy'] = accuracy.item()
            
            # 轉換為 numpy 數組以使用 sklearn 的指標
            y_true = targets.cpu().numpy()
            y_pred = predictions.cpu().numpy()
            
            # F1分數 (macro平均)
            try:
                from sklearn.metrics import f1_score
                metrics['f1_score'] = f1_score(y_true, y_pred, average='macro')
            except Exception as e:
                logger.warning(f"計算F1分數時出錯: {str(e)}")
                metrics['f1_score'] = 0.0
            
            # 精確度 (macro平均)
            try:
                from sklearn.metrics import precision_score
                metrics['precision'] = precision_score(y_true, y_pred, average='macro')
            except Exception as e:
                logger.warning(f"計算精確度時出錯: {str(e)}")
                metrics['precision'] = 0.0
            
            # 召回率 (macro平均)
            try:
                from sklearn.metrics import recall_score
                metrics['recall'] = recall_score(y_true, y_pred, average='macro')
            except Exception as e:
                logger.warning(f"計算召回率時出錯: {str(e)}")
                metrics['recall'] = 0.0
        else:
            # 回歸任務，計算MSE和MAE
            try:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                y_true = targets.cpu().numpy()
                y_pred = predictions.cpu().numpy()
                metrics['mse'] = mean_squared_error(y_true, y_pred)
                metrics['mae'] = mean_absolute_error(y_true, y_pred)
                metrics['r2'] = r2_score(y_true, y_pred)
            except Exception as e:
                logger.warning(f"計算回歸指標時出錯: {str(e)}")
                metrics['mse'] = metrics['mae'] = metrics['r2'] = 0.0
        
        return metrics

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
        loss_type = loss_config.get('type', 'MSELoss')
        loss_params = loss_config.get('parameters', {})
        
        logger.info(f"使用損失函數: {loss_type}，參數: {loss_params}")
        
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