"""
存檔管理器：集中管理框架中的各種存檔邏輯

該模塊提供了一個統一的接口來處理各種存檔需求，包括模型檢查點、激活值、梯度、訓練日誌等。
透過集中管理存檔邏輯，可以減少代碼重複，並提供更一致的存檔路徑和格式。
"""

import os
import torch
import json
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class SaveManager:
    """存檔管理器，用於集中管理框架中的各種存檔邏輯"""
    
    def __init__(self, base_dir: str, experiment_name: Optional[str] = None, create_subdirs: bool = True):
        """初始化存檔管理器
        
        Args:
            base_dir: 基本輸出目錄
            experiment_name: 實驗名稱，如果提供，會作為子目錄
            create_subdirs: 是否創建標準子目錄結構
        """
        self.base_dir = base_dir
        self.experiment_name = experiment_name
        
        # 如果提供了實驗名稱，將其添加到基本目錄
        if experiment_name:
            self.experiment_dir = os.path.join(base_dir, experiment_name)
        else:
            self.experiment_dir = base_dir
            
        # 確保目錄存在
        os.makedirs(self.experiment_dir, exist_ok=True)
        logger.info(f"初始化存檔管理器，實驗目錄: {self.experiment_dir}")
        
        # 創建標準子目錄
        if create_subdirs:
            self.subdirs = {
                'models': os.path.join(self.experiment_dir, 'models'),
                'hooks': os.path.join(self.experiment_dir, 'hooks'),
                'logs': os.path.join(self.experiment_dir, 'logs'),
                'tensorboard': os.path.join(self.experiment_dir, 'tensorboard_logs'),
                'results': os.path.join(self.experiment_dir, 'results'),
                'datasets': os.path.join(self.experiment_dir, 'datasets'),
                'feature_vectors': os.path.join(self.experiment_dir, 'feature_vectors')
            }
            
            for name, path in self.subdirs.items():
                os.makedirs(path, exist_ok=True)
                logger.debug(f"創建子目錄: {path}")
        else:
            self.subdirs = {}
    
    def get_path(self, subdir: str, filename: str) -> str:
        """獲取特定子目錄中的文件路徑
        
        Args:
            subdir: 子目錄名稱
            filename: 文件名稱
            
        Returns:
            str: 完整文件路徑
        """
        if subdir in self.subdirs:
            return os.path.join(self.subdirs[subdir], filename)
        else:
            # 如果子目錄不存在於預定義的目錄中，創建它
            full_subdir = os.path.join(self.experiment_dir, subdir)
            os.makedirs(full_subdir, exist_ok=True)
            return os.path.join(full_subdir, filename)
    
    def save_config(self, config: Dict[str, Any], filename: str = 'config.json') -> str:
        """保存配置
        
        Args:
            config: 配置字典
            filename: 保存的文件名
            
        Returns:
            str: 保存的文件路徑
        """
        save_path = os.path.join(self.experiment_dir, filename)
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"配置已保存到: {save_path}")
        return save_path
    
    def save_results(self, results: Dict[str, Any], filename: str = 'results.json') -> str:
        """保存實驗結果
        
        Args:
            results: 結果字典
            filename: 保存的文件名
            
        Returns:
            str: 保存的文件路徑
        """
        save_path = self.get_path('results', filename)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"實驗結果已保存到: {save_path}")
        return save_path
    
    def save_model(self, model: torch.nn.Module, filename: str = 'model.pth', 
                  additional_data: Optional[Dict[str, Any]] = None, epoch: Optional[int] = None) -> str:
        """保存模型
        
        Args:
            model: PyTorch 模型
            filename: 保存的文件名
            additional_data: 要與模型一起保存的額外數據
            epoch: 當前 epoch 編號
            
        Returns:
            str: 保存的文件路徑
        """
        if epoch is not None:
            # 如果提供了 epoch，則在文件名中添加 epoch 編號
            name, ext = os.path.splitext(filename)
            filename = f"{name}_epoch_{epoch}{ext}"
            
        save_path = self.get_path('models', filename)
        
        # 準備要保存的數據
        save_data = {
            'model_state_dict': model.state_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        # 添加額外數據
        if additional_data:
            save_data.update(additional_data)
            
        torch.save(save_data, save_path)
        logger.info(f"模型已保存到: {save_path}")
        return save_path
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                       epoch: int, loss: float, filename: str = 'checkpoint.pth',
                       additional_data: Optional[Dict[str, Any]] = None) -> str:
        """保存檢查點
        
        Args:
            model: PyTorch 模型
            optimizer: 優化器
            epoch: 當前 epoch 編號
            loss: 當前損失
            filename: 保存的文件名
            additional_data: 要與檢查點一起保存的額外數據
            
        Returns:
            str: 保存的文件路徑
        """
        # 在文件名中添加 epoch 編號
        name, ext = os.path.splitext(filename)
        checkpoint_filename = f"{name}_epoch_{epoch}{ext}"
            
        save_path = self.get_path('models', checkpoint_filename)
        
        # 準備要保存的數據
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }
        
        # 添加額外數據
        if additional_data:
            checkpoint.update(additional_data)
            
        torch.save(checkpoint, save_path)
        logger.info(f"檢查點已保存到: {save_path}")
        return save_path
    
    def save_hook_data(self, data: Dict[str, Any], hook_type: str,
                       epoch: int, batch: Optional[int] = None,
                       name: Optional[str] = None) -> str:
        """保存鉤子數據
        
        Args:
            data: 要保存的數據
            hook_type: 鉤子類型 (activation, gradient, etc.)
            epoch: 當前 epoch 編號
            batch: 當前批次索引
            name: 識別此數據的名稱
            
        Returns:
            str: 保存的文件路徑
        """
        # 創建 epoch 子目錄
        epoch_dir = os.path.join(self.get_path('hooks', f'epoch_{epoch}'))
        os.makedirs(epoch_dir, exist_ok=True)
        
        # 構建文件名
        filename_parts = []
        if name:
            clean_name = name.replace('.', '_')  # 替換可能導致路徑問題的字符
            filename_parts.append(clean_name)
        
        filename_parts.append(hook_type)
        
        if batch is not None:
            filename_parts.append(f'batch_{batch}')
        else:
            filename_parts.append('all')
            
        filename = f"{'_'.join(filename_parts)}.pt"
        save_path = os.path.join(epoch_dir, filename)
        
        # 保存數據
        torch.save(data, save_path)
        logger.info(f"{hook_type} 數據已保存到: {save_path}")
        return save_path
    
    def save_activations(self, activations: Dict[str, torch.Tensor], 
                        epoch: int, batch: Optional[int] = None) -> List[str]:
        """保存激活值
        
        Args:
            activations: 層名到激活值的映射
            epoch: 當前 epoch 編號
            batch: 當前批次索引
            
        Returns:
            List[str]: 保存的文件路徑列表
        """
        saved_paths = []
        for name, tensor in activations.items():
            save_path = self.save_hook_data(
                data=tensor,
                hook_type='activation',
                epoch=epoch,
                batch=batch,
                name=name
            )
            saved_paths.append(save_path)
        return saved_paths
    
    def save_gradients(self, gradients: Dict[str, torch.Tensor],
                      epoch: int, batch: Optional[int] = None) -> List[str]:
        """保存梯度
        
        Args:
            gradients: 參數名到梯度的映射
            epoch: 當前 epoch 編號
            batch: 當前批次索引
            
        Returns:
            List[str]: 保存的文件路徑列表
            
        Description:
            保存每個參數的梯度張量(.pt)、詳細統計量(.json，包含分位數)和直方圖數據(.pt)。
        """
        saved_paths = []
        # 中文註解：確保epoch目錄存在
        epoch_dir = os.path.join(self.get_path('hooks', f'epoch_{epoch}'))
        os.makedirs(epoch_dir, exist_ok=True)
        
        # 定義內部函數用於安全計算分位數（防止記憶體溢出）
        def safe_quantile(tensor, q):
            """以記憶體友好的方式計算分位數
            
            Args:
                tensor: 輸入張量
                q: 分位數值 (例如 [0.25, 0.5, 0.75])
                
            Returns:
                torch.Tensor: 計算的分位數結果
            """
            # 將輸入張量移至CPU，以防它在GPU上
            if tensor.device.type != 'cpu':
                tensor = tensor.cpu()
                
            # 攤平張量
            flat = tensor.view(-1)
            
            # 設定批次大小 (每批次最多處理這麼多元素)
            chunk_size = 1000000  # 100萬個元素
            
            # 如果張量很小，直接計算
            if flat.numel() <= chunk_size:
                return torch.quantile(flat, q, interpolation='linear')
            
            # 對於大張量，使用近似方法
            # 1. 收集所有批次的分位數
            quantiles_list = []
            num_chunks = (flat.numel() + chunk_size - 1) // chunk_size  # 向上取整
            
            # 2. 從每個批次中收集樣本
            samples = []
            step = max(1, flat.numel() // (chunk_size // 10))  # 採樣步長
            indices = torch.arange(0, flat.numel(), step)
            
            # 限制樣本數量
            if indices.numel() > chunk_size:
                indices = indices[:chunk_size]
                
            samples = flat[indices]
            
            # 3. 使用樣本計算分位數
            try:
                return torch.quantile(samples, q, interpolation='linear')
            except Exception as e:
                logger.warning(f"使用樣本計算分位數時出錯: {e}，返回估計值")
                # 返回一個估計值（最小值、中值和最大值的插值）
                if isinstance(q, torch.Tensor):
                    min_val = float(torch.min(samples).item())
                    max_val = float(torch.max(samples).item())
                    median = float(torch.median(samples).item())
                    
                    # 根據q的值估計分位數
                    results = []
                    for qi in q:
                        qi_float = float(qi.item())
                        if qi_float <= 0.25:
                            # 在最小值和中位數之間插值
                            val = min_val + (median - min_val) * (qi_float / 0.5)
                        else:
                            # 在中位數和最大值之間插值
                            val = median + (max_val - median) * ((qi_float - 0.5) / 0.5)
                        results.append(val)
                    return torch.tensor(results)
                else:
                    # 如果q不是張量，返回一個合理的預設值
                    return torch.tensor([0.0])
        
        for name, tensor in gradients.items():
            if tensor is not None:
                # ========== 1. 保存原始梯度張量 ========== #
                save_path = self.save_hook_data(
                    data=tensor,
                    hook_type='gradient',
                    epoch=epoch,
                    batch=batch,
                    name=name
                )
                saved_paths.append(save_path)
                
                # ========== 2. 保存詳細統計信息 (含分位數) ========== #
                # 中文註解：計算基礎統計量與分位數
                try:
                    flat_tensor = tensor.view(-1)
                    q = torch.tensor([0.25, 0.5, 0.75])
                    
                    # 使用記憶體友好的方式計算分位數
                    quantiles = safe_quantile(flat_tensor, q)
                    
                    stats = {
                        'mean': float(torch.mean(tensor).item()),
                        'std': float(torch.std(tensor).item()),
                        'min': float(torch.min(tensor).item()),
                        'max': float(torch.max(tensor).item()),
                        'norm': float(torch.norm(tensor).item()),
                        'quantile_25': float(quantiles[0].item()),
                        'quantile_50': float(quantiles[1].item()), # 中位數
                        'quantile_75': float(quantiles[2].item()),
                        'timestamp': datetime.now().isoformat(),
                        'epoch': epoch,
                        'batch': batch
                    }
                    
                    # 中文註解：定義統計量檔案路徑與名稱
                    stats_filename = os.path.join(
                        epoch_dir,
                        os.path.splitext(os.path.basename(save_path))[0] + '_stats.json'
                    )
                    
                    # 中文註解：寫入JSON檔案
                    with open(stats_filename, 'w') as f:
                        json.dump(stats, f, indent=2)
                    saved_paths.append(stats_filename)
                
                except Exception as e:
                    logger.warning(f"計算或保存參數 '{name}' 的梯度統計信息時出錯: {e}")

                # ========== 3. 保存直方圖數據 ========== #
                try:
                    # 中文註解：計算直方圖數據
                    # 如果張量太大，使用採樣
                    if tensor.numel() > 1000000:  # 超過100萬個元素時採樣
                        step = max(1, tensor.numel() // 100000)  # 採樣步長
                        indices = torch.arange(0, tensor.numel(), step)
                        samples = tensor.view(-1)[indices]
                        hist_counts, hist_bins = torch.histogram(samples.cpu(), bins=100) # 可調整bins數量
                    else:
                        hist_counts, hist_bins = torch.histogram(tensor.cpu(), bins=100) # 可調整bins數量
                    
                    hist_data = {
                        'hist': hist_counts,
                        'bin_edges': hist_bins,
                        'timestamp': datetime.now().isoformat(),
                        'epoch': epoch,
                        'batch': batch
                    }
                    
                    # 中文註解：定義直方圖檔案路徑與名稱
                    hist_filename = os.path.join(
                        epoch_dir,
                        os.path.splitext(os.path.basename(save_path))[0] + '_hist.pt'
                    )
                    
                    # 中文註解：儲存直方圖數據
                    torch.save(hist_data, hist_filename)
                    saved_paths.append(hist_filename)
                except Exception as e:
                    logger.warning(f"計算或保存參數 '{name}' 的梯度直方圖時出錯: {e}")
                
        return saved_paths
    
    def save_batch_data(self, batch_data: Dict[str, Any], epoch: int, batch: int) -> str:
        """保存批次數據
        
        Args:
            batch_data: 批次數據
            epoch: 當前 epoch 編號
            batch: 當前批次索引
            
        Returns:
            str: 保存的文件路徑
        """
        filename = f"batch_{batch}_data.pt"
        
        # 創建 epoch 子目錄
        epoch_dir = os.path.join(self.get_path('hooks', f'epoch_{epoch}'))
        os.makedirs(epoch_dir, exist_ok=True)
        
        save_path = os.path.join(epoch_dir, filename)
        
        # 添加時間戳
        batch_data_with_timestamp = {
            **batch_data,
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'batch': batch
        }
        
        torch.save(batch_data_with_timestamp, save_path)
        logger.info(f"批次數據已保存到: {save_path}")
        return save_path
    
    def save_epoch_summary(self, epoch: int, train_logs: Optional[Dict[str, Any]] = None,
                          val_logs: Optional[Dict[str, Any]] = None) -> str:
        """保存 epoch 摘要
        
        Args:
            epoch: 當前 epoch 編號
            train_logs: 訓練日誌
            val_logs: 驗證日誌
            
        Returns:
            str: 保存的文件路徑
        """
        # 創建 epoch 子目錄
        epoch_dir = os.path.join(self.get_path('hooks', f'epoch_{epoch}'))
        os.makedirs(epoch_dir, exist_ok=True)
        
        save_path = os.path.join(epoch_dir, 'epoch_summary.pt')
        
        summary_data = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat()
        }
        
        # 添加訓練和驗證日誌
        if train_logs:
            summary_data['train'] = train_logs
        
        if val_logs:
            summary_data['val'] = val_logs
            
        torch.save(summary_data, save_path)
        logger.info(f"Epoch {epoch} 摘要已保存到: {save_path}")
        return save_path
    
    def save_training_summary(self, history: Dict[str, List], total_epochs: int) -> str:
        """保存訓練摘要
        
        Args:
            history: 訓練歷史記錄
            total_epochs: 總訓練 epoch 數
            
        Returns:
            str: 保存的文件路徑
        """
        # 將保存路徑更改為 hooks 子目錄，並將文件名更改為 training_summary.pt
        save_path = self.get_path('hooks', 'training_summary.pt') 
        
        # 準備要保存的數據 (確保所有數據都是可序列化的，儘管對於 torch.save 這不那麼嚴格)
        # 原始的 history 數據結構可能已經適合 torch.save
        summary_data = {
            'history': history, # 直接保存 history dict
            'total_epochs': total_epochs,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            torch.save(summary_data, save_path)
            logger.info(f"訓練摘要已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存訓練摘要到 {save_path} 時出錯: {e}", exc_info=True)
            # 如果 torch.save 失敗，可以考慮回退到JSON，但需要確保 history 可序列化
            # 此處為簡化，僅記錄錯誤
            # save_path_json = self.get_path('hooks', 'training_summary_error.json')
            # try:
            #     with open(save_path_json, 'w') as f_json:
            #         json.dump(summary_data, f_json, indent=2, default=str) # 添加 default=str 以處理無法序列化的類型
            #     logger.warning(f"由於torch.save失敗，訓練摘要已嘗試保存為JSON到: {save_path_json}")
            # except Exception as e_json:
            #     logger.error(f"嘗試將訓練摘要保存為JSON也失敗: {e_json}")
            return ""
        return save_path
    
    def save_evaluation_results(self, results: Dict[str, Any], mode: Optional[str] = None) -> str:
        """保存評估結果
        
        Args:
            results: 評估結果字典
            mode: 評估模式，如 'test', 'val' 等
            
        Returns:
            str: 保存的文件路徑
            
        Description:
            保存評估結果到標準位置，用於後續分析
            
        References:
            無
        """
        if mode:
            filename = f'evaluation_results_{mode}.json'
        else:
            filename = 'evaluation_results.json'
        
        save_path = self.get_path('results', filename)
        
        # 如果結果包含張量，將其轉換為列表
        results_json = {}
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                results_json[key] = value.tolist()
            else:
                results_json[key] = value
        
        # 添加時間戳
        results_json['timestamp'] = datetime.now().isoformat()
        
        with open(save_path, 'w') as f:
            json.dump(results_json, f, indent=2)
            
        logger.info(f"評估結果已保存到: {save_path}")
        return save_path
    
    def get_tensorboard_log_dir(self) -> str:
        """獲取 TensorBoard 日誌目錄
        
        Returns:
            str: TensorBoard 日誌目錄
        """
        return self.get_path('tensorboard', '')
    
    def get_experiment_dir(self) -> str:
        """獲取實驗目錄
        
        Returns:
            str: 實驗目錄
        """
        return self.experiment_dir
    
    def save_gns_stats(self, gns_stats: Dict[str, Any], epoch: int) -> str:
        """保存 GNS 統計量
        
        Args:
            gns_stats: GNS 統計量字典
            epoch: 當前 epoch 編號
        
        Returns:
            str: 保存的文件路徑
        
        Description:
            將 GNS 統計量以 JSON 格式儲存於 hooks/epoch_{n}/gns_stats_epoch_{n}.json
        References:
            https://arxiv.org/abs/2006.08536
        """
        # 中文註解：建立 hooks/epoch_{n} 目錄
        epoch_dir = os.path.join(self.get_path('hooks', f'epoch_{epoch}'))
        os.makedirs(epoch_dir, exist_ok=True)
        
        filename = f"gns_stats_epoch_{epoch}.json"
        save_path = os.path.join(epoch_dir, filename)
        
        # 寫入 GNS 統計量
        with open(save_path, 'w') as f:
            json.dump(gns_stats, f, indent=2)
        logger.info(f"GNS 統計量已保存到: {save_path}")
        return save_path
        
    def save_dataset_info(self, dataset_type: str, dataset: Any, config: Dict[str, Any] = None, 
                          file_paths: List[str] = None) -> str:
        """保存資料集資訊
        
        Args:
            dataset_type: 資料集類型 ('train', 'val', 'test')
            dataset: 資料集物件
            config: 配置字典，包含資料集相關設定
            file_paths: 資料集中使用的檔案路徑列表
            
        Returns:
            str: 保存的文件路徑
            
        Description:
            保存資料集的詳細資訊，包括大小、分布統計、分割方式、預處理參數和檔案路徑
            
        References:
            無
        """
        # 創建 datasets 子目錄
        datasets_dir = os.path.join(self.experiment_dir, 'datasets')
        os.makedirs(datasets_dir, exist_ok=True)
        
        # 準備資料集資訊
        dataset_info = {
            'type': dataset_type,
            'size': len(dataset),
            'timestamp': datetime.now().isoformat()
        }
        
        # 添加配置資訊（如果有）
        if config:
            # 提取資料集相關的配置
            data_config = config.get('data', {})
            dataset_info['config'] = {
                'type': data_config.get('type', 'unknown'),
                'splits': data_config.get('splits', {}),
                'preprocessing': data_config.get('preprocessing', {}),
                'transforms': data_config.get('transforms', {}).get(dataset_type, {})
            }
        
        # 嘗試獲取更多資料集資訊
        try:
            # 類別分布（如果是分類任務）
            if hasattr(dataset, 'get_class_distribution'):
                dataset_info['class_distribution'] = dataset.get_class_distribution()
            elif hasattr(dataset, 'class_counts'):
                dataset_info['class_distribution'] = dataset.class_counts
            else:
                # 嘗試手動計算類別分布
                try:
                    labels = []
                    for i in range(min(len(dataset), 1000)):  # 最多取1000個樣本避免過大
                        sample = dataset[i]
                        if isinstance(sample, dict) and 'label' in sample:
                            labels.append(sample['label'])
                        elif isinstance(sample, tuple) and len(sample) > 1:
                            labels.append(sample[1])
                    
                    if labels:
                        if isinstance(labels[0], torch.Tensor):
                            labels = [label.item() if label.numel() == 1 else label.tolist() for label in labels]
                        
                        # 計算類別計數
                        from collections import Counter
                        class_counts = Counter(labels)
                        dataset_info['class_distribution'] = {str(k): v for k, v in class_counts.items()}
                except Exception as e:
                    logger.warning(f"計算類別分布時出錯: {e}")
            
            # 特徵維度（如果適用）
            if hasattr(dataset, 'get_feature_dim'):
                dataset_info['feature_dim'] = dataset.get_feature_dim()
            elif hasattr(dataset, 'feature_dim'):
                dataset_info['feature_dim'] = dataset.feature_dim
                
            # 樣本資訊（如果有患者ID）
            if hasattr(dataset, 'samples') and isinstance(dataset.samples, list):
                # 提取樣本摘要資訊（避免過大）
                sample_summary = []
                for i, sample in enumerate(dataset.samples[:100]):  # 最多取100個樣本
                    if isinstance(sample, dict):
                        # 過濾掉大型資料欄位
                        filtered_sample = {k: v for k, v in sample.items() 
                                          if not isinstance(v, (np.ndarray, torch.Tensor)) or 
                                          (isinstance(v, (np.ndarray, torch.Tensor)) and v.size < 100)}
                        sample_summary.append(filtered_sample)
                    else:
                        sample_summary.append(str(sample))
                    if i >= 99:  # 只取前100個
                        break
                
                if sample_summary:
                    dataset_info['sample_summary'] = sample_summary
                    dataset_info['total_samples'] = len(dataset.samples)
        except Exception as e:
            logger.warning(f"獲取額外資料集資訊時出錯: {e}")
            
        # 添加檔案路徑（如果有）
        if file_paths:
            # 如果路徑太多，只保存一部分
            if len(file_paths) > 100:
                dataset_info['file_paths_sample'] = file_paths[:100]
                dataset_info['total_files'] = len(file_paths)
            else:
                dataset_info['file_paths'] = file_paths
                
        # 保存為PT檔案（保留完整資訊，包括張量）
        pt_filename = f"{dataset_type}_dataset_info.pt"
        pt_save_path = os.path.join(datasets_dir, pt_filename)
        torch.save(dataset_info, pt_save_path)
        logger.info(f"{dataset_type} 資料集資訊已保存到: {pt_save_path}")
        
        # 同時保存為JSON檔案（易於查看，但不包括張量）
        try:
            json_filename = f"{dataset_type}_dataset_info.json"
            json_save_path = os.path.join(datasets_dir, json_filename)
            
            # 處理不能序列化為JSON的物件
            json_safe_info = {}
            for k, v in dataset_info.items():
                if isinstance(v, (np.ndarray, torch.Tensor)):
                    if v.size < 1000:  # 避免過大
                        json_safe_info[k] = v.tolist() if hasattr(v, 'tolist') else str(v)
                    else:
                        json_safe_info[k] = f"{type(v).__name__} of shape {v.shape}"
                elif isinstance(v, dict):
                    # 遞迴處理字典
                    json_safe_dict = {}
                    for dk, dv in v.items():
                        if isinstance(dv, (np.ndarray, torch.Tensor)):
                            if hasattr(dv, 'shape') and hasattr(dv, 'size') and dv.size < 1000:
                                json_safe_dict[dk] = dv.tolist() if hasattr(dv, 'tolist') else str(dv)
                            else:
                                json_safe_dict[dk] = f"{type(dv).__name__} of shape {getattr(dv, 'shape', 'unknown')}"
                        else:
                            try:
                                # 嘗試JSON序列化
                                json.dumps({dk: dv})
                                json_safe_dict[dk] = dv
                            except (TypeError, OverflowError):
                                json_safe_dict[dk] = str(dv)
                    json_safe_info[k] = json_safe_dict
                else:
                    try:
                        # 嘗試JSON序列化
                        json.dumps({k: v})
                        json_safe_info[k] = v
                    except (TypeError, OverflowError):
                        json_safe_info[k] = str(v)
            
            with open(json_save_path, 'w', encoding='utf-8') as f:
                json.dump(json_safe_info, f, ensure_ascii=False, indent=2)
            logger.info(f"{dataset_type} 資料集資訊(JSON)已保存到: {json_save_path}")
        except Exception as e:
            logger.warning(f"保存資料集資訊為JSON時出錯: {e}")
        
        return pt_save_path
        
    def save_datasets_statistics(self, train_dataset=None, val_dataset=None, test_dataset=None, 
                               config: Dict[str, Any] = None) -> str:
        """保存所有資料集的綜合統計資訊
        
        Args:
            train_dataset: 訓練資料集
            val_dataset: 驗證資料集
            test_dataset: 測試資料集
            config: 配置字典
            
        Returns:
            str: 保存的文件路徑
            
        Description:
            計算並保存所有資料集的綜合統計資訊，便於比較不同資料集的特性
            
        References:
            無
        """
        # 創建 datasets 子目錄
        datasets_dir = os.path.join(self.experiment_dir, 'datasets')
        os.makedirs(datasets_dir, exist_ok=True)
        
        # 準備綜合統計資訊
        stats = {
            'timestamp': datetime.now().isoformat(),
            'datasets': {}
        }
        
        # 添加配置資訊（如果有）
        if config:
            data_config = config.get('data', {})
            stats['data_config'] = {
                'type': data_config.get('type', 'unknown'),
                'splits': data_config.get('splits', {}),
                'preprocessing': data_config.get('preprocessing', {}),
            }
        
        # 收集各資料集的基本資訊
        datasets = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
        
        for name, dataset in datasets.items():
            if dataset is not None:
                stats['datasets'][name] = {
                    'size': len(dataset)
                }
                
                # 嘗試獲取更多資訊
                try:
                    # 類別分布
                    if hasattr(dataset, 'get_class_distribution'):
                        stats['datasets'][name]['class_distribution'] = dataset.get_class_distribution()
                    elif hasattr(dataset, 'class_counts'):
                        stats['datasets'][name]['class_distribution'] = dataset.class_counts
                        
                    # 特徵維度
                    if hasattr(dataset, 'get_feature_dim'):
                        stats['datasets'][name]['feature_dim'] = dataset.get_feature_dim()
                    elif hasattr(dataset, 'feature_dim'):
                        stats['datasets'][name]['feature_dim'] = dataset.feature_dim
                except Exception as e:
                    logger.warning(f"獲取{name}資料集統計資訊時出錯: {e}")
        
        # 保存為JSON檔案
        save_path = os.path.join(datasets_dir, 'dataset_statistics.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"資料集綜合統計資訊已保存到: {save_path}")
        
        return save_path

    # 新增方法：保存特徵向量數據
    def save_feature_vector_data(self, data: Dict[str, Any], layer_name: str, epoch: int, 
                                 dataset_name: str, suffix: str) -> str:
        """保存特定層的特徵向量及其分析結果
        
        Args:
            data (Dict[str, Any]): 要保存的數據字典 (例如包含 'activations', 'targets', 'tsne', 'cosine_similarity')
            layer_name (str): 層的名稱
            epoch (int): 當前的 epoch
            dataset_name (str): 數據集名稱 (例如 'train', 'val', 'test')
            suffix (str): 文件後綴，用於區分不同類型的數據 (例如 'features', 'tsne', 'cosine_similarity')
            
        Returns:
            str: 保存的文件路徑
        """
        # 確保 feature_vectors 和 epoch 子目錄存在
        epoch_feature_dir = os.path.join(self.get_path('feature_vectors', ''), f'epoch_{epoch}')
        os.makedirs(epoch_feature_dir, exist_ok=True)
        
        # 構建文件名
        clean_layer_name = layer_name.replace('.', '_') # 確保層名稱適用於文件名
        filename = f"layer_{clean_layer_name}_{dataset_name}_{suffix}.pt"
        save_path = os.path.join(epoch_feature_dir, filename)
        
        try:
            torch.save(data, save_path)
            logger.info(f"特徵向量數據已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存特徵向量數據到 {save_path} 時出錯: {e}", exc_info=True)
            return "" # 或者拋出異常
        return save_path

# 中文註解：這是save_manager.py的Minimal Executable Unit，檢查SaveManager能正確建立目錄、保存模型/結果/激活值，並測試錯誤路徑時的報錯
if __name__ == "__main__":
    """
    Description: Minimal Executable Unit for save_manager.py，檢查SaveManager能正確建立目錄、保存模型/結果/激活值，並測試錯誤路徑時的報錯。
    Args: None
    Returns: None
    References: 無
    """
    import torch
    import torch.nn as nn
    import shutil
    import os
    import logging
    logging.basicConfig(level=logging.INFO)
    test_dir = "test_save_manager"
    try:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        mgr = SaveManager(test_dir, experiment_name="exp1")
        # 測試save_config
        cfg_path = mgr.save_config({"a": 1}, filename="cfg.json")
        print(f"save_config測試成功: {os.path.exists(cfg_path)}")
        # 測試save_results
        res_path = mgr.save_results({"acc": 0.9}, filename="res.json")
        print(f"save_results測試成功: {os.path.exists(res_path)}")
        # 測試save_model
        class DummyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(2, 1)
            def forward(self, x):
                return self.fc(x)
        model = DummyNet()
        model_path = mgr.save_model(model, filename="model.pth")
        print(f"save_model測試成功: {os.path.exists(model_path)}")
        # 測試save_activations
        acts = {"fc": torch.randn(2, 2)}
        act_paths = mgr.save_activations(acts, epoch=0, batch=0)
        print(f"save_activations測試成功: {all(os.path.exists(p) for p in act_paths)}")
    except Exception as e:
        print(f"SaveManager遇到錯誤（預期行為）: {e}")
    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir) 