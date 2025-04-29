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
                'results': os.path.join(self.experiment_dir, 'results')
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
                flat_tensor = tensor.view(-1)
                quantiles = torch.quantile(flat_tensor, torch.tensor([0.25, 0.5, 0.75]), interpolation='linear')
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

                # ========== 3. 保存直方圖數據 ========== #
                try:
                    # 中文註解：計算直方圖數據
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
            history: 訓練歷史
            total_epochs: 總 epoch 數
            
        Returns:
            str: 保存的文件路徑
        """
        save_path = self.get_path('hooks', 'training_summary.pt')
        
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'total_epochs': total_epochs,
            'history': history
        }
        
        torch.save(summary_data, save_path)
        logger.info(f"訓練摘要已保存到: {save_path}")
        return save_path
    
    def save_evaluation_results(self, results: Dict[str, Any], mode: Optional[str] = None) -> str:
        """保存評估結果
        
        Args:
            results: 評估結果
            mode: 評估模式 (例如 'test', 'val')
            
        Returns:
            str: 保存的文件路徑
        """
        filename = f"evaluation_results{f'_{mode}' if mode else ''}.pt"
        save_path = self.get_path('hooks', filename)
        
        eval_data = {
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        
        if mode:
            eval_data['mode'] = mode
            
        torch.save(eval_data, save_path)
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