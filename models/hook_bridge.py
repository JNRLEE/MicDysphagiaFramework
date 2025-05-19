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
        importlib.import_module("sbp_analyzer")
        return True
    except ImportError:
        return False


class SimpleActivationHook:
    """簡易激活值鉤子，用於捕獲並存儲模型中間層的激活值
    
    無需依賴外部庫，直接存儲數據到指定目錄
    """
    
    def __init__(
        self,
        model: nn.Module,
        layer_names: Optional[List[str]] = None,
        save_manager: Optional[SaveManager] = None,
    ):
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
    
    def __init__(
        self,
        model: nn.Module,
        param_names: Optional[List[str]] = None,
        save_manager: Optional[SaveManager] = None,
    ):
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
                    "mean": float(np.mean(grad_np)),
                    "std": float(np.std(grad_np)),
                    "min": float(np.min(grad_np)),
                    "max": float(np.max(grad_np)),
                    "norm": float(torch.norm(grad).item()),
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
    
    def __init__(
        self,
        model: nn.Module,
                 monitored_layers: Optional[List[str]] = None,
                 monitored_params: Optional[List[str]] = None,
        output_dir: Optional[str] = None,  # 保留以兼容舊用法，但優先使用save_manager
        save_manager: Optional[SaveManager] = None,
        feature_vector_layers: Optional[List[str]] = None,
        tsne_perplexity: int = 30,  # 新增：t-SNE perplexity 參數
    ):  # 新增：特徵向量捕獲層
        """初始化模型鉤子管理器
        
        Args:
            model: 要監控的 PyTorch 模型
            monitored_layers: 要監控激活值的層名稱列表 (用於SimpleActivationHook)
            monitored_params: 要監控梯度的參數名稱列表
            output_dir: 輸出目錄（如果提供save_manager則不使用）
            save_manager: 已存在的SaveManager實例，如果提供則優先使用
            feature_vector_layers: 要捕獲特徵向量的層名稱列表
            tsne_perplexity: t-SNE計算的perplexity參數，默認為30
        """
        self.model = model
        self.output_dir = output_dir
        
        # 如果提供了save_manager，直接使用；否則根據output_dir創建
        if save_manager:
            self.save_manager = save_manager
            if output_dir:
                logger.info(f"使用提供的SaveManager，忽略output_dir參數: {output_dir}")
        elif output_dir:
            # 創建SaveManager實例（如果output_dir提供）
            self.save_manager = SaveManager(output_dir)
            logger.info(f"創建新的SaveManager，輸出目錄: {output_dir}")
        else:
            self.save_manager = None
            logger.warning("未提供save_manager或output_dir，將無法保存數據")
        
        # 創建激活值和梯度鉤子
        self.activation_hook = SimpleActivationHook(
            model, monitored_layers, self.save_manager
        )
        self.gradient_hook = SimpleGradientHook(
            model, monitored_params, self.save_manager
        )

        # 初始化特徵向量捕獲相關
        self.feature_vector_layers = feature_vector_layers
        self.feature_vectors: Dict[str, List[torch.Tensor]] = (
            {layer: [] for layer in self.feature_vector_layers}
            if self.feature_vector_layers
            else {}
        )
        self.feature_hooks: List[Tuple[str, Any]] = []  # 用於存儲特徵向量鉤子的句柄
        
        # 記錄當前批次數據
        self.current_epoch = 0
        self.current_batch = 0
        self.inputs = None
        self.outputs = None
        self.targets = None
        self.loss = None
        
        logger.info(f"模型鉤子管理器已初始化")
        if self.feature_vector_layers:
            logger.info(f"將捕獲以下層的特徵向量: {self.feature_vector_layers}")

        # 設置t-SNE perplexity默認值
        self.tsne_perplexity = tsne_perplexity

        # 添加一個屬性來存儲由Callback收集的目標
        self.all_targets_for_features: Optional[List[torch.Tensor]] = None

    def _find_module_by_name(self, name: str) -> Optional[nn.Module]:
        """通過名稱查找模型中的模塊"""
        if name == "":
            return self.model
        for n, m in self.model.named_modules():
            if n == name:
                return m
        logger.warning(f"未找到名稱為 '{name}' 的模塊")
        return None

    def register_feature_hooks(self):
        """註冊特徵向量捕獲鉤子

        Description:
            為在 feature_vector_layers 中指定的層註冊前向鉤子以捕獲其輸出。
            此方法應在評估開始前調用。
        """
        if not self.feature_vector_layers or not self.save_manager:
            if self.feature_vector_layers:
                logger.warning("未配置SaveManager，無法註冊特徵向量鉤子")
            return

        # 先移除舊的特徵鉤子，以防重複註冊
        self._remove_feature_hooks()

        for layer_name in self.feature_vector_layers:
            module = self._find_module_by_name(layer_name)
            if module:
                hook = module.register_forward_hook(
                    lambda mod, inp, out, name=layer_name: self._feature_hook_fn(
                        name, out
                    )
                )
                self.feature_hooks.append((layer_name, hook))
                logger.info(f"已為層 '{layer_name}' 註冊特徵向量捕獲鉤子")
            else:
                logger.warning(f"無法為層 '{layer_name}' 註冊特徵向量鉤子：未找到該層")

    def _feature_hook_fn(
        self, name: str, output: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ):
        """特徵向量捕獲回調函數

        Args:
            name: 層名稱
            output: 層的輸出

        Description:
            此函數在註冊的層前向傳播後被調用。
            它將層的輸出（特徵向量）存儲起來，僅在模型處於評估模式時執行。
        """
        if self.model.training:  # 只在評估模式下捕獲
            return

        current_output: torch.Tensor
        if isinstance(output, tuple):
            current_output = output[0]  # 通常取第一個輸出
        else:
            current_output = output

        if name in self.feature_vectors:
            self.feature_vectors[name].append(current_output.detach().cpu())
            logger.debug(f"捕獲到層 '{name}' 的特徵向量，形狀: {current_output.shape}")
        else:
            logger.warning(f"嘗試為未初始化的層 '{name}' 捕獲特徵向量")

    def _save_feature_vectors(
        self,
        epoch: int,
        collected_targets: Optional[List[torch.Tensor]],
        dataset_name: str = "eval",
        compute_similarity: bool = True,
        compute_tsne: bool = True,
        label_names: Optional[List[str]] = None,
        label_mapping: Optional[Dict[int, str]] = None,
        label_field: Optional[str] = None,
    ):
        """
        收集、處理並保存指定層的特徵向量，並可選計算相似度和t-SNE。
        會將特徵向量保存到 `feature_vectors/epoch_{epoch}/layer_{layer_name}_features.pt`。
        如果啟用，相似度矩陣和t-SNE結果也會保存。

        Args:
            epoch (int): 當前輪次。
            collected_targets (Optional[List[torch.Tensor]]): 從Callback收集到的目標列表。
            dataset_name (str): 數據集名稱。
            compute_similarity (bool): 是否計算餘弦相似度。
            compute_tsne (bool): 是否計算t-SNE。
            label_names (Optional[List[str]]): 原始標籤文本列表。
            label_mapping (Optional[Dict[int, str]]): 標籤映射字典。
            label_field (Optional[str]]): 標籤欄位名稱。
        """
        if not self.save_manager:
            logger.warning("未配置存檔管理器，無法保存特徵向量。")
            return

        if not self.feature_vectors:
            logger.info(
                f"在 {dataset_name} (epoch {epoch}) 上沒有要處理的當前特徵向量。"
            )
            return

        collected_targets_tensor: Optional[torch.Tensor] = None
        # 使用傳入的 collected_targets
        if collected_targets is not None:
            if isinstance(collected_targets, list) and all(
                isinstance(t, torch.Tensor) for t in collected_targets
            ):
                if collected_targets:
                    try:
                        collected_targets_tensor = torch.cat(
                            collected_targets, dim=0
                        ).cpu()
                    except Exception as e:
                        logger.warning(
                            f"拼接為 {dataset_name} (epoch {epoch}) 收集的目標張量列表時出錯: {e}"
                        )
            elif isinstance(
                collected_targets, torch.Tensor
            ):  # 如果已經是拼接好的Tensor
                collected_targets_tensor = collected_targets.cpu()
            else:
                logger.warning(
                    f"為 {dataset_name} (epoch {epoch}) 收集的目標數據類型不受支持: {type(collected_targets)}，將忽略目標。"
                )

        logger.info(
            f"開始為 {dataset_name} (epoch {epoch}) 保存特徵向量。監控的層: {list(self.feature_vectors.keys())}"
        )
        
        # 記錄標籤信息
        if label_field:
            logger.info(f"標籤欄位: {label_field}")
        if label_mapping:
            logger.info(f"標籤映射: {label_mapping}")
        if label_names:
            logger.info(f"標籤名稱: {label_names}")

        for (
            layer_name,
            batch_features_list,
        ) in self.feature_vectors.items():
            if not batch_features_list:
                logger.debug(
                    f"層 '{layer_name}' 在 {dataset_name} (epoch {epoch}) 上沒有收集到特徵向量，跳過保存。"
                )
                continue

            try:
                # 從 batch_features_list 創建原始（可能未展平）的特徵張量
                raw_features_tensor = torch.cat(batch_features_list, dim=0)

                # 複製一份用於保存，這份可能會被展平
                features_to_save_tensor = raw_features_tensor.clone()
                original_shape = features_to_save_tensor.shape  # 記錄展平前的形狀

                if features_to_save_tensor.dim() > 2:
                    features_to_save_tensor = features_to_save_tensor.view(
                        features_to_save_tensor.size(0), -1
                    )
                    logger.debug(
                        f"層 '{layer_name}' 的特徵張量已從 {original_shape} 展平為 {features_to_save_tensor.shape} 以便保存"
                    )
                elif features_to_save_tensor.dim() == 1:
                    features_to_save_tensor = features_to_save_tensor.unsqueeze(1)
                    logger.debug(
                        f"層 '{layer_name}' 的特徵張量已從 {original_shape} 調整為 {features_to_save_tensor.shape} 以便保存"
                    )

                final_targets_for_layer = None
                if collected_targets_tensor is not None:
                    if collected_targets_tensor.size(0) == features_to_save_tensor.size(
                        0
                    ):  # 比較展平後用於保存的特徵數量
                        final_targets_for_layer = collected_targets_tensor
                    else:
                        logger.warning(
                            f"層 '{layer_name}' 的特徵數量 ({features_to_save_tensor.size(0)}) 與目標數量 ({collected_targets_tensor.size(0)}) 不匹配。將不會為此層保存目標。"
                        )

                feature_data_to_save = {
                    "layer_name": layer_name,
                    "activations": features_to_save_tensor,  # 保存（可能）展平的特徵
                    "targets": final_targets_for_layer,
                    "timestamp": datetime.now().isoformat(),
                    "epoch": epoch,
                    "dataset_name": dataset_name, # 雖然文檔檔名不含，但內容可包含
                }

                # 添加標籤相關信息到特徵數據
                if label_names is not None:
                    feature_data_to_save["label_names"] = label_names
                if label_mapping is not None:
                    feature_data_to_save["label_mapping"] = label_mapping
                if label_field is not None:
                    feature_data_to_save["label_field"] = label_field

                logger.debug(
                    f"準備保存層 '{layer_name}' 的特徵數據，包含鍵: {list(feature_data_to_save.keys())}"
                )

                # 使用 save_feature_vector_data 方法保存特徵
                save_path_features = self.save_manager.save_feature_vector_data(
                    data=feature_data_to_save,
                    layer_name=layer_name,
                    epoch=epoch,
                    dataset_name=dataset_name,
                    suffix="features",
                )
                if not save_path_features:
                    logger.error(f"為層 '{layer_name}' 保存 features 時返回了空路徑。")
                else:
                    logger.info(f"已保存特徵向量到: {save_path_features}")
                    # 成功保存 features.pt 後，更新 feature_analysis.json
                    self._update_feature_analysis_summary(
                        layer_name,
                        raw_features_tensor,  # 傳遞原始、未展平的特徵
                        final_targets_for_layer,
                        epoch,
                        dataset_name,
                    )

                # 如果啟用了相似度計算
                if (
                    compute_similarity
                    and features_to_save_tensor.dim() > 1 # 確保不是單一純量特徵
                    and features_to_save_tensor.size(0) > 1 # 確保至少有兩個樣本
                ):
                    # 使用 features_to_save_tensor (已展平) 進行計算
                    similarity_results = self._compute_cosine_similarity(
                        features_to_save_tensor, final_targets_for_layer
                    )
                    
                    # 合併元數據
                    cosine_data_to_save = {
                        "layer_name": layer_name,
                        "epoch": epoch,
                        "dataset_name": dataset_name,
                        # similarity_matrix 和其他統計數據由 _compute_cosine_similarity 返回
                        **similarity_results # 解包 _compute_cosine_similarity 的結果
                    }
                    
                    # 添加標籤相關信息到相似度數據
                    if label_names is not None:
                        cosine_data_to_save["label_names"] = label_names
                    if label_mapping is not None:
                        cosine_data_to_save["label_mapping"] = label_mapping
                    if label_field is not None:
                        cosine_data_to_save["label_field"] = label_field
                    
                    cosine_path = self.save_manager.save_feature_vector_data(
                        data=cosine_data_to_save,
                        layer_name=layer_name,
                        epoch=epoch,
                        dataset_name=dataset_name,
                        suffix="cosine_similarity",
                    )
                    if cosine_path:
                        logger.info(f"已保存餘弦相似度矩陣到: {cosine_path}")
                elif compute_similarity: # 如果啟用了但條件不滿足
                    logger.warning(
                        f"層 '{layer_name}' (epoch {epoch}, dataset {dataset_name}) 的特徵向量只有一個樣本或維度不足，無法計算餘弦相似度。"
                    )

                # 如果啟用了 t-SNE 計算
                if (
                    compute_tsne
                    and features_to_save_tensor.dim() > 1 # 確保不是單一純量特徵
                ):
                    if (
                        features_to_save_tensor.shape[0] > self.tsne_perplexity + 1
                        and features_to_save_tensor.shape[0] > 2 # t-SNE 至少需要少量樣本
                    ):
                        # 使用 features_to_save_tensor (已展平) 進行計算
                        # 將 final_targets_for_layer 和標籤信息傳遞給 _compute_tsne
                        tsne_results = self._compute_tsne(
                            features_to_save_tensor,
                            targets=final_targets_for_layer,
                            perplexity=self.tsne_perplexity,
                            label_names=label_names,
                            label_mapping=label_mapping,
                            label_field=label_field,
                        )

                        # 合併元數據
                        tsne_data_to_save = {
                            "layer_name": layer_name,
                            "epoch": epoch,
                            "dataset_name": dataset_name,
                            # targets, tsne_coordinates 等由 _compute_tsne 返回
                            **tsne_results # 解包 _compute_tsne 的結果
                        }
                        
                        # 如果 _compute_tsne 沒有返回 targets，但我們有 final_targets_for_layer，也添加
                        if 'targets' not in tsne_data_to_save and final_targets_for_layer is not None:
                            tsne_data_to_save['targets'] = final_targets_for_layer

                        tsne_path = self.save_manager.save_feature_vector_data(
                            data=tsne_data_to_save,
                            layer_name=layer_name,
                            epoch=epoch,
                            dataset_name=dataset_name,
                            suffix="tsne",
                        )
                        if tsne_path:
                            logger.info(f"已保存t-SNE結果到: {tsne_path}")
                    else:
                        logger.warning(
                            f"層 '{layer_name}' (epoch {epoch}, dataset {dataset_name}) 的樣本數量不足 ({features_to_save_tensor.shape[0]}) 無法進行 t-SNE (perplexity={self.tsne_perplexity})。"
                        )
                elif compute_tsne: # 如果啟用了但條件不滿足
                    logger.warning(
                        f"層 '{layer_name}' (epoch {epoch}, dataset {dataset_name}) 的特徵維度不足，無法進行t-SNE。"
                    )

            except Exception as e:
                logger.error(
                    f"為層 '{layer_name}' (epoch {epoch}, dataset {dataset_name}) 保存特徵向量或進行分析時出錯: {e}",
                    exc_info=True,
                )

    def _compute_cosine_similarity(
        self, features: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """計算特徵向量間的餘弦相似度 (已整合)
        Args:
            features: 特徵向量，形狀為 [樣本數, 特徵維度] (應為展平後的)
            targets: 目標標籤，形狀為 [樣本數]
        Returns:
            Dict[str, Any]: 包含相似度矩陣和相關統計資訊的字典
        """
        try:
            features_np = features.cpu().numpy()
            similarity_matrix = cosine_similarity(features_np)
            result = {
                "similarity_matrix": torch.tensor(similarity_matrix),
                "timestamp": datetime.now().isoformat(),
                "num_samples": features.shape[0],
            }
            if targets is not None and len(targets) == len(features):
                try:
                    targets_np = targets.cpu().numpy()
                    unique_targets = np.unique(targets_np)
                    if len(unique_targets) < 1:  # 允許單類別，但不會計算類間相似度
                        logger.warning(
                            f"目標標籤只有一個或沒有類別 {unique_targets}，無法計算類間相似度，但仍會計算類內相似度"
                        )
                        # return result # 不直接返回，繼續計算可能的統計

                    intra_class_similarities = []
                    inter_class_similarities = []
                    centroids = {}
                    class_specific_similarities = {}

                    for cls_val in unique_targets:
                        cls_indices = np.where(targets_np == cls_val)[0]
                        if len(cls_indices) > 0:
                            cls_features = features_np[cls_indices]
                            centroids[cls_val] = np.mean(
                                cls_features, axis=0, keepdims=True
                            )
                            if len(cls_indices) > 1:
                                cls_similarity_matrix = similarity_matrix[
                                    np.ix_(cls_indices, cls_indices)
                                ]
                                mask = ~np.eye(
                                    cls_similarity_matrix.shape[0], dtype=bool
                                )
                                if (
                                    cls_similarity_matrix[mask].size > 0
                                ):  # 確保有元素可計算mean
                                    mean_sim = float(cls_similarity_matrix[mask].mean())
                                    intra_class_similarities.append(mean_sim)
                                    class_specific_similarities[int(cls_val)] = mean_sim
                                else:
                                    logger.debug(
                                        f"類別 {cls_val} 只有一個樣本或樣本間無有效相似度可計算均值"
                                    )

                    if len(unique_targets) > 1:  # 只有多於一個類別時才計算類間相似度
                        for i, cls1_val in enumerate(unique_targets):
                            for cls2_val in unique_targets[i + 1 :]:
                                cls1_indices = np.where(targets_np == cls1_val)[0]
                                cls2_indices = np.where(targets_np == cls2_val)[0]
                                if len(cls1_indices) > 0 and len(cls2_indices) > 0:
                                    cross_similarity = similarity_matrix[
                                        np.ix_(cls1_indices, cls2_indices)
                                    ]
                                    if cross_similarity.size > 0:
                                        inter_class_similarities.append(
                                            float(cross_similarity.mean())
                                        )

                    centroid_similarities = []
                    if centroids:  # 確保質心已計算
                        for i, sample_feature in enumerate(features_np):
                            cls_label = targets_np[i]
                            if cls_label in centroids:
                                centroid_sim_val = cosine_similarity(
                                    [sample_feature], centroids[cls_label]
                                )[0][0]
                                centroid_similarities.append(float(centroid_sim_val))

                    result["classes"] = [int(c) for c in unique_targets.tolist()]
                    if intra_class_similarities:
                        result["intra_class_avg_similarity"] = float(
                            np.mean(intra_class_similarities)
                        )
                    if inter_class_similarities:
                        result["inter_class_avg_similarity"] = float(
                            np.mean(inter_class_similarities)
                        )
                    if centroid_similarities:
                        result["sample_to_centroid_avg_similarity"] = float(
                            np.mean(centroid_similarities)
                        )
                    if class_specific_similarities:
                        result["intra_class_similarities_by_class"] = (
                            class_specific_similarities
                        )
                except Exception as e_cls_sim:
                    logger.error(
                        f"計算類別相關餘弦相似度時出錯: {e_cls_sim}", exc_info=True
                    )
            return result
        except Exception as e:
            logger.error(f"計算餘弦相似度時出現未預期的錯誤: {e}", exc_info=True)
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "num_samples": features.shape[0] if hasattr(features, "shape") else 0,
            }

    def _compute_tsne(
        self,
        features: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        perplexity: int = 30,
        label_names: Optional[List[str]] = None,
        label_mapping: Optional[Dict[int, str]] = None,
        label_field: Optional[str] = None,
    ) -> Dict[str, Any]:
        """使用t-SNE計算特徵向量的二維嵌入
        Args:
            features: 特徵向量，形狀為 [樣本數, 特徵維度] (應為展平後的)
            targets: 目標標籤，形狀為 [樣本數]
            perplexity: t-SNE的困惑度參數
            label_names: 原始標籤文本列表
            label_mapping: 標籤映射字典，例如 {0: "正常", 1: "輕度", 2: "中度"}
            label_field: 標籤欄位名稱，例如 "score", "DrLee_Evaluation"
        Returns:
            Dict[str, Any]: 包含t-SNE座標和相關統計資訊的字典
        """
        try:
            features_np = features.cpu().numpy()
            n_samples, n_features_dim = features_np.shape

            if n_samples <= 1:
                logger.warning(f"t-SNE計算：樣本數 ({n_samples}) 過少，無法執行t-SNE。")
                return {
                    "timestamp": datetime.now().isoformat(),
                    "error": "Too few samples for t-SNE",
                    "num_samples": n_samples,
                }

            # PCA降維以加速t-SNE，並處理高維數據
            # 確保PCA的n_components小於樣本數和特徵數
            pca_n_components = min(n_samples - 1, n_features_dim, 50)
            if (
                pca_n_components < 2 and n_features_dim >= 2
            ):  # 如果原始維度至少為2，PCA組件至少為2
                pca_n_components = min(
                    n_features_dim, n_samples - 1
                )  # 確保組件數不超過樣本數-1

            features_pca_np = features_np
            explained_variance_ratio = None
            if (
                n_features_dim > pca_n_components and pca_n_components > 0
            ):  # 僅當原始維度大於PCA組件數時執行PCA
                from sklearn.decomposition import PCA

                pca = PCA(n_components=pca_n_components, random_state=42)
                features_pca_np = pca.fit_transform(features_np)
                explained_variance_ratio = pca.explained_variance_ratio_
                logger.info(
                    f"t-SNE計算：使用PCA降維到{pca_n_components}個組件 (樣本數={n_samples}, 原始特徵數={n_features_dim})"
                )
            else:
                logger.info(
                    f"t-SNE計算：未執行PCA或PCA組件數不足 (樣本數={n_samples}, 原始特徵數={n_features_dim}, PCA組件={pca_n_components})，直接使用原特徵或PCA後特徵"
                )

            # 調整t-SNE的perplexity，確保它小於樣本數
            actual_perplexity = min(perplexity, n_samples - 1)
            if actual_perplexity <= 0:  # 如果樣本太少，無法運行TSNE
                logger.warning(
                    f"t-SNE計算：調整後的perplexity ({actual_perplexity}) 過小，無法執行t-SNE。樣本數: {n_samples}"
                )
                return {
                    "timestamp": datetime.now().isoformat(),
                    "error": "Perplexity too small for t-SNE after adjustment",
                    "num_samples": n_samples,
                    "original_dim": n_features_dim,
                }

            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=actual_perplexity,
                n_iter=300,
            )  # n_iter可以調整
            tsne_result_coords = tsne.fit_transform(features_pca_np)

            result = {
                "tsne_coordinates": torch.tensor(tsne_result_coords),
                "timestamp": datetime.now().isoformat(),
                "num_samples": n_samples,
                "original_dim": n_features_dim,  # 記錄原始維度
            }
            if explained_variance_ratio is not None:
                result["pca_explained_variance_ratio"] = torch.tensor(
                    explained_variance_ratio
                )
            if targets is not None:
                result["targets"] = targets.cpu()
            
            # 添加標籤相關信息
            if label_names is not None:
                result["label_names"] = label_names
            if label_mapping is not None:
                result["label_mapping"] = label_mapping
            if label_field is not None:
                result["label_field"] = label_field
            
            return result
        except ImportError:
            logger.error(
                "t-SNE計算失敗：scikit-learn未安裝或無法導入。請安裝scikit-learn以使用此功能。"
            )
            return {"error": "scikit-learn not available"}
        except Exception as e:
            logger.error(f"計算t-SNE時出現錯誤: {e}", exc_info=True)
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "num_samples": features.shape[0] if hasattr(features, "shape") else 0,
            }

    def _update_feature_analysis_summary(
        self,
        layer_name: str,
        features: torch.Tensor,
        targets: Optional[torch.Tensor],
        epoch: int,
        dataset_name: str,
    ):
        """更新特徵分析摘要 (已整合)
        Args:
            layer_name: 層名稱
            features: 特徵向量 (原始，非展平)
            targets: 目標標籤
            epoch: 當前epoch
            dataset_name: 數據集名稱 (用於日誌記錄，但不再用於檔名)
        """
        if not self.save_manager:
            return

        # 摘要文件不再包含 dataset_name 以符合 framework_data_structure.md
        summary_filename = "feature_analysis.json"  # <--- 修改檔名
        summary_path = self.save_manager.get_path("feature_vectors", summary_filename)

        summary = {}
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r") as f:
                    summary = json.load(f)
            except json.JSONDecodeError:
                logger.warning(
                    f"無法解析現有的特徵分析摘要文件: {summary_path}，將創建新的摘要。"
                )
                summary = {}

        if layer_name not in summary:
            summary[layer_name] = {}
        if str(epoch) not in summary[layer_name]:
            summary[layer_name][str(epoch)] = {}

        # 使用展平後的特徵計算統計量
        flattened_features_for_stats = features.view(features.shape[0], -1)

        layer_epoch_summary = {
            "num_samples": features.shape[0],
            "feature_dim_original": list(features.shape[1:]),  # 保存原始多維形狀
            "feature_dim_flattened": flattened_features_for_stats.shape[1],
            "timestamp": datetime.now().isoformat(),
            "dataset_name": dataset_name,
        }

        if flattened_features_for_stats.numel() > 0:  # 確保張量非空
            layer_epoch_summary["feature_norm_mean"] = float(
                torch.norm(flattened_features_for_stats, dim=1).mean().item()
            )
            layer_epoch_summary["feature_std"] = float(
                flattened_features_for_stats.std().item()
            )
        else:
            layer_epoch_summary["feature_norm_mean"] = None
            layer_epoch_summary["feature_std"] = None

        if targets is not None:
            unique_target_labels, counts = torch.unique(targets, return_counts=True)
            layer_epoch_summary["num_classes"] = len(unique_target_labels)
            layer_epoch_summary["class_distribution"] = {
                int(t): int(c)
                for t, c in zip(unique_target_labels.tolist(), counts.tolist())
            }

        summary[layer_name][str(epoch)] = layer_epoch_summary

        try:
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            logger.error(
                f"保存特徵分析摘要到 {summary_path} 時出錯: {e}", exc_info=True
            )

    def clear_feature_vectors(self):
        """清除已捕獲的特徵向量緩存"""
        for layer_name in self.feature_vectors:
            self.feature_vectors[layer_name] = []
        logger.debug("已清除所有緩存的特徵向量")

    def _remove_feature_hooks(self):
        """移除所有已註冊的特徵向量鉤子"""
        for layer_name, hook in self.feature_hooks:
            hook.remove()
            logger.debug(f"已移除層 '{layer_name}' 的特徵向量鉤子")
        self.feature_hooks = []
        if self.feature_vector_layers:  # 僅當實際監控時打印
            logger.info("已移除所有特徵向量鉤子")

    def update_batch_data(
        self,
                         inputs: Optional[torch.Tensor] = None,
                         outputs: Optional[torch.Tensor] = None,
                         targets: Optional[torch.Tensor] = None,
                         loss: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
        batch_idx: Optional[int] = None,
    ):
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
            self.inputs = (
                inputs.detach().cpu() if isinstance(inputs, torch.Tensor) else inputs
            )
        if outputs is not None:
            self.outputs = (
                outputs.detach().cpu() if isinstance(outputs, torch.Tensor) else outputs
            )
        if targets is not None:
            self.targets = (
                targets.detach().cpu() if isinstance(targets, torch.Tensor) else targets
            )
        if loss is not None:
            self.loss = loss.detach().cpu() if isinstance(loss, torch.Tensor) else loss
        if epoch is not None:
            self.current_epoch = epoch
        if batch_idx is not None:
            self.current_batch = batch_idx
            
        logger.debug(
            f"更新批次數據: epoch={self.current_epoch}, batch={self.current_batch}, loss={self.loss}"
        )
    
    def save_current_data(self):
        """保存當前數據到文件"""
        if not self.save_manager:
            logger.warning("未設置存檔管理器，無法保存數據")
            return
        
        # 保存激活值和梯度
        self.activation_hook.save_activations(self.current_epoch, self.current_batch)
        self.gradient_hook.save_gradients(self.current_epoch, self.current_batch)

        # 保存特徵向量 (將在 on_evaluation_end 中處理，以確保是評估數據)
        # 如果需要在每個批次保存，可以在這裡調用 self._save_feature_vectors(self.current_epoch, "batch_data")
        
        # 保存批次數據
        batch_data = {
            "epoch": self.current_epoch,
            "batch": self.current_batch,
            "timestamp": datetime.now().isoformat(),
        }
        
        # 只保存非 None 的數據
        if self.loss is not None:
            batch_data["loss"] = self.loss
        
        # 輸入/輸出/目標可能很大，只保存統計信息或小樣本
        if self.outputs is not None and isinstance(self.outputs, torch.Tensor):
            batch_data["outputs_stats"] = {
                "shape": list(self.outputs.shape),
                "mean": float(torch.mean(self.outputs).item()),
                "std": float(torch.std(self.outputs).item()),
                "min": float(torch.min(self.outputs).item()),
                "max": float(torch.max(self.outputs).item()),
            }
            # 保存小樣本（最多 10 個樣本）
            max_samples = min(10, self.outputs.shape[0])
            batch_data["outputs_sample"] = self.outputs[:max_samples].numpy().tolist()
        
        if self.targets is not None and isinstance(self.targets, torch.Tensor):
            batch_data["targets_stats"] = {
                "shape": list(self.targets.shape),
                "mean": (
                    float(torch.mean(self.targets).item())
                    if self.targets.dtype.is_floating_point
                    else None
                ),
                "std": (
                    float(torch.std(self.targets).item())
                    if self.targets.dtype.is_floating_point
                    else None
                ),
                "min": float(torch.min(self.targets).item()),
                "max": float(torch.max(self.targets).item()),
            }
            # 保存小樣本（最多 10 個樣本）
            max_samples = min(10, self.targets.shape[0])
            batch_data["targets_sample"] = self.targets[:max_samples].numpy().tolist()
        
        # 使用 SaveManager 保存批次數據
        self.save_manager.save_batch_data(
            batch_data, self.current_epoch, self.current_batch
        )
    
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
        self.clear_feature_vectors()  # 新增：清除特徵向量
        self.inputs = None
        self.outputs = None
        self.targets = None
        self.loss = None
    
    def remove_hooks(self):
        """移除所有已註冊的鉤子"""
        self.activation_hook.remove_hooks()
        self.gradient_hook.remove_hooks()
        self.clear_feature_vectors()  # 新增：移除特徵鉤子
    
    def __del__(self):
        """析構時移除鉤子"""
        self.remove_hooks()


# 以下是用於與 trainers 模組銜接的函數 ---------------------


class SimpleModelAnalyticsCallback(CallbackInterface):
    """簡易模型分析回調，用於在訓練過程中收集並保存模型數據
    
    該類實現了 CallbackInterface 接口定義的所有方法，能夠在訓練過程中
    捕獲中間層激活值和梯度，支持離線分析和可視化。無需依賴外部庫，直接存儲數據到指定目錄。
    """
    
    def __init__(
        self,
        model: nn.Module = None,
        output_dir: str = "results",
                monitored_layers: Optional[List[str]] = None,
                monitored_params: Optional[List[str]] = None,
        feature_vector_layers: Optional[List[str]] = None,
        save_frequency: int = 1,
        # 新增配置參數，用於控制特徵分析計算
        compute_similarity: bool = True,
        compute_tsne: bool = True,
        tsne_perplexity: int = 30,
    ):
        """初始化模型分析回調
        Args:
            # ... (其他参数)
            compute_similarity: 是否計算特徵向量的餘弦相似度
            compute_tsne: 是否計算特徵向量的t-SNE嵌入
            tsne_perplexity: t-SNE計算的困惑度參數
        """
        self.model = model
        self.output_dir = output_dir
        self.monitored_layers = monitored_layers
        self.monitored_params = monitored_params
        self.feature_vector_layers = feature_vector_layers
        self.save_frequency = save_frequency
        self.hook_manager: Optional[SimpleModelHookManager] = None
        self.save_manager: Optional[SaveManager] = None
        self.current_epoch = 0
        self.current_dataset_name_for_features = "eval"

        # 保存特徵分析的配置
        self.compute_similarity = compute_similarity
        self.compute_tsne = compute_tsne
        self.tsne_perplexity = tsne_perplexity

        logger.info(
            f"初始化模型分析回調，輸出目錄: {output_dir}, 保存頻率: {save_frequency} epoch"
        )
        if feature_vector_layers:
            logger.info(f"將捕獲以下層的特徵向量: {feature_vector_layers}")
            logger.info(f"  計算餘弦相似度: {self.compute_similarity}")
            logger.info(f"  計算t-SNE: {self.compute_tsne}")
            if self.compute_tsne:
                logger.info(f"  t-SNE Perplexity: {self.tsne_perplexity}")
    
    def on_train_begin(self, model: nn.Module, logs: Dict[str, Any] = None) -> None:
        """訓練開始時的回調
        
        Args:
            model: 訓練的模型
            logs: 日誌字典
        """
        if self.model is None:
            self.model = model
        
        # 如果輸出目錄是相對路徑，嘗試從日誌中獲取基本輸出目錄
        if logs and "tensorboard_writer" in logs:
            tb_writer = logs["tensorboard_writer"]
            if hasattr(tb_writer, "log_dir"):
                parent_dir = os.path.dirname(os.path.dirname(tb_writer.log_dir))
                self.output_dir = parent_dir
                logger.info(
                    f"從TensorBoard日誌獲取輸出目錄 (實驗根目錄): {self.output_dir}"
                )
        
        # 創建存檔管理器
        self.save_manager = SaveManager(self.output_dir)
        
        # 創建鉤子管理器
        self.hook_manager = SimpleModelHookManager(
            self.model,
            self.monitored_layers,
            self.monitored_params,
            save_manager=self.save_manager,
            feature_vector_layers=self.feature_vector_layers,  # 傳遞特徵向量層
            tsne_perplexity=self.tsne_perplexity,  # 傳遞 t-SNE perplexity 參數
        )
        logger.info(
            f"模型分析回調已初始化，輸出目錄設定為: {self.save_manager.experiment_dir if self.save_manager else '未知'}"
        )
    
    def on_epoch_begin(
        self, epoch: int, model: nn.Module, logs: Dict[str, Any] = None
    ) -> None:
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
    
    def on_batch_begin(
        self,
        batch: int,
        model: nn.Module,
        inputs: torch.Tensor = None,
        targets: torch.Tensor = None,
        logs: Dict[str, Any] = None,
    ) -> None:
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
    
    def on_batch_end(
        self,
        batch: int,
        model: nn.Module,
        inputs: torch.Tensor = None,
        targets: torch.Tensor = None,
        outputs: torch.Tensor = None,
        loss: torch.Tensor = None,
        logs: Dict[str, Any] = None,
    ) -> None:
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
                batch_idx=batch,
            )

            # 為特徵向量分析收集目標 (僅在評估階段且特徵捕獲已啟用時)
            # 假設 on_evaluation_begin 已被調用，且模型處於 eval模式
            if (
                not model.training
                and self.hook_manager.feature_hooks
                and hasattr(self.hook_manager, "all_targets_for_features")
            ):
                if targets is not None:
                    self.hook_manager.all_targets_for_features.append(
                        targets.detach().cpu()
                    )

            # 每 N 個批次保存一次數據 (例如梯度和SimpleActivationHook的激活值)
            # save_this_batch = False # 根據您的邏輯決定是否每個批次都保存梯度等
            # ...

    def on_evaluation_begin(
        self, model: nn.Module, logs: Dict[str, Any] = None
    ) -> None:
        """評估開始時的回調"""
        if self.hook_manager:
            if self.hook_manager.model is not model:
                self.hook_manager.model = model
                logger.info("Hook manager中的模型已更新為評估模型")

            self.current_dataset_name_for_features = logs.get("dataset_name", "eval")
            self.hook_manager.current_epoch = logs.get(
                "epoch", self.current_epoch
            )  # 確保hook_manager知道當前epoch
            logger.info(
                f"評估開始，數據集: {self.current_dataset_name_for_features}, Epoch: {self.hook_manager.current_epoch}。將為此數據集註冊特徵向量鉤子。"
            )
            self.hook_manager.register_feature_hooks()
            self.hook_manager.clear_feature_vectors()

            # 如果需要，在這裡收集 targets 以便後續分析
            # 例如，可以初始化一個列表來存儲每個批次的 targets
            if self.hook_manager.feature_vector_layers:  # 僅當需要捕獲特徵時
                self.hook_manager.all_targets_for_features = []

    def on_evaluation_end(
        self,
        model: nn.Module,
        results: Dict[str, Any] = None,
        logs: Dict[str, Any] = None,
    ) -> None:
        """評估結束時的回調"""
        if self.hook_manager:
            current_eval_epoch = logs.get("epoch", self.current_epoch)
            dataset_name = logs.get(
                "dataset_name", self.current_dataset_name_for_features
            )
            logger.info(
                f"評估結束，數據集: {dataset_name}, Epoch: {current_eval_epoch}。將保存捕獲的特徵向量。"
            )

            # 從 self.hook_manager 獲取收集到的 targets
            targets_for_saving = (
                self.hook_manager.all_targets_for_features
                if hasattr(self.hook_manager, "all_targets_for_features")
                else None
            )

            # 從配置中獲取標籤相關資訊
            config = logs.get('config', {})
            label_field = None
            label_mapping = None
            label_names = None
            
            # 嘗試從不同可能的配置位置獲取標籤信息
            dataset_config = config.get('dataset', {})
            
            # 對於索引化數據集，從index配置獲取標籤信息
            if dataset_config.get('data_source', {}).get('use_index_csv', False):
                index_config = dataset_config.get('data_source', {}).get('index', {})
                label_field = index_config.get('label_field')
                
                if 'label_mapping' in index_config:
                    label_mapping = index_config.get('label_mapping')
                    # 確保鍵是整數
                    if isinstance(label_mapping, dict):
                        label_mapping = {int(k): v for k, v in label_mapping.items()}
                    
                    # 提取標籤名稱列表
                    if label_mapping:
                        label_names = list(label_mapping.values())
            
            # 對於分類任務，從class_names獲取標籤信息
            if not label_field and dataset_config.get('is_classification', False):
                class_names = dataset_config.get('class_names', [])
                if class_names:
                    label_names = class_names
                    label_mapping = {i: name for i, name in enumerate(class_names)}
                    label_field = 'class'
            
            # 對於回歸任務，設置標籤字段為score並創建範圍標籤
            if not label_field and dataset_config.get('target_type') == 'regression':
                label_field = 'score'
                
                # 為回歸任務創建範圍標籤
                if targets_for_saving:
                    try:
                        all_targets = torch.cat(targets_for_saving, dim=0)
                        min_val = all_targets.min().item()
                        max_val = all_targets.max().item()
                        
                        # 創建範圍段
                        third = (max_val - min_val) / 3
                        label_mapping = {
                            0: f"{min_val:.1f}-{(min_val+third):.1f}",
                            1: f"{(min_val+third):.1f}-{(min_val+2*third):.1f}",
                            2: f"{(min_val+2*third):.1f}-{max_val:.1f}"
                        }
                        label_names = list(label_mapping.values())
                    except Exception as e:
                        logger.warning(f"創建回歸數據的標籤映射時出錯: {e}")
            
            # 如果沒有從配置獲取到標籤信息，使用默認值 - DrLee_Evaluation
            if not label_field:
                label_field = "DrLee_Evaluation"
                label_mapping = {
                    0: "聽起來正常",
                    1: "輕度異常",
                    2: "重度異常"
                }
                label_names = list(label_mapping.values())
                logger.info(f"未從配置中找到標籤信息，使用默認值: DrLee_Evaluation")
            
            logger.info(f"將使用以下標籤信息保存特徵向量：")
            logger.info(f"  標籤欄位: {label_field}")
            logger.info(f"  標籤名稱: {label_names}")
            logger.info(f"  標籤映射: {label_mapping}")

            # 調用feature_vector保存方法
            self.hook_manager._save_feature_vectors(
                epoch=current_eval_epoch,
                collected_targets=targets_for_saving,
                dataset_name=dataset_name,
                compute_similarity=self.compute_similarity,
                compute_tsne=self.compute_tsne,
                label_names=label_names,
                label_mapping=label_mapping,
                label_field=label_field
            )
            
            self.hook_manager._remove_feature_hooks()

    def on_epoch_end(
        self,
        epoch: int,
        model: nn.Module,
        train_logs: Dict[str, Any] = None,
        val_logs: Dict[str, Any] = None,
        logs: Dict[str, Any] = None,
    ) -> None:
        """epoch 結束時的回調"""
        logger.info(f"SimpleModelAnalyticsCallback: 結束 epoch {epoch}")

        # 保存驗證集預測結果 (如果存在)
        if val_logs and "metrics" in val_logs and self.save_manager:
            metrics_data = val_logs["metrics"]
            validation_outputs = metrics_data.get("outputs")
            validation_targets = metrics_data.get("targets")
            validation_predictions = metrics_data.get("predictions")
        
            if validation_outputs is not None and validation_targets is not None:
                logger.info(
                    f"SimpleModelAnalyticsCallback: Epoch {epoch} 檢測到驗證集 outputs 和 targets，準備保存 epoch_{epoch}_validation_predictions.pt"
                )
                try:
                    validation_results_to_save = {
                        "outputs": (
                            validation_outputs.cpu()
                            if isinstance(validation_outputs, torch.Tensor)
                            else validation_outputs
                        ),
                        "targets": (
                            validation_targets.cpu()
                            if isinstance(validation_targets, torch.Tensor)
                            else validation_targets
                        ),
                    }
                    if validation_predictions is not None:
                        validation_results_to_save["predictions"] = (
                            validation_predictions.cpu()
                            if isinstance(validation_predictions, torch.Tensor)
                            else validation_predictions
                        )

                    hooks_dir = self.save_manager.get_path("hooks", "")  # 獲取hooks子目錄
                    os.makedirs(hooks_dir, exist_ok=True)
                    save_path = os.path.join(
                        hooks_dir, f"epoch_{epoch}_validation_predictions.pt"
                    )
                    torch.save(validation_results_to_save, save_path)
                    logger.info(
                        f"SimpleModelAnalyticsCallback: Epoch {epoch} 驗證集預測結果已保存到: {save_path}"
                    )
                except Exception as e:  # 確保 try 有對應的 except
                    logger.error(
                        f"SimpleModelAnalyticsCallback: 保存 Epoch {epoch} 驗證集預測結果時出錯: {e}",
                        exc_info=True,
                    )
            else:  # 這個 else 對應 if validation_outputs is not None...
                logger.warning(
                    f"SimpleModelAnalyticsCallback: Epoch {epoch} 未能從 val_logs['metrics'] 中獲取 'outputs' 或 'targets'，無法保存 epoch_{epoch}_validation_predictions.pt"
                )
        elif not self.save_manager:  # 這個 elif 對應 if val_logs and 'metrics' in val_logs...
            logger.warning(
                f"SimpleModelAnalyticsCallback: Epoch {epoch} - SaveManager 未初始化，無法保存驗證預測。"
            )

        # 如果達到保存頻率，保存 epoch 級別的摘要數據 (如梯度、GNS等)
        if (
            (epoch + 1) % self.save_frequency == 0
            and self.save_manager
            and self.hook_manager
        ):
            logger.info(
                f"SimpleModelAnalyticsCallback: Epoch {epoch} 達到保存頻率，調用 save_manager.save_epoch_summary"
            )
            self.save_manager.save_epoch_summary(epoch, train_logs, val_logs)

    def on_train_end(
        self,
        model: nn.Module,
        history: Dict[str, List] = None,
        logs: Dict[str, Any] = None,
    ) -> None:
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
                    self.save_manager.save_training_summary(
                        history, self.current_epoch + 1
                    )
                except Exception as e:
                    logger.error(f"保存訓練摘要時出錯: {e}")
                    
                    # 備用方案：保存為 PT 檔案而非 JSON
                    try:
                        save_path = self.save_manager.get_path(
                            "results", "training_summary.pt"
                        )
                        torch.save(
                            {
                                "total_epochs": self.current_epoch + 1,
                                "timestamp": datetime.now().isoformat(),
                            },
                            save_path,
                        )
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
    
    def __init__(self, save_manager: SaveManager, dataset_name: str = "test"):
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
        self.all_targets: List[torch.Tensor] = []
        self.all_predictions: List[torch.Tensor] = []
        self.all_raw_outputs: List[torch.Tensor] = []  # 改名以清晰表示這是原始模型輸出
        self.metrics: Dict[str, Any] = {}
        self.batch_count = 0
    
    def on_evaluation_begin(
        self, model: nn.Module, logs: Dict[str, Any] = None
    ) -> None:
        """評估開始時的處理"""
        self.all_targets = []
        self.all_predictions = []
        self.all_raw_outputs = []  # 清空原始輸出
        self.metrics = {}
        self.batch_count = 0
        logger.info(
            f"EvaluationResultsHook: 開始收集 {self.dataset_name} 數據集的評估結果 (Epoch: {logs.get('epoch', 'N/A') if logs else 'N/A'})"
        )

    def on_batch_end(
        self,
        batch: int,
        model: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        loss: Optional[torch.Tensor] = None,
        logs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """評估批次結束時的處理"""
        current_phase = logs.get("phase", "unknown") if logs else "unknown"
        # 確保只在 self.dataset_name 匹配當前評估的 phase (或 dataset_name) 時收集
        # 或者，如果 logs 中直接提供了 dataset_name，則優先使用它
        active_dataset_name = (
            logs.get("dataset_name", current_phase) if logs else current_phase
        )

        if active_dataset_name == self.dataset_name:
            if targets is not None:
                self.all_targets.append(targets.detach().cpu())
            
            if outputs is not None:
                self.all_raw_outputs.append(outputs.detach().cpu())  # 存儲原始模型輸出 (logits)

            # 批次計數器只應在正確的 dataset_name 時增加
            self.batch_count += 1
            logger.debug(
                f"EvaluationResultsHook ({self.dataset_name}): 處理完批次 {batch}, "
                f"已收集 {self.batch_count} 個批次數據。Raw outputs: {len(self.all_raw_outputs)}, "
                f"Targets: {len(self.all_targets)}"
            )
        else:
            logger.debug(
                f"EvaluationResultsHook: 跳過批次 {batch}，因為目標數據集 '{self.dataset_name}' "
                f"與當前評估數據集 '{active_dataset_name}' 不匹配。"
            )

    def on_evaluation_end(
        self,
        model: nn.Module,
        results: Optional[Dict[str, Any]] = None,
        logs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """評估結束時的處理"""
        current_epoch = logs.get("epoch", -1) if logs else -1
        current_dataset_name = (
            logs.get("dataset_name", "unknown") if logs else "unknown"
        )

        if current_dataset_name != self.dataset_name:
            logger.debug(
                f"EvaluationResultsHook: 跳過 on_evaluation_end，因為目標數據集 '{self.dataset_name}' 与 logs 中的數據集 '{current_dataset_name}' 不匹配。"
            )
            return
            
        logger.info(
            f"EvaluationResultsHook: {self.dataset_name} 數據集評估結束 (Epoch: {current_epoch})。收集了 {self.batch_count} 個批次的數據"
        )

        experiment_config = logs.get("config", {}) if logs else {}
        metrics_from_trainer = logs.get("metrics", results if results else {})

        final_targets_tensor: Optional[torch.Tensor] = None
        final_raw_outputs_tensor: Optional[torch.Tensor] = None
        final_predictions_tensor: Optional[torch.Tensor] = None
        final_probabilities_tensor: Optional[torch.Tensor] = None

        if self.all_targets:
            try:
                final_targets_tensor = torch.cat(self.all_targets, dim=0)
            except Exception as e:
                logger.error(
                    f"EvaluationResultsHook ({self.dataset_name}): 拼接 all_targets 時出錯: {e}",
                    exc_info=True,
                )

        if self.all_raw_outputs:
            try:
                final_raw_outputs_tensor = torch.cat(self.all_raw_outputs, dim=0)
            except Exception as e:
                logger.error(
                    f"EvaluationResultsHook ({self.dataset_name}): 拼接 all_raw_outputs 時出錯: {e}",
                    exc_info=True,
                )

        if final_raw_outputs_tensor is not None:
            is_classification_task = (
                experiment_config.get("model", {})
                .get("parameters", {})
                .get("is_classification", True)
            )
            if is_classification_task:
                logger.debug(
                    f"EvaluationResultsHook ({self.dataset_name}): 分類任務，處理原始輸出以獲取預測和概率。原始輸出形狀: {final_raw_outputs_tensor.shape}"
                )
                if (
                    final_raw_outputs_tensor.dim() > 1
                    and final_raw_outputs_tensor.size(1) > 1
                ):
                    final_probabilities_tensor = torch.softmax(
                        final_raw_outputs_tensor, dim=1
                    )
                    final_predictions_tensor = torch.argmax(
                        final_probabilities_tensor, dim=1
                    )
                elif final_raw_outputs_tensor.dim() > 0:
                    final_probabilities_tensor = torch.sigmoid(final_raw_outputs_tensor)
                    final_predictions_tensor = (final_probabilities_tensor > 0.5).long()
                    if (
                        final_probabilities_tensor.dim() == 2
                        and final_probabilities_tensor.size(1) == 1
                    ):
                        final_probabilities_tensor = final_probabilities_tensor.squeeze(1)
                        final_predictions_tensor = final_predictions_tensor.squeeze(1)
                else:
                    logger.warning(
                        f"EvaluationResultsHook ({self.dataset_name}): 分類任務的原始輸出維度無法識別，跳過概率/預測轉換。形狀: {final_raw_outputs_tensor.shape}"
                    )
            else:
                logger.debug(
                    f"EvaluationResultsHook ({self.dataset_name}): 回歸任務，預測和概率直接使用原始輸出。"
                )
                final_predictions_tensor = final_raw_outputs_tensor
                final_probabilities_tensor = final_raw_outputs_tensor
        elif not self.all_raw_outputs and self.all_predictions:
            logger.warning(
                f"EvaluationResultsHook ({self.dataset_name}): final_raw_outputs_tensor 為空，但 self.all_predictions 有數據，嘗試使用它。"
            )
            try:
                potential_probs_or_logits = torch.cat(self.all_predictions, dim=0)
                is_classification_task = (
                    experiment_config.get("model", {})
                    .get("parameters", {})
                    .get("is_classification", True)
                )
                if is_classification_task:
                    if (
                        potential_probs_or_logits.dim() > 1
                        and potential_probs_or_logits.size(1) > 1
                    ):
                        final_probabilities_tensor = torch.softmax(
                            potential_probs_or_logits, dim=1
                        )
                        final_predictions_tensor = torch.argmax(
                            final_probabilities_tensor, dim=1
                        )
                    elif potential_probs_or_logits.dim() > 0:
                        final_probabilities_tensor = torch.sigmoid(
                            potential_probs_or_logits
                        )
                        final_predictions_tensor = (final_probabilities_tensor > 0.5).long()
                        if (
                            final_probabilities_tensor.dim() == 2
                            and final_probabilities_tensor.size(1) == 1
                        ):
                            final_probabilities_tensor = final_probabilities_tensor.squeeze(1)
                            final_predictions_tensor = final_predictions_tensor.squeeze(1)
                else:
                    final_predictions_tensor = potential_probs_or_logits
                    final_probabilities_tensor = potential_probs_or_logits
            except Exception as e:
                logger.error(
                    f"EvaluationResultsHook ({self.dataset_name}): 處理 self.all_predictions 時出錯: {e}",
                    exc_info=True,
                )
                final_predictions_tensor = None
                final_probabilities_tensor = None

        if final_targets_tensor is not None and final_predictions_tensor is not None:
            logger.info(
                f"EvaluationResultsHook ({self.dataset_name}): 準備保存結果。Targets: {final_targets_tensor.shape}, Predictions: {final_predictions_tensor.shape}"
            )
            eval_results_to_save = {
                "targets": final_targets_tensor.cpu(),
                "predictions": final_predictions_tensor.cpu(),
                "timestamp": datetime.now().isoformat(),
                "epoch": current_epoch,
            }
            if final_probabilities_tensor is not None:
                eval_results_to_save["probabilities"] = final_probabilities_tensor.cpu()

            calculated_metrics = {
                k: v
                for k, v in metrics_from_trainer.items()
                if k not in ["outputs", "targets", "predictions"]
            }
            if calculated_metrics:
                eval_results_to_save["metrics"] = calculated_metrics

            if not self.save_manager:
                logger.error(
                    f"EvaluationResultsHook ({self.dataset_name}): SaveManager 未初始化，無法保存。"
                )
                return

            hooks_dir = self.save_manager.get_path("hooks", "")
            os.makedirs(hooks_dir, exist_ok=True)
            save_path = os.path.join(
                hooks_dir, f"evaluation_results_{self.dataset_name}.pt"
            )
            try:
                torch.save(eval_results_to_save, save_path)
                logger.info(
                    f"EvaluationResultsHook: {self.dataset_name} 數據集的評估結果已保存到: {save_path}"
                )

                if self.dataset_name == "test":
                    root_save_path = os.path.join(
                        self.save_manager.experiment_dir, "test_predictions.pt"
                    )
                    try:
                        torch.save(eval_results_to_save, root_save_path)
                        logger.info(
                            f"EvaluationResultsHook: {self.dataset_name} 測試結果副本已保存到: {root_save_path}"
                        )
                    except Exception as e:
                        logger.error(
                            f"EvaluationResultsHook ({self.dataset_name}): 保存到 {root_save_path} 時出錯: {e}",
                            exc_info=True,
                        )
            except Exception as e:
                logger.error(
                    f"EvaluationResultsHook ({self.dataset_name}): 保存到 {save_path} 時出錯: {e}",
                    exc_info=True,
                )
        else:
            logger.warning(
                f"EvaluationResultsHook ({self.dataset_name}): 未能獲取足夠的 targets 或 predictions/raw_outputs 數據 ({'targets missing' if final_targets_tensor is None else ''}{' predictions missing' if final_predictions_tensor is None else ''})，無法保存評估結果。"
            )

        self.all_targets = []
        self.all_predictions = []
        self.all_raw_outputs = []
        self.batch_count = 0


def create_analyzer_callback(
    output_dir: str = "results",
    monitored_layers: Optional[List[str]] = None,
    monitored_params: Optional[List[str]] = None,
    save_frequency: int = 1,
    capture_evaluation_results: bool = False,
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
                save_frequency=save_frequency,
            )
        except ImportError as e:
            logger.warning(
                f"無法導入 SBP_analyzer.callbacks.ModelAnalyticsCallback: {e}"
            )
            logger.warning("將使用內置的簡易模型分析器")
    
    # 回退到內置的簡易分析器
    analytics_callback = SimpleModelAnalyticsCallback(
        output_dir=output_dir,
        monitored_layers=monitored_layers,
        monitored_params=monitored_params,
        save_frequency=save_frequency,
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
    output_dir = config.get("global", {}).get("output_dir", "results")
    
    # 從training部分獲取save_every作為默認保存頻率
    default_save_frequency = config.get("training", {}).get("save_every", 1)
    
    # 檢查是否配置了 hook 部分
    hooks_config = config.get("hooks", {})
    # 如果配置中沒有hooks部分或hooks部分為空，直接返回空列表
    if "hooks" not in config or not config["hooks"]:
            logger.info("配置中沒有hooks部分或hooks為空，不添加任何回調")
            return []
    
    # 特徵向量捕獲層（如果 activation_capture 啟用）s
    feature_vector_layers_for_analytics = None
    if hooks_config.get("activation_capture", {}).get("enabled", False):
        activation_config = hooks_config.get("activation_capture", {})
        feature_vector_layers_for_analytics = activation_config.get("target_layers", [])
        if feature_vector_layers_for_analytics:
            logger.info(
                f"檢測到 activation_capture 配置，將傳遞 target_layers 給 ModelAnalyticsCallback: {feature_vector_layers_for_analytics}"
            )
        # 注意：activation_capture 的其他配置（如 compute_similarity, tsne）將不再由此處直接處理
        # 這些功能需要被整合到 SimpleModelAnalyticsCallback 或專門的可視化鉤子中（如果需要保留）

    # 模型分析回調 (整合了梯度、SimpleActivationHook的激活值以及新的特徵向量捕獲)
    if hooks_config.get("model_analytics", {}).get("enabled", False):
        analytics_config = hooks_config.get("model_analytics", {})
        monitored_layers = analytics_config.get("monitored_layers", [])
        monitored_params = analytics_config.get("monitored_params", [])
        save_frequency = analytics_config.get("save_frequency", default_save_frequency)
        current_feature_vector_layers = analytics_config.get(
            "feature_vector_layers", feature_vector_layers_for_analytics
        )

        # 從 model_analytics 配置中讀取特徵分析相關參數
        compute_similarity_cfg = analytics_config.get("compute_similarity", True)
        compute_tsne_cfg = analytics_config.get("compute_tsne", True)
        tsne_perplexity_cfg = analytics_config.get("tsne_perplexity", 30)

        # 讀取標籤相關配置 (如果存在)
        debug_mode = analytics_config.get("debug_mode", False)

        logger.info(f"ModelAnalyticsCallback 將使用 save_frequency: {save_frequency}")
        if current_feature_vector_layers:
            logger.info(
                f"ModelAnalyticsCallback 將捕獲以下層的特徵向量: {current_feature_vector_layers}"
            )
            logger.info(f"  計算餘弦相似度 (來自配置): {compute_similarity_cfg}")
            logger.info(f"  計算t-SNE (來自配置): {compute_tsne_cfg}")
            if compute_tsne_cfg:
                logger.info(f"  t-SNE Perplexity (來自配置): {tsne_perplexity_cfg}")
        
        if debug_mode:
            logger.info(f"ModelAnalyticsCallback 將在調試模式下運行，啟用更詳細的日誌和檢查")

        # 創建 SaveManager 實例
        save_manager = SaveManager(base_dir=output_dir, create_subdirs=True)
        logger.info(f"為回調創建了SaveManager，根目錄: {output_dir}")

        analytics_callback_instance = SimpleModelAnalyticsCallback(
            output_dir=output_dir,
            monitored_layers=monitored_layers,
            monitored_params=monitored_params,
            feature_vector_layers=current_feature_vector_layers,
            save_frequency=save_frequency,
            compute_similarity=compute_similarity_cfg,  # 傳遞配置值
            compute_tsne=compute_tsne_cfg,  # 傳遞配置值
            tsne_perplexity=tsne_perplexity_cfg,  # 傳遞配置值
        )
        callbacks.append(analytics_callback_instance)

    # 評估結果捕獲 (EvaluationResultsHook)
    if hooks_config.get("evaluation_capture", {}).get("enabled", False):
        eval_capture_config = hooks_config.get("evaluation_capture", {})
        datasets_to_capture = eval_capture_config.get("datasets", ["test"])
        # 如果還沒有創建 SaveManager，則創建一個
        if "save_manager" not in locals():
            save_manager = SaveManager(base_dir=output_dir, create_subdirs=True)
            logger.info(
                f"為EvaluationResultsHook創建了SaveManager，根目錄: {output_dir}"
            )

        for dataset_name in datasets_to_capture:
            callbacks.append(EvaluationResultsHook(save_manager, dataset_name))
            logger.info(f"已添加 EvaluationResultsHook 用於數據集: {dataset_name}")

    # 移除獨立的 ActivationCaptureHook 實例化，因為其功能已整合
    # if hooks_config.get('activation_capture', {}).get('enabled', False):
    #     # ... (舊的 ActivationCaptureHook 實例化代碼已刪除)
    #     pass
    
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
