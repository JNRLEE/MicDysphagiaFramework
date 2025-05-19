#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
測試PCA降維功能的腳本
此腳本創建模擬特徵數據，並驗證PCA降維的效果
"""

import numpy as np
import torch
from data.feature_dataset import FeatureDataset
import tempfile
import os
import pandas as pd
import logging
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data(temp_dir, feature_dims=[2000, 3000, 2500], num_samples=100):
    """創建測試數據
    
    Args:
        temp_dir: 臨時目錄
        feature_dims: 要創建的特徵向量維度列表
        num_samples: 要創建的樣本數量
        
    Returns:
        tuple: (特徵路徑列表, 測試索引CSV路徑)
    """
    feature_paths = []
    file_paths = []
    labels = []
    scores = []
    
    # 根據num_samples創建樣本，循環使用feature_dims
    for i in range(num_samples):
        # 選擇當前樣本的特徵維度
        dim = feature_dims[i % len(feature_dims)]
        
        # 創建有一定結構的特徵（不是純隨機）
        base = np.sin(np.linspace(0, 10*np.pi, dim)) * 2  # 基礎正弦波形
        freq_factor = 0.5 + (i % 5) * 0.2  # 添加一些頻率變化
        phase = (i % 3) * np.pi / 4  # 添加相位變化
        base = np.sin(np.linspace(0, 10*np.pi * freq_factor, dim) + phase) * 2
        noise = np.random.normal(0, 0.5, dim)  # 添加一些噪聲
        feat = base + noise
        
        # 保存測試特徵
        feat_path = os.path.join(temp_dir, f'feat{i+1}.npy')
        np.save(feat_path, feat)
        feature_paths.append(feat_path)
        
        # 創建文件路徑（雖然是虛擬的）
        file_paths.append(os.path.join(temp_dir, f'dir{i+1}'))
        
        # 創建標籤和分數
        label = '正常' if i % 3 == 0 else '中度' if i % 3 == 1 else '重度'
        labels.append(label)
        scores.append((i % 10) * 3)  # 0-27的分數
        
        # 只為前10個樣本打印日誌，避免輸出過多
        if i < 10:
            logger.info(f"創建特徵向量 {i+1}: shape={feat.shape}, 保存至 {feat_path}")
    
    # 如果樣本較多，只顯示總結
    if num_samples > 10:
        logger.info(f"總共創建了 {num_samples} 個特徵向量，維度分別為 {feature_dims}")
    
    # 創建測試索引CSV
    test_csv_path = os.path.join(temp_dir, 'test_index.csv')
    test_data = pd.DataFrame({
        'file_path': file_paths,
        'features_path': feature_paths,
        'DrLee_Evaluation': labels,
        'score': scores
    })
    test_data.to_csv(test_csv_path, index=False)
    logger.info(f"創建測試索引CSV: {test_csv_path}, 包含 {len(feature_paths)} 個樣本")
    
    return feature_paths, test_csv_path

def visualize_pca_components(features_array, pca):
    """可視化PCA主成分
    
    Args:
        features_array: 原始特徵數組
        pca: 已訓練的PCA模型
    """
    # 創建圖形
    plt.figure(figsize=(15, 10))
    
    # 繪製前10個主成分的解釋方差比
    plt.subplot(2, 2, 1)
    n_components = min(10, len(pca.explained_variance_ratio_))
    plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_[:n_components] * 100)
    plt.title('前10個主成分的解釋方差百分比')
    plt.xlabel('主成分')
    plt.ylabel('解釋方差百分比 (%)')
    
    # 繪製累積解釋方差
    plt.subplot(2, 2, 2)
    cumulative = np.cumsum(pca.explained_variance_ratio_) * 100
    plt.plot(range(1, len(cumulative) + 1), cumulative)
    plt.title('累積解釋方差')
    plt.xlabel('主成分數量')
    plt.ylabel('累積解釋方差 (%)')
    plt.axhline(y=95, color='r', linestyle='--', label='95% 閾值')
    plt.legend()
    
    # 繪製原始數據的前兩個主成分投影
    plt.subplot(2, 2, 3)
    transformed = pca.transform(features_array)
    plt.scatter(transformed[:, 0], transformed[:, 1])
    plt.title('數據在前2個主成分上的投影')
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    
    # 繪製前兩個主成分的方向向量
    plt.subplot(2, 2, 4)
    if len(features_array[0]) > 500:  # 如果特徵維度很高，只展示部分特徵
        components = pca.components_[:2, :500]
        plt.plot(components[0], label='主成分1 (前500維)')
        plt.plot(components[1], label='主成分2 (前500維)')
    else:
        components = pca.components_[:2]
        plt.plot(components[0], label='主成分1')
        plt.plot(components[1], label='主成分2')
    plt.title('前2個主成分的方向向量')
    plt.legend()
    
    # 保存圖片
    plt.tight_layout()
    plt.savefig('pca_visualization.png')
    logger.info("PCA可視化已保存到 pca_visualization.png")

def main():
    """主函數"""
    # 創建臨時目錄
    with tempfile.TemporaryDirectory() as temp_dir:
        # 創建測試數據 - 增加樣本數量到100個
        feature_paths, test_csv_path = create_test_data(temp_dir, num_samples=100)
        
        # 測試不同的填充和壓縮模式
        test_modes = [
            ('center', None),   # 置中填充，無壓縮
            ('center', 'pca'),  # 置中填充，PCA壓縮
            ('right', 'pca')    # 右側填充，PCA壓縮
        ]
        
        for padding_mode, compression_method in test_modes:
            logger.info(f"\n===== 測試模式: padding_mode={padding_mode}, compression_method={compression_method} =====")
            
            # 創建配置
            config = {
                'data': {
                    'preprocessing': {
                        'features': {
                            'normalize': False,
                            'max_feature_dim': 1024,
                            'padding_mode': padding_mode,
                            'compression_method': compression_method,
                            'target_dim': 64  # 較小的維度便於測試
                        }
                    }
                }
            }
            
            # 創建數據集
            dataset = FeatureDataset(
                index_path=test_csv_path,
                label_field="DrLee_Evaluation",
                config=config,
                is_train=True,  # 設為True以初始化PCA
            )
            
            # 檢查數據集
            logger.info(f"數據集大小: {len(dataset)}")
            
            # 加載幾個樣本，檢查其維度
            feature_dims = []
            for i in range(min(len(dataset), 3)):
                features, label = dataset[i]
                feature_dims.append(features.shape[0])
                logger.info(f"樣本 {i}: 特徵維度={features.shape}, 標籤={label}")
            
            # 檢查所有樣本的維度是否一致
            if len(set(feature_dims)) == 1:
                logger.info(f"所有樣本的維度一致: {feature_dims[0]}")
            else:
                logger.warning(f"樣本維度不一致: {feature_dims}")
            
            # 如果使用了PCA，進行額外檢查
            if compression_method == 'pca' and dataset.pca is not None:
                # 顯示PCA壓縮效果
                explained_variance = sum(dataset.pca.explained_variance_ratio_) * 100
                logger.info(f"PCA ({dataset.target_dim}個主成分) 解釋了 {explained_variance:.2f}% 的方差")
                
                # 驗證PCA壓縮效果
                # 加載原始特徵
                original_features = []
                
                # 只使用前5個樣本進行可視化
                visual_paths = feature_paths[:5]
                
                # 處理特徵以便可視化
                max_length = 0
                raw_features = []
                
                # 先加載原始特徵並找到最大長度
                for path in visual_paths:
                    feat = np.load(path)
                    raw_features.append(feat)
                    max_length = max(max_length, len(feat))
                
                # 對齊所有特徵長度（用於可視化）
                for i, feat in enumerate(raw_features):
                    if padding_mode == 'center':
                        # 使用與數據集相同的處理方式
                        feat = dataset._center_pad_features(feat, max_length)
                        original_features.append(feat)
                    else:
                        # 對於右側填充，需要手動對齊長度
                        padded = np.zeros(max_length)
                        padded[:len(feat)] = feat
                        original_features.append(padded)
                
                # 堆疊為二維數組
                features_array = np.vstack(original_features)
                
                # 可視化PCA結果
                visualize_pca_components(features_array, dataset.pca)

if __name__ == "__main__":
    main() 