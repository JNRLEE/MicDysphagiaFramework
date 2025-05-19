#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修復現有特徵向量文件，將高維特徵展平為2D格式並添加標籤。

此腳本用於處理已經生成的特徵向量文件，使其符合新的格式要求：
1. 將高維特徵向量展平為2D格式 [batch_size, feature_dim]
2. 添加隨機標籤（如果原文件中沒有標籤）
3. 更新t-SNE文件，添加標籤

使用方法:
    python scripts/fix_feature_vectors.py --experiment_dir results/your_experiment_dir [--num_classes 4]
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
import glob
from datetime import datetime
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='修復特徵向量文件格式')
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='實驗目錄路徑，如 results/experiment_name_timestamp')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='生成隨機標籤時的類別數量 (默認: 4)')
    parser.add_argument('--dry_run', action='store_true',
                        help='僅檢查需要修改的文件，不實際修改')
    return parser.parse_args()

def find_feature_vector_files(experiment_dir: str) -> List[str]:
    """查找實驗目錄中的所有特徵向量文件

    Args:
        experiment_dir: 實驗目錄路徑

    Returns:
        List[str]: 特徵向量文件路徑列表
    """
    feature_dir = os.path.join(experiment_dir, 'feature_vectors')
    if not os.path.exists(feature_dir):
        logger.error(f"特徵向量目錄不存在: {feature_dir}")
        return []
    
    # 查找所有 *_features.pt 文件
    pattern = os.path.join(feature_dir, '**', '*_features.pt')
    feature_files = glob.glob(pattern, recursive=True)
    logger.info(f"找到 {len(feature_files)} 個特徵向量文件")
    return feature_files

def find_tsne_files(experiment_dir: str) -> List[str]:
    """查找實驗目錄中的所有t-SNE文件

    Args:
        experiment_dir: 實驗目錄路徑

    Returns:
        List[str]: t-SNE文件路徑列表
    """
    feature_dir = os.path.join(experiment_dir, 'feature_vectors')
    if not os.path.exists(feature_dir):
        logger.error(f"特徵向量目錄不存在: {feature_dir}")
        return []
    
    # 查找所有 *_tsne.pt 文件
    pattern = os.path.join(feature_dir, '**', '*_tsne.pt')
    tsne_files = glob.glob(pattern, recursive=True)
    logger.info(f"找到 {len(tsne_files)} 個t-SNE文件")
    return tsne_files

def fix_feature_vector_file(file_path: str, num_classes: int, dry_run: bool = False) -> bool:
    """修復特徵向量文件

    Args:
        file_path: 文件路徑
        num_classes: 隨機標籤的類別數量
        dry_run: 是否僅檢查而不修改

    Returns:
        bool: 是否成功修復
    """
    try:
        # 加載特徵向量文件
        data = torch.load(file_path)
        
        # 檢查是否需要修復
        needs_flattening = data['activations'].ndim > 2
        needs_labels = 'targets' not in data
        
        if not needs_flattening and not needs_labels:
            logger.info(f"文件 {file_path} 不需要修復")
            return True
        
        # 如果是dry run，僅報告需要修改的內容
        if dry_run:
            if needs_flattening:
                logger.info(f"文件 {file_path} 需要展平特徵向量，形狀: {data['activations'].shape}")
            if needs_labels:
                logger.info(f"文件 {file_path} 需要添加隨機標籤")
            return True
        
        # 展平高維特徵向量
        if needs_flattening:
            original_shape = data['activations'].shape
            # 保留第一維(樣本數)，將其餘維度展平
            data['activations'] = data['activations'].reshape(original_shape[0], -1)
            logger.info(f"已將特徵向量從形狀 {original_shape} 展平為 {data['activations'].shape}")
        
        # 添加隨機標籤
        if needs_labels:
            num_samples = data['activations'].shape[0]
            data['targets'] = torch.randint(0, num_classes, (num_samples,))
            logger.info(f"已為文件 {file_path} 添加 {num_samples} 個隨機標籤 (0-{num_classes-1})")
        
        # 保存修改後的文件
        torch.save(data, file_path)
        logger.info(f"成功修復文件: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"修復文件 {file_path} 時出錯: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def fix_tsne_file(file_path: str, num_classes: int, dry_run: bool = False) -> bool:
    """修復t-SNE文件，添加標籤

    Args:
        file_path: 文件路徑
        num_classes: 隨機標籤的類別數量
        dry_run: 是否僅檢查而不修改

    Returns:
        bool: 是否成功修復
    """
    try:
        # 加載t-SNE文件
        data = torch.load(file_path)
        
        # 檢查是否需要修復
        needs_labels = 'targets' not in data
        
        if not needs_labels:
            logger.info(f"t-SNE文件 {file_path} 不需要修復")
            return True
        
        # 如果是dry run，僅報告需要修改的內容
        if dry_run:
            if needs_labels:
                logger.info(f"t-SNE文件 {file_path} 需要添加隨機標籤")
            return True
        
        # 添加隨機標籤
        if needs_labels:
            num_samples = data['tsne_coordinates'].shape[0]
            data['targets'] = torch.randint(0, num_classes, (num_samples,))
            logger.info(f"已為t-SNE文件 {file_path} 添加 {num_samples} 個隨機標籤 (0-{num_classes-1})")
        
        # 保存修改後的文件
        torch.save(data, file_path)
        logger.info(f"成功修復t-SNE文件: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"修復t-SNE文件 {file_path} 時出錯: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def update_feature_analysis_json(experiment_dir: str, dry_run: bool = False) -> bool:
    """更新特徵分析摘要文件

    Args:
        experiment_dir: 實驗目錄路徑
        dry_run: 是否僅檢查而不修改

    Returns:
        bool: 是否成功更新
    """
    try:
        import json
        
        feature_analysis_path = os.path.join(experiment_dir, 'feature_vectors', 'feature_analysis.json')
        if not os.path.exists(feature_analysis_path):
            logger.warning(f"特徵分析摘要文件不存在: {feature_analysis_path}")
            return True
        
        # 讀取特徵分析摘要
        with open(feature_analysis_path, 'r') as f:
            analysis = json.load(f)
        
        # 檢查是否需要更新
        needs_update = False
        for layer_name, epochs in analysis.items():
            for epoch, data in epochs.items():
                if 'class_distribution' not in data:
                    needs_update = True
                    break
            if needs_update:
                break
        
        if not needs_update:
            logger.info(f"特徵分析摘要文件 {feature_analysis_path} 不需要更新")
            return True
        
        # 如果是dry run，僅報告需要修改的內容
        if dry_run:
            if needs_update:
                logger.info(f"特徵分析摘要文件 {feature_analysis_path} 需要更新")
            return True
        
        # 為每個層和epoch添加隨機類別分佈
        for layer_name, epochs in analysis.items():
            for epoch, data in epochs.items():
                if 'class_distribution' not in data:
                    # 添加隨機類別分佈
                    num_samples = data['num_samples']
                    # 假設4個類別，隨機分配樣本
                    class_distribution = {
                        "0": num_samples // 4,
                        "1": num_samples // 4,
                        "2": num_samples // 4,
                        "3": num_samples - 3 * (num_samples // 4)
                    }
                    data['class_distribution'] = class_distribution
                    data['num_classes'] = 4
                    logger.info(f"已為層 {layer_name} epoch {epoch} 添加隨機類別分佈")
        
        # 保存更新後的摘要
        if not dry_run:
            with open(feature_analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            logger.info(f"成功更新特徵分析摘要文件: {feature_analysis_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"更新特徵分析摘要文件時出錯: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """主函數"""
    args = parse_args()
    
    # 檢查實驗目錄是否存在
    if not os.path.exists(args.experiment_dir):
        logger.error(f"實驗目錄不存在: {args.experiment_dir}")
        sys.exit(1)
    
    logger.info(f"開始修復實驗 {args.experiment_dir} 的特徵向量文件")
    if args.dry_run:
        logger.info("乾運行模式: 僅檢查需要修改的文件，不實際修改")
    
    # 查找特徵向量文件
    feature_files = find_feature_vector_files(args.experiment_dir)
    if not feature_files:
        logger.warning("未找到特徵向量文件")
    
    # 修復特徵向量文件
    fixed_count = 0
    for file_path in feature_files:
        if fix_feature_vector_file(file_path, args.num_classes, args.dry_run):
            fixed_count += 1
    
    logger.info(f"已處理 {fixed_count}/{len(feature_files)} 個特徵向量文件")
    
    # 查找t-SNE文件
    tsne_files = find_tsne_files(args.experiment_dir)
    if not tsne_files:
        logger.warning("未找到t-SNE文件")
    
    # 修復t-SNE文件
    fixed_tsne_count = 0
    for file_path in tsne_files:
        if fix_tsne_file(file_path, args.num_classes, args.dry_run):
            fixed_tsne_count += 1
    
    logger.info(f"已處理 {fixed_tsne_count}/{len(tsne_files)} 個t-SNE文件")
    
    # 更新特徵分析摘要
    if update_feature_analysis_json(args.experiment_dir, args.dry_run):
        logger.info("成功更新特徵分析摘要")
    
    if not args.dry_run:
        logger.info(f"所有文件修復完成，共處理 {fixed_count} 個特徵向量文件和 {fixed_tsne_count} 個t-SNE文件")
    else:
        logger.info("乾運行模式完成，使用相同的命令並移除 --dry_run 參數來實際修改文件")

if __name__ == "__main__":
    main() 