#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TSNE視覺化工具：使用標籤信息為TSNE散點圖著色

此腳本讀取TSNE文件，並使用其中的標籤信息創建一個帶有標籤顏色的散點圖。
如果TSNE文件中包含標籤映射，則使用標籤映射作為圖例標題。
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse


def visualize_tsne(tsne_file_path, output_dir=None, title=None):
    """
    讀取TSNE文件並可視化結果，使用標籤信息為點著色
    
    Args:
        tsne_file_path: TSNE文件路徑
        output_dir: 輸出目錄，如果為None則顯示而不保存
        title: 圖表標題，如果為None則自動生成
    """
    # 檢查文件是否存在
    if not os.path.exists(tsne_file_path):
        print(f"錯誤：找不到文件 {tsne_file_path}")
        return False

    # 加載TSNE文件
    try:
        data = torch.load(tsne_file_path, weights_only=False)
        print(f"成功加載TSNE文件: {tsne_file_path}")
        print(f"文件包含以下鍵: {list(data.keys())}")
    except Exception as e:
        print(f"加載文件時出錯: {e}")
        return False

    # 檢查必要的鍵是否存在
    if 'tsne_coordinates' not in data:
        print("錯誤：文件不包含 'tsne_coordinates' 鍵")
        return False

    # 提取TSNE坐標
    tsne_coords = data['tsne_coordinates'].numpy()
    
    # 提取標籤信息
    targets = None
    if 'targets' in data:
        targets = data['targets'].numpy()
    
    # 提取標籤映射和標籤名稱
    label_mapping = data.get('label_mapping', None)
    label_names = data.get('label_names', None)
    label_field = data.get('label_field', "未知標籤")
    
    # 創建圖表
    plt.figure(figsize=(12, 10))
    
    # 決定如何為點著色
    if targets is not None:
        # 獲取唯一標籤
        unique_labels = np.unique(targets)
        n_labels = len(unique_labels)
        
        # 創建顏色映射
        cmap = plt.cm.get_cmap('tab10' if n_labels <= 10 else 'tab20', n_labels)
        
        # 為每個標籤繪製點
        for i, label in enumerate(unique_labels):
            mask = targets == label
            label_text = str(label)
            
            # 如果有標籤映射，使用映射後的標籤名稱
            if label_mapping is not None and int(label) in label_mapping:
                label_text = f"{label}: {label_mapping[int(label)]}"
            elif label_names is not None and i < len(label_names):
                label_text = f"{label}: {label_names[i]}"
                
            plt.scatter(
                tsne_coords[mask, 0], 
                tsne_coords[mask, 1],
                c=[cmap(i)],
                label=label_text,
                alpha=0.7,
                s=80
            )
    else:
        # 如果沒有標籤，所有點使用同一顏色
        plt.scatter(tsne_coords[:, 0], tsne_coords[:, 1], c='blue', alpha=0.5, s=50)
    
    # 添加圖例和標題
    if targets is not None and (label_mapping is not None or label_names is not None):
        plt.legend(title=f"{label_field}", fontsize=10, title_fontsize=12)
    
    # 設置標題
    if title is None:
        # 從文件中提取信息自動生成標題
        layer_name = data.get('layer_name', 'unknown_layer')
        epoch = data.get('epoch', 'unknown_epoch')
        dataset_name = data.get('dataset_name', 'unknown_dataset')
        title = f"t-SNE 可視化 - {layer_name} (Epoch {epoch}, {dataset_name})"
    
    plt.title(title, fontsize=14)
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 保存或顯示圖表
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        file_name = os.path.basename(tsne_file_path).replace('.pt', '.png')
        output_path = os.path.join(output_dir, file_name)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"圖表已保存到: {output_path}")
    else:
        plt.show()
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TSNE視覺化工具')
    parser.add_argument('tsne_file', help='TSNE文件路徑')
    parser.add_argument('--output', '-o', help='輸出目錄，默認為當前目錄下的 tsne_plots/', default='tsne_plots/')
    parser.add_argument('--title', '-t', help='圖表標題，默認自動生成')
    parser.add_argument('--display', '-d', action='store_true', help='顯示圖表而不保存')
    
    args = parser.parse_args()
    
    output_dir = None if args.display else args.output
    visualize_tsne(args.tsne_file, output_dir, args.title) 