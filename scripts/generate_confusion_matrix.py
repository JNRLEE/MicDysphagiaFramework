"""
生成混淆矩陣視覺化腳本
根據模型測試集的預測結果生成混淆矩陣，並保存為圖像文件
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import json
import argparse

def generate_confusion_matrix(result_dir, normalize=None):
    """
    根據測試結果生成混淆矩陣，並保存為PNG圖像
    
    Args:
        result_dir: 結果目錄路徑
        normalize: 混淆矩陣標準化方式，可為None, 'true', 'pred', 'all'
        
    Returns:
        分析目錄路徑
    """
    # 構建相關路徑
    hooks_dir = os.path.join(result_dir, 'hooks')
    test_pred_path = os.path.join(result_dir, 'test_predictions.pt')
    
    # 檢查測試預測文件是否存在
    if not os.path.exists(test_pred_path):
        test_pred_path = os.path.join(hooks_dir, 'evaluation_results_test.pt')
        if not os.path.exists(test_pred_path):
            print(f"無法找到測試預測文件: {test_pred_path}")
            return None
    
    # 創建分析目錄結構
    analysis_dir = os.path.join(result_dir, 'analysis')
    performance_dir = os.path.join(analysis_dir, 'performance')
    os.makedirs(performance_dir, exist_ok=True)
    
    # 加載測試結果
    test_data = torch.load(test_pred_path)
    y_true = test_data.get('targets', test_data.get('labels')).cpu().numpy()
    y_pred = test_data.get('predictions').cpu().numpy()
    
    # 嘗試獲取類別名稱
    class_names = None
    config_path = os.path.join(result_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            # 獲取標籤欄位名稱
            label_field = config.get('data', {}).get('label_field', 'DrLee_Evaluation')
            
            # 嘗試找到標籤映射
            if label_field == 'DrLee_Evaluation':
                class_names = ['聽起來正常', '輕度異常', '重度異常']
            elif label_field == 'DrTai_Evaluation':
                class_names = ['吞嚥障礙', '正常', '無OR 輕微吞嚥障礙', '重度吞嚥障礙']
    
    if class_names is None:
        # 默認使用數字作為類別名稱
        n_classes = max(max(y_true), max(y_pred)) + 1
        class_names = [str(i) for i in range(n_classes)]
    
    # 計算混淆矩陣
    cm = confusion_matrix(y_true, y_pred)
    
    # 創建一個新的圖
    plt.figure(figsize=(10, 8))
    
    # 如果需要標準化
    if normalize:
        if normalize == 'true':
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = '混淆矩陣 (按真實標籤標準化)'
            fmt = '.2f'
        elif normalize == 'pred':
            cm_normalized = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
            title = '混淆矩陣 (按預測標籤標準化)'
            fmt = '.2f'
        elif normalize == 'all':
            cm_normalized = cm.astype('float') / cm.sum()
            title = '混淆矩陣 (按所有樣本標準化)'
            fmt = '.2f'
        
        # 對於含有零的行，避免除以零
        cm_normalized = np.nan_to_num(cm_normalized)
        
        # 使用seaborn創建熱力圖
        sns.heatmap(cm_normalized, annot=True, fmt=fmt, cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
    else:
        title = '混淆矩陣'
        # 使用scikit-learn的ConfusionMatrixDisplay
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap='Blues', values_format='d')
        plt.title(title)
    
    plt.ylabel('真實標籤')
    plt.xlabel('預測標籤')
    plt.tight_layout()
    
    # 保存不同版本的混淆矩陣
    if normalize:
        output_path = os.path.join(performance_dir, f'confusion_matrix_test_normalized_by_{normalize}.png')
    else:
        output_path = os.path.join(performance_dir, 'confusion_matrix_test.png')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩陣已保存到: {output_path}")
    
    # 關閉圖形以釋放內存
    plt.close()
    
    return analysis_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='為模型測試結果生成混淆矩陣')
    parser.add_argument('--result_dir', type=str, required=True, help='實驗結果目錄路徑')
    parser.add_argument('--normalize', type=str, default=None, choices=[None, 'true', 'pred', 'all'], 
                        help='混淆矩陣標準化方式，可為None, true, pred, all')
    
    args = parser.parse_args()
    
    if args.normalize == 'None':
        args.normalize = None
    
    generate_confusion_matrix(args.result_dir, args.normalize) 