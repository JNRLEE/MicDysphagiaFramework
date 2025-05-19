"""
訓練器模組
提供各種模型訓練和評估的實現：

1. 訓練器接口:
   - 標準化訓練流程
   - 支持回調機制

2. PyTorch訓練器:
   - 支持分類和回歸任務
   - 提供分批訓練、評估和預測功能
   - 整合TensorBoard監控
   - 自動保存檢查點和最佳模型

3. 訓練功能:
   - 支持多種學習率調度器
   - 支持早停機制
   - 支持梯度裁剪
   - 提供混合精度訓練

4. 評估功能:
   - 計算準確率、精確率、召回率、F1分數等評估指標
   - 繪製混淆矩陣和ROC曲線
   - 支持交叉驗證
"""

# 訓練器工廠
from .trainer_factory import create_trainer

# PyTorch訓練器
from .pytorch_trainer import PyTorchTrainer

__all__ = [
    # 訓練器工廠
    'create_trainer',
    
    # PyTorch訓練器
    'PyTorchTrainer'
] 