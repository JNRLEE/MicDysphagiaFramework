"""
損失函數模組
提供各種損失函數和工廠類來支持深度學習模型的訓練：

1. 標準損失函數:
   - MSELoss: 均方誤差損失
   - CrossEntropyLoss: 交叉熵損失
   - BCELoss: 二元交叉熵損失

2. 排序損失函數:
   - 成對方法(Pairwise):
     - PairwiseRankingLoss: 成對排序損失
   - 列表方法(Listwise):
     - ListwiseRankingLoss: 列表排序損失
     - LambdaRankLoss: LambdaRank損失

3. 組合損失函數:
   - CombinedLoss: 多種損失函數的線性組合

4. 特殊損失函數:
   - FocalLoss: 焦點損失，解決類別不平衡問題
   - WeightedMSELoss: 加權均方誤差損失
"""

# 損失工廠
from .loss_factory import create_loss_function as create_loss, LossFactory

# 排序損失函數
from .ranking_losses import (
    PairwiseRankingLoss, ListwiseRankingLoss, LambdaRankLoss
)

# 組合損失函數
from .combined_losses import (
    CombinedLoss, WeightedMSELoss, FocalLoss
)

# 可用損失函數列表
available_losses = LossFactory.list_available_losses()

__all__ = [
    # 損失工廠
    'create_loss',
    'LossFactory',
    'available_losses',
    
    # 排序損失函數 - 基類
    'PairwiseRankingLoss',
    'ListwiseRankingLoss',
    
    # 排序損失函數 - 列表方法
    'LambdaRankLoss',
    
    # 組合損失函數
    'CombinedLoss',
    
    # 特殊損失函數
    'WeightedMSELoss',
    'FocalLoss'
] 