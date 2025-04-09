"""
損失函數模組，提供各種損失函數和工廠類來支持深度學習模型的訓練。
包括標準PyTorch損失函數、排序損失函數和組合損失函數。
"""

from .loss_factory import LossFactory
from .ranking_losses import PairwiseRankingLoss, ListwiseRankingLoss, LambdaRankLoss
from .combined_losses import CombinedLoss, WeightedMSELoss, FocalLoss

# 導出所有可用的損失函數名稱，方便查詢
available_losses = LossFactory.list_available_losses() 