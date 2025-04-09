"""
此腳本用於測試損失函數的運作。
生成模擬資料並傳遞給各種損失函數，確認其正常工作。
"""

import torch
import logging
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

# 引入loss模組
from losses import (
    LossFactory, 
    PairwiseRankingLoss, 
    ListwiseRankingLoss, 
    LambdaRankLoss, 
    CombinedLoss, 
    WeightedMSELoss, 
    FocalLoss
)

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_losses")

# 設定隨機種子，保證結果可複現
def set_seed(seed: int = 42):
    """設定隨機種子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_test_data(
    batch_size: int = 16, 
    score_range: Tuple[int, int] = (0, 40),
    device: str = "cpu"
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    生成測試資料，模擬模型輸出和對應的標籤
    
    Args:
        batch_size: 批次大小
        score_range: 分數範圍
        device: 設備
        
    Returns:
        模型預測值和標籤資料字典
    """
    # 生成隨機分數作為標籤
    scores = torch.randint(
        score_range[0], 
        score_range[1], 
        (batch_size,), 
        dtype=torch.float32, 
        device=device
    )
    
    # 生成預測值 (加入一些噪聲)
    predictions = scores + torch.randn(batch_size, device=device) * 5.0
    
    # 確保至少有一些預測是接近準確的
    for i in range(batch_size // 4):
        idx = random.randint(0, batch_size - 1)
        predictions[idx] = scores[idx] + torch.randn(1, device=device) * 0.5
    
    # 組織模型預測輸出
    model_output = predictions.view(batch_size, 1)
    
    # 組織標籤資料
    targets = {
        'score': scores,
        'patient_id': [f"P{i:03d}" for i in range(batch_size)],
        'selection': [random.choice(["a", "b", "c", "d"]) for _ in range(batch_size)]
    }
    
    return model_output, targets

def test_pytorch_standard_losses():
    """測試PyTorch標準損失函數"""
    logger.info("測試PyTorch標準損失函數...")
    
    batch_size = 16
    model_output, targets = generate_test_data(batch_size)
    
    # 測試MSELoss
    mse_loss_config = {'type': 'MSELoss', 'parameters': {}}
    mse_loss = LossFactory.get_loss(mse_loss_config)
    mse_result = mse_loss(model_output, targets['score'].unsqueeze(1))
    logger.info(f"MSELoss: {mse_result.item():.4f}")
    
    # 測試L1Loss
    l1_loss_config = {'type': 'L1Loss', 'parameters': {}}
    l1_loss = LossFactory.get_loss(l1_loss_config)
    l1_result = l1_loss(model_output, targets['score'].unsqueeze(1))
    logger.info(f"L1Loss: {l1_result.item():.4f}")
    
    # 測試SmoothL1Loss
    smooth_l1_loss_config = {'type': 'SmoothL1Loss', 'parameters': {}}
    smooth_l1_loss = LossFactory.get_loss(smooth_l1_loss_config)
    smooth_l1_result = smooth_l1_loss(model_output, targets['score'].unsqueeze(1))
    logger.info(f"SmoothL1Loss: {smooth_l1_result.item():.4f}")
    
    # 測試HuberLoss
    huber_loss_config = {'type': 'HuberLoss', 'parameters': {'delta': 1.0}}
    huber_loss = LossFactory.get_loss(huber_loss_config)
    huber_result = huber_loss(model_output, targets['score'].unsqueeze(1))
    logger.info(f"HuberLoss: {huber_result.item():.4f}")
    
    # 執行分類損失測試
    # 生成分類標籤
    class_labels = torch.randint(0, 5, (batch_size,))
    class_predictions = torch.randn(batch_size, 5)  # 分數預測
    
    # 測試CrossEntropyLoss
    ce_loss_config = {'type': 'CrossEntropyLoss', 'parameters': {}}
    ce_loss = LossFactory.get_loss(ce_loss_config)
    ce_result = ce_loss(class_predictions, class_labels)
    logger.info(f"CrossEntropyLoss: {ce_result.item():.4f}")
    
    # 測試BCEWithLogitsLoss
    binary_labels = torch.randint(0, 2, (batch_size, 1)).float()
    binary_predictions = torch.randn(batch_size, 1)
    
    bce_loss_config = {'type': 'BCEWithLogitsLoss', 'parameters': {}}
    bce_loss = LossFactory.get_loss(bce_loss_config)
    bce_result = bce_loss(binary_predictions, binary_labels)
    logger.info(f"BCEWithLogitsLoss: {bce_result.item():.4f}")
    
    # 所有損失函數應該都成功執行，並返回有效的損失值
    logger.info("PyTorch標準損失函數測試完成!")

def test_ranking_losses():
    """測試排序損失函數"""
    logger.info("測試排序損失函數...")
    
    # 生成有序分數數據
    batch_size = 16
    model_output, targets = generate_test_data(batch_size)
    
    # 測試PairwiseRankingLoss
    for strategy in ['random', 'score_diff', 'hard_negative']:
        for use_exp in [False, True]:
            logger.info(f"測試PairwiseRankingLoss (strategy={strategy}, use_exp={use_exp})...")
            pairwise_config = {
                'type': 'PairwiseRankingLoss', 
                'parameters': {
                    'margin': 1.0,
                    'sampling_ratio': 0.5,
                    'sampling_strategy': strategy,
                    'use_exp': use_exp
                }
            }
            pairwise_loss = LossFactory.get_loss(pairwise_config)
            pairwise_result = pairwise_loss(model_output, targets['score'].unsqueeze(1))
            logger.info(f"PairwiseRankingLoss ({strategy}, use_exp={use_exp}): {pairwise_result.item():.4f}")
    
    # 測試ListwiseRankingLoss
    for method in ['listnet', 'listmle', 'approxndcg']:
        logger.info(f"測試ListwiseRankingLoss (method={method})...")
        listwise_config = {
            'type': 'ListwiseRankingLoss', 
            'parameters': {
                'method': method,
                'temperature': 1.0,
                'k': 10,
                'group_size': 0,
                'stochastic': True
            }
        }
        listwise_loss = LossFactory.get_loss(listwise_config)
        listwise_result = listwise_loss(model_output, targets['score'].unsqueeze(1))
        logger.info(f"ListwiseRankingLoss ({method}): {listwise_result.item():.4f}")
    
    # 測試LambdaRankLoss
    logger.info("測試LambdaRankLoss...")
    lambdarank_config = {
        'type': 'LambdaRankLoss', 
        'parameters': {
            'sigma': 1.0,
            'k': 10,
            'sampling_ratio': 0.5
        }
    }
    lambdarank_loss = LossFactory.get_loss(lambdarank_config)
    lambdarank_result = lambdarank_loss(model_output, targets['score'].unsqueeze(1))
    logger.info(f"LambdaRankLoss: {lambdarank_result.item():.4f}")
    
    # 所有排序損失函數應該都成功執行，並返回有效的損失值
    logger.info("排序損失函數測試完成!")

def test_combined_losses():
    """測試組合損失函數"""
    logger.info("測試組合損失函數...")
    
    batch_size = 16
    model_output, targets = generate_test_data(batch_size)
    
    # 測試WeightedMSELoss
    weights = torch.ones(batch_size)
    # 強調某些樣本
    weights[0:batch_size//4] = 2.0
    
    weighted_mse_config = {
        'type': 'WeightedMSELoss', 
        'parameters': {'reduction': 'mean'}
    }
    weighted_mse_loss = LossFactory.get_loss(weighted_mse_config)
    weighted_mse_result = weighted_mse_loss(
        model_output, 
        targets['score'].unsqueeze(1), 
        weights=weights
    )
    logger.info(f"WeightedMSELoss: {weighted_mse_result.item():.4f}")
    
    # 測試FocalLoss (二分類情況)
    binary_labels = torch.randint(0, 2, (batch_size,))
    binary_predictions = torch.randn(batch_size, 1)
    
    focal_loss_config = {
        'type': 'FocalLoss', 
        'parameters': {
            'alpha': 0.25,
            'gamma': 2.0
        }
    }
    focal_loss = LossFactory.get_loss(focal_loss_config)
    focal_loss_result = focal_loss(binary_predictions, binary_labels)
    logger.info(f"FocalLoss (二分類): {focal_loss_result.item():.4f}")
    
    # 測試CombinedLoss
    # 創建多個損失函數的組合
    losses = {
        'mse': LossFactory.get_loss({'type': 'MSELoss', 'parameters': {}}),
        'pairwise': LossFactory.get_loss({
            'type': 'PairwiseRankingLoss', 
            'parameters': {'margin': 1.0}
        }),
        'listwise': LossFactory.get_loss({
            'type': 'ListwiseRankingLoss', 
            'parameters': {'method': 'listnet'}
        })
    }
    
    weights = {
        'mse': 1.0,
        'pairwise': 0.5,
        'listwise': 0.3
    }
    
    combined_loss = CombinedLoss(losses, weights)
    combined_result = combined_loss(model_output, targets['score'].unsqueeze(1))
    logger.info(f"CombinedLoss: {combined_result.item():.4f}")
    
    # 測試自適應權重
    adaptive_combined_loss = CombinedLoss(
        losses, 
        weights, 
        adaptive_weights=True,
        weight_update_freq=5,
        weight_update_ratio=0.2
    )
    
    # 模擬多次更新
    for i in range(10):
        adaptive_result = adaptive_combined_loss(model_output, targets['score'].unsqueeze(1))
        if i % 5 == 0:
            logger.info(f"自適應CombinedLoss (步驟 {i}): {adaptive_result.item():.4f}, 權重: {adaptive_combined_loss.weights}")
    
    # 從配置創建組合損失函數
    combined_config = {
        'combined': {
            'mse_loss': {
                'type': 'MSELoss',
                'parameters': {},
                'weight': 1.0
            },
            'ranking_loss': {
                'type': 'PairwiseRankingLoss',
                'parameters': {'margin': 0.5},
                'weight': 0.5
            }
        }
    }
    
    factory_combined_loss = LossFactory.create_from_config(combined_config)
    factory_result = factory_combined_loss(model_output, targets['score'].unsqueeze(1))
    logger.info(f"從配置創建的CombinedLoss: {factory_result.item():.4f}")
    
    # 測試獲取單獨的損失值
    individual_losses = factory_combined_loss.get_individual_losses(
        model_output, 
        targets['score'].unsqueeze(1)
    )
    logger.info(f"單獨的損失值: {individual_losses}")
    
    logger.info("組合損失函數測試完成!")

def test_loss_factory_create_from_config():
    """測試從配置創建損失函數"""
    logger.info("測試從配置創建損失函數...")
    
    # 單一損失函數配置
    mse_config = {
        'type': 'MSELoss',
        'parameters': {}
    }
    
    mse_loss = LossFactory.create_from_config(mse_config)
    assert isinstance(mse_loss, torch.nn.MSELoss)
    logger.info("成功從配置創建MSELoss")
    
    # 自定義損失函數配置
    pairwise_config = {
        'type': 'PairwiseRankingLoss',
        'parameters': {
            'margin': 1.0,
            'sampling_strategy': 'score_diff'
        }
    }
    
    pairwise_loss = LossFactory.create_from_config(pairwise_config)
    assert isinstance(pairwise_loss, PairwiseRankingLoss)
    logger.info("成功從配置創建PairwiseRankingLoss")
    
    # 組合損失函數配置
    combined_config = {
        'combined': {
            'mse': {
                'type': 'MSELoss',
                'parameters': {},
                'weight': 1.0
            },
            'l1': {
                'type': 'L1Loss',
                'parameters': {},
                'weight': 0.5
            }
        }
    }
    
    combined_loss = LossFactory.create_from_config(combined_config)
    assert isinstance(combined_loss, CombinedLoss)
    logger.info("成功從配置創建CombinedLoss")
    
    # 獲取可用損失函數列表
    available_losses = LossFactory.list_available_losses()
    logger.info(f"可用的損失函數: {available_losses}")
    
    logger.info("從配置創建損失函數測試完成!")

def main():
    """主函數"""
    set_seed(42)
    logger.info("開始測試損失函數...")
    
    # 測試PyTorch標準損失函數
    test_pytorch_standard_losses()
    
    # 測試排序損失函數
    test_ranking_losses()
    
    # 測試組合損失函數
    test_combined_losses()
    
    # 測試從配置創建損失函數
    test_loss_factory_create_from_config()
    
    logger.info("所有損失函數測試完成!")

if __name__ == "__main__":
    main() 