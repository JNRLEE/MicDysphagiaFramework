"""
此測試模組用於驗證損失函數的功能性。
通過生成模擬的模型輸出和標籤數據，測試損失函數的計算過程。
"""

import os
import sys
import json
import torch
import numpy as np
import random
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple

# 將項目根目錄添加到sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 導入損失函數
from losses import LossFactory, available_losses, PairwiseRankingLoss, ListwiseRankingLoss, LambdaRankLoss, CombinedLoss

# 設置隨機種子，確保可重現性
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class LossFunctionTester:
    """
    損失函數測試類，用於測試各種損失函數的功能。
    通過生成模擬的模型輸出和標籤數據，驗證損失函數的計算過程。
    """
    
    def __init__(self, batch_size: int = 16, num_batches: int = 5):
        """
        初始化損失函數測試器
        
        Args:
            batch_size: 批次大小
            num_batches: 測試的批次數量
        """
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 從model_data_bridging_report.json讀取模型輸出信息
        self.model_output_info = self._load_model_output_info()
        
        # 生成測試數據
        self.test_data = self._generate_test_data()
        
        # 測試結果存儲
        self.results = {}
    
    def _load_model_output_info(self) -> Dict[str, Any]:
        """
        加載模型數據橋接報告，用於模擬真實模型輸出
        
        Returns:
            模型輸出信息字典
        """
        try:
            # 讀取模型數據橋接報告
            report_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), 
                '../modelconstruct_test/model_data_bridging_report.json'
            ))
            with open(report_path, 'r') as f:
                report = json.load(f)
            return report
        except Exception as e:
            print(f"警告: 無法加載模型數據橋接報告: {e}")
            # 如果無法加載報告，則使用默認值
            return {
                "swin_transformer": {"output_shape": "(1,)", "output_value": 5.0},
                "fcnn": {"output_shape": "(1,)", "output_value": 3.2},
                "cnn": {"output_shape": "(1,)", "output_value": 4.1},
                "resnet": {"output_shape": "(1,)", "output_value": 2.8}
            }
    
    def _generate_test_data(self) -> List[Dict[str, torch.Tensor]]:
        """
        生成測試數據
        
        Returns:
            測試數據批次列表
        """
        test_data = []
        
        for _ in range(self.num_batches):
            # 生成隨機的模型預測值（這裡假設模型輸出為回歸值）
            predictions = torch.rand(self.batch_size, 1) * 10  # 0-10範圍內的值
            
            # 生成隨機的真實標籤（EAT-10得分）
            targets = torch.randint(0, 11, (self.batch_size, 1)).float()  # 0-10整數值
            
            # 生成隨機的患者ID
            patient_ids = [f"P{random.randint(100, 999)}" for _ in range(self.batch_size)]
            
            # 生成隨機的動作選擇
            selections = [random.choice(['swallow', 'drink', 'eat']) for _ in range(self.batch_size)]
            
            # 將所有數據轉移到設備
            predictions = predictions.to(self.device)
            targets = targets.to(self.device)
            
            test_data.append({
                'predictions': predictions,
                'targets': targets,
                'patient_ids': patient_ids,
                'selections': selections
            })
        
        return test_data
    
    def test_standard_losses(self) -> Dict[str, float]:
        """
        測試PyTorch標準損失函數
        
        Returns:
            測試結果字典
        """
        standard_losses = [
            "MSELoss", "L1Loss", "CrossEntropyLoss", "BCELoss", 
            "BCEWithLogitsLoss", "SmoothL1Loss", "HuberLoss"
        ]
        
        results = {}
        
        for loss_name in standard_losses:
            try:
                # 創建損失函數
                loss_config = {'type': loss_name, 'parameters': {}}
                
                # 對於需要特殊處理的損失函數，調整參數
                if loss_name == "CrossEntropyLoss":
                    # 對於交叉熵損失，需要整數標籤和多類別輸出
                    continue  # 這裡跳過，回歸問題不適合使用交叉熵
                
                if loss_name == "BCELoss":
                    # 對於二元交叉熵，需要將輸出和標籤歸一化到0-1範圍
                    loss_config['parameters'] = {'reduction': 'mean'}
                    # 對數據進行特殊處理
                    for batch in self.test_data:
                        batch['normalized_predictions'] = torch.sigmoid(batch['predictions'])
                        batch['normalized_targets'] = batch['targets'] / 10.0  # 將0-10的值歸一化到0-1範圍
                
                if loss_name == "BCEWithLogitsLoss":
                    # 對於帶logits的二元交叉熵，標籤需要歸一化到0-1範圍
                    for batch in self.test_data:
                        batch['normalized_targets'] = batch['targets'] / 10.0
                
                # 獲取損失函數
                loss_fn = LossFactory.get_loss(loss_config)
                
                # 計算每個批次的損失並取平均值
                total_loss = 0.0
                num_valid_batches = 0
                
                for batch in self.test_data:
                    try:
                        if loss_name == "BCELoss":
                            loss = loss_fn(batch['normalized_predictions'], batch['normalized_targets'])
                        elif loss_name == "BCEWithLogitsLoss":
                            loss = loss_fn(batch['predictions'], batch['normalized_targets'])
                        else:
                            loss = loss_fn(batch['predictions'], batch['targets'])
                        
                        total_loss += loss.item()
                        num_valid_batches += 1
                    except Exception as e:
                        print(f"  - 批次處理錯誤 ({loss_name}): {e}")
                
                if num_valid_batches > 0:
                    avg_loss = total_loss / num_valid_batches
                    results[loss_name] = avg_loss
                    print(f"  - {loss_name}: {avg_loss:.6f}")
                else:
                    print(f"  - {loss_name}: 無法計算損失")
            
            except Exception as e:
                print(f"  - {loss_name} 測試失敗: {e}")
        
        return results
    
    def test_ranking_losses(self) -> Dict[str, float]:
        """
        測試排序損失函數
        
        Returns:
            測試結果字典
        """
        ranking_losses = [
            {"name": "PairwiseRankingLoss", "config": {"margin": 0.3, "sampling_ratio": 0.5, "sampling_strategy": "score_diff"}},
            {"name": "PairwiseRankingLoss", "config": {"margin": 0.0, "sampling_ratio": 0.3, "sampling_strategy": "random"}},
            {"name": "PairwiseRankingLoss", "config": {"margin": 0.2, "sampling_ratio": 0.4, "sampling_strategy": "hard_negative", "use_exp": True}},
            {"name": "ListwiseRankingLoss", "config": {"method": "listnet", "temperature": 1.0, "group_size": 4}},
            {"name": "ListwiseRankingLoss", "config": {"method": "listmle", "temperature": 0.5, "stochastic": True}},
            {"name": "ListwiseRankingLoss", "config": {"method": "approxndcg", "k": 5}},
            {"name": "LambdaRankLoss", "config": {"sigma": 1.0, "k": 5, "sampling_ratio": 0.3}}
        ]
        
        results = {}
        
        for loss_info in ranking_losses:
            loss_name = loss_info["name"]
            config = loss_info["config"]
            
            try:
                # 創建損失函數
                loss_fn = None
                
                if loss_name == "PairwiseRankingLoss":
                    loss_fn = PairwiseRankingLoss(**config)
                elif loss_name == "ListwiseRankingLoss":
                    loss_fn = ListwiseRankingLoss(**config)
                elif loss_name == "LambdaRankLoss":
                    loss_fn = LambdaRankLoss(**config)
                
                if loss_fn is None:
                    print(f"  - {loss_name} 創建失敗")
                    continue
                
                # 計算每個批次的損失並取平均值
                total_loss = 0.0
                num_valid_batches = 0
                
                for batch in self.test_data:
                    try:
                        loss = loss_fn(batch['predictions'], batch['targets'])
                        total_loss += loss.item()
                        num_valid_batches += 1
                    except Exception as e:
                        print(f"  - 批次處理錯誤 ({loss_name}): {e}")
                
                if num_valid_batches > 0:
                    avg_loss = total_loss / num_valid_batches
                    loss_key = f"{loss_name}_{config.get('sampling_strategy', '')}" if loss_name == "PairwiseRankingLoss" else \
                              f"{loss_name}_{config.get('method', '')}" if loss_name == "ListwiseRankingLoss" else \
                              loss_name
                    results[loss_key] = avg_loss
                    print(f"  - {loss_key}: {avg_loss:.6f}")
                else:
                    print(f"  - {loss_name} ({config}): 無法計算損失")
            
            except Exception as e:
                print(f"  - {loss_name} ({config}) 測試失敗: {e}")
        
        return results
    
    def test_combined_loss(self) -> Dict[str, float]:
        """
        測試組合損失函數
        
        Returns:
            測試結果字典
        """
        # 創建一些基本損失函數
        mse_loss = LossFactory.get_loss({'type': 'MSELoss', 'parameters': {}})
        l1_loss = LossFactory.get_loss({'type': 'L1Loss', 'parameters': {}})
        pairwise_loss = PairwiseRankingLoss(margin=0.3, sampling_ratio=0.3, sampling_strategy='score_diff')
        listwise_loss = ListwiseRankingLoss(method='listnet', temperature=1.0)
        
        # 創建不同的組合
        combined_configs = [
            {
                "name": "MSE+L1",
                "losses": {"mse": mse_loss, "l1": l1_loss},
                "weights": {"mse": 0.7, "l1": 0.3}
            },
            {
                "name": "MSE+PairwiseRanking",
                "losses": {"mse": mse_loss, "pairwise": pairwise_loss},
                "weights": {"mse": 0.5, "pairwise": 0.5}
            },
            {
                "name": "MSE+ListwiseRanking",
                "losses": {"mse": mse_loss, "listwise": listwise_loss},
                "weights": {"mse": 0.5, "listwise": 0.5}
            },
            {
                "name": "MSE+L1+PairwiseRanking",
                "losses": {"mse": mse_loss, "l1": l1_loss, "pairwise": pairwise_loss},
                "weights": {"mse": 0.4, "l1": 0.2, "pairwise": 0.4}
            }
        ]
        
        results = {}
        
        for config in combined_configs:
            try:
                # 創建組合損失函數
                combined_loss = CombinedLoss(
                    losses=config["losses"],
                    weights=config["weights"]
                )
                
                # 計算每個批次的損失並取平均值
                total_loss = 0.0
                individual_losses = {name: 0.0 for name in config["losses"].keys()}
                num_valid_batches = 0
                
                for batch in self.test_data:
                    try:
                        # 計算組合損失
                        loss = combined_loss(batch['predictions'], batch['targets'])
                        total_loss += loss.item()
                        
                        # 獲取各個損失函數的值
                        indiv_losses = combined_loss.get_individual_losses(batch['predictions'], batch['targets'])
                        for name, value in indiv_losses.items():
                            individual_losses[name] += value
                        
                        num_valid_batches += 1
                    except Exception as e:
                        print(f"  - 批次處理錯誤 ({config['name']}): {e}")
                
                if num_valid_batches > 0:
                    avg_loss = total_loss / num_valid_batches
                    avg_individual_losses = {
                        name: value / num_valid_batches for name, value in individual_losses.items()
                    }
                    
                    results[config["name"]] = {
                        "combined": avg_loss,
                        "individual": avg_individual_losses
                    }
                    
                    print(f"  - {config['name']}: 組合損失 = {avg_loss:.6f}")
                    for name, value in avg_individual_losses.items():
                        print(f"    - {name}: {value:.6f}")
                else:
                    print(f"  - {config['name']}: 無法計算損失")
            
            except Exception as e:
                print(f"  - {config['name']} 測試失敗: {e}")
        
        return results
    
    def test_all_losses(self) -> Dict[str, Any]:
        """
        測試所有損失函數
        
        Returns:
            完整測試結果
        """
        print("=== 測試標準損失函數 ===")
        standard_results = self.test_standard_losses()
        
        print("\n=== 測試排序損失函數 ===")
        ranking_results = self.test_ranking_losses()
        
        print("\n=== 測試組合損失函數 ===")
        combined_results = self.test_combined_loss()
        
        # 組合所有結果
        all_results = {
            "standard_losses": standard_results,
            "ranking_losses": ranking_results,
            "combined_losses": combined_results
        }
        
        self.results = all_results
        return all_results
    
    def visualize_results(self) -> None:
        """
        可視化測試結果
        
        保存結果圖表
        """
        if not self.results:
            print("警告: 沒有測試結果可視化")
            return
        
        # 創建保存目錄
        output_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        # 當前時間戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 可視化標準損失函數結果
        if self.results.get("standard_losses"):
            plt.figure(figsize=(10, 6))
            losses = list(self.results["standard_losses"].keys())
            values = list(self.results["standard_losses"].values())
            
            plt.bar(losses, values)
            plt.title("標準損失函數測試結果")
            plt.xlabel("損失函數")
            plt.ylabel("損失值")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, f"standard_losses_{timestamp}.png"))
            plt.close()
        
        # 可視化排序損失函數結果
        if self.results.get("ranking_losses"):
            plt.figure(figsize=(12, 6))
            losses = list(self.results["ranking_losses"].keys())
            values = list(self.results["ranking_losses"].values())
            
            plt.bar(losses, values)
            plt.title("排序損失函數測試結果")
            plt.xlabel("損失函數")
            plt.ylabel("損失值")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, f"ranking_losses_{timestamp}.png"))
            plt.close()
        
        # 可視化組合損失函數結果
        if self.results.get("combined_losses"):
            plt.figure(figsize=(10, 8))
            losses = list(self.results["combined_losses"].keys())
            combined_values = [d["combined"] for d in self.results["combined_losses"].values()]
            
            plt.bar(losses, combined_values)
            plt.title("組合損失函數測試結果")
            plt.xlabel("損失函數組合")
            plt.ylabel("組合損失值")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, f"combined_losses_{timestamp}.png"))
            plt.close()
            
            # 可視化組合損失函數中的各個損失函數
            for loss_name, loss_data in self.results["combined_losses"].items():
                if "individual" in loss_data:
                    plt.figure(figsize=(8, 5))
                    indiv_losses = list(loss_data["individual"].keys())
                    indiv_values = list(loss_data["individual"].values())
                    
                    plt.bar(indiv_losses, indiv_values)
                    plt.title(f"{loss_name} - 各損失函數值")
                    plt.xlabel("損失函數")
                    plt.ylabel("損失值")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    plt.savefig(os.path.join(output_dir, f"{loss_name}_individual_{timestamp}.png"))
                    plt.close()
        
        print(f"測試結果圖表已保存到: {output_dir}")


def main():
    """
    主函數，運行損失函數測試
    """
    print("=== 開始測試損失函數 ===")
    print(f"可用的損失函數: {available_losses}")
    
    # 創建測試器
    tester = LossFunctionTester(batch_size=32, num_batches=3)
    
    # 運行測試
    all_results = tester.test_all_losses()
    
    # 可視化結果
    tester.visualize_results()
    
    # 將結果保存為JSON
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(os.path.join(output_dir, f"loss_test_results_{timestamp}.json"), 'w') as f:
        # 將張量轉換為可序列化的格式
        serializable_results = json.dumps(all_results, default=lambda o: float(o) if isinstance(o, (torch.Tensor, np.float32, np.float64)) else o, indent=2)
        f.write(serializable_results)
    
    print("\n=== 測試完成 ===")


if __name__ == "__main__":
    main() 