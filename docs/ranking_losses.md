# 排序損失函數 (Ranking Loss Functions)

本文檔詳細介紹了框架中實現的各種排序損失函數，包括它們的數學原理、特點以及應用場景。

## 目錄

- [介紹](#介紹)
- [成對排序損失函數](#成對排序損失函數)
  - [邊界排序損失 (Margin Ranking Loss)](#邊界排序損失-margin-ranking-loss)
  - [RankNet 損失](#ranknet-損失)
- [列表排序損失函數](#列表排序損失函數)
  - [ListMLE 損失](#listmle-損失)
  - [近似 NDCG 損失](#近似-ndcg-損失)
  - [LambdaRank 損失](#lambdarank-損失)
- [組合排序損失函數](#組合排序損失函數)
- [損失函數選擇指南](#損失函數選擇指南)
- [實際應用示例](#實際應用示例)
- [參考文獻](#參考文獻)

## 介紹

排序學習 (Learning to Rank) 是機器學習的一個重要分支，特別適用於需要對項目進行排序的任務，例如吞嚥障礙評估的嚴重程度排序、搜索引擎結果排序等。排序學習的核心是設計合適的損失函數，使模型能夠學習到正確的排序關係。

排序損失函數通常可以分為三類：

1. **成對方法 (Pointwise Approach)**: 將排序問題轉化為分類或回歸問題，獨立預測每個項目的相關性分數。
2. **成對方法 (Pairwise Approach)**: 關注項目間的相對順序，通過優化項目對的正確排序關係。
3. **列表方法 (Listwise Approach)**: 直接優化整個排序列表的質量度量，如NDCG、MAP等。

## 成對排序損失函數

### 邊界排序損失 (Margin Ranking Loss)

邊界排序損失是一種成對損失函數，通過比較每對項目間的預測分數差異與真實排序關係，來優化模型。

**數學定義**:

對於一對項目 $i$ 和 $j$，以及它們的相對關係標籤 $y_{ij} \in \{-1, 1\}$，邊界排序損失定義為：

$$L(s_i, s_j, y_{ij}) = \max(0, -y_{ij} \cdot (s_i - s_j) + \text{margin})$$

其中 $s_i$ 和 $s_j$ 是模型對項目 $i$ 和 $j$ 的預測分數，$\text{margin}$ 是一個超參數，控制對錯誤排序的懲罰強度。

**特點**:
- 簡單直觀，計算效率高
- 只關注相對順序，不直接優化絕對分數
- 通過邊界參數可調整對錯誤排序的懲罰程度

**適用場景**:
- 當任務只關心項目間的相對排序而非絕對分數時
- 適合二元相關性場景 (如項目是否相關)
- 計算資源有限的環境

### RankNet 損失

RankNet 是一種基於概率的成對排序損失函數，由 Burges 等人在 2005 年提出。它使用 logistic 函數將分數差轉換為排序概率。

**數學定義**:

給定項目對 $(i, j)$ 及其預測分數 $s_i$ 和 $s_j$，模型預測 $i$ 應排在 $j$ 之前的概率為：

$$P_{ij} = \frac{1}{1 + e^{-\sigma(s_i - s_j)}}$$

其中 $\sigma$ 是一個縮放因子，控制概率曲線的陡峭程度。

真實的目標概率 $\bar{P}_{ij}$ 基於標籤 $y_{ij}$：

$$\bar{P}_{ij} = \frac{1 + y_{ij}}{2}$$

損失函數則是這兩個概率的交叉熵：

$$L(s_i, s_j, y_{ij}) = -\bar{P}_{ij} \log P_{ij} - (1 - \bar{P}_{ij}) \log (1 - P_{ij})$$

化簡後可得：

$$L(s_i, s_j, y_{ij}) = \log(1 + e^{-\sigma \cdot y_{ij} \cdot (s_i - s_j)})$$

**特點**:
- 使用概率框架，提供了平滑的梯度
- 對噪聲標籤較為穩健
- 是許多更先進排序算法的基礎

**適用場景**:
- 有噪聲或主觀標籤的排序任務
- 需要排序概率而非僅是排序結果的應用
- 吞嚥障礙評估中，不同評估者對同一患者的評分可能存在差異的情況

## 列表排序損失函數

### ListMLE 損失

List Minimum Likelihood Estimation (ListMLE) 是由 Xia 等人在 2008 年提出的一種列表排序損失函數，它基於 Plackett-Luce 模型，直接建模了整個排序列表的概率。

**數學定義**:

給定項目集合 $\{x_1, x_2, ..., x_n\}$ 和模型預測的分數 $\{s_1, s_2, ..., s_n\}$，以及真實排序 $\pi = (\pi(1), \pi(2), ..., \pi(n))$，ListMLE 損失定義為負對數似然：

$$L(s, \pi) = -\log P(\pi | s) = -\sum_{i=1}^n \log \frac{\exp(s_{\pi(i)})}{\sum_{j=i}^n \exp(s_{\pi(j)})}$$

這個損失函數鼓勵模型給排序靠前的項目更高的分數。

**特點**:
- 直接優化整個排序列表
- 考慮了位置信息
- 計算複雜度較低，為 $O(n\log n)$

**適用場景**:
- 需要直接優化整個排序列表質量的任務
- 排序位置很重要的應用，如吞嚥障礙嚴重程度評估
- 訓練數據包含完整排序的情況

### 近似 NDCG 損失

Normalized Discounted Cumulative Gain (NDCG) 是評估排序質量的常用指標，但它不可微。近似 NDCG 損失通過使用 softmax 函數近似排序過程，使得可以直接優化 NDCG。

**數學定義**:

NDCG@k 定義為：

$$\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}$$

其中 DCG@k 是：

$$\text{DCG@k} = \sum_{i=1}^k \frac{2^{r_{\pi(i)}} - 1}{\log_2(i+1)}$$

$r_{\pi(i)}$ 是排在位置 $i$ 的項目的相關性，IDCG@k 是理想情況下的 DCG@k。

近似 NDCG 損失使用 softmax 建模排序過程：

$$\hat{\pi}_i = \frac{\exp(s_i/\tau)}{\sum_j \exp(s_j/\tau)}$$

其中 $\tau$ 是溫度參數，控制近似的平滑程度。

損失函數為：

$$L(s, r) = 1 - \sum_{i=1}^k \frac{2^{r_i} - 1}{\log_2(i+1)} \cdot \hat{\pi}_i$$

**特點**:
- 直接優化 NDCG 評估指標
- 通過溫度參數可調整近似的平滑程度
- 考慮了位置和相關性分數的非線性增益

**適用場景**:
- NDCG 是評估指標的任務
- 項目相關性有多個等級的排序問題
- 排序位置非常重要的應用，如吞嚥障礙嚴重程度評估

### LambdaRank 損失

LambdaRank 是 RankNet 的改進版本，通過將排序評估指標（如 NDCG）的變化納入梯度計算，直接優化排序質量。

**數學定義**:

LambdaRank 不直接定義損失函數，而是定義梯度：

$$\lambda_{ij} = \frac{\partial L}{\partial s_i} = -\sigma \cdot \frac{|\Delta \text{NDCG}_{ij}|}{1 + e^{\sigma(s_i - s_j)}} \cdot y_{ij}$$

其中 $|\Delta \text{NDCG}_{ij}|$ 是交換項目 $i$ 和 $j$ 導致的 NDCG 變化絕對值。

**特點**:
- 直接優化排序評估指標
- 結合了成對和列表方法的優勢
- 對於提高排序質量非常有效

**適用場景**:
- 需要直接優化 NDCG 等排序評估指標的任務
- 有充足計算資源的環境
- 需要最高排序質量的應用

## 組合排序損失函數

組合排序損失結合了多種損失函數的優勢，通過加權方式平衡不同的優化目標。

**數學定義**:

給定多個損失函數 $L_1, L_2, ..., L_m$ 和對應的權重 $w_1, w_2, ..., w_m$，組合損失定義為：

$$L = \sum_{i=1}^m w_i \cdot L_i$$

當使用自適應權重時，權重可以動態調整：

$$w_i = \frac{\exp(-\alpha \cdot L_i)}{\sum_{j=1}^m \exp(-\alpha \cdot L_j)}$$

其中 $\alpha$ 是溫度參數。

**特點**:
- 結合多種損失函數的優勢
- 可以平衡多種優化目標
- 通過自適應權重機制自動調整各損失的重要性

**適用場景**:
- 複雜排序任務，需要兼顧多種評估指標
- 需要在不同階段關注不同優化目標的訓練過程
- 模型較為複雜，容易過擬合單一損失函數的情況

## 損失函數選擇指南

選擇合適的排序損失函數取決於多種因素：

1. **任務特性**:
   - 只關心相對順序？考慮成對損失函數
   - 需要優化特定評估指標？選擇對應的列表損失函數
   - 需要平衡多個目標？考慮組合損失函數

2. **數據特性**:
   - 有噪聲標籤？選擇對噪聲穩健的損失函數（如 RankNet）
   - 只有成對標籤？選擇成對損失函數
   - 有完整排序標籤？考慮列表損失函數

3. **計算資源**:
   - 資源有限？選擇計算開銷較小的損失函數（如邊界排序損失）
   - 資源充足？考慮更複雜但效果更好的損失函數（如 LambdaRank）

4. **模型大小和容量**:
   - 簡單模型？避免過於複雜的損失函數
   - 複雜模型？考慮更強力的正則化損失函數

## 實際應用示例

在吞嚥障礙嚴重程度評估中，排序損失函數可以用於以下場景：

1. **嚴重程度排序**:
   - 使用 ListMLE 或 ApproxNDCG 損失函數直接優化患者吞嚥障礙嚴重程度的排序
   - 適用於需要將患者按障礙嚴重程度排序的情況

2. **治療方案優先順序**:
   - 使用 LambdaRank 損失函數優化治療方案的優先順序
   - 確保最適合的治療方案排在前面

3. **多指標評估**:
   - 使用組合損失函數同時考慮多個評估指標
   - 平衡臨床相關性和統計顯著性

4. **專家一致性建模**:
   - 使用 RankNet 損失函數處理多專家評估結果
   - 建模專家間的一致性和分歧

## 參考文獻

1. Burges, C. J. (2010). From RankNet to LambdaRank to LambdaMART: An overview. Learning, 11(23-581), 81.

2. Cao, Z., Qin, T., Liu, T. Y., Tsai, M. F., & Li, H. (2007, June). Learning to rank: from pairwise approach to listwise approach. In Proceedings of the 24th international conference on Machine learning (pp. 129-136).

3. Xia, F., Liu, T. Y., Wang, J., Zhang, W., & Li, H. (2008, July). Listwise approach to learning to rank: theory and algorithm. In Proceedings of the 25th international conference on Machine learning (pp. 1192-1199).

4. Wang, J., Wang, S., Wang, L., & Chan, A. (2018). Heuristic ranking in constrained recommendation setting. arXiv preprint arXiv:1806.06372.

5. Bruch, S., Zoghi, M., Bendersky, M., & Najork, M. (2019, November). Revisiting approximate metric optimization in the age of deep neural networks. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 1241-1244). 