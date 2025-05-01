# 損失函數數學原理

本文檔深入解析MicDysphagiaFramework中實現的各類損失函數的數學原理，特別是排序學習損失函數的理論基礎。

## 標準損失函數

### 均方誤差損失 (MSELoss)

均方誤差(Mean Squared Error, MSE)是回歸問題中最常用的損失函數：

$$L_{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

其中：
- $n$ 是樣本數量
- $y_i$ 是第 $i$ 個樣本的真實值
- $\hat{y}_i$ 是模型對第 $i$ 個樣本的預測值

MSE對誤差的平方進行懲罰，對異常值特別敏感。對MSE求導得到的梯度與誤差成正比，使得較大的誤差獲得更強的修正信號。

### L1損失 (L1Loss)

L1損失計算預測值與真實值之間的絕對差：

$$L_{L1} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

相比MSE，L1損失對異常值不那麼敏感，但在誤差為零附近梯度變化較為劇烈，可能導致訓練不穩定。

### 平滑L1損失 (SmoothL1Loss)

SmoothL1Loss結合了MSE和L1Loss的優點，在誤差較小時表現如MSE，在誤差較大時表現如L1Loss：

$$L_{SmoothL1} = 
\begin{cases} 
0.5(y_i - \hat{y}_i)^2, & \text{if } |y_i - \hat{y}_i| < 1 \\
|y_i - \hat{y}_i| - 0.5, & \text{otherwise}
\end{cases}$$

這樣設計使得損失函數在誤差較小時有平滑的導數，在誤差較大時梯度不會過大。

### 交叉熵損失 (CrossEntropyLoss)

交叉熵損失用於多類別分類任務：

$$L_{CE} = -\sum_{i=1}^{n}\sum_{c=1}^{C}y_{i,c}\log(\hat{y}_{i,c})$$

其中：
- $C$ 是類別數
- $y_{i,c}$ 是指示第 $i$ 個樣本是否屬於類別 $c$ 的二元變量
- $\hat{y}_{i,c}$ 是模型預測第 $i$ 個樣本屬於類別 $c$ 的概率

### 二元交叉熵 (BCELoss)

二元交叉熵用於二分類問題：

$$L_{BCE} = -\frac{1}{n}\sum_{i=1}^{n}[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

其中 $\hat{y}_i$ 應該在 $(0,1)$ 範圍內，通常通過sigmoid函數實現。BCEWithLogitsLoss則是將sigmoid函數整合到損失計算中，提高數值穩定性。

## 排序學習損失函數

排序學習(Learning to Rank, LTR)在信息檢索和推薦系統中廣泛應用，主要關注項目間的相對順序，而非絕對分數。在吞嚥障礙評估中，準確排序患者的嚴重程度同樣重要。

### Pairwise排序學習

#### 理論基礎

Pairwise方法將排序問題轉換為分類問題，預測兩個項目之間的相對順序。假設 $x_i > x_j$ 表示項目 $i$ 應排在項目 $j$ 之前，則模型訓練目標是最大化以下概率：

$$P(x_i > x_j | y_i > y_j)$$

其中 $y_i$ 和 $y_j$ 是真實標籤值（如EAT-10評分）。

#### 1. PairwiseRankingLoss

我們實現的PairwiseRankingLoss基於邊際排序損失(Margin Ranking Loss)原理：

$$L_{pair} = \frac{1}{|P|}\sum_{(i,j) \in P}\max(0, -\text{sign}(y_i - y_j) \cdot (\hat{y}_i - \hat{y}_j) + \text{margin})$$

其中：
- $P$ 是所有可能的樣本對集合
- $\text{sign}(y_i - y_j)$ 指示真實排序方向
- $\hat{y}_i$ 和 $\hat{y}_j$ 是模型預測值
- margin是要求預測值之間的最小差距

若使用指數加權，損失函數變為：

$$L_{pair} = \frac{1}{|P|}\sum_{(i,j) \in P}\max(0, -\text{sign}(y_i - y_j) \cdot (\hat{y}_i - \hat{y}_j) + \text{margin}) \cdot \exp(|y_i - y_j|)$$

這樣對真實差距大的樣本對給予更高權重。

#### 採樣策略數學表述

1. **按分數差異採樣** (score_diff)：
   優先選擇真實分數差異大的樣本對，採樣概率與分數差成正比：
   $$P_{select}(i,j) \propto |y_i - y_j|$$

2. **隨機採樣** (random)：
   均勻隨機採樣，每對樣本被選中的概率相等：
   $$P_{select}(i,j) = \text{constant}$$

3. **硬負例採樣** (hard_negative)：
   優先選擇模型預測錯誤或接近的樣本對，採樣概率與預測難度成正比：
   $$P_{select}(i,j) \propto \max(0, -\text{sign}(y_i - y_j) \cdot (\hat{y}_i - \hat{y}_j) + \text{margin})$$

### Listwise排序學習

#### 理論基礎

Listwise方法直接優化整個排序列表，更接近真實評估指標（如NDCG, MAP等）。假設 $\pi$ 是一個排序，$P(\pi|x)$ 是給定特徵 $x$ 下 $\pi$ 的概率，則模型目標是最大化正確排序的概率。

#### 1. ListNet

ListNet基於排序概率分布的交叉熵：

$$L_{listnet} = -\sum_{\pi} P(\pi|y) \log P(\pi|\hat{y})$$

為簡化計算，通常使用Top-1排序概率：

$$P(i|\hat{y}) = \frac{\exp(\hat{y}_i/\tau)}{\sum_{j=1}^{n}\exp(\hat{y}_j/\tau)}$$

其中 $\tau$ 是溫度參數，控制概率分布的平滑程度。損失函數簡化為：

$$L_{listnet} = -\sum_{i=1}^{n} P(i|y) \log P(i|\hat{y})$$

#### 2. ListMLE

ListMLE(Maximum Likelihood Estimation)最大化正確排序的似然：

$$L_{listmle} = -\log P(\pi_y|\hat{y})$$

其中 $\pi_y$ 是按真實標籤排序的序列。對於一個排序 $\pi = [j_1, j_2, ..., j_n]$，其概率可表示為：

$$P(\pi|\hat{y}) = \prod_{k=1}^{n} \frac{\exp(\hat{y}_{j_k}/\tau)}{\sum_{l=k}^{n}\exp(\hat{y}_{j_l}/\tau)}$$

即連續選擇剩餘項目中概率最高的項目。

#### 3. ApproxNDCG

ApproxNDCG損失函數直接優化NDCG(Normalized Discounted Cumulative Gain)指標，NDCG定義為：

$$NDCG@k = \frac{DCG@k}{IDCG@k}$$

其中：
$$DCG@k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i+1)}$$

$rel_i$ 是排在第 $i$ 位的項目的相關性分數。問題是NDCG不可微，因此使用softmax函數創建一個平滑近似：

$$\hat{\pi}_i = \frac{\exp(\hat{y}_i/\tau)}{\sum_{j=1}^{n}\exp(\hat{y}_j/\tau)}$$

然後計算期望的DCG：

$$E[DCG@k] = \sum_{i=1}^{n}(2^{y_i} - 1)\sum_{j=1}^{k}\frac{\hat{\pi}_{i,j}}{\log_2(j+1)}$$

其中 $\hat{\pi}_{i,j}$ 是項目 $i$ 排在位置 $j$ 的概率。

### LambdaRank損失

LambdaRank是RankNet的擴展，不直接定義損失函數，而是定義梯度（lambda）：

$$\lambda_{ij} = \frac{\partial L}{\partial s_i} = \frac{\partial L}{\partial P_{ij}} \cdot \frac{\partial P_{ij}}{\partial s_i}$$

其中 $P_{ij}$ 是模型預測項目 $i$ 應排在項目 $j$ 之前的概率，通常使用以下定義：

$$P_{ij} = \frac{1}{1 + e^{-\sigma(s_i - s_j)}}$$

LambdaRank的創新在於將評估指標（如NDCG）的變化納入梯度計算：

$$\lambda_{ij} = \frac{-\sigma}{1 + e^{\sigma(s_i - s_j)}} \cdot |\Delta NDCG_{ij}|$$

其中 $\Delta NDCG_{ij}$ 是交換項目 $i$ 和 $j$ 導致的NDCG變化。這使得模型更關注對最終評估指標影響較大的樣本對。

## 組合損失函數

組合損失函數將多個損失函數加權組合：

$$L_{combined} = \sum_{i=1}^{m} w_i L_i$$

其中：
- $L_i$ 是第 $i$ 個損失函數
- $w_i$ 是第 $i$ 個損失函數的權重，且 $\sum_{i=1}^{m} w_i = 1$

### 自適應權重調整

自適應權重調整基於損失函數的相對變化，實時調整權重：

$$w_i^{(t+1)} = w_i^{(t)} \cdot (1 - r) + r \cdot \frac{L_i^{(t)}/L_i^{(0)}}{\sum_{j=1}^{m}L_j^{(t)}/L_j^{(0)}}$$

其中：
- $w_i^{(t)}$ 是第 $t$ 次迭代時第 $i$ 個損失函數的權重
- $L_i^{(t)}$ 是第 $t$ 次迭代時第 $i$ 個損失函數的值
- $L_i^{(0)}$ 是第 $i$ 個損失函數的初始值
- $r$ 是權重更新率

這種方法使得訓練過程更加平衡，避免某一損失函數主導整個優化過程。

## 數學視角下的損失函數選擇

### 回歸任務的最優損失函數

從統計學角度，最優損失函數取決於數據分布：
- 正態分布數據：MSELoss是極大似然估計器
- 拉普拉斯分布數據：L1Loss是極大似然估計器
- 混合數據或存在異常值：HuberLoss或SmoothL1Loss通常更穩健

### 排序任務的最優損失函數

排序任務的最優損失函數取決於評估指標和任務特性：
- 成對關係重要：PairwiseRankingLoss效果好
- 需要優化特定指標：LambdaRank或ApproxNDCG直接優化相應指標
- 整體排序一致性：ListMLE更關注全局排序

### 組合損失理論基礎

組合損失函數可以視為多目標優化問題，找到帕累托最優解(Pareto optimal solution)：

$$\min_{\theta} \{L_1(\theta), L_2(\theta), ..., L_m(\theta)\}$$

線性加權是最常用的多目標優化方法，但權重選擇至關重要。自適應權重調整可以看作一種動態多目標優化策略，在訓練過程中平衡各個目標。

## 實用案例分析

### 案例一：EAT-10評分預測

在EAT-10評分預測中，我們關注絕對評分準確性和相對排序：

$$L = 0.7 \cdot MSE + 0.3 \cdot PairwiseRanking$$

這種組合能同時優化分數精度和相對嚴重程度排序。

### 案例二：吞嚥障礙分類與排序

同時處理離散分類和連續排序：

$$L = 0.5 \cdot BCE + 0.5 \cdot ListwiseRanking$$

這種組合既考慮類別準確性，又考慮類別內排序一致性。

### 案例三：自適應權重的多任務學習

在多任務學習中，損失函數間尺度差異可能較大：

$$L = w_1 \cdot MSE + w_2 \cdot LambdaRank$$

使用自適應權重調整，可以根據每個任務的學習難度動態調整權重，促進更平衡的優化。

## 結論

損失函數設計是模型訓練的核心，特別是在吞嚥障礙評估等需要同時考慮多種目標的任務中。理解各類損失函數的數學原理，有助於選擇最適合任務特性的損失函數，或設計更有效的組合損失函數。通過數學分析和實驗驗證相結合，可以找到最優的損失函數配置，提高模型性能。 