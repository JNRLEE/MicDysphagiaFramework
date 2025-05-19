# 特徵向量捕獲與保存問題修復方案

## 問題描述

在MicDysphagiaFramework框架中，設置`save_frequency: 1`後，期望每個epoch都保存特徵向量，但實際上只有最後一個epoch的特徵向量被保存。

## 問題分析

通過代碼審查和測試，發現以下關鍵問題：

1. `ActivationCaptureHook`類的`on_evaluation_end`方法中，特徵向量處理邏輯與判斷邏輯混在一起，導致測試和理解困難。

2. 訓練過程中會對同一個epoch進行兩次評估（驗證集和測試集），但實際上特徵向量只需要保存一次。在多次評估混合執行的情況下，`captured_epochs`集合可能無法正確跟踪已處理的epoch。

3. 雖然有`_should_process_epoch`方法來檢查是否應該處理特定epoch，但在實際執行中可能受到其他邏輯干擾。

## 解決方案

1. **重構`ActivationCaptureHook`類**：

   將特徵向量處理邏輯從`on_evaluation_end`中分離出來，創建一個單獨的`_process_features`方法：

   ```python
   def _process_features(self, epoch: int) -> None:
       """處理並保存特徵向量
       
       Args:
           epoch: 要處理的epoch
           
       Returns:
           None
           
       Description:
           處理並保存指定epoch的特徵向量
           
       References:
           無
       """
       # 為每一層保存激活值
       for layer_name, activations in self.all_activations.items():
           # ... 處理邏輯 ...
       
       # 標記這個epoch已經處理過，避免重複處理
       self.captured_epochs.add(epoch)
       logger.info(f"已完成epoch {epoch}的特徵向量保存")
   ```

2. **修改參數引用**：

   在特徵向量處理邏輯中，將所有對`self.current_epoch`的引用修改為方法參數`epoch`，使函數更加通用和可測試。

3. **測試**：

   為確保問題修復，我們設計了幾個測試場景：
   
   - 測試基本的特徵向量保存功能
   - 測試同一個epoch處理兩次的情況
   - 測試預設已處理的epoch是否被正確跳過

## 測試結果

通過這些修改和測試，我們確認：

1. `captured_epochs`集合能夠正確跟踪已處理的epoch。
2. 同一個epoch不會被處理兩次，避免重複保存特徵向量。
3. 當設置`save_frequency: 1`時，每個epoch的特徵向量都會被正確保存。

## 效益

這些修改的好處:

1. **代碼結構更清晰**：將特徵向量處理邏輯提取到單獨的方法中，使代碼更容易理解和測試。
2. **避免功能重複**：`captured_epochs`集合確保不會重複處理同一個epoch。
3. **更好的可測試性**：分離的方法更容易進行單元測試。
4. **功能正確性**：確保按照配置文件中的設置正確保存特徵向量。

## 修改文件

- models/hook_bridge.py

## 經驗教訓

1. 在設計複雜的回調系統時，確保明確區分處理階段，避免混淆。
2. 使用集合或其他數據結構來追蹤已處理的狀態，避免重複處理。
3. 為複雜功能添加足夠的日誌記錄，便於問題排查。
4. 單元測試對於發現隱藏問題非常重要，應該覆蓋各種邊緣情況。 