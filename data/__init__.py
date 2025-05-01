"""
音頻和特徵數據處理模組
功能：
1. 提供音頻、頻譜圖和特徵數據讀取功能
2. 支持多種數據格式和轉換方式
3. 支持按患者ID拆分數據集

數據讀取邏輯說明：
1. WAV音頻檔讀取：
   - 從患者資料夾中尋找名為"Probe0_RX_IN_TDM4CH0.wav"的音頻檔
   - 使用librosa.load加載音頻數據，根據配置進行重採樣和裁剪
   - 從資料夾中的info.json獲取患者ID、吞嚥動作類型和EAT-10分數

2. NPZ特徵檔讀取：
   - 尋找副檔名為.npz的特徵文件
   - 使用np.load讀取文件，支持直接讀取'features'鍵或從文件名推斷患者ID
   - 支持處理不同維度的特徵向量，自動截斷過大的特徵

3. 頻譜圖圖像讀取：
   - 從患者資料夾中尋找.png格式的頻譜圖文件
   - 使用PIL.Image讀取圖像並轉換為RGB格式
   - 應用配置的圖像轉換（調整大小、標準化等）

4. 患者信息(info.json)讀取：
   - 使用utils.patient_info_loader模組統一處理患者資料夾中的info.json檔案
   - 該模組自動尋找非"WavTokenizer_tokens_info.json"的JSON文件
   - 提取標準化的患者信息，包含患者ID、EAT-10分數和動作選擇
   - 支持按患者ID拆分數據集，確保相同患者的數據不會同時出現在訓練和測試集
"""

from .audio_dataset import AudioDataset
from .feature_dataset import FeatureDataset
from .spectrogram_dataset import SpectrogramDataset

__all__ = [
    'AudioDataset', 
    'FeatureDataset', 
    'SpectrogramDataset'
] 