# Batch Analysis Summary Report

Date: 2025-05-01 10:06:40

Total experiments analyzed: 9

## Experiment Comparison

| Experiment | Model Type | Parameters | Epochs | Best Val Loss | Convergence | Overfitting | GNS Value | Recommended Batch Size |
|------------|------------|------------|--------|---------------|-------------|-------------|-----------|------------------------|
| audio_swin_regression_20250426_215039 | swin_transformer | 27618684 | 4 | 5339.5859375 | Underfitting (Training loss higher than validation loss) | No | N/A | N/A |
| audio_swin_classification_20250428_181427 | swin_transformer | 27626766 | N/A | N/A | N/A | No | N/A | N/A |
| audio_fcnn_regression_20250417_143054 | cnn | 25817857 | 12 | 33.244327545166016 | Underfitting (Training loss higher than validation loss) | Yes | N/A | N/A |
| audio_fcnn_regression_20250426_215615 | fcnn | 690945 | 11 | 36.11079406738281 | Underfitting (Training loss higher than validation loss) | Yes | 0.0 | 8 |
| audio_swin_classification_20250426_215346 | swin_transformer | 27626766 | 12 | 48.41275405883789 | Moderate | Yes | 0.0 | 8 |
| audio_swin_classification_20250427_210101 | swin_transformer | 27626766 | 12 | 48.41275405883789 | Moderate | Yes | 0.0 | 8 |
| audio_swin_classification_20250428_171600 | swin_transformer | 27626766 | 12 | 48.41275405883789 | Moderate | Yes | 0.0 | 8 |
| audio_swin_regression_20250417_142912 | swin_transformer | 27618684 | 4 | 5339.5859375 | Underfitting (Training loss higher than validation loss) | No | N/A | N/A |
| audio_swin_classification_20250427_183314 | swin_transformer | 27626766 | 12 | 48.41275405883789 | Moderate | Yes | 0.0 | 8 |

## Detailed Experiment Summaries

### audio_swin_regression_20250426_215039

**Configuration**
- Model Type: swin_transformer
- Data Type: audio
- Task Type: regression

**Model Structure**
- Total Parameters: 27618684
- Layer Count: 0

**Training Performance**
- Epochs Trained: 4
- Best Validation Loss: 5339.5859375
- Convergence Status: Underfitting (Training loss higher than validation loss)
- Overfitting: No

[Detailed Analysis Report](audio_swin_regression_20250426_215039/custom_analysis/analysis_report.md)

---

### audio_swin_classification_20250428_181427

**Configuration**
- Model Type: swin_transformer
- Data Type: audio
- Task Type: classification

**Model Structure**
- Total Parameters: 27626766
- Layer Count: 0

**Training Performance**
- Epochs Trained: N/A
- Best Validation Loss: N/A
- Convergence Status: N/A
- Overfitting: No

[Detailed Analysis Report](audio_swin_classification_20250428_181427/custom_analysis/analysis_report.md)

---

### audio_fcnn_regression_20250417_143054

**Configuration**
- Model Type: cnn
- Data Type: audio
- Task Type: regression

**Model Structure**
- Total Parameters: 25817857
- Layer Count: 0

**Training Performance**
- Epochs Trained: 12
- Best Validation Loss: 33.244327545166016
- Convergence Status: Underfitting (Training loss higher than validation loss)
- Overfitting: Yes (Epoch 3)

[Detailed Analysis Report](audio_fcnn_regression_20250417_143054/custom_analysis/analysis_report.md)

---

### audio_fcnn_regression_20250426_215615

**Configuration**
- Model Type: fcnn
- Data Type: audio
- Task Type: regression

**Model Structure**
- Total Parameters: 690945
- Layer Count: 0

**Training Performance**
- Epochs Trained: 11
- Best Validation Loss: 36.11079406738281
- Convergence Status: Underfitting (Training loss higher than validation loss)
- Overfitting: Yes (Epoch 2)

**GNS Analysis**
- Average GNS Value: 0.0
- GNS Trend: Stable
- Recommended Batch Size: 8

[Detailed Analysis Report](audio_fcnn_regression_20250426_215615/custom_analysis/analysis_report.md)

---

### audio_swin_classification_20250426_215346

**Configuration**
- Model Type: swin_transformer
- Data Type: audio
- Task Type: classification

**Model Structure**
- Total Parameters: 27626766
- Layer Count: 0

**Training Performance**
- Epochs Trained: 12
- Best Validation Loss: 48.41275405883789
- Convergence Status: Moderate
- Overfitting: Yes (Epoch 8)

**GNS Analysis**
- Average GNS Value: 0.0
- GNS Trend: Stable
- Recommended Batch Size: 8

[Detailed Analysis Report](audio_swin_classification_20250426_215346/custom_analysis/analysis_report.md)

---

### audio_swin_classification_20250427_210101

**Configuration**
- Model Type: swin_transformer
- Data Type: audio
- Task Type: classification

**Model Structure**
- Total Parameters: 27626766
- Layer Count: 0

**Training Performance**
- Epochs Trained: 12
- Best Validation Loss: 48.41275405883789
- Convergence Status: Moderate
- Overfitting: Yes (Epoch 8)

**GNS Analysis**
- Average GNS Value: 0.0
- GNS Trend: Stable
- Recommended Batch Size: 8

[Detailed Analysis Report](audio_swin_classification_20250427_210101/custom_analysis/analysis_report.md)

---

### audio_swin_classification_20250428_171600

**Configuration**
- Model Type: swin_transformer
- Data Type: audio
- Task Type: classification

**Model Structure**
- Total Parameters: 27626766
- Layer Count: 0

**Training Performance**
- Epochs Trained: 12
- Best Validation Loss: 48.41275405883789
- Convergence Status: Moderate
- Overfitting: Yes (Epoch 8)

**GNS Analysis**
- Average GNS Value: 0.0
- GNS Trend: Stable
- Recommended Batch Size: 8

[Detailed Analysis Report](audio_swin_classification_20250428_171600/custom_analysis/analysis_report.md)

---

### audio_swin_regression_20250417_142912

**Configuration**
- Model Type: swin_transformer
- Data Type: audio
- Task Type: regression

**Model Structure**
- Total Parameters: 27618684
- Layer Count: 0

**Training Performance**
- Epochs Trained: 4
- Best Validation Loss: 5339.5859375
- Convergence Status: Underfitting (Training loss higher than validation loss)
- Overfitting: No

[Detailed Analysis Report](audio_swin_regression_20250417_142912/custom_analysis/analysis_report.md)

---

### audio_swin_classification_20250427_183314

**Configuration**
- Model Type: swin_transformer
- Data Type: audio
- Task Type: classification

**Model Structure**
- Total Parameters: 27626766
- Layer Count: 0

**Training Performance**
- Epochs Trained: 12
- Best Validation Loss: 48.41275405883789
- Convergence Status: Moderate
- Overfitting: Yes (Epoch 8)

**GNS Analysis**
- Average GNS Value: 0.0
- GNS Trend: Stable
- Recommended Batch Size: 8

[Detailed Analysis Report](audio_swin_classification_20250427_183314/custom_analysis/analysis_report.md)

---

## Overall Recommendations

- **Best Performing Model**: audio_fcnn_regression_20250417_143054 with validation loss 33.244327545166016

**Batch Size Recommendations**:
- For audio_fcnn_regression_20250426_215615: Low GNS value detected. Consider decreasing batch size for faster convergence.
- For audio_swin_classification_20250426_215346: Low GNS value detected. Consider decreasing batch size for faster convergence.
- For audio_swin_classification_20250427_210101: Low GNS value detected. Consider decreasing batch size for faster convergence.
- For audio_swin_classification_20250428_171600: Low GNS value detected. Consider decreasing batch size for faster convergence.
- For audio_swin_classification_20250427_183314: Low GNS value detected. Consider decreasing batch size for faster convergence.

**Regularization Recommendations**:
- The following experiments show signs of overfitting and may benefit from regularization techniques:
  - audio_fcnn_regression_20250417_143054
  - audio_fcnn_regression_20250426_215615
  - audio_swin_classification_20250426_215346
  - audio_swin_classification_20250427_210101
  - audio_swin_classification_20250428_171600
  - audio_swin_classification_20250427_183314
- Consider using techniques like dropout, L2 regularization, or early stopping.
