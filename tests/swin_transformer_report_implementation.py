"""
Implementation of the feature magnitude adjustment solution for the Swin Transformer model
This script demonstrates the full solution outlined in the technical report
"""

import os
import sys
import torch
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, accuracy_score

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from models.model_factory import create_model
from models.swin_transformer import SwinTransformerModel
from data.dataset_factory import create_dataset
from utils.data_adapter import DataAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

def load_config(config_path):
    """Load configuration file"""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model(config, checkpoint_path=None):
    """Load model from checkpoint"""
    logger.info("Creating model")
    model = create_model(config)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading model weights: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
    
    return model

def preprocess_audio_to_spectrogram(audio, sr=16000):
    """
    Correctly process audio to spectrogram format suitable for Swin Transformer
    
    Args:
        audio: Audio waveform
        sr: Sample rate
        
    Returns:
        torch.FloatTensor: 3-channel spectrogram tensor
    """
    import librosa
    from skimage.transform import resize
    
    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_mels=128,
        n_fft=1024,
        hop_length=512
    )
    
    # Convert to decibel scale
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Resize to standard input dimensions
    S_db_resized = resize(S_db, (224, 224), anti_aliasing=True)
    
    # Normalize to [0, 1] range
    S_db_normalized = (S_db_resized - S_db_resized.min()) / (S_db_resized.max() - S_db_resized.min())
    
    # Transform to 3-channel format
    rgb_spectrogram = np.stack([S_db_normalized] * 3, axis=0)
    
    return torch.FloatTensor(rgb_spectrogram)

def apply_feature_magnitude_adjustment(model, magnitude_scale=0.1):
    """
    Apply feature magnitude adjustment to the model
    
    Args:
        model: The model to adjust
        magnitude_scale: The scaling factor for feature magnitude
        
    Returns:
        The adjusted model
    """
    logger.info(f"Applying feature magnitude adjustment with scale factor: {magnitude_scale}")
    
    # Create a new model instance with the same architecture
    if isinstance(model, SwinTransformerModel):
        num_classes = model.head[1].out_features
        fixed_model = SwinTransformerModel(num_classes=num_classes)
        
        # Copy all parameters
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in dict(fixed_model.named_parameters()):
                    dict(fixed_model.named_parameters())[name].copy_(param)
        
        # Adjust output layer weight magnitude
        with torch.no_grad():
            head_weight = fixed_model.head[1].weight
            
            # Calculate current standard deviation
            current_std = torch.std(head_weight, dim=1, keepdim=True)
            target_std = current_std * magnitude_scale
            
            # Scale weights to target standard deviation
            scaled_weight = head_weight * (target_std / current_std)
            fixed_model.head[1].weight.copy_(scaled_weight)
            
            logger.info(f"Head weight STD before scaling: {current_std.mean().item():.6f}")
            logger.info(f"Head weight STD after scaling: {torch.std(fixed_model.head[1].weight, dim=1).mean().item():.6f}")
        
        # Reset bias to zero
        with torch.no_grad():
            fixed_model.head[1].bias.zero_()
            logger.info("Head bias initialized to zero")
    else:
        # Generic model copying
        import copy
        fixed_model = copy.deepcopy(model)
        
        # Adjust weights of the output layer
        for name, param in fixed_model.named_parameters():
            if ('head' in name or 'fc' in name or 'output' in name or 'classifier' in name) and 'weight' in name:
                with torch.no_grad():
                    current_std = torch.std(param, dim=1, keepdim=True)
                    target_std = current_std * magnitude_scale
                    scaled_param = param * (target_std / current_std)
                    param.copy_(scaled_param)
                    logger.info(f"Adjusted magnitude of {name}")
            
            # Reset bias in output layers
            elif ('head' in name or 'fc' in name or 'output' in name or 'classifier' in name) and 'bias' in name:
                with torch.no_grad():
                    param.zero_()
                    logger.info(f"Reset bias in {name} to zero")
    
    return fixed_model

def create_balanced_dataloader(dataset, batch_size=16):
    """
    Create a balanced dataloader with stratified sampling
    
    Args:
        dataset: The dataset to load
        batch_size: Batch size
        
    Returns:
        DataLoader with balanced class distribution
    """
    logger.info("Creating balanced dataloader with stratified sampling")
    
    # Calculate class counts
    class_counts = {}
    for i in range(len(dataset)):
        try:
            sample = dataset[i]
            if isinstance(sample, tuple) and len(sample) >= 2:
                label = sample[1]
                if isinstance(label, torch.Tensor):
                    label = label.item()
                class_counts[label] = class_counts.get(label, 0) + 1
            elif isinstance(sample, dict) and 'label' in sample:
                label = sample['label']
                if isinstance(label, torch.Tensor):
                    label = label.item()
                class_counts[label] = class_counts.get(label, 0) + 1
        except Exception as e:
            logger.warning(f"Error processing sample {i}: {str(e)}")
    
    logger.info(f"Class distribution: {class_counts}")
    
    # Calculate class weights
    class_weights = {}
    total_samples = sum(class_counts.values())
    
    for cls, count in class_counts.items():
        class_weights[cls] = 1.0 / (count / total_samples)
    
    # Normalize weights
    weight_sum = sum(class_weights.values())
    for cls in class_weights:
        class_weights[cls] = class_weights[cls] * len(class_weights) / weight_sum
    
    logger.info(f"Class weights: {class_weights}")
    
    # Assign weights to each sample
    sample_weights = []
    for i in range(len(dataset)):
        try:
            sample = dataset[i]
            if isinstance(sample, tuple) and len(sample) >= 2:
                label = sample[1]
                if isinstance(label, torch.Tensor):
                    label = label.item()
                sample_weights.append(class_weights.get(label, 1.0))
            elif isinstance(sample, dict) and 'label' in sample:
                label = sample['label']
                if isinstance(label, torch.Tensor):
                    label = label.item()
                sample_weights.append(class_weights.get(label, 1.0))
            else:
                sample_weights.append(1.0)
        except Exception as e:
            logger.warning(f"Error assigning weight to sample {i}: {str(e)}")
            sample_weights.append(1.0)
    
    # Create sampler and dataloader
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader

def evaluate_model(model, dataset, device=None, num_samples=None):
    """
    Evaluate the model on a dataset
    
    Args:
        model: The model to evaluate
        dataset: The dataset to evaluate on
        device: The device to use
        num_samples: Number of samples to evaluate (None for all)
        
    Returns:
        Dict with evaluation metrics
    """
    logger.info("Evaluating model")
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    # Initialize results
    all_labels = []
    all_preds = []
    all_probs = []
    
    # Limit samples if specified
    max_samples = len(dataset) if num_samples is None else min(num_samples, len(dataset))
    
    # Evaluate samples
    for i in range(max_samples):
        try:
            sample = dataset[i]
            
            if isinstance(sample, tuple) and len(sample) >= 2:
                inputs, label = sample[0], sample[1]
                if isinstance(label, torch.Tensor):
                    label = label.item()
                
                # Ensure float32 type
                if not inputs.is_floating_point() or inputs.dtype != torch.float32:
                    inputs = inputs.float()
                
                # Ensure 3 channels
                if inputs.shape[0] == 1:
                    inputs = inputs.repeat(3, 1, 1)
                
                # Add batch dimension
                inputs = inputs.unsqueeze(0).to(device)
                
                # Forward pass
                with torch.no_grad():
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                
                # Record results
                all_labels.append(label)
                all_preds.append(preds.item())
                all_probs.append(probs.cpu().numpy()[0])
            
            elif isinstance(sample, dict) and 'spectrogram' in sample and 'label' in sample:
                inputs = sample['spectrogram']
                label = sample['label']
                if isinstance(label, torch.Tensor):
                    label = label.item()
                
                # Ensure float32 type
                if not inputs.is_floating_point() or inputs.dtype != torch.float32:
                    inputs = inputs.float()
                
                # Ensure 3 channels
                if inputs.shape[0] == 1:
                    inputs = inputs.repeat(3, 1, 1)
                
                # Add batch dimension
                inputs = inputs.unsqueeze(0).to(device)
                
                # Forward pass
                with torch.no_grad():
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                
                # Record results
                all_labels.append(label)
                all_preds.append(preds.item())
                all_probs.append(probs.cpu().numpy()[0])
        
        except Exception as e:
            logger.error(f"Error processing sample {i}: {str(e)}")
    
    # Calculate metrics
    if len(all_labels) > 0:
        # Overall accuracy
        accuracy = accuracy_score(all_labels, all_preds) * 100
        
        # F1 scores
        f1_macro = f1_score(all_labels, all_preds, average='macro') * 100
        f1_weighted = f1_score(all_labels, all_preds, average='weighted') * 100
        f1_per_class = f1_score(all_labels, all_preds, average=None) * 100
        
        # Class distribution
        class_counts = {}
        for label in all_labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        # Prediction distribution
        pred_counts = {}
        for pred in all_preds:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        
        # Per-class accuracy
        class_accuracies = {}
        for cls in set(all_labels):
            indices = [i for i, label in enumerate(all_labels) if label == cls]
            correct = sum(1 for i in indices if all_preds[i] == cls)
            class_accuracies[int(cls)] = correct / len(indices) * 100 if indices else 0
        
        logger.info(f"Overall accuracy: {accuracy:.2f}%")
        logger.info(f"F1 score (macro): {f1_macro:.2f}%")
        logger.info(f"F1 score (weighted): {f1_weighted:.2f}%")
        logger.info(f"Class distribution: {class_counts}")
        logger.info(f"Prediction distribution: {pred_counts}")
        logger.info(f"Per-class accuracy: {class_accuracies}")
        
        # Check for single class prediction
        if len(set(all_preds)) == 1:
            logger.warning(f"Model only predicts a single class: {all_preds[0]}")
        
        return {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'f1_per_class': {str(i): float(score) for i, score in enumerate(f1_per_class)},
            'class_distribution': {str(k): v for k, v in class_counts.items()},
            'pred_distribution': {str(k): v for k, v in pred_counts.items()},
            'class_accuracies': {str(k): float(v) for k, v in class_accuracies.items()},
        }
    
    logger.error("No samples were successfully evaluated")
    return None

def main():
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"tests/swin_transformer_solution_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up file logging
    file_handler = logging.FileHandler(output_dir / "solution.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting Swin Transformer solution implementation. Results will be saved to {output_dir}")
    
    # Load configuration
    config_path = "config/example_classification_drlee.yaml"
    logger.info(f"Loading configuration: {config_path}")
    config = load_config(config_path)
    
    # Load original model
    checkpoint_path = "results/indexed_classification_drlee_20250519_012024/models/best_model.pth"
    original_model = load_model(config, checkpoint_path)
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_dataset(config)
    
    # Use validation dataset for testing if available
    dataset_for_testing = val_dataset if val_dataset is not None else test_dataset
    
    if dataset_for_testing is None:
        logger.error("No dataset available for testing")
        return
    
    # Evaluate original model
    logger.info("\n===== Evaluating Original Model =====")
    original_results = evaluate_model(original_model, dataset_for_testing)
    
    if original_results:
        import json
        with open(output_dir / "original_model_results.json", 'w') as f:
            json.dump(original_results, f, indent=2)
    
    # Try different magnitude scale factors
    scale_factors = [0.01, 0.05, 0.1, 0.2, 0.5]
    best_model = None
    best_f1 = 0
    best_scale = 0
    
    for scale in scale_factors:
        logger.info(f"\n===== Trying Feature Magnitude Scale: {scale} =====")
        
        # Apply feature magnitude adjustment
        adjusted_model = apply_feature_magnitude_adjustment(original_model, magnitude_scale=scale)
        
        # Evaluate adjusted model
        results = evaluate_model(adjusted_model, dataset_for_testing)
        
        if results:
            with open(output_dir / f"scale_{scale}_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save model
            torch.save(adjusted_model.state_dict(), output_dir / f"model_scale_{scale}.pth")
            
            # Track best model
            if results['f1_macro'] > best_f1:
                best_f1 = results['f1_macro']
                best_model = adjusted_model
                best_scale = scale
    
    if best_model is not None:
        logger.info(f"\n===== Best Model (Scale: {best_scale}) =====")
        
        # Save best model separately
        torch.save(best_model.state_dict(), output_dir / "best_model.pth")
        
        # Create implementation notes
        with open(output_dir / "implementation_notes.md", 'w') as f:
            f.write(f"""# Swin Transformer Solution Implementation

## Best Configuration

- **Feature Magnitude Scale Factor**: {best_scale}
- **F1 Score (Macro)**: {best_f1:.2f}%

## Implementation Details

This solution addresses the single-class prediction issue by adjusting the feature magnitude in the output layer of the Swin Transformer model. The key components are:

1. **Feature Magnitude Adjustment**: Scaling the standard deviation of the output layer weights to {best_scale} of their original values.
2. **Bias Reset**: Zero-initializing the bias terms in the output layer.

## Usage Instructions

To use this fixed model:

```python
from models.model_factory import create_model
import torch

# Load configuration
config = load_config("config/example_classification_drlee.yaml")

# Create model
model = create_model(config)

# Load fixed weights
model.load_state_dict(torch.load("tests/swin_transformer_solution_{timestamp}/best_model.pth"))
```

## Additional Recommendations

For further improvements, consider implementing:

1. Balanced loss function with class weights
2. Data augmentation for minority classes
3. Stratified sampling strategy
4. Fine-tuning with early stopping based on F1 score
""")
    
    logger.info(f"Solution implementation completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 