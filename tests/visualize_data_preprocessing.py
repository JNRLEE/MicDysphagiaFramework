"""
Data Preprocessing Visualization Test Script
Purpose: Visualize various stages of audio data preprocessing to check for issues in the conversion process
This script generates images for each stage of data preprocessing, allowing you to analyze spectrogram quality and converted RGB images
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from PIL import Image
import pandas as pd
from pathlib import Path
import tqdm
import random
from datetime import datetime

# Add project root directory to path for importing project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset_factory import create_dataset
from utils.data_adapter import DataAdapter
from utils.data_index_loader import DataIndexLoader

# Set output directory
OUTPUT_DIR = Path('tests/data_preprocessing_visualization')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_subdir = OUTPUT_DIR / f"test_{timestamp}"
output_subdir.mkdir(exist_ok=True)

# Set configuration parameters
config = {
    "index_path": "data/metadata/selection_groups/biscuit_20250516_230749.csv",
    "label_field": "DrLee_Evaluation",
    "filter_criteria": {},
    "audio": {
        "sample_rate": 16000,
        "duration": 5.0,
        "normalize": True
    },
    "features": {
        "method": "mel_spectrogram",
        "n_mels": 128,
        "n_fft": 1024,
        "hop_length": 512,
        "log_mel": True
    },
    "input_size": [224, 224]
}

def save_audio_waveform(audio, sr, filepath, title="Audio Waveform"):
    """Save audio waveform plot"""
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    
def save_spectrogram(spec, filepath, title="Spectrogram", aspect='auto'):
    """Save spectrogram plot"""
    plt.figure(figsize=(10, 6))
    
    if len(spec.shape) == 2:  # Single channel spectrogram
        plt.imshow(spec, aspect=aspect, origin='lower', cmap='viridis')
    else:  # Multi-channel spectrogram (possibly RGB)
        plt.imshow(np.transpose(spec, (1, 2, 0)), aspect=aspect)
    
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    
def save_rgb_channels(rgb_tensor, filepath_prefix):
    """Save RGB channel images separately"""
    if len(rgb_tensor.shape) == 3 and rgb_tensor.shape[0] == 3:
        for i, channel_name in enumerate(['R', 'G', 'B']):
            plt.figure(figsize=(8, 8))
            plt.imshow(rgb_tensor[i], cmap='viridis')
            plt.colorbar()
            plt.title(f'{channel_name} Channel')
            plt.tight_layout()
            plt.savefig(f"{filepath_prefix}_{channel_name}.png")
            plt.close()

def save_as_image(tensor, filepath):
    """Save tensor as image file"""
    # Normalize to 0-255 range
    if len(tensor.shape) == 2:  # Single channel
        img_array = np.uint8(255 * (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8))
        img = Image.fromarray(img_array)
        img.save(filepath)
    elif len(tensor.shape) == 3 and tensor.shape[0] == 3:  # Three channels (RGB)
        # Convert from (C, H, W) to (H, W, C)
        tensor = np.transpose(tensor, (1, 2, 0))
        img_array = np.uint8(255 * (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8))
        img = Image.fromarray(img_array)
        img.save(filepath)

def process_audio_sample(audio_path, sample_info, output_dir):
    """Process single audio sample and save visualization results for each stage"""
    # Create sample output directory
    sample_output = output_dir / f"sample_{Path(audio_path).stem}"
    sample_output.mkdir(exist_ok=True)
    
    # Save sample information
    with open(sample_output / "info.txt", "w") as f:
        f.write(f"Audio file: {audio_path}\n")
        for key, value in sample_info.items():
            f.write(f"{key}: {value}\n")
    
    # 1. Load original audio
    try:
        audio, sr = librosa.load(audio_path, sr=config["audio"]["sample_rate"], 
                             duration=config["audio"]["duration"])
        
        # Save audio waveform
        save_audio_waveform(audio, sr, sample_output / "1_waveform.png", 
                          title=f"Audio Waveform - {Path(audio_path).stem}")
        
        # 2. Generate spectrogram (single channel)
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_fft=config["features"]["n_fft"], 
            hop_length=config["features"]["hop_length"], 
            n_mels=config["features"]["n_mels"]
        )
        
        # Convert to logarithmic scale (dB)
        if config["features"]["log_mel"]:
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Save spectrogram image
        save_spectrogram(mel_spec, sample_output / "2_mel_spectrogram.png", 
                        title=f"Mel Spectrogram (Single Channel) - {Path(audio_path).stem}")
        save_as_image(mel_spec, sample_output / "2_mel_spectrogram_raw.png")
        
        # 3. Resize spectrogram to match model input (224x224)
        # Convert spectrogram to tensor and resize
        tensor_spec = torch.from_numpy(mel_spec).unsqueeze(0)  # Add channel dimension (1, H, W)
        
        # Use pytorch's interpolation function to resize
        resized_spec = torch.nn.functional.interpolate(
            tensor_spec.unsqueeze(0),  # Add batch dimension (1, 1, H, W)
            size=tuple(config["input_size"]),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0).numpy()  # Remove batch and channel dimensions, get (H, W)
        
        # Save resized spectrogram
        save_spectrogram(resized_spec, sample_output / "3_resized_spectrogram.png", 
                        title=f"Resized Spectrogram (224x224) - {Path(audio_path).stem}")
        save_as_image(resized_spec, sample_output / "3_resized_spectrogram_raw.png")
        
        # 4. Convert to 3-channel (RGB) input
        # Method 1: Directly copy same spectrogram to 3 channels
        rgb_spec_1 = np.stack([resized_spec] * 3, axis=0)
        save_spectrogram(rgb_spec_1, sample_output / "4a_rgb_spectrogram_same.png", 
                        title=f"RGB Spectrogram (Same Channel) - {Path(audio_path).stem}")
        save_rgb_channels(rgb_spec_1, str(sample_output / "4a_rgb_channel"))
        save_as_image(rgb_spec_1, sample_output / "4a_rgb_spectrogram_same_raw.png")
        
        # Method 2: Use actual project's DataAdapter for conversion
        # This is the method actually used in training
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()  # [1, audio_length]
        
        # Use DataAdapter to convert audio to spectrogram
        spec_config = config.get("features", {})
        
        # Use DataAdapter to convert audio to spectrogram
        model_dummy_config = {"data": {"preprocessing": {"spectrogram": spec_config}}}
        spectrograms = DataAdapter.convert_audio_to_spectrogram(audio_tensor, spec_config)
        
        # Ensure spectrogram is 3-channel (required by visual model)
        if spectrograms.size(1) == 1:
            spectrograms = spectrograms.repeat(1, 3, 1, 1)
            
        # Resize to match model input
        if (spectrograms.size(2) != config["input_size"][0] or
            spectrograms.size(3) != config["input_size"][1]):
            spectrograms = torch.nn.functional.interpolate(
                spectrograms,
                size=tuple(config["input_size"]),
                mode='bilinear',
                align_corners=False
            )
        
        rgb_numpy = spectrograms.squeeze(0).numpy()
        
        # Save RGB image after adapter conversion
        save_spectrogram(rgb_numpy, sample_output / "4b_rgb_spectrogram_adapter.png", 
                        title=f"RGB Spectrogram (Adapter Conversion) - {Path(audio_path).stem}")
        save_rgb_channels(rgb_numpy, str(sample_output / "4b_rgb_channel"))
        save_as_image(rgb_numpy, sample_output / "4b_rgb_spectrogram_adapter_raw.png")
        
        # Simulate actual training data flow
        batch = {"audio": audio_tensor, "label": sample_info.get("類別", "")}
        # Use DataAdapter for batch adaptation
        adapted_batch = DataAdapter.adapt_batch(batch, "swin_transformer", model_dummy_config)
        
        # Check if adapted batch contains spectrogram
        if "spectrogram" in adapted_batch and adapted_batch["spectrogram"] is not None:
            adapted_spec = adapted_batch["spectrogram"].squeeze(0).numpy()
            save_spectrogram(adapted_spec, sample_output / "5_adapted_spectrogram.png", 
                            title=f"Adapted Spectrogram - {Path(audio_path).stem}")
            save_rgb_channels(adapted_spec, str(sample_output / "5_adapted_channel"))
            save_as_image(adapted_spec, sample_output / "5_adapted_spectrogram_raw.png")
        
        # 5. Save spectrogram statistics
        stats = {
            "Original Audio": {
                "Shape": audio.shape,
                "Min": audio.min(),
                "Max": audio.max(),
                "Mean": audio.mean(),
                "Std": audio.std()
            },
            "Spectrogram (Single Channel)": {
                "Shape": mel_spec.shape,
                "Min": mel_spec.min(),
                "Max": mel_spec.max(),
                "Mean": mel_spec.mean(),
                "Std": mel_spec.std()
            },
            "Resized Spectrogram": {
                "Shape": resized_spec.shape,
                "Min": resized_spec.min(),
                "Max": resized_spec.max(),
                "Mean": resized_spec.mean(),
                "Std": resized_spec.std()
            },
            "RGB Spectrogram (Method 1)": {
                "Shape": rgb_spec_1.shape,
                "Min": rgb_spec_1.min(),
                "Max": rgb_spec_1.max(),
                "Mean": rgb_spec_1.mean(),
                "Std": rgb_spec_1.std()
            },
            "RGB Spectrogram (Real Training Flow)": {
                "Shape": rgb_numpy.shape,
                "Min": rgb_numpy.min(),
                "Max": rgb_numpy.max(),
                "Mean": rgb_numpy.mean(),
                "Std": rgb_numpy.std()
            }
        }
        
        with open(sample_output / "stats.txt", "w") as f:
            for stage, stage_stats in stats.items():
                f.write(f"\n{stage}:\n")
                for key, value in stage_stats.items():
                    f.write(f"  {key}: {value}\n")

        return True
        
    except Exception as e:
        print(f"Error processing sample {audio_path}: {e}")
        with open(sample_output / "error.txt", "w") as f:
            f.write(f"Processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function: Load dataset and process samples"""
    print(f"Starting data preprocessing visualization test, results will be saved in {output_subdir}")
    
    # Use DataIndexLoader class to read data
    try:
        # First try using standard DataIndexLoader class
        loader = DataIndexLoader(config["index_path"], verify_paths=False)
        df = loader.get_all_data()
    except Exception as e:
        print(f"Error reading data with DataIndexLoader class: {str(e)}")
        print("Switching to simple pandas direct reading...")
        # If failed, use pandas to read CSV directly
        df = pd.read_csv(config["index_path"])
    
    print(f"Index file contains {len(df)} records")
    
    # Check CSV file column names
    print(f"CSV file columns: {df.columns.tolist()}")
    
    # Check if audio path column exists
    path_column = None
    for candidate in ['wav_path', 'file_path', 'path']:
        if candidate in df.columns:
            path_column = candidate
            print(f"Found audio path column: '{path_column}'")
            break
    
    if path_column is None:
        print("Error: Cannot find audio path column in CSV file.")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Get samples from each class
    label_field = config["label_field"]
    
    if label_field in df.columns:
        labels = df[label_field].unique()
        print(f"Classes ({label_field}): {labels}")
        
        # Select some samples from each class
        samples_per_class = 3
        selected_samples = []
        
        for label in labels:
            class_samples = df[df[label_field] == label]
            if len(class_samples) > 0:
                # Randomly select samples
                selected = class_samples.sample(min(samples_per_class, len(class_samples)))
                selected_samples.append(selected)
        
        all_selected = pd.concat(selected_samples).reset_index(drop=True)
        print(f"Selected {len(all_selected)} samples for visualization")
        
        # Process each selected sample
        for _, row in all_selected.iterrows():
            audio_path = row[path_column]
            sample_info = {
                "類別": row[label_field],
                "患者ID": row.get('patient_id', 'unknown'),
                "序號": row.get('data_id', 'unknown'),
                "選擇": row.get('selection', 'unknown')
            }
            print(f"Processing sample: {audio_path} (Class: {sample_info['類別']})")
            process_audio_sample(audio_path, sample_info, output_subdir)
            
        print(f"Processing complete, all results saved in {output_subdir}")
    else:
        print(f"Error: Cannot find field '{label_field}' in index file")

if __name__ == "__main__":
    main() 