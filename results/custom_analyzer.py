#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Custom analysis script for analyzing a single experiment directory.
This script analyzes:
1. Model structure analysis
2. Training history analysis
3. GNS (Gradient Noise Scale) analysis
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import torch
import seaborn as sns
import argparse
import re
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CustomAnalyzer:
    """Custom analyzer for analyzing experiment data"""
    
    def __init__(self, experiment_dir):
        """
        Initialize the analyzer
        
        Args:
            experiment_dir (str): Path to the experiment directory
        """
        self.experiment_dir = experiment_dir
        self.analysis_dir = os.path.join(experiment_dir, 'custom_analysis')
        Path(self.analysis_dir).mkdir(exist_ok=True)
        self.results = {}
    
    def load_config(self):
        """
        Load experiment configuration
        
        Returns:
            dict: Experiment configuration
        """
        config_path = os.path.join(self.experiment_dir, 'config.json')
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.results['config'] = config
            logger.info("Successfully loaded experiment configuration")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return None
    
    def load_model_structure(self):
        """
        Load model structure information
        
        Returns:
            dict: Model structure information
        """
        model_structure_path = os.path.join(self.experiment_dir, 'model_structure.json')
        try:
            with open(model_structure_path, 'r') as f:
                model_structure = json.load(f)
            self.results['model_structure'] = model_structure
            logger.info("Successfully loaded model structure information")
            return model_structure
        except Exception as e:
            logger.error(f"Error loading model structure: {e}")
            return None
    
    def load_training_history(self):
        """
        Load training history
        
        Returns:
            dict: Training history record
        """
        history_path = os.path.join(self.experiment_dir, 'training_history.json')
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            self.results['training_history'] = history
            logger.info("Successfully loaded training history")
            return history
        except Exception as e:
            logger.error(f"Error loading training history: {e}")
            return None
    
    def analyze_model_structure(self):
        """
        Analyze model structure
        
        Returns:
            dict: Model structure analysis results
        """
        model_structure = self.results.get('model_structure')
        if not model_structure:
            model_structure = self.load_model_structure()
            if not model_structure:
                logger.error("Cannot analyze model structure, failed to load model structure information")
                return None
        
        # Extract model statistics
        try:
            total_params = model_structure.get('total_parameters', 0)
            layer_info = model_structure.get('layer_info', [])
            
            # Calculate parameter ratio for each layer
            layer_param_ratio = {}
            for layer in layer_info:
                layer_name = layer.get('name', 'unknown')
                layer_params = layer.get('parameters', 0)
                if total_params > 0:
                    layer_param_ratio[layer_name] = layer_params / total_params * 100
            
            # Calculate parameter distribution statistics
            sorted_layers = sorted(layer_info, key=lambda x: x.get('parameters', 0), reverse=True)
            top_layers = sorted_layers[:5]  # Top 5 layers with most parameters
            
            # Generate parameter distribution plot
            plt.figure(figsize=(10, 6))
            x = [layer.get('name', 'unknown') for layer in top_layers]
            y = [layer.get('parameters', 0) for layer in top_layers]
            plt.bar(x, y)
            plt.title('Model Parameter Distribution (Top 5 Layers)')
            plt.xlabel('Layer Name')
            plt.ylabel('Parameter Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save image
            params_plot_path = os.path.join(self.analysis_dir, 'model_params_distribution.png')
            plt.savefig(params_plot_path)
            plt.close()
            
            analysis_result = {
                'total_parameters': total_params,
                'layer_count': len(layer_info),
                'layer_param_ratio': layer_param_ratio,
                'top_heavy_layers': [layer.get('name') for layer in top_layers],
                'params_distribution_plot': params_plot_path
            }
            
            self.results['model_structure_analysis'] = analysis_result
            logger.info("Completed model structure analysis")
            return analysis_result
        
        except Exception as e:
            logger.error(f"Error analyzing model structure: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def analyze_training_history(self):
        """
        Analyze training history
        
        Returns:
            dict: Training history analysis results
        """
        history = self.results.get('training_history')
        if not history:
            history = self.load_training_history()
            if not history:
                logger.error("Cannot analyze training history, failed to load training history")
                return None
        
        try:
            # Get training and validation losses
            train_loss = history.get('train_loss', [])
            val_loss = history.get('val_loss', [])
            best_val_loss = history.get('best_val_loss', float('inf'))
            training_time = history.get('training_time', 0)
            
            # Calculate training metrics
            if train_loss and val_loss:
                epochs = list(range(len(train_loss)))
                final_train_loss = train_loss[-1] if train_loss else None
                final_val_loss = val_loss[-1] if val_loss else None
                
                # Calculate average convergence speed (loss decrease per epoch)
                train_loss_diffs = [train_loss[i] - train_loss[i+1] for i in range(len(train_loss)-1) if train_loss[i] > train_loss[i+1]]
                avg_convergence_speed = sum(train_loss_diffs) / len(train_loss_diffs) if train_loss_diffs else 0
                
                # Detect overfitting
                overfitting_detected = False
                overfitting_epoch = None
                
                # If validation loss starts increasing while training loss continues to decrease, overfitting may be occurring
                for i in range(1, min(len(train_loss), len(val_loss))):
                    if train_loss[i] < train_loss[i-1] and val_loss[i] > val_loss[i-1]:
                        consecutive_overfit = 0
                        for j in range(i, min(len(train_loss), len(val_loss))):
                            if train_loss[j] < train_loss[j-1] and val_loss[j] > val_loss[j-1]:
                                consecutive_overfit += 1
                            else:
                                consecutive_overfit = 0
                            
                            if consecutive_overfit >= 2:  # If this pattern occurs for 2 consecutive epochs, consider it overfitting
                                overfitting_detected = True
                                overfitting_epoch = j - 1
                                break
                        
                        if overfitting_detected:
                            break
                
                # Plot loss curves
                plt.figure(figsize=(10, 6))
                plt.plot(epochs, train_loss, 'b-', label='Training Loss')
                plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
                if overfitting_detected and overfitting_epoch is not None:
                    plt.axvline(x=overfitting_epoch, color='g', linestyle='--', label=f'Overfitting Detected (Epoch {overfitting_epoch})')
                plt.title('Training and Validation Loss Curves')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                
                # Save image
                loss_plot_path = os.path.join(self.analysis_dir, 'training_loss_curve.png')
                plt.savefig(loss_plot_path)
                plt.close()
                
                # Calculate training-validation loss difference
                train_val_diff = [abs(train_loss[i] - val_loss[i]) for i in range(min(len(train_loss), len(val_loss)))]
                avg_train_val_diff = sum(train_val_diff) / len(train_val_diff) if train_val_diff else 0
                
                # Determine convergence status
                if final_train_loss is not None and final_val_loss is not None:
                    if abs(final_train_loss - final_val_loss) < 0.1 * final_val_loss:
                        convergence_status = "Good (Training and validation losses are close)"
                    elif final_train_loss < 0.5 * final_val_loss:
                        convergence_status = "Overfitting (Training loss much lower than validation loss)"
                    elif final_train_loss > final_val_loss * 1.2:
                        convergence_status = "Underfitting (Training loss higher than validation loss)"
                    else:
                        convergence_status = "Moderate"
                else:
                    convergence_status = "Unknown (Missing final loss values)"
            
                analysis_result = {
                    'epochs_trained': len(train_loss),
                    'final_train_loss': final_train_loss,
                    'final_val_loss': final_val_loss,
                    'best_val_loss': best_val_loss,
                    'training_time': training_time,
                    'avg_convergence_speed': avg_convergence_speed,
                    'avg_train_val_diff': avg_train_val_diff,
                    'overfitting_detected': overfitting_detected,
                    'overfitting_epoch': overfitting_epoch,
                    'convergence_status': convergence_status,
                    'loss_curve_plot': loss_plot_path
                }
                
                self.results['training_history_analysis'] = analysis_result
                logger.info("Completed training history analysis")
                return analysis_result
            else:
                logger.error("Training history missing loss records")
                return None
        
        except Exception as e:
            logger.error(f"Error analyzing training history: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def analyze_gns(self):
        """
        Analyze GNS (Gradient Noise Scale)
        
        Returns:
            dict: GNS analysis results
        """
        logger.info("Starting GNS analysis...")
        # Find all GNS data files
        gns_files = []
        hooks_dir = os.path.join(self.experiment_dir, 'hooks')
        
        try:
            if os.path.exists(hooks_dir):
                logger.info(f"Scanning hooks directory: {hooks_dir}")
                for epoch_dir in os.listdir(hooks_dir):
                    epoch_path = os.path.join(hooks_dir, epoch_dir)
                    logger.info(f"Checking epoch directory: {epoch_path}")
                    if os.path.isdir(epoch_path) and epoch_dir.startswith('epoch_'):
                        for file in os.listdir(epoch_path):
                            if file.startswith('gns_stats_epoch_'):
                                gns_file_path = os.path.join(epoch_path, file)
                                logger.info(f"Found GNS data file: {gns_file_path}")
                                gns_files.append(gns_file_path)
            
            if not gns_files:
                logger.warning("No GNS data files found")
                return None
            
            # Load GNS data
            gns_data = []
            for file_path in gns_files:
                try:
                    logger.info(f"Attempting to read GNS data file: {file_path}")
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            gns_data.append(data)
                            logger.info(f"Successfully loaded GNS data from {file_path}")
                    except PermissionError:
                        logger.warning(f"Permission denied when trying to read {file_path}. Skipping.")
                        continue
                except Exception as e:
                    logger.error(f"Cannot read GNS data file {file_path}: {e}")
            
            if not gns_data:
                logger.error("All GNS data files failed to load")
                return None
            
            # Extract GNS statistics
            logger.info("Extracting GNS statistics...")
            epochs = [data.get('epoch') for data in gns_data]
            gns_values = [data.get('gns', 0) for data in gns_data]
            total_vars = [data.get('total_var', 0) for data in gns_data]
            mean_norm_sqs = [data.get('mean_norm_sq', 0) for data in gns_data]
            
            # Calculate GNS average and trend
            avg_gns = sum(gns_values) / len(gns_values) if gns_values else 0
            
            # GNS trend analysis
            gns_trend = "Increasing" if len(gns_values) > 1 and gns_values[-1] > gns_values[0] else "Decreasing" if len(gns_values) > 1 and gns_values[-1] < gns_values[0] else "Stable"
            
            # Plot GNS trend
            logger.info("Generating GNS trend plot...")
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, gns_values, 'g-o', label='GNS Value')
            plt.title('GNS (Gradient Noise Scale) Trend')
            plt.xlabel('Epochs')
            plt.ylabel('GNS Value')
            plt.grid(True)
            plt.legend()
            
            # Save image
            gns_plot_path = os.path.join(self.analysis_dir, 'gns_trend.png')
            plt.savefig(gns_plot_path)
            plt.close()
            logger.info(f"GNS trend plot saved to {gns_plot_path}")
            
            # Batch size recommendation
            # Higher GNS values suggest higher gradient noise, recommend increasing batch size
            # Lower GNS values suggest lower gradient noise, can consider reducing batch size for faster convergence
            current_batch_size = self.results.get('config', {}).get('data', {}).get('dataloader', {}).get('batch_size', 16)
            
            if avg_gns > 10:
                recommended_batch_size = current_batch_size * 2
                recommendation = "High GNS value detected. Consider increasing batch size to reduce gradient noise."
            elif avg_gns < 0.1:
                recommended_batch_size = max(current_batch_size // 2, 1)
                recommendation = "Low GNS value detected. Consider decreasing batch size for faster convergence."
            else:
                recommended_batch_size = current_batch_size
                recommendation = "GNS value is moderate. Current batch size appears appropriate."
            
            # Plot gradient statistics
            logger.info("Generating gradient statistics plot...")
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, total_vars, 'b-o', label='Total Gradient Variance')
            plt.plot(epochs, mean_norm_sqs, 'r-o', label='Mean Squared Norm')
            plt.title('Gradient Statistics Trend')
            plt.xlabel('Epochs')
            plt.ylabel('Value (Log Scale)')
            plt.yscale('log')
            plt.grid(True)
            plt.legend()
            
            # Save image
            grad_stats_plot_path = os.path.join(self.analysis_dir, 'gradient_stats.png')
            plt.savefig(grad_stats_plot_path)
            plt.close()
            logger.info(f"Gradient statistics plot saved to {grad_stats_plot_path}")
            
            analysis_result = {
                'epochs_analyzed': epochs,
                'gns_values': gns_values,
                'average_gns': avg_gns,
                'gns_trend': gns_trend,
                'total_variances': total_vars,
                'mean_norm_squares': mean_norm_sqs,
                'current_batch_size': current_batch_size,
                'recommended_batch_size': recommended_batch_size,
                'recommendation': recommendation,
                'gns_plot': gns_plot_path,
                'grad_stats_plot': grad_stats_plot_path
            }
            
            self.results['gns_analysis'] = analysis_result
            logger.info("Completed GNS analysis")
            return analysis_result
        
        except Exception as e:
            logger.error(f"Error analyzing GNS: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def generate_report(self):
        """
        Generate analysis report
        
        Returns:
            str: Report file path
        """
        report_path = os.path.join(self.analysis_dir, 'analysis_report.md')
        
        try:
            # Get analysis results
            config = self.results.get('config', {})
            model_structure_analysis = self.results.get('model_structure_analysis', {})
            training_history_analysis = self.results.get('training_history_analysis', {})
            gns_analysis = self.results.get('gns_analysis', {})
            
            # Format report content
            report_content = f"""# Audio Classification Model Analysis Report

## 1. Experiment Overview

- **Experiment Name:** {config.get('global', {}).get('experiment_name', 'Unknown')}
- **Model Type:** {config.get('model', {}).get('type', 'Unknown')}
- **Data Type:** {config.get('data', {}).get('type', 'Unknown')}
- **Task Type:** {config.get('data', {}).get('filtering', {}).get('task_type', 'Unknown')}

## 2. Model Structure Analysis

- **Total Parameters:** {model_structure_analysis.get('total_parameters', 'N/A')}
- **Layer Count:** {model_structure_analysis.get('layer_count', 'N/A')}
- **Layers with Most Parameters:**
"""
            
            # Add top layers by parameter count
            top_layers = model_structure_analysis.get('top_heavy_layers', [])
            for layer in top_layers:
                report_content += f"  - {layer}\n"
            
            # Add parameter distribution plot
            params_plot = model_structure_analysis.get('params_distribution_plot')
            if params_plot:
                rel_path = os.path.relpath(params_plot, self.analysis_dir)
                report_content += f"\n![Model Parameter Distribution]({rel_path})\n"
            
            # Add training history analysis
            report_content += f"""
## 3. Training History Analysis

- **Epochs Trained:** {training_history_analysis.get('epochs_trained', 'N/A')}
- **Training Time:** {training_history_analysis.get('training_time', 'N/A')} seconds
- **Final Training Loss:** {training_history_analysis.get('final_train_loss', 'N/A')}
- **Final Validation Loss:** {training_history_analysis.get('final_val_loss', 'N/A')}
- **Best Validation Loss:** {training_history_analysis.get('best_val_loss', 'N/A')}
- **Convergence Speed:** {training_history_analysis.get('avg_convergence_speed', 'N/A')} (average loss decrease per epoch)
- **Train-Validation Difference:** {training_history_analysis.get('avg_train_val_diff', 'N/A')} (average difference)
- **Convergence Status:** {training_history_analysis.get('convergence_status', 'N/A')}
- **Overfitting Detected:** {'Yes (Epoch ' + str(training_history_analysis.get('overfitting_epoch', 'N/A')) + ')' if training_history_analysis.get('overfitting_detected') else 'No'}
"""
            
            # Add loss curve plot
            loss_plot = training_history_analysis.get('loss_curve_plot')
            if loss_plot:
                rel_path = os.path.relpath(loss_plot, self.analysis_dir)
                report_content += f"\n![Training and Validation Loss Curves]({rel_path})\n"
            
            # Add GNS analysis
            if gns_analysis:
                report_content += f"""
## 4. GNS (Gradient Noise Scale) Analysis

- **Epochs Analyzed:** {', '.join(map(str, gns_analysis.get('epochs_analyzed', [])))}
- **Average GNS Value:** {gns_analysis.get('average_gns', 'N/A')}
- **GNS Trend:** {gns_analysis.get('gns_trend', 'N/A')}
- **Current Batch Size:** {gns_analysis.get('current_batch_size', 'N/A')}
- **Recommended Batch Size:** {gns_analysis.get('recommended_batch_size', 'N/A')}
- **Recommendation:** {gns_analysis.get('recommendation', 'N/A')}
"""
                
                # Add GNS trend plot
                gns_plot = gns_analysis.get('gns_plot')
                if gns_plot:
                    rel_path = os.path.relpath(gns_plot, self.analysis_dir)
                    report_content += f"\n![GNS Trend]({rel_path})\n"
                
                # Add gradient statistics plot
                grad_stats_plot = gns_analysis.get('grad_stats_plot')
                if grad_stats_plot:
                    rel_path = os.path.relpath(grad_stats_plot, self.analysis_dir)
                    report_content += f"\n![Gradient Statistics Trend]({rel_path})\n"
            
            # Add conclusions and recommendations
            report_content += """
## 5. Conclusions and Recommendations

"""
            
            # Generate conclusions and recommendations based on analysis results
            # Model structure recommendations
            if model_structure_analysis:
                total_params = model_structure_analysis.get('total_parameters', 0)
                if total_params > 10000000:  # Over 10 million parameters
                    report_content += "- **Model Complexity:** The model has a large number of parameters. Consider using a smaller model or pruning techniques to reduce parameter count.\n"
                elif total_params < 100000:  # Under 100k parameters
                    report_content += "- **Model Complexity:** The model has relatively few parameters. Consider increasing model capacity to improve performance.\n"
                else:
                    report_content += "- **Model Complexity:** The model parameter count is moderate.\n"
            
            # Training recommendations
            if training_history_analysis:
                convergence_status = training_history_analysis.get('convergence_status', '')
                overfitting_detected = training_history_analysis.get('overfitting_detected', False)
                
                if 'Overfitting' in convergence_status or overfitting_detected:
                    report_content += "- **Training Process:** Overfitting detected. Consider adding regularization (e.g., Dropout, L2 regularization) or implementing early stopping.\n"
                elif 'Underfitting' in convergence_status:
                    report_content += "- **Training Process:** Underfitting detected. Consider increasing model capacity or extending training duration.\n"
                else:
                    report_content += "- **Training Process:** Training process appears normal with good convergence.\n"
            
            # GNS recommendations
            if gns_analysis:
                report_content += f"- **Batch Size:** {gns_analysis.get('recommendation', '')}\n"
            
            # Write report file
            with open(report_path, 'w') as f:
                f.write(report_content)
            
            logger.info(f"Analysis report generated: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def run_all_analysis(self):
        """
        Run all analyses and generate report
        
        Returns:
            dict: Complete analysis results
        """
        logger.info(f"Starting analysis of experiment: {self.experiment_dir}")
        
        # Load data
        self.load_config()
        self.load_model_structure()
        self.load_training_history()
        
        # Run analyses
        self.analyze_model_structure()
        self.analyze_training_history()
        
        # Try to run GNS analysis, but continue if it fails
        try:
            logger.info("Starting GNS analysis (this may take some time)...")
            gns_result = self.analyze_gns()
            if gns_result is None:
                logger.warning("GNS analysis did not produce results, continuing without GNS data")
        except Exception as e:
            logger.error(f"GNS analysis failed: {e}")
            logger.error("Continuing without GNS data")
            import traceback
            logger.error(traceback.format_exc())
        
        # Generate report
        logger.info("Generating final report...")
        report_path = self.generate_report()
        
        logger.info(f"Analysis complete, report saved to: {report_path}")
        return self.results

def main():
    """Main function to analyze a single experiment directory"""
    parser = argparse.ArgumentParser(description='Analyze a single experiment directory')
    parser.add_argument('--experiment_dir', type=str, default='results/custom_audio_fcnn_classification_20250430_225309',
                        help='Path to the experiment directory to analyze')
    args = parser.parse_args()
    
    experiment_dir = args.experiment_dir
    if not os.path.exists(experiment_dir):
        logger.error(f"Experiment directory not found: {experiment_dir}")
        return
    
    logger.info(f"Analyzing experiment: {experiment_dir}")
    
    # Create analyzer and run analysis
    analyzer = CustomAnalyzer(experiment_dir)
    results = analyzer.run_all_analysis()
    
    if results:
        logger.info(f"Analysis complete. Results saved to: {os.path.join(experiment_dir, 'custom_analysis')}")
    else:
        logger.error("Analysis failed to complete successfully")

if __name__ == "__main__":
    main() 