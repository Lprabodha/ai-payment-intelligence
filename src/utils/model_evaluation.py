"""
Model Evaluation Utility
Provides comprehensive accuracy metrics calculation and visualization for all AI models
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)

# Set style for better plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")


class ModelEvaluator:
    """Comprehensive model evaluation with metrics and visualizations"""
    
    def __init__(self, model_name, output_dir="/src/data/models"):
        self.model_name = model_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metrics = {}
        self.history = {}
        
    def evaluate_classification(
        self, y_true, y_pred, y_pred_proba=None, 
        threshold=0.5, save_images=True
    ):
        """
        Evaluate classification model with comprehensive metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels (binary)
            y_pred_proba: Predicted probabilities (optional)
            threshold: Classification threshold
            save_images: Whether to save visualization images
        """
        # Convert to binary if needed
        if y_pred_proba is not None:
            y_pred_binary = (y_pred_proba >= threshold).astype(int)
        elif y_pred is not None:
            y_pred_binary = y_pred
        else:
            raise ValueError("Either y_pred or y_pred_proba must be provided")
            
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'precision': precision_score(y_true, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true, y_pred_binary, zero_division=0),
            'f1_score': f1_score(y_true, y_pred_binary, zero_division=0),
            'threshold': threshold
        }
        
        # Add probability-based metrics if available
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
            
            # Find optimal threshold
            precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_pred_proba)
            f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-9)
            optimal_idx = np.argmax(f1_scores)
            metrics['optimal_threshold'] = float(thresholds[optimal_idx])
            metrics['optimal_f1'] = float(f1_scores[optimal_idx])
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_binary)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['true_negatives'] = int(cm[0, 0])
        metrics['false_positives'] = int(cm[0, 1])
        metrics['false_negatives'] = int(cm[1, 0])
        metrics['true_positives'] = int(cm[1, 1])
        
        # Classification report
        report = classification_report(y_true, y_pred_binary, output_dict=True, zero_division=0)
        metrics['classification_report'] = report
        
        self.metrics = metrics
        
        # Save visualizations
        if save_images:
            self._plot_classification_metrics(y_true, y_pred_binary, y_pred_proba)
        
        return metrics
    
    def evaluate_regression(
        self, y_true, y_pred, save_images=True
    ):
        """
        Evaluate regression model with comprehensive metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            save_images: Whether to save visualization images
        """
        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2_score': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred),
            'mean_absolute_error': mean_absolute_error(y_true, y_pred),
            'mean_squared_error': mean_squared_error(y_true, y_pred)
        }
        
        # Additional statistics
        residuals = y_true - y_pred
        metrics['mean_residual'] = float(np.mean(residuals))
        metrics['std_residual'] = float(np.std(residuals))
        metrics['max_error'] = float(np.max(np.abs(residuals)))
        
        self.metrics = metrics
        
        # Save visualizations
        if save_images:
            self._plot_regression_metrics(y_true, y_pred)
        
        return metrics
    
    def _plot_classification_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Create comprehensive classification visualization plots"""
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Confusion Matrix
        ax1 = plt.subplot(2, 3, 1)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        ax1.set_title(f'{self.model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # 2. ROC Curve
        if y_pred_proba is not None:
            ax2 = plt.subplot(2, 3, 2)
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            ax2.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.4f})')
            ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title(f'{self.model_name} - ROC Curve', fontsize=14, fontweight='bold')
            ax2.legend(loc="lower right")
            ax2.grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        if y_pred_proba is not None:
            ax3 = plt.subplot(2, 3, 3)
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = average_precision_score(y_true, y_pred_proba)
            ax3.plot(recall, precision, color='blue', lw=2,
                    label=f'PR curve (AUC = {pr_auc:.4f})')
            ax3.set_xlabel('Recall')
            ax3.set_ylabel('Precision')
            ax3.set_title(f'{self.model_name} - Precision-Recall Curve', fontsize=14, fontweight='bold')
            ax3.legend(loc="lower left")
            ax3.grid(True, alpha=0.3)
        
        # 4. Metrics Bar Chart
        ax4 = plt.subplot(2, 3, 4)
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        metric_values = [
            self.metrics.get('accuracy', 0),
            self.metrics.get('precision', 0),
            self.metrics.get('recall', 0),
            self.metrics.get('f1_score', 0)
        ]
        bars = ax4.bar(metric_names, metric_values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
        ax4.set_ylim([0, 1])
        ax4.set_ylabel('Score')
        ax4.set_title(f'{self.model_name} - Classification Metrics', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Prediction Distribution
        ax5 = plt.subplot(2, 3, 5)
        if y_pred_proba is not None:
            ax5.hist(y_pred_proba[y_true == 0], bins=30, alpha=0.7, label='Negative', color='blue')
            ax5.hist(y_pred_proba[y_true == 1], bins=30, alpha=0.7, label='Positive', color='red')
        else:
            ax5.hist(y_pred, bins=30, alpha=0.7, label='Predictions', color='blue')
        ax5.set_xlabel('Predicted Probability' if y_pred_proba is not None else 'Predicted Label')
        ax5.set_ylabel('Frequency')
        ax5.set_title(f'{self.model_name} - Prediction Distribution', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Metrics Summary Table
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Create metrics table
        table_data = []
        if 'accuracy' in self.metrics:
            table_data.append(['Accuracy', f"{self.metrics['accuracy']:.4f}"])
        if 'precision' in self.metrics:
            table_data.append(['Precision', f"{self.metrics['precision']:.4f}"])
        if 'recall' in self.metrics:
            table_data.append(['Recall', f"{self.metrics['recall']:.4f}"])
        if 'f1_score' in self.metrics:
            table_data.append(['F1 Score', f"{self.metrics['f1_score']:.4f}"])
        if 'roc_auc' in self.metrics:
            table_data.append(['ROC AUC', f"{self.metrics['roc_auc']:.4f}"])
        if 'pr_auc' in self.metrics:
            table_data.append(['PR AUC', f"{self.metrics['pr_auc']:.4f}"])
        if 'optimal_threshold' in self.metrics:
            table_data.append(['Optimal Threshold', f"{self.metrics['optimal_threshold']:.4f}"])
        
        table = ax6.table(cellText=table_data, colLabels=['Metric', 'Value'],
                         cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        ax6.set_title(f'{self.model_name} - Metrics Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.suptitle(f'{self.model_name} - Model Evaluation Report', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Save figure
        image_path = os.path.join(self.output_dir, f'{self.model_name}_evaluation.png')
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved evaluation image: {image_path}")
    
    def _plot_regression_metrics(self, y_true, y_pred):
        """Create comprehensive regression visualization plots"""
        fig = plt.figure(figsize=(20, 12))
        
        residuals = y_true - y_pred
        
        # 1. Actual vs Predicted
        ax1 = plt.subplot(2, 3, 1)
        ax1.scatter(y_true, y_pred, alpha=0.6, s=50)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title(f'{self.model_name} - Actual vs Predicted', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add R² score
        r2 = r2_score(y_true, y_pred)
        ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Residuals Plot
        ax2 = plt.subplot(2, 3, 2)
        ax2.scatter(y_pred, residuals, alpha=0.6, s=50)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title(f'{self.model_name} - Residuals Plot', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Residuals Distribution
        ax3 = plt.subplot(2, 3, 3)
        ax3.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(x=0, color='r', linestyle='--', lw=2)
        ax3.set_xlabel('Residuals')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'{self.model_name} - Residuals Distribution', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Metrics Bar Chart
        ax4 = plt.subplot(2, 3, 4)
        metric_names = ['MAE', 'RMSE', 'R²', 'MAPE']
        metric_values = [
            self.metrics.get('mae', 0),
            self.metrics.get('rmse', 0),
            self.metrics.get('r2_score', 0),
            self.metrics.get('mape', 0) if self.metrics.get('mape', 0) < 100 else 0
        ]
        bars = ax4.bar(metric_names, metric_values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
        ax4.set_ylabel('Score')
        ax4.set_title(f'{self.model_name} - Regression Metrics', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Prediction Distribution
        ax5 = plt.subplot(2, 3, 5)
        ax5.hist(y_true, bins=30, alpha=0.7, label='Actual', color='blue')
        ax5.hist(y_pred, bins=30, alpha=0.7, label='Predicted', color='red')
        ax5.set_xlabel('Value')
        ax5.set_ylabel('Frequency')
        ax5.set_title(f'{self.model_name} - Value Distribution', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Metrics Summary Table
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Create metrics table
        table_data = [
            ['MAE', f"{self.metrics.get('mae', 0):.4f}"],
            ['MSE', f"{self.metrics.get('mse', 0):.4f}"],
            ['RMSE', f"{self.metrics.get('rmse', 0):.4f}"],
            ['R² Score', f"{self.metrics.get('r2_score', 0):.4f}"],
            ['MAPE', f"{self.metrics.get('mape', 0):.4f}%"],
            ['Mean Residual', f"{self.metrics.get('mean_residual', 0):.4f}"],
            ['Std Residual', f"{self.metrics.get('std_residual', 0):.4f}"]
        ]
        
        table = ax6.table(cellText=table_data, colLabels=['Metric', 'Value'],
                         cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        ax6.set_title(f'{self.model_name} - Metrics Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.suptitle(f'{self.model_name} - Model Evaluation Report', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Save figure
        image_path = os.path.join(self.output_dir, f'{self.model_name}_evaluation.png')
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved evaluation image: {image_path}")
    
    def plot_training_history(self, history, save_image=True):
        """Plot training history (loss, accuracy, reward, etc.)"""
        if not history:
            return
        
        # Determine number of subplots needed
        num_plots = 0
        plot_keys = []
        
        if 'loss' in history or 'reward' in history:
            num_plots += 1
            if 'loss' in history:
                plot_keys.append('loss')
            if 'reward' in history:
                plot_keys.append('reward')
        
        if 'accuracy' in history or 'acc' in history:
            num_plots += 1
            plot_keys.append('accuracy' if 'accuracy' in history else 'acc')
        
        if num_plots == 0:
            return
        
        fig, axes = plt.subplots(1, num_plots, figsize=(15, 5))
        if num_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot loss or reward
        if 'loss' in history:
            axes[plot_idx].plot(history['loss'], label='Training Loss', color='blue')
            if 'val_loss' in history:
                axes[plot_idx].plot(history['val_loss'], label='Validation Loss', color='red')
            axes[plot_idx].set_xlabel('Epoch/Step')
            axes[plot_idx].set_ylabel('Loss')
            axes[plot_idx].set_title(f'{self.model_name} - Training Loss', fontsize=12, fontweight='bold')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1
        elif 'reward' in history:
            axes[plot_idx].plot(history['reward'], label='Reward', color='green')
            axes[plot_idx].set_xlabel('Episode')
            axes[plot_idx].set_ylabel('Reward')
            axes[plot_idx].set_title(f'{self.model_name} - Training Reward', fontsize=12, fontweight='bold')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1
        
        # Plot accuracy if available
        if 'accuracy' in history or 'acc' in history:
            acc_key = 'accuracy' if 'accuracy' in history else 'acc'
            axes[plot_idx].plot(history[acc_key], label='Training Accuracy', color='blue')
            if f'val_{acc_key}' in history:
                axes[plot_idx].plot(history[f'val_{acc_key}'], label='Validation Accuracy', color='red')
            axes[plot_idx].set_xlabel('Epoch')
            axes[plot_idx].set_ylabel('Accuracy')
            axes[plot_idx].set_title(f'{self.model_name} - Training Accuracy', fontsize=12, fontweight='bold')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_image:
            image_path = os.path.join(self.output_dir, f'{self.model_name}_training_history.png')
            plt.savefig(image_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✅ Saved training history image: {image_path}")
        else:
            plt.close()
    
    def save_metrics(self, filename=None):
        """Save metrics to JSON file"""
        if filename is None:
            filename = f'{self.model_name}_metrics.json'
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Add timestamp
        metrics_with_meta = {
            'model_name': self.model_name,
            'evaluated_at': datetime.now().isoformat(),
            'metrics': self.metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_with_meta, f, indent=4)
        
        print(f"  ✅ Saved metrics: {filepath}")
        return filepath

