"""
cervix_confmtrx.py
Generate confusion matrix and detailed evaluation metrics for CLIVP-Cervix model
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    precision_recall_curve
)
import json
import argparse
from pathlib import Path
import pickle
from torch.utils.data import DataLoader
from PIL import Image

from models.clivp_cervix import create_model
from preprocess_cervix import CervixPreprocessor


class CervixEvaluator:
    def __init__(self, model_path, mode=1, device="cuda"):
        """
        Initialize evaluator for CLIVP-Cervix model.
        
        Args:
            model_path: Path to trained model checkpoint
            mode: 0 for image only, 1 for multimodal
            device: cuda or cpu
        """
        self.mode = mode
        self.device = device if torch.cuda.is_available() else "cpu"
        self.class_names = ['HSIL', 'LSIL', 'NILM', 'SCC']
        
        # Load model
        self.model = self.load_model(model_path)
        self.model.eval()
        
    def load_model(self, model_path):
        """Load trained model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = create_model(
            num_classes=4,
            mode=self.mode,
            clip_model=checkpoint.get('args', {}).get('clip_model', 'ViT-B/32'),
            freeze_clip=True
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def evaluate_dataset(self, data_loader):
        """Evaluate model on a dataset."""
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in data_loader:
                if 'features' in batch:
                    # Preprocessed features
                    features = batch['features'].to(self.device)
                    labels = batch['label'].to(self.device)
                    outputs = self.model(features)
                else:
                    # Raw images
                    images = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)
                    texts = batch.get('text', None)
                    if texts is not None:
                        texts = texts.to(self.device)
                    outputs = self.model(images, texts)
                
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create annotation text
        annotations = np.empty_like(cm, dtype=str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
        
        # Plot heatmap
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    cbar_kws={'label': 'Count'})
        
        plt.title(f'Confusion Matrix - CLIVP-Cervix (Mode {self.mode})', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Add accuracy text
        accuracy = accuracy_score(y_true, y_pred)
        plt.text(0.5, -0.15, f'Overall Accuracy: {accuracy:.3f}', 
                ha='center', transform=plt.gca().transAxes, fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
        
        return cm
    
    def plot_roc_curves(self, y_true, y_probs, save_path=None):
        """Plot ROC curves for each class."""
        plt.figure(figsize=(10, 8))
        
        # Convert to one-hot encoding
        n_classes = len(self.class_names)
        y_true_binary = np.eye(n_classes)[y_true]
        
        # Calculate ROC curve for each class
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curves - CLIVP-Cervix (Mode {self.mode})', fontsize=16)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to: {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curves(self, y_true, y_probs, save_path=None):
        """Plot Precision-Recall curves for each class."""
        plt.figure(figsize=(10, 8))
        
        # Convert to one-hot encoding
        n_classes = len(self.class_names)
        y_true_binary = np.eye(n_classes)[y_true]
        
        # Calculate PR curve for each class
        for i, class_name in enumerate(self.class_names):
            precision, recall, _ = precision_recall_curve(y_true_binary[:, i], y_probs[:, i])
            avg_precision = auc(recall, precision)
            
            plt.plot(recall, precision, label=f'{class_name} (AP = {avg_precision:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curves - CLIVP-Cervix (Mode {self.mode})', fontsize=16)
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PR curves saved to: {save_path}")
        
        plt.show()
    
    def generate_classification_report(self, y_true, y_pred, save_path=None):
        """Generate detailed classification report."""
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Convert to DataFrame for better visualization
        df_report = pd.DataFrame(report).transpose()
        
        # Format the DataFrame
        df_report = df_report.round(3)
        
        print("\nClassification Report:")
        print("=" * 70)
        print(df_report)
        
        if save_path:
            # Save as CSV
            csv_path = save_path.replace('.txt', '.csv')
            df_report.to_csv(csv_path)
            
            # Save as formatted text
            with open(save_path, 'w') as f:
                f.write("CLIVP-Cervix Classification Report\n")
                f.write(f"Mode: {self.mode} ({'Image+Text' if self.mode == 1 else 'Image Only'})\n")
                f.write("=" * 70 + "\n\n")
                f.write(df_report.to_string())
                f.write("\n\n" + "=" * 70 + "\n")
                f.write(f"Overall Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
            
            print(f"\nClassification report saved to: {save_path}")
        
        return report
    
    def plot_class_distribution(self, y_true, y_pred, save_path=None):
        """Plot true vs predicted class distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # True distribution
        true_counts = pd.Series(y_true).value_counts().sort_index()
        ax1.bar(self.class_names, true_counts.values, color='skyblue', edgecolor='black')
        ax1.set_title('True Class Distribution', fontsize=14)
        ax1.set_xlabel('Class', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        
        # Add value labels
        for i, v in enumerate(true_counts.values):
            ax1.text(i, v + 10, str(v), ha='center', fontsize=10)
        
        # Predicted distribution
        pred_counts = pd.Series(y_pred).value_counts().sort_index()
        ax2.bar(self.class_names, pred_counts.values, color='lightcoral', edgecolor='black')
        ax2.set_title('Predicted Class Distribution', fontsize=14)
        ax2.set_xlabel('Class', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        
        # Add value labels
        for i, v in enumerate(pred_counts.values):
            ax2.text(i, v + 10, str(v), ha='center', fontsize=10)
        
        plt.suptitle(f'Class Distribution Comparison - CLIVP-Cervix (Mode {self.mode})', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class distribution plot saved to: {save_path}")
        
        plt.show()
    
    def analyze_misclassifications(self, y_true, y_pred, y_probs, data_info=None, top_k=5):
        """Analyze most confident misclassifications."""
        misclassified_idx = np.where(y_true != y_pred)[0]
        
        if len(misclassified_idx) == 0:
            print("No misclassifications found!")
            return
        
        # Get confidence scores for misclassified samples
        misclassified_confidences = []
        for idx in misclassified_idx:
            pred_class = y_pred[idx]
            confidence = y_probs[idx, pred_class]
            misclassified_confidences.append({
                'index': idx,
                'true_class': self.class_names[y_true[idx]],
                'pred_class': self.class_names[pred_class],
                'confidence': confidence,
                'true_class_prob': y_probs[idx, y_true[idx]]
            })
        
        # Sort by confidence (descending)
        misclassified_confidences.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"\nTop {top_k} Most Confident Misclassifications:")
        print("=" * 80)
        
        for i, item in enumerate(misclassified_confidences[:top_k]):
            print(f"\n{i+1}. Sample Index: {item['index']}")
            print(f"   True Class: {item['true_class']}")
            print(f"   Predicted Class: {item['pred_class']} (confidence: {item['confidence']:.3f})")
            print(f"   True Class Probability: {item['true_class_prob']:.3f}")
            
            if data_info and item['index'] < len(data_info):
                info = data_info[item['index']]
                if 'image_path' in info:
                    print(f"   Image: {info['image_path']}")
        
        # Create confusion pairs analysis
        confusion_pairs = {}
        for item in misclassified_confidences:
            pair = (item['true_class'], item['pred_class'])
            if pair not in confusion_pairs:
                confusion_pairs[pair] = 0
            confusion_pairs[pair] += 1
        
        print("\n\nMost Common Confusion Pairs:")
        print("=" * 50)
        sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
        for (true_class, pred_class), count in sorted_pairs[:10]:
            print(f"{true_class} → {pred_class}: {count} cases")
    
    def save_predictions(self, y_true, y_pred, y_probs, data_info=None, save_path=None):
        """Save all predictions to a CSV file."""
        results = []
        
        for i in range(len(y_true)):
            result = {
                'index': i,
                'true_class': self.class_names[y_true[i]],
                'pred_class': self.class_names[y_pred[i]],
                'correct': y_true[i] == y_pred[i],
                'confidence': y_probs[i, y_pred[i]]
            }
            
            # Add class probabilities
            for j, class_name in enumerate(self.class_names):
                result[f'prob_{class_name}'] = y_probs[i, j]
            
            # Add data info if available
            if data_info and i < len(data_info):
                info = data_info[i]
                if 'image_path' in info:
                    result['image_path'] = info['image_path']
                if 'description' in info and self.mode == 1:
                    result['description'] = info['description']
            
            results.append(result)
        
        df_results = pd.DataFrame(results)
        
        if save_path:
            df_results.to_csv(save_path, index=False)
            print(f"\nPredictions saved to: {save_path}")
        
        return df_results


def main():
    parser = argparse.ArgumentParser(description='Generate confusion matrix and metrics for CLIVP-Cervix')
    parser.add_argument('--mode', type=int, default=1, choices=[0, 1],
                       help='0: image only, 1: image+text')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default='data/processed/val_data_mode1.pkl',
                       help='Path to validation data')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    with open(args.data_path, 'rb') as f:
        val_data = pickle.load(f)
    
    # Create dataset and dataloader
    from train import CervixDataset
    val_dataset = CervixDataset(val_data, mode=args.mode)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize evaluator
    evaluator = CervixEvaluator(args.checkpoint, mode=args.mode)
    
    # Evaluate
    print("Evaluating model...")
    y_pred, y_true, y_probs = evaluator.evaluate_dataset(val_loader)
    
    # Generate all visualizations and reports
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Confusion Matrix
    cm_path = os.path.join(args.output_dir, f'confusion_matrix_mode{args.mode}_{timestamp}.png')
    evaluator.plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
    
    # 2. ROC Curves
    roc_path = os.path.join(args.output_dir, f'roc_curves_mode{args.mode}_{timestamp}.png')
    evaluator.plot_roc_curves(y_true, y_probs, save_path=roc_path)
    
    # 3. Precision-Recall Curves
    pr_path = os.path.join(args.output_dir, f'pr_curves_mode{args.mode}_{timestamp}.png')
    evaluator.plot_precision_recall_curves(y_true, y_probs, save_path=pr_path)
    
    # 4. Classification Report
    report_path = os.path.join(args.output_dir, f'classification_report_mode{args.mode}_{timestamp}.txt')
    evaluator.generate_classification_report(y_true, y_pred, save_path=report_path)
    
    # 5. Class Distribution
    dist_path = os.path.join(args.output_dir, f'class_distribution_mode{args.mode}_{timestamp}.png')
    evaluator.plot_class_distribution(y_true, y_pred, save_path=dist_path)
    
    # 6. Misclassification Analysis
    evaluator.analyze_misclassifications(y_true, y_pred, y_probs, val_data)
    
    # 7. Save all predictions
    pred_path = os.path.join(args.output_dir, f'predictions_mode{args.mode}_{timestamp}.csv')
    evaluator.save_predictions(y_true, y_pred, y_probs, val_data, save_path=pred_path)
    
    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
