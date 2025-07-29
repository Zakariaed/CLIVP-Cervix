"""
10fold_train.py
10-fold cross-validation training for CLIVP-Cervix model
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pickle
from tqdm import tqdm
import argparse
import json
from datetime import datetime
import clip

from models.clivp_cervix import create_model


class CervixDataset(Dataset):
    """Dataset class for preprocessed cervical cell features."""
    
    def __init__(self, data, mode=1, transform=None):
        self.data = data
        self.mode = mode
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        features = torch.tensor(item['features'], dtype=torch.float32)
        label = torch.tensor(item['class_label'], dtype=torch.long)
        
        sample = {
            'features': features,
            'label': label,
            'class': item['class'],
            'image_path': item['image_path']
        }
        
        if self.mode == 1 and 'description' in item:
            sample['description'] = item['description']
            
        return sample


class CLIPCervixDataset(Dataset):
    """Dataset class for raw images and text (for end-to-end training)."""
    
    def __init__(self, csv_path, preprocess, mode=1):
        self.df = pd.read_csv(csv_path)
        self.preprocess = preprocess
        self.mode = mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load and preprocess image
        image = self.preprocess(Image.open(row['image_path']))
        
        # Prepare output
        sample = {
            'image': image,
            'label': row['class_label'],
            'class': row['class']
        }
        
        if self.mode == 1:
            # Tokenize text
            text = clip.tokenize([row['description']], truncate=True).squeeze(0)
            sample['text'] = text
            
        return sample


def train_epoch(model, dataloader, criterion, optimizer, device, mode=1):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        if isinstance(batch['features'], torch.Tensor):
            # Using preprocessed features
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
        else:
            # Using raw images and text
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            texts = batch.get('text', None)
            
            if texts is not None:
                texts = texts.to(device)
            
            optimizer.zero_grad()
            
            if hasattr(model, 'forward_contrastive') and mode == 1 and texts is not None:
                outputs, contrastive_loss = model.forward_contrastive(images, texts)
                classification_loss = criterion(outputs, labels)
                loss = classification_loss + contrastive_loss
            else:
                outputs = model(images, texts)
                loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = total_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def validate_epoch(model, dataloader, criterion, device, mode=1):
    """Validate model for one epoch."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            if isinstance(batch['features'], torch.Tensor):
                # Using preprocessed features
                features = batch['features'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
            else:
                # Using raw images and text
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                texts = batch.get('text', None)
                
                if texts is not None:
                    texts = texts.to(device)
                
                outputs = model(images, texts)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    epoch_loss = total_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    return epoch_loss, epoch_acc, precision, recall, f1, all_preds, all_labels, all_probs


def perform_10fold_cv(args):
    """Perform 10-fold cross-validation."""
    
    # Load preprocessed data
    with open(args.data_path, 'rb') as f:
        all_data = pickle.load(f)
    
    # Extract features and labels for stratification
    all_features = np.array([item['features'] for item in all_data])
    all_labels = np.array([item['class_label'] for item in all_data])
    
    # Initialize 10-fold cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
    
    # Store results
    fold_results = []
    all_fold_predictions = []
    all_fold_labels = []
    all_fold_probs = []
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Perform 10-fold CV
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_features, all_labels)):
        print(f"\n{'='*50}")
        print(f"FOLD {fold + 1}/10")
        print(f"{'='*50}")
        
        # Split data
        train_data = [all_data[i] for i in train_idx]
        val_data = [all_data[i] for i in val_idx]
        
        # Create datasets
        train_dataset = CervixDataset(train_data, mode=args.mode)
        val_dataset = CervixDataset(val_data, mode=args.mode)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers
        )
        
        # Create model
        model = create_model(
            num_classes=4,
            mode=args.mode,
            clip_model=args.clip_model,
            dropout=args.dropout,
            freeze_clip=args.freeze_clip
        ).to(device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training loop
        best_val_acc = 0
        best_model_state = None
        patience_counter = 0
        
        fold_train_history = []
        fold_val_history = []
        
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            
            # Train
            train_loss, train_acc, _, _ = train_epoch(
                model, train_loader, criterion, optimizer, device, args.mode
            )
            
            # Validate
            val_loss, val_acc, val_prec, val_rec, val_f1, val_preds, val_labels, val_probs = validate_epoch(
                model, val_loader, criterion, device, args.mode
            )
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Save history
            fold_train_history.append({
                'epoch': epoch + 1,
                'loss': train_loss,
                'acc': train_acc
            })
            
            fold_val_history.append({
                'epoch': epoch + 1,
                'loss': val_loss,
                'acc': val_acc,
                'precision': val_prec,
                'recall': val_rec,
                'f1': val_f1
            })
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Val Precision: {val_prec:.4f}, Val Recall: {val_rec:.4f}, Val F1: {val_f1:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                
                # Save best model for this fold
                torch.save({
                    'model_state_dict': best_model_state,
                    'fold': fold,
                    'val_acc': best_val_acc,
                    'args': args
                }, f"checkpoints/fold_{fold}_best_model.pth")
                
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print("Early stopping triggered")
                    break
        
        # Load best model and get final predictions
        model.load_state_dict(best_model_state)
        _, _, _, _, _, final_preds, final_labels, final_probs = validate_epoch(
            model, val_loader, criterion, device, args.mode
        )
        
        # Store fold results
        fold_results.append({
            'fold': fold + 1,
            'best_val_acc': best_val_acc,
            'train_history': fold_train_history,
            'val_history': fold_val_history,
            'confusion_matrix': confusion_matrix(final_labels, final_preds),
            'final_predictions': final_preds,
            'final_labels': final_labels,
            'final_probs': final_probs
        })
        
        all_fold_predictions.extend(final_preds)
        all_fold_labels.extend(final_labels)
        all_fold_probs.extend(final_probs)
    
    # Calculate overall metrics
    overall_acc = accuracy_score(all_fold_labels, all_fold_predictions)
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        all_fold_labels, all_fold_predictions, average='weighted'
    )
    overall_cm = confusion_matrix(all_fold_labels, all_fold_predictions)
    
    # Class-wise metrics
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        all_fold_labels, all_fold_predictions, average=None
    )
    
    # Save results
    results = {
        'overall_metrics': {
            'accuracy': overall_acc,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'confusion_matrix': overall_cm.tolist()
        },
        'class_metrics': {
            'precision': class_precision.tolist(),
            'recall': class_recall.tolist(),
            'f1': class_f1.tolist(),
            'support': class_support.tolist()
        },
        'fold_results': fold_results,
        'all_predictions': all_fold_predictions,
        'all_labels': all_fold_labels,
        'all_probs': all_fold_probs,
        'args': vars(args)
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"results/10fold_results_mode{args.mode}_{timestamp}.json"
    os.makedirs("results", exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("10-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall F1-Score: {overall_f1:.4f}")
    print("\nClass-wise Metrics:")
    class_names = ['HSIL', 'LSIL', 'NILM', 'SCC']
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: Precision={class_precision[i]:.4f}, "
              f"Recall={class_recall[i]:.4f}, F1={class_f1[i]:.4f}")
    
    print(f"\nResults saved to: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='10-fold CV training for CLIVP-Cervix')
    parser.add_argument('--data_path', type=str, default='data/processed/processed_features_mode1.pkl',
                       help='Path to preprocessed data')
    parser.add_argument('--mode', type=int, default=1, choices=[0, 1],
                       help='0: image only, 1: image+text')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs per fold')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32',
                       help='CLIP model variant')
    parser.add_argument('--freeze_clip', action='store_true',
                       help='Freeze CLIP weights')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Run 10-fold CV
    perform_10fold_cv(args)


if __name__ == "__main__":
    main()
