"""
train.py
Single model training script for CLIVP-Cervix
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle
from tqdm import tqdm
import argparse
import json
from datetime import datetime
import wandb
from PIL import Image
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


class Trainer:
    def __init__(self, model, train_loader, val_loader, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs
        )
        
        # Initialize tracking
        self.best_val_acc = 0
        self.best_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        # Initialize wandb if enabled
        if args.use_wandb:
            wandb.init(
                project="clivp-cervix",
                name=f"mode_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=vars(args)
            )
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Handle different data formats
            if 'features' in batch:
                # Preprocessed features
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
            else:
                # Raw images and text
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                texts = batch.get('text', None)
                
                if texts is not None:
                    texts = texts.to(self.device)
                
                self.optimizer.zero_grad()
                
                if hasattr(self.model, 'forward_contrastive') and self.args.mode == 1 and texts is not None:
                    outputs, contrastive_loss = self.model.forward_contrastive(images, texts)
                    classification_loss = self.criterion(outputs, labels)
                    loss = classification_loss + contrastive_loss
                else:
                    outputs = self.model(images, texts)
                    loss = self.criterion(outputs, labels)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch in pbar:
                if 'features' in batch:
                    # Preprocessed features
                    features = batch['features'].to(self.device)
                    labels = batch['label'].to(self.device)
                    outputs = self.model(features)
                else:
                    # Raw images and text
                    images = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)
                    texts = batch.get('text', None)
                    
                    if texts is not None:
                        texts = texts.to(self.device)
                    
                    outputs = self.model(images, texts)
                
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = total_loss / len(self.val_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        return epoch_loss, epoch_acc, precision, recall, f1
    
    def train(self):
        """Full training loop."""
        print(f"Training on {self.device}")
        print(f"Total epochs: {self.args.epochs}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.args.epochs):
            print(f"\nEpoch {epoch + 1}/{self.args.epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc, val_prec, val_rec, val_f1 = self.validate_epoch()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Update learning rate
            self.scheduler.step()
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Val Precision: {val_prec:.4f}, Val Recall: {val_rec:.4f}, Val F1: {val_f1:.4f}")
            print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Log to wandb
            if self.args.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_precision': val_prec,
                    'val_recall': val_rec,
                    'val_f1': val_f1,
                    'lr': self.scheduler.get_last_lr()[0]
                })
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                self.save_checkpoint(epoch, val_acc, is_best=True)
                print(f"New best model! Val Acc: {val_acc:.4f}")
            
            # Regular checkpoint
            if (epoch + 1) % self.args.save_freq == 0:
                self.save_checkpoint(epoch, val_acc, is_best=False)
        
        print(f"\nTraining completed!")
        print(f"Best Val Acc: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
        
        # Save final model
        self.save_checkpoint(self.args.epochs - 1, val_acc, is_best=False, final=True)
        
        # Save training history
        self.save_training_history()
    
    def save_checkpoint(self, epoch, val_acc, is_best=False, final=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
            'args': self.args
        }
        
        if is_best:
            path = os.path.join(self.args.checkpoint_dir, 'best_model.pth')
        elif final:
            path = os.path.join(self.args.checkpoint_dir, 'final_model.pth')
        else:
            path = os.path.join(self.args.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def save_training_history(self):
        """Save training history."""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'args': vars(self.args)
        }
        
        history_path = os.path.join(self.args.checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Training history saved: {history_path}")


def main():
    parser = argparse.ArgumentParser(description='Train CLIVP-Cervix model')
    
    # Data arguments
    parser.add_argument('--train_data', type=str, default='data/processed/train_data_mode1.pkl',
                       help='Path to training data')
    parser.add_argument('--val_data', type=str, default='data/processed/val_data_mode1.pkl',
                       help='Path to validation data')
    parser.add_argument('--mode', type=int, default=1, choices=[0, 1],
                       help='0: image only, 1: image+text')
    
    # Model arguments
    parser.add_argument('--clip_model', type=str, default='ViT-B/32',
                       help='CLIP model variant')
    parser.add_argument('--freeze_clip', action='store_true',
                       help='Freeze CLIP weights')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Other arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Save checkpoint frequency')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load data
    with open(args.train_data, 'rb') as f:
        train_data = pickle.load(f)
    
    with open(args.val_data, 'rb') as f:
        val_data = pickle.load(f)
    
    # Create datasets
    train_dataset = CervixDataset(train_data, mode=args.mode)
    val_dataset = CervixDataset(val_data, mode=args.mode)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = create_model(
        num_classes=4,
        mode=args.mode,
        clip_model=args.clip_model,
        dropout=args.dropout,
        freeze_clip=args.freeze_clip
    )
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, args)
    
    # Train model
    trainer.train()
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
