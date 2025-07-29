"""
dataset.py
Dataset utilities for CLIVP-Cervix
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import clip
from typing import Dict, List, Tuple, Optional


class CervixImageDataset(Dataset):
    """Dataset for loading cervical cell images directly."""
    
    def __init__(self, 
                 csv_path: str,
                 transform: Optional[transforms.Compose] = None,
                 mode: int = 1,
                 clip_model: str = "ViT-B/32"):
        """
        Initialize dataset.
        
        Args:
            csv_path: Path to CSV with image paths and descriptions
            transform: Image transformations
            mode: 0 for image only, 1 for image+text
            clip_model: CLIP model variant for preprocessing
        """
        self.df = pd.read_csv(csv_path)
        self.mode = mode
        self.transform = transform
        
        # Load CLIP preprocessing
        _, self.clip_preprocess = clip.load(clip_model, device="cpu")
        
        # Class mapping
        self.class_to_idx = {
            'HSIL': 0,
            'LSIL': 1,
            'NILM': 2,
            'SCC': 3
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        image_path = row['image_path']
        image = Image.open(image_path).convert('RGB')
        
        # Apply CLIP preprocessing
        if self.clip_preprocess:
            image = self.clip_preprocess(image)
        elif self.transform:
            image = self.transform(image)
        
        # Get label
        label = row['class_label']
        
        sample = {
            'image': image,
            'label': label,
            'class': row['class'],
            'image_path': image_path
        }
        
        # Add text if in multimodal mode
        if self.mode == 1 and 'description' in row:
            text = clip.tokenize([row['description']], truncate=True).squeeze(0)
            sample['text'] = text
            sample['description'] = row['description']
        
        return sample


class BalancedBatchSampler(torch.utils.data.Sampler):
    """Sampler that ensures balanced classes in each batch."""
    
    def __init__(self, dataset, batch_size, num_classes=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes = num_classes
        
        # Get indices for each class
        self.class_indices = {i: [] for i in range(num_classes)}
        for idx in range(len(dataset)):
            label = dataset[idx]['label']
            self.class_indices[label].append(idx)
        
        # Ensure each class has enough samples
        self.min_class_size = min(len(indices) for indices in self.class_indices.values())
        self.samples_per_class = batch_size // num_classes
        
    def __iter__(self):
        # Shuffle indices for each class
        for class_idx in self.class_indices:
            np.random.shuffle(self.class_indices[class_idx])
        
        # Generate balanced batches
        num_batches = self.min_class_size // self.samples_per_class
        
        for batch_idx in range(num_batches):
            batch = []
            for class_idx in range(self.num_classes):
                start_idx = batch_idx * self.samples_per_class
                end_idx = start_idx + self.samples_per_class
                batch.extend(self.class_indices[class_idx][start_idx:end_idx])
            
            np.random.shuffle(batch)
            yield batch
    
    def __len__(self):
        return (self.min_class_size // self.samples_per_class) * self.batch_size


def create_data_loaders(
    train_csv: str,
    val_csv: str,
    batch_size: int = 32,
    mode: int = 1,
    num_workers: int = 4,
    balanced_sampling: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        train_csv: Path to training CSV
        val_csv: Path to validation CSV
        batch_size: Batch size
        mode: 0 for image only, 1 for multimodal
        num_workers: Number of data loading workers
        balanced_sampling: Whether to use balanced batch sampling
        
    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = CervixImageDataset(train_csv, mode=mode)
    val_dataset = CervixImageDataset(val_csv, mode=mode)
    
    # Create data loaders
    if balanced_sampling:
        train_sampler = BalancedBatchSampler(train_dataset, batch_size)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


class AugmentationPipeline:
    """Data augmentation pipeline for cervical cell images."""
    
    def __init__(self, mode='train'):
        """
        Initialize augmentation pipeline.
        
        Args:
            mode: 'train' or 'val'
        """
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomRotation(degrees=30),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1)
                ),
                transforms.RandomGrayscale(p=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __call__(self, image):
        return self.transform(image)


def analyze_dataset_statistics(csv_path: str) -> Dict:
    """
    Analyze dataset statistics.
    
    Args:
        csv_path: Path to dataset CSV
        
    Returns:
        Dictionary with dataset statistics
    """
    df = pd.read_csv(csv_path)
    
    stats = {
        'total_samples': len(df),
        'class_distribution': df['class'].value_counts().to_dict(),
        'class_percentages': (df['class'].value_counts() / len(df) * 100).to_dict(),
        'unique_descriptions': df['description'].nunique() if 'description' in df.columns else 0,
        'avg_description_length': df['description'].str.len().mean() if 'description' in df.columns else 0
    }
    
    # Check for missing files
    missing_files = []
    for path in df['image_path']:
        if not os.path.exists(path):
            missing_files.append(path)
    
    stats['missing_files'] = missing_files
    stats['missing_files_count'] = len(missing_files)
    
    return stats


def split_dataset(
    csv_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify: bool = True,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        csv_path: Path to full dataset CSV
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        stratify: Whether to stratify by class
        random_seed: Random seed for reproducibility
        
    Returns:
        train_df, val_df, test_df
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"
    
    df = pd.read_csv(csv_path)
    
    if stratify:
        from sklearn.model_selection import train_test_split
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            stratify=df['class'],
            random_state=random_seed
        )
        
        # Second split: train vs val
        val_size = val_ratio / (train_ratio + val_ratio)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size,
            stratify=train_val_df['class'],
            random_state=random_seed
        )
    else:
        # Random split
        df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        
        n_total = len(df)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_df = df[:n_train]
        val_df = df[n_train:n_train + n_val]
        test_df = df[n_train + n_val:]
    
    # Save splits
    output_dir = os.path.dirname(csv_path)
    train_df.to_csv(os.path.join(output_dir, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_split.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_split.csv'), index=False)
    
    print(f"Dataset split completed:")
    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    # Example usage
    csv_path = "data/descriptions.csv"
    
    # Analyze dataset
    stats = analyze_dataset_statistics(csv_path)
    print("Dataset Statistics:")
    for key, value in stats.items():
        if key != 'missing_files':
            print(f"{key}: {value}")
    
    # Split dataset
    train_df, val_df, test_df = split_dataset(csv_path)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        "data/train_split.csv",
        "data/val_split.csv",
        batch_size=32,
        mode=1
    )
