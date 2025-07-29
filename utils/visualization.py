"""
visualization.py
Visualization utilities for CLIVP-Cervix
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import pandas as pd
from PIL import Image
import cv2


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """
    Plot training history with loss and accuracy curves.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    epochs = range(1, len(history['train_losses']) + 1)
    ax1.plot(epochs, history['train_losses'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_losses'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_accs'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_accs'], 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Mark best epoch
    best_epoch = history.get('best_epoch', np.argmax(history['val_accs']) + 1)
    best_acc = history.get('best_val_acc', max(history['val_accs']))
    ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
    ax2.text(best_epoch, best_acc, f'Best: {best_acc:.3f}', 
             ha='center', va='bottom', color='g')
    
    plt.suptitle('CLIVP-Cervix Training History', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.show()


def visualize_attention_maps(
    model,
    image_path: str,
    text: str,
    device: str = "cuda",
    save_path: Optional[str] = None
):
    """
    Visualize attention maps from the model.
    
    Args:
        model: CLIVP-Cervix model
        image_path: Path to input image
        text: Text description
        device: Device to run on
        save_path: Path to save visualization
    """
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    _, preprocess = clip.load("ViT-B/32", device=device)
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Tokenize text
    text_tokens = clip.tokenize([text], truncate=True).to(device)
    
    # Get attention weights (this is a simplified version)
    with torch.no_grad():
        # Get image features with attention
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)
        
        # Calculate similarity (as proxy for attention)
        similarity = F.cosine_similarity(image_features, text_features)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Attention visualization (simplified)
    ax2.imshow(image)
    ax2.set_title(f'Similarity Score: {similarity.item():.3f}')
    ax2.axis('off')
    
    # Add text description
    plt.figtext(0.5, 0.02, f"Text: {text[:100]}...", 
                ha='center', fontsize=10, wrap=True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention map saved to: {save_path}")
    
    plt.show()


def plot_class_activation_maps(
    model,
    image_path: str,
    class_idx: int,
    device: str = "cuda",
    save_path: Optional[str] = None
):
    """
    Generate Class Activation Maps (CAM) for visualization.
    
    Args:
        model: CLIVP-Cervix model
        image_path: Path to input image
        class_idx: Target class index
        device: Device to run on
        save_path: Path to save visualization
    """
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # This is a placeholder for actual CAM implementation
    # In practice, you'd need to modify the model to extract intermediate features
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image_rgb)
    ax.set_title(f'Class Activation Map - Class {class_idx}')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_predictions(
    model,
    data_loader,
    num_samples: int = 16,
    device: str = "cuda",
    save_path: Optional[str] = None
):
    """
    Visualize model predictions on sample images.
    
    Args:
        model: CLIVP-Cervix model
        data_loader: DataLoader with samples
        num_samples: Number of samples to visualize
        device: Device to run on
        save_path: Path to save visualization
    """
    model.eval()
    class_names = ['HSIL', 'LSIL', 'NILM', 'SCC']
    
    samples = []
    with torch.no_grad():
        for batch in data_loader:
            if len(samples) >= num_samples:
                break
                
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            texts = batch.get('text', None)
            
            if texts is not None:
                texts = texts.to(device)
            
            outputs = model(images, texts)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            for i in range(min(len(images), num_samples - len(samples))):
                samples.append({
                    'image': images[i].cpu(),
                    'true_label': labels[i].cpu().item(),
                    'pred_label': preds[i].cpu().item(),
                    'probs': probs[i].cpu().numpy(),
                    'image_path': batch.get('image_path', [None] * len(images))[i]
                })
    
    # Create visualization grid
    n_cols = 4
    n_rows = (num_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    
    for idx, sample in enumerate(samples):
        ax = axes[idx]
        
        # Denormalize image for display
        img = sample['image'].permute(1, 2, 0)
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        img = torch.clamp(img, 0, 1)
        
        ax.imshow(img)
        
        true_class = class_names[sample['true_label']]
        pred_class = class_names[sample['pred_label']]
        confidence = sample['probs'][sample['pred_label']]
        
        # Color code based on correctness
        color = 'green' if sample['true_label'] == sample['pred_label'] else 'red'
        
        ax.set_title(f'True: {true_class}\nPred: {pred_class} ({confidence:.2f})',
                    color=color, fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(len(samples), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('CLIVP-Cervix Predictions', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions visualization saved to: {save_path}")
    
    plt.show()


def plot_feature_distribution(
    features: np.ndarray,
    labels: np.ndarray,
    method: str = 'tsne',
    save_path: Optional[str] = None
):
    """
    Plot feature distribution using dimensionality reduction.
    
    Args:
        features: Feature array (n_samples, n_features)
        labels: Label array (n_samples,)
        method: 'tsne' or 'pca'
        save_path: Path to save plot
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    class_names = ['HSIL', 'LSIL', 'NILM', 'SCC']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    # Reduce dimensions
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced_features = reducer.fit_transform(features)
        title = 't-SNE Feature Distribution'
    else:
        reducer = PCA(n_components=2, random_state=42)
        reduced_features = reducer.fit_transform(features)
        title = f'PCA Feature Distribution (Explained Var: {reducer.explained_variance_ratio_.sum():.2f})'
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(class_names):
        mask = labels == i
        plt.scatter(reduced_features[mask, 0], reduced_features[mask, 1],
                   c=colors[i], label=class_name, alpha=0.6, s=50)
    
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature distribution plot saved to: {save_path}")
    
    plt.show()


def create_model_architecture_diagram(save_path: Optional[str] = None):
    """
    Create a diagram of the CLIVP-Cervix model architecture.
    
    Args:
        save_path: Path to save the diagram
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Define components
    components = {
        'Image Input': (2, 7),
        'Text Input': (2, 5),
        'CLIP Image Encoder': (4, 7),
        'CLIP Text Encoder': (4, 5),
        'Image Projection': (6, 7),
        'Text Projection': (6, 5),
        'Cross-Modal Attention': (8, 6),
        'Feature Fusion': (10, 6),
        'Classifier': (12, 6),
        'Output (4 classes)': (14, 6)
    }
    
    # Draw components
    for name, (y, x) in components.items():
        if 'Input' in name:
            rect = plt.Rectangle((x-0.8, y-0.3), 1.6, 0.6, 
                               fill=True, facecolor='lightblue', edgecolor='black')
        elif 'CLIP' in name:
            rect = plt.Rectangle((x-0.8, y-0.3), 1.6, 0.6, 
                               fill=True, facecolor='lightgreen', edgecolor='black')
        elif 'Output' in name:
            rect = plt.Rectangle((x-0.8, y-0.3), 1.6, 0.6, 
                               fill=True, facecolor='lightcoral', edgecolor='black')
        else:
            rect = plt.Rectangle((x-0.8, y-0.3), 1.6, 0.6, 
                               fill=True, facecolor='lightyellow', edgecolor='black')
        
        ax.add_patch(rect)
        ax.text(x, y, name, ha='center', va='center', fontsize=10)
    
    # Draw connections
    connections = [
        ('Image Input', 'CLIP Image Encoder'),
        ('Text Input', 'CLIP Text Encoder'),
        ('CLIP Image Encoder', 'Image Projection'),
        ('CLIP Text Encoder', 'Text Projection'),
        ('Image Projection', 'Cross-Modal Attention'),
        ('Text Projection', 'Cross-Modal Attention'),
        ('Cross-Modal Attention', 'Feature Fusion'),
        ('Feature Fusion', 'Classifier'),
        ('Classifier', 'Output (4 classes)')
    ]
    
    for start, end in connections:
        y1, x1 = components[start]
        y2, x2 = components[end]
        ax.arrow(x1, y1 + 0.3, x2 - x1, y2 - y1 - 0.6, 
                head_width=0.2, head_length=0.2, fc='black', ec='black')
    
    ax.set_xlim(3, 9)
    ax.set_ylim(0, 16)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('CLIVP-Cervix Model Architecture', fontsize=16, pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Architecture diagram saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Visualization utilities for CLIVP-Cervix")
    
    # Create architecture diagram
    create_model_architecture_diagram("model_architecture.png")
