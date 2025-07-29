"""
clivp_cervix.py
CLIP-based Vision-Language Model for Cervical Cell Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from typing import Optional, Tuple

class CLIVPCervix(nn.Module):
    """
    CLIP-based Vision-Language model for cervical cell classification.
    Combines visual and textual features for improved classification performance.
    """
    
    def __init__(self, 
                 num_classes=4,
                 clip_model="ViT-B/32",
                 feature_dim=512,
                 hidden_dim=256,
                 dropout=0.2,
                 mode=1,
                 freeze_clip=True):
        """
        Initialize CLIVP-Cervix model.
        
        Args:
            num_classes: Number of cervical cell classes (4)
            clip_model: CLIP model variant
            feature_dim: CLIP feature dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
            mode: 0 for image only, 1 for multimodal
            freeze_clip: Whether to freeze CLIP weights
        """
        super(CLIVPCervix, self).__init__()
        
        self.num_classes = num_classes
        self.mode = mode
        self.feature_dim = feature_dim
        
        # Load CLIP model
        self.clip_model, _ = clip.load(clip_model, device="cuda" if torch.cuda.is_available() else "cpu")
        
        # Freeze CLIP if specified
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # Feature projection layers
        if self.mode == 1:  # Multimodal
            # Separate projections for image and text
            self.image_projection = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            )
            
            self.text_projection = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            )
            
            # Cross-modal attention
            self.cross_attention = CrossModalAttention(hidden_dim)
            
            # Fusion layer
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            )
            
            classifier_input_dim = hidden_dim
            
        else:  # Image only
            self.image_projection = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            )
            
            classifier_input_dim = hidden_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Learnable temperature for contrastive loss
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
    def encode_image(self, image):
        """Encode image using CLIP vision encoder."""
        image_features = self.clip_model.encode_image(image)
        return F.normalize(image_features, p=2, dim=-1)
    
    def encode_text(self, text):
        """Encode text using CLIP text encoder."""
        text_features = self.clip_model.encode_text(text)
        return F.normalize(text_features, p=2, dim=-1)
    
    def forward(self, image, text=None, return_features=False):
        """
        Forward pass through the model.
        
        Args:
            image: Preprocessed image tensor
            text: Tokenized text (optional, for mode=1)
            return_features: Whether to return intermediate features
            
        Returns:
            logits: Classification logits
            features: Optional intermediate features
        """
        # Encode image
        image_features = self.encode_image(image)
        projected_image = self.image_projection(image_features)
        
        if self.mode == 1 and text is not None:
            # Encode text
            text_features = self.encode_text(text)
            projected_text = self.text_projection(text_features)
            
            # Apply cross-modal attention
            attended_image, attended_text = self.cross_attention(projected_image, projected_text)
            
            # Fuse features
            fused_features = torch.cat([attended_image, attended_text], dim=-1)
            final_features = self.fusion(fused_features)
            
        else:
            # Image only
            final_features = projected_image
        
        # Classification
        logits = self.classifier(final_features)
        
        if return_features:
            return logits, final_features
        return logits
    
    def get_image_text_similarity(self, image, text):
        """Calculate cosine similarity between image and text embeddings."""
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        similarity = torch.cosine_similarity(image_features, text_features, dim=-1)
        return similarity


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for image-text fusion."""
    
    def __init__(self, hidden_dim, num_heads=8):
        super(CrossModalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head attention layers
        self.image_to_text_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.text_to_image_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, image_features, text_features):
        """
        Apply bidirectional cross-modal attention.
        
        Args:
            image_features: [batch_size, hidden_dim]
            text_features: [batch_size, hidden_dim]
            
        Returns:
            attended_image: Image features attended by text
            attended_text: Text features attended by image
        """
        # Reshape for attention (add sequence dimension)
        image_features = image_features.unsqueeze(1)  # [batch, 1, hidden_dim]
        text_features = text_features.unsqueeze(1)    # [batch, 1, hidden_dim]
        
        # Image attended by text
        attended_image, _ = self.image_to_text_attn(
            query=image_features,
            key=text_features,
            value=text_features
        )
        attended_image = self.norm1(attended_image + image_features)
        
        # Text attended by image
        attended_text, _ = self.text_to_image_attn(
            query=text_features,
            key=image_features,
            value=image_features
        )
        attended_text = self.norm2(attended_text + text_features)
        
        # Remove sequence dimension
        attended_image = attended_image.squeeze(1)
        attended_text = attended_text.squeeze(1)
        
        return attended_image, attended_text


class ContrastiveLoss(nn.Module):
    """Contrastive loss for aligning image and text representations."""
    
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, image_features, text_features):
        """
        Calculate contrastive loss between image and text features.
        
        Args:
            image_features: Normalized image features [batch_size, feature_dim]
            text_features: Normalized text features [batch_size, feature_dim]
            
        Returns:
            loss: Contrastive loss value
        """
        batch_size = image_features.size(0)
        
        # Calculate similarity matrix
        similarity_matrix = torch.matmul(image_features, text_features.T) / self.temperature
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=image_features.device)
        
        # Calculate loss
        loss_i2t = F.cross_entropy(similarity_matrix, labels)
        loss_t2i = F.cross_entropy(similarity_matrix.T, labels)
        
        loss = (loss_i2t + loss_t2i) / 2
        
        return loss


class CLIVPCervixWithContrastive(CLIVPCervix):
    """Extended model with contrastive learning capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_loss = ContrastiveLoss(temperature=0.07)
        
    def forward_contrastive(self, image, text, alpha=0.5):
        """
        Forward pass with contrastive loss.
        
        Args:
            image: Preprocessed image tensor
            text: Tokenized text
            alpha: Weight for contrastive loss
            
        Returns:
            logits: Classification logits
            total_loss: Combined classification and contrastive loss
        """
        # Get features
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        # Get classification logits
        logits = self.forward(image, text)
        
        # Calculate contrastive loss
        contrastive_loss = self.contrastive_loss(image_features, text_features)
        
        return logits, contrastive_loss * alpha


def create_model(num_classes=4, mode=1, pretrained_path=None, **kwargs):
    """
    Factory function to create CLIVP-Cervix model.
    
    Args:
        num_classes: Number of classes
        mode: 0 for image only, 1 for multimodal
        pretrained_path: Path to pretrained weights
        **kwargs: Additional model arguments
        
    Returns:
        model: CLIVP-Cervix model instance
    """
    model = CLIVPCervixWithContrastive(
        num_classes=num_classes,
        mode=mode,
        **kwargs
    )
    
    if pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained weights from {pretrained_path}")
    
    return model
