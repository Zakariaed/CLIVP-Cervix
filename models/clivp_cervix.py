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
