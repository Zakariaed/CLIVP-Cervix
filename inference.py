"""
inference.py
Inference script for CLIVP-Cervix model
"""

import os
import torch
import clip
import numpy as np
from PIL import Image
import argparse
from typing import Dict, List, Optional
import json

from models.clivp_cervix import create_model


class CervixClassifier:
    """Inference class for CLIVP-Cervix model."""
    
    def __init__(self, 
                 checkpoint_path: str,
                 mode: int = 1,
                 device: str = "cuda"):
        """
        Initialize classifier.
        
        Args:
            checkpoint_path: Path to model checkpoint
            mode: 0 for image only, 1 for multimodal
            device: Device to run on
        """
        self.mode = mode
        self.device = device if torch.cuda.is_available() else "cpu"
        self.class_names = ['HSIL', 'LSIL', 'NILM', 'SCC']
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        # Load CLIP preprocessing
        _, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Default descriptions for each class
        self.default_descriptions = {
            'HSIL': "High-grade dysplastic cervical cells with enlarged hyperchromatic nuclei and high nuclear-cytoplasmic ratio",
            'LSIL': "Low-grade dysplastic cervical cells with mild nuclear enlargement and perinuclear halos",
            'NILM': "Normal cervical epithelial cells with regular nuclear size and normal nuclear-cytoplasmic ratio",
            'SCC': "Malignant squamous cells with marked nuclear pleomorphism and abnormal keratinization"
        }
        
    def _load_model(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract model configuration
        args = checkpoint.get('args', {})
        
        # Create model
        model = create_model(
            num_classes=4,
            mode=self.mode,
            clip_model=args.get('clip_model', 'ViT-B/32'),
            dropout=args.get('dropout', 0.2),
            freeze_clip=True
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def predict_single(self, 
                      image_path: str,
                      text_description: Optional[str] = None) -> Dict:
        """
        Predict class for a single image.
        
        Args:
            image_path: Path to image
            text_description: Optional custom text description
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Prepare text if in multimodal mode
        text_tokens = None
        if self.mode == 1:
            if text_description is None:
                # Use all default descriptions and average
                text_list = list(self.default_descriptions.values())
            else:
                text_list = [text_description]
            
            text_tokens = clip.tokenize(text_list, truncate=True).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            if self.mode == 1 and text_tokens is not None:
                # For multiple descriptions, average the predictions
                all_logits = []
                for i in range(text_tokens.shape[0]):
                    logits = self.model(image_tensor, text_tokens[i:i+1])
                    all_logits.append(logits)
                
                logits = torch.stack(all_logits).mean(dim=0)
            else:
                logits = self.model(image_tensor)
            
            probs = torch.softmax(logits, dim=1).squeeze(0)
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()
        
        # Prepare results
        results = {
            'predicted_class': self.class_names[pred_idx],
            'confidence': confidence,
            'probabilities': {
                class_name: prob.item() 
                for class_name, prob in zip(self.class_names, probs)
            },
            'image_path': image_path,
            'mode': 'multimodal' if self.mode == 1 else 'image_only'
        }
        
        if text_description:
            results['text_description'] = text_description
        
        return results
    
    def predict_batch(self, 
                     image_paths: List[str],
                     text_descriptions: Optional[List[str]] = None) -> List[Dict]:
        """
        Predict classes for multiple images.
        
        Args:
            image_paths: List of image paths
            text_descriptions: Optional list of text descriptions
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            text = text_descriptions[i] if text_descriptions else None
            result = self.predict_single(image_path, text)
            results.append(result)
        
        return results
    
    def predict_with_uncertainty(self, 
                               image_path: str,
                               n_forward: int = 10,
                               dropout_rate: float = 0.2) -> Dict:
        """
        Predict with uncertainty estimation using MC Dropout.
        
        Args:
            image_path: Path to image
            n_forward: Number of forward passes
            dropout_rate: Dropout rate for uncertainty
            
        Returns:
            Dictionary with predictions and uncertainty
        """
        # Enable dropout for uncertainty estimation
        self.model.train()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Multiple forward passes
        all_probs = []
        
        with torch.no_grad():
            for _ in range(n_forward):
                if self.mode == 1:
                    # Use all default descriptions
                    text_list = list(self.default_descriptions.values())
                    text_tokens = clip.tokenize(text_list, truncate=True).to(self.device)
                    
                    # Average over all descriptions
                    logits_list = []
                    for i in range(text_tokens.shape[0]):
                        logits = self.model(image_tensor, text_tokens[i:i+1])
                        logits_list.append(logits)
                    
                    logits = torch.stack(logits_list).mean(dim=0)
                else:
                    logits = self.model(image_tensor)
                
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        # Calculate statistics
        all_probs = np.array(all_probs).squeeze()
        mean_probs = all_probs.mean(axis=0)
        std_probs = all_probs.std(axis=0)
        
        pred_idx = np.argmax(mean_probs)
        
        # Calculate entropy as uncertainty measure
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8))
        
        # Set model back to eval mode
        self.model.eval()
        
        results = {
            'predicted_class': self.class_names[pred_idx],
            'confidence': mean_probs[pred_idx],
            'uncertainty': std_probs[pred_idx],
            'entropy': entropy,
            'mean_probabilities': {
                class_name: prob 
                for class_name, prob in zip(self.class_names, mean_probs)
            },
            'std_probabilities': {
                class_name: std 
                for class_name, std in zip(self.class_names, std_probs)
            },
            'image_path': image_path
        }
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Run inference with CLIVP-Cervix model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image or directory')
    parser.add_argument('--mode', type=int, default=1, choices=[0, 1],
                       help='0: image only, 1: multimodal')
    parser.add_argument('--text', type=str, default=None,
                       help='Custom text description (for mode=1)')
    parser.add_argument('--uncertainty', action='store_true',
                       help='Enable uncertainty estimation')
    parser.add_argument('--batch', action='store_true',
                       help='Process directory of images')
    parser.add_argument('--output', type=str, default='predictions.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = CervixClassifier(
        checkpoint_path=args.checkpoint,
        mode=args.mode
    )
    
    # Process single image or batch
    if args.batch and os.path.isdir(args.image):
        # Get all image files in directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []
        
        for file in os.listdir(args.image):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(args.image, file))
        
        print(f"Found {len(image_paths)} images to process")
        
        # Batch prediction
        results = classifier.predict_batch(image_paths)
        
    else:
        # Single image prediction
        if args.uncertainty:
            results = classifier.predict_with_uncertainty(args.image)
        else:
            results = classifier.predict_single(args.image, args.text)
    
    # Display results
    if isinstance(results, list):
        print(f"\nProcessed {len(results)} images:")
        for result in results:
            print(f"\nImage: {os.path.basename(result['image_path'])}")
            print(f"Predicted: {result['predicted_class']} ({result['confidence']:.3f})")
    else:
        print(f"\nImage: {os.path.basename(results['image_path'])}")
        print(f"Predicted: {results['predicted_class']} ({results['confidence']:.3f})")
        print("\nClass Probabilities:")
        for class_name, prob in results.get('mean_probabilities', results['probabilities']).items():
            if 'uncertainty' in results:
                std = results['std_probabilities'][class_name]
                print(f"  {class_name}: {prob:.3f} ± {std:.3f}")
            else:
                print(f"  {class_name}: {prob:.3f}")
        
        if 'entropy' in results:
            print(f"\nUncertainty (entropy): {results['entropy']:.3f}")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
            '
