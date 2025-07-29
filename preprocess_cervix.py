"""
preprocess_cervix.py
Preprocess cervical cell images and text descriptions for CLIP-based training.
"""

import os
import torch
import clip
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle
import argparse
from pathlib import Path
import cv2
from torchvision import transforms

class CervixPreprocessor:
    def __init__(self, mode=1, model_name="ViT-B/32", device="cuda"):
        """
        Initialize preprocessor for cervical cell images.
        
        Args:
            mode: 0 for image only, 1 for image+text
            model_name: CLIP model variant
            device: cuda or cpu
        """
        self.mode = mode
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        
        # Custom preprocessing for medical images
        self.medical_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Initialized preprocessor with mode={mode} on {self.device}")
        
    def preprocess_image(self, image_path):
        """Apply preprocessing to cervical cell images."""
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply CLIP preprocessing
        clip_processed = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Extract CLIP features
        with torch.no_grad():
            image_features = self.model.encode_image(clip_processed)
            image_features = image_features.cpu().numpy().squeeze()
            
        return image_features
    
    def enhance_image(self, image_path):
        """Apply medical image enhancement techniques."""
        # Read image with OpenCV
        img = cv2.imread(str(image_path))
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Good for medical images
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(enhanced)
    
    def preprocess_text(self, text):
        """Preprocess text descriptions using CLIP text encoder."""
        # Tokenize and encode text
        text_tokens = clip.tokenize([text], truncate=True).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features.cpu().numpy().squeeze()
            
        return text_features
    
    def create_multimodal_features(self, image_features, text_features):
        """Combine image and text features for multimodal representation."""
        # Normalize features
        image_features = image_features / np.linalg.norm(image_features)
        text_features = text_features / np.linalg.norm(text_features)
        
        # Concatenate features
        combined_features = np.concatenate([image_features, text_features])
        
        # Alternative: weighted combination
        # weight = 0.7  # Adjust based on modality importance
        # combined_features = weight * image_features + (1 - weight) * text_features
        
        return combined_features
    
    def process_dataset(self, csv_path, output_dir, enhance_images=True):
        """Process entire dataset and save features."""
        # Load descriptions
        df = pd.read_csv(csv_path)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each sample
        processed_data = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
            try:
                # Process image
                if enhance_images:
                    # Enhance medical image first
                    enhanced_img = self.enhance_image(row['image_path'])
                    # Save enhanced image temporarily
                    temp_path = f"/tmp/enhanced_{idx}.png"
                    enhanced_img.save(temp_path)
                    image_features = self.preprocess_image(temp_path)
                    os.remove(temp_path)
                else:
                    image_features = self.preprocess_image(row['image_path'])
                
                if self.mode == 1:
                    # Process text
                    text_features = self.preprocess_text(row['description'])
                    
                    # Combine features
                    features = self.create_multimodal_features(image_features, text_features)
                else:
                    # Image only
                    features = image_features
                
                # Store processed data
                processed_data.append({
                    'features': features,
                    'class': row['class'],
                    'class_label': row['class_label'],
                    'image_path': row['image_path'],
                    'description': row['description'] if self.mode == 1 else None
                })
                
            except Exception as e:
                print(f"Error processing {row['image_path']}: {e}")
                continue
        
        # Save processed features
        output_file = os.path.join(output_dir, f'processed_features_mode{self.mode}.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(processed_data, f)
        
        print(f"Saved {len(processed_data)} processed samples to {output_file}")
        
        # Save feature statistics
        self.save_statistics(processed_data, output_dir)
        
        return processed_data
    
    def save_statistics(self, processed_data, output_dir):
        """Save dataset statistics for analysis."""
        # Extract all features
        features = np.array([item['features'] for item in processed_data])
        labels = np.array([item['class_label'] for item in processed_data])
        
        # Calculate statistics
        stats = {
            'num_samples': len(processed_data),
            'feature_dim': features.shape[1],
            'mode': self.mode,
            'class_distribution': pd.Series(labels).value_counts().to_dict(),
            'feature_mean': features.mean(axis=0),
            'feature_std': features.std(axis=0)
        }
        
        # Save statistics
        stats_file = os.path.join(output_dir, f'dataset_stats_mode{self.mode}.pkl')
        with open(stats_file, 'wb') as f:
            pickle.dump(stats, f)
        
        # Print statistics
        print("\nDataset Statistics:")
        print(f"Total samples: {stats['num_samples']}")
        print(f"Feature dimension: {stats['feature_dim']}")
        print("Class distribution:")
        for class_label, count in stats['class_distribution'].items():
            class_names = {0: 'HSIL', 1: 'LSIL', 2: 'NILM', 3: 'SCC'}
            print(f"  {class_names[class_label]}: {count} samples")

def create_train_val_split(processed_data, val_ratio=0.2, random_seed=42):
    """Create train/validation split maintaining class balance."""
    np.random.seed(random_seed)
    
    # Group by class
    class_groups = {}
    for item in processed_data:
        class_label = item['class_label']
        if class_label not in class_groups:
            class_groups[class_label] = []
        class_groups[class_label].append(item)
    
    train_data = []
    val_data = []
    
    # Split each class
    for class_label, items in class_groups.items():
        np.random.shuffle(items)
        split_idx = int(len(items) * (1 - val_ratio))
        train_data.extend(items[:split_idx])
        val_data.extend(items[split_idx:])
    
    return train_data, val_data

def main():
    parser = argparse.ArgumentParser(description='Preprocess cervical cell dataset')
    parser.add_argument('--mode', type=int, default=1, choices=[0, 1],
                       help='0: image only, 1: image+text')
    parser.add_argument('--csv_path', type=str, default='data/descriptions.csv',
                       help='Path to descriptions CSV')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory for processed features')
    parser.add_argument('--enhance', action='store_true',
                       help='Apply medical image enhancement')
    parser.add_argument('--model', type=str, default='ViT-B/32',
                       help='CLIP model variant')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = CervixPreprocessor(mode=args.mode, model_name=args.model)
    
    # Process dataset
    processed_data = preprocessor.process_dataset(
        args.csv_path, 
        args.output_dir, 
        enhance_images=args.enhance
    )
    
    # Create train/val split
    train_data, val_data = create_train_val_split(processed_data)
    
    # Save splits
    with open(os.path.join(args.output_dir, f'train_data_mode{args.mode}.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    
    with open(os.path.join(args.output_dir, f'val_data_mode{args.mode}.pkl'), 'wb') as f:
        pickle.dump(val_data, f)
    
    print(f"\nTrain samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

if __name__ == "__main__":
    main()
