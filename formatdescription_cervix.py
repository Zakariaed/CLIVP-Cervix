"""
formatdescription_cervix.py
Generate textual descriptions for cervical cell images based on their class labels.
This script creates rich, medically-informed descriptions for each image class.
"""

import os
import pandas as pd
from pathlib import Path
import json

# Define rich textual descriptions for each cervical cell class
CLASS_DESCRIPTIONS = {
    "HSIL": {
        "full_name": "High squamous intra-epithelial lesion",
        "primary_description": "High-grade dysplastic cervical cells showing severe abnormalities with enlarged hyperchromatic nuclei, irregular nuclear membranes, coarse chromatin pattern, and significantly increased nuclear-cytoplasmic ratio",
        "cellular_features": [
            "markedly enlarged nuclei",
            "hyperchromatic nuclear staining",
            "irregular nuclear membranes",
            "coarse and clumped chromatin",
            "high nuclear-cytoplasmic ratio",
            "nuclear crowding and overlapping",
            "loss of cellular polarity"
        ],
        "clinical_significance": "pre-cancerous lesion requiring immediate clinical attention",
        "cytological_grade": "high-grade dysplasia"
    },
    "LSIL": {
        "full_name": "Low squamous intra-epithelial lesion",
        "primary_description": "Low-grade dysplastic cervical cells with mild nuclear abnormalities including slight nuclear enlargement, irregular nuclear contours, hyperchromasia, and characteristic perinuclear halos indicating HPV cytopathic effect",
        "cellular_features": [
            "mild nuclear enlargement",
            "slightly irregular nuclear contours",
            "fine chromatin pattern",
            "perinuclear halos (koilocytosis)",
            "binucleation or multinucleation",
            "mild hyperchromasia",
            "preserved nuclear-cytoplasmic ratio"
        ],
        "clinical_significance": "low-grade pre-cancerous changes often associated with HPV infection",
        "cytological_grade": "low-grade dysplasia"
    },
    "NILM": {
        "full_name": "Negative for Intraepithelial malignancy",
        "primary_description": "Normal cervical squamous epithelial cells with regular nuclear size and shape, smooth nuclear membranes, fine evenly distributed chromatin, and normal nuclear-cytoplasmic ratio without any dysplastic features",
        "cellular_features": [
            "regular nuclear size",
            "smooth nuclear membranes",
            "fine, evenly distributed chromatin",
            "normal nuclear-cytoplasmic ratio",
            "well-defined cell borders",
            "transparent cytoplasm",
            "normal cellular maturation pattern"
        ],
        "clinical_significance": "benign cellular changes with no evidence of dysplasia or malignancy",
        "cytological_grade": "normal cytology"
    },
    "SCC": {
        "full_name": "Squamous cell carcinoma",
        "primary_description": "Malignant squamous cells displaying severe nuclear abnormalities with marked pleomorphism, irregular chromatin distribution, prominent nucleoli, abnormal mitoses, and evidence of keratinization in invasive carcinoma",
        "cellular_features": [
            "marked nuclear pleomorphism",
            "irregular chromatin clumping",
            "prominent nucleoli",
            "abnormal mitotic figures",
            "tumor diathesis background",
            "keratinizing or non-keratinizing patterns",
            "bizarre cell shapes"
        ],
        "clinical_significance": "invasive malignancy requiring immediate oncological intervention",
        "cytological_grade": "malignant"
    }
}

def generate_varied_descriptions(class_name, num_variations=5):
    """Generate multiple varied descriptions for a single class to enhance training diversity."""
    base_info = CLASS_DESCRIPTIONS[class_name]
    variations = []
    
    # Template variations
    templates = [
        "Cytological image showing {full_name}: {primary_description}",
        "Cervical cytology displaying {cytological_grade} with {feature_list}",
        "Microscopic view of {full_name} characterized by {feature_subset}",
        "{clinical_significance} presenting as cells with {feature_list}",
        "Pap smear showing {cytological_grade}: {short_features} in cervical epithelial cells"
    ]
    
    for i in range(num_variations):
        template = templates[i % len(templates)]
        
        # Create feature subsets
        features = base_info["cellular_features"]
        if i % 2 == 0:
            feature_list = ", ".join(features[:4])
        else:
            feature_list = ", ".join(features[2:6] if len(features) > 5 else features)
        
        feature_subset = " and ".join(features[1:3])
        short_features = ", ".join(features[:2])
        
        description = template.format(
            full_name=base_info["full_name"],
            primary_description=base_info["primary_description"],
            cytological_grade=base_info["cytological_grade"],
            clinical_significance=base_info["clinical_significance"],
            feature_list=feature_list,
            feature_subset=feature_subset,
            short_features=short_features
        )
        
        variations.append(description)
    
    return variations

def create_description_csv(data_dir="data/cervix-dataset", output_file="data/descriptions.csv"):
    """Create CSV file with image paths and their corresponding text descriptions."""
    
    data = []
    class_dirs = ["HSIL", "LSIL", "NILM", "SCC"]
    
    for class_name in class_dirs:
        class_path = Path(data_dir) / class_name
        
        if not class_path.exists():
            print(f"Warning: Directory {class_path} not found. Skipping...")
            continue
            
        # Get all image files
        image_files = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png")) + list(class_path.glob("*.jpeg"))
        
        print(f"Processing {len(image_files)} images from class: {class_name}")
        
        # Generate varied descriptions
        descriptions = generate_varied_descriptions(class_name, num_variations=10)
        
        for idx, img_file in enumerate(image_files):
            # Cycle through description variations
            description = descriptions[idx % len(descriptions)]
            
            data.append({
                "image_path": str(img_file),
                "class": class_name,
                "class_label": class_dirs.index(class_name),
                "description": description,
                "full_class_name": CLASS_DESCRIPTIONS[class_name]["full_name"]
            })
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"\nDescription CSV created successfully!")
    print(f"Total images processed: {len(df)}")
    print(f"Class distribution:")
    print(df['class'].value_counts())
    
    # Also save the class descriptions as JSON for reference
    with open("data/class_descriptions.json", "w") as f:
        json.dump(CLASS_DESCRIPTIONS, f, indent=2)
    
    return df

def create_augmented_descriptions(df, augmentation_factor=3):
    """Create additional augmented text descriptions for better training."""
    augmented_data = []
    
    for _, row in df.iterrows():
        # Original description
        augmented_data.append(row.to_dict())
        
        # Create augmented versions
        class_name = row['class']
        base_info = CLASS_DESCRIPTIONS[class_name]
        
        for i in range(augmentation_factor - 1):
            aug_row = row.to_dict().copy()
            
            # Create different description styles
            if i == 0:
                # Technical description
                aug_row['description'] = f"Cytopathological examination reveals {base_info['full_name']} " \
                                       f"with the following features: {', '.join(base_info['cellular_features'][:3])}"
            else:
                # Clinical description
                aug_row['description'] = f"Clinical cytology showing {base_info['clinical_significance']} " \
                                       f"manifesting as {base_info['cytological_grade']} with characteristic " \
                                       f"{', '.join(base_info['cellular_features'][3:5])}"
            
            augmented_data.append(aug_row)
    
    augmented_df = pd.DataFrame(augmented_data)
    augmented_df.to_csv("data/descriptions_augmented.csv", index=False)
    
    return augmented_df

if __name__ == "__main__":
    # Generate main descriptions
    df = create_description_csv()
    
    # Generate augmented descriptions
    aug_df = create_augmented_descriptions(df)
    print(f"\nAugmented dataset created with {len(aug_df)} samples")
    
    # Display sample descriptions
    print("\nSample descriptions for each class:")
    for class_name in ["HSIL", "LSIL", "NILM", "SCC"]:
        sample = df[df['class'] == class_name].iloc[0]
        print(f"\n{class_name}:")
        print(f"Description: {sample['description'][:200]}...")
