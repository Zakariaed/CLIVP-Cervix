# CLIVP-Cervix: Cervical Cell Classification using Vision Language Models (VLMs)

A CLIP-based PyTorch implementation for cervical cell classification, leveraging vision-language models for improved performance on cytopathology images.

This is the official repository for "LEVERAGING VISION LANGUAGE MODELS FOR CERVICAL CELL CLASSIFICATION IN CYTOPATHOLOGY".

## Overview

CLIVP-Cervix adapts the power of CLIP (Contrastive Language-Image Pretraining) for cervical cell classification, achieving superior performance by combining visual features with rich textual descriptions of cytological characteristics.

## Dataset

The model is trained on the cervix-dataset with 4 classes:
- High squamous intra-epithelial lesion (HSIL)
- Low squamous intra-epithelial lesion (LSIL)
- Negative for Intraepithelial malignancy (NILM)
- Squamous cell carcinoma (SCC)

## Requirements

```bash
torch>=1.9.0
torchvision>=0.10.0
transformers>=4.16.0
clip @ git+https://github.com/openai/CLIP.git
pandas>=1.3.0
numpy>=1.21.0
opencv-python>=4.5.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
tqdm>=4.62.0
pillow>=8.3.0
```

## Installation

```bash
git clone https://github.com/yourusername/CLIVP-Cervix.git
cd CLIVP-Cervix
pip install -r requirements.txt
```

## Dataset Preparation

1. Download the cervix-dataset and place it in the `data/` folder
2. Run the preprocessing scripts:

```bash
# Generate text descriptions for cervical cell images
python formatdescription_cervix.py

# Preprocess images and text data
python preprocess_cervix.py --mode 1
```

### Preprocessing Modes:
- Mode 0: Image features only
- Mode 1: Image and text features (recommended)

## Training

### 10-Fold Cross-Validation Training
```bash
python 10fold_train.py --dataset cervix --mode 1 --epochs 50 --batch_size 32
```

### Single Model Training
```bash
python train.py --dataset cervix --mode 1 --epochs 50 --batch_size 32
```

## Evaluation

Generate confusion matrix and performance metrics:
```bash
python cervix_confmtrx.py --mode 1 --checkpoint best_model.pth
```

## Model Architecture

CLIVP-Cervix uses a dual-stream architecture:
1. **Visual Encoder**: CLIP's Vision Transformer (ViT-B/32) for extracting image features
2. **Text Encoder**: CLIP's Text Transformer for encoding cytological descriptions
3. **Fusion Module**: Combines visual and textual features for final classification

## Text Descriptions

The model uses rich textual descriptions of cytological features, including:
- Cell morphology (size, shape, nuclear-cytoplasmic ratio)
- Nuclear characteristics (chromatin pattern, nuclear membrane)
- Cytoplasmic features
- Architectural patterns

Example descriptions:
- HSIL: "High-grade dysplastic cells with enlarged hyperchromatic nuclei, irregular nuclear membranes, coarse chromatin, and high nuclear-cytoplasmic ratio"
- LSIL: "Low-grade dysplastic cells with mild nuclear enlargement, slightly irregular nuclear contours, and perinuclear halos (koilocytosis)"
- NILM: "Normal squamous epithelial cells with regular nuclear size, smooth nuclear membranes, and normal nuclear-cytoplasmic ratio"
- SCC: "Malignant squamous cells with marked nuclear pleomorphism, irregular chromatin, prominent nucleoli, and abnormal keratinization"

## Performance

Results on cervix-dataset using 10-fold cross-validation:

| Model | Mode | Accuracy | Precision | Recall | F1-Score |
|-------|------|----------|-----------|---------|----------|
| CLIVP-Cervix | Image only | TBD | TBD | TBD | TBD |
| CLIVP-Cervix | Image+Text | TBD | TBD | TBD | TBD |

## Project Structure

```
CLIVP-Cervix/
├── data/
│   ├── cervix-dataset/
│   │   ├── HSIL/
│   │   ├── LSIL/
│   │   ├── NILM/
│   │   └── SCC/
│   └── descriptions.csv
├── models/
│   ├── clivp_cervix.py
│   ├── encoders.py
│   └── fusion.py
├── utils/
│   ├── dataset.py
│   ├── metrics.py
│   └── visualization.py
├── formatdescription_cervix.py
├── preprocess_cervix.py
├── 10fold_train.py
├── train.py
├── cervix_confmtrx.py
├── requirements.txt
└── README.md
```

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{clivp-cervix2024,
  title={Leveraging Vision Language Models for Cervical Cell Classification in Cytopathology},
  author={Your Name},
  booktitle={Conference Name},
  year={2024}
}
```

## Acknowledgments

- This work is inspired by the CLIVP-FER implementation
- CLIP model by OpenAI
- Cervix dataset contributors

## License

MIT License

## Contact

For questions and suggestions, please open an issue or contact [your-email@example.com]
