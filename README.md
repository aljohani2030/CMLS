Copy the content below and save as README.md
markdown
# CMLS: Ensemble Deep Learning Model with Attention-Based Feature Fusion for Deepfake Facial Image Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Paper](https://img.shields.io/badge/Paper-The_Visual_Computer-orange)

> Official implementation of the paper published in **The Visual Computer (2026)**

---

## 📖 Overview

This repository contains the official implementation of **CMLS (Collaborative Mutual Learning Supervision)** for deepfake face detection. The proposed model combines multiple deep learning architectures (CNN, MobileNetV2, LSTM) with attention-based feature fusion to achieve state-of-the-art detection performance.

### Key Results

| Dataset | Accuracy |
|---------|----------|
| Celeb-DF | **98.10%** |
| FaceForensics++ | **95.50%** |

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU training)
- 2× NVIDIA Tesla T4 GPUs (recommended)

### Steps

```bash
# Clone repository
git clone https://github.com/[your-username]/CMLS-Deepfake-Detection.git
cd CMLS-Deepfake-Detection

# Create virtual environment
conda create -n cmls python=3.9
conda activate cmls

# Install dependencies
pip install -r requirements.txt
Requirements
Create requirements.txt:

torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
pillow>=10.0.0

Directory Structure
text
./data/
├── Celeb-DF/
│   ├── train/
│   │   ├── real/
│   │   └── fake/
│   ├── val/
│   └── test/
└── FaceForensics++/
    ├── train/
    ├── val/
    └── test/

Data Split (as per manuscript)
Training: 60%
Validation: 10%
Testing: 30%

🚀 Usage
Configuration
Edit config.py:

python
class Config:
    # Paths
    CELEB_DF_PATH = './data/Celeb-DF'
    FFPP_PATH = './data/FaceForensics++'
    
    # Training
    IMG_SIZE = 224
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    DROPOUT = 0.2
    
    # Hardware
    DEVICE = 'cuda'  # or 'cpu'
Training
bash
# Train on Celeb-DF
python train.py --dataset celeb_df --epochs 50 --batch_size 32

# Train on FaceForensics++
python train.py --dataset ffpp --epochs 50 --batch_size 32
Evaluation
bash
# Evaluate on Celeb-DF
python evaluate.py --model_path ./checkpoints/best_model.pth --dataset celeb_df

# Evaluate on FaceForensics++
python evaluate.py --model_path ./checkpoints/best_model.pth --dataset ffpp

📊 Results
Table 3: Classification Performance
FaceForensics++

Model	Accuracy	Precision	Recall	F1	AUC
CNN	88.30%	87.50%	88.00%	87.70%	0.88
MobileNetV2	85.60%	84.80%	85.20%	85.00%	0.85
LSTM	83.50%	83.00%	82.50%	82.70%	0.83
Proposed Ensemble	95.50%	95.10%	95.00%	95.00%	0.95
Celeb-DF

Model	Accuracy	Precision	Recall	F1	AUC
CNN	96.40%	96.20%	96.00%	96.10%	0.96
MobileNetV2	94.70%	94.50%	94.30%	94.40%	0.95
LSTM	91.40%	91.00%	90.80%	90.90%	0.91
Proposed Ensemble	98.10%	98.00%	97.50%	97.70%	0.96
Table 4 & 5: Ablation Studies
Exp	Components	FF++ Acc	Celeb-DF Acc
1	BottleNeck only	78.2%	83.2%
2	+ CNN	81.5%	88.5%
3	+ DenseBlock	84.1%	79.7%
4	+ Channel Attention	86.3%	89.5%
5	+ Feature Fusion	88.7%	91.2%
6	Full Ensemble	95.5%	98.1%
Table 6: Dataset Size Impact
Dataset	Data Fraction	Accuracy	F1	AUC
FaceForensics++	25%	72.4%	0.71	0.78
50%	79.2%	0.78	0.85
75%	86.1%	0.86	0.86
100%	95.5%	0.93	0.95
Celeb-DF	25%	85.0%	0.84	0.85
50%	90.0%	0.89	0.91
75%	96.1%	0.96	0.94
100%	98.1%	0.98	0.96
📁 Project Structure
text
CMLS-Deepfake-Detection/
├── config.py                 # Configuration settings
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── generate_all_results.py   # Generate all manuscript results
├── visualize.py              # Visualization utilities
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── CITATION.bib              # Citation format
├── LICENSE                   # MIT License
├── models/
│   ├── __init__.py
│   ├── cnn_branch.py
│   ├── transformer_branch.py
│   ├── mamba_branch.py
│   ├── attention_fusion.py
│   └── mutual_learning_loss.py
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   └── preprocess.py
└── checkpoints/              # Saved models
