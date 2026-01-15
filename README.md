# Contrastive Self-Supervised Learning for OOD Detection in Satellite Imagery

[![WACV 2026](https://img.shields.io/badge/WACV-2026-blue)](https://wacv2026.thecvf.com/)
[![Workshop](https://img.shields.io/badge/Workshop-GeoCV-green)](https://www.grss-ieee.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Official implementation of "Contrastive Self-Supervised Learning for Out-of-Distribution Detection in Satellite Imagery: When Simpler is Better"**

Accepted at WACV 2026 GeoCV Workshop

## 🎯 Overview

This repository contains the code and models for our comprehensive study on out-of-distribution (OOD) detection in satellite imagery. We compare contrastive self-supervised pretraining (SimCLR on BigEarthNet-S2) versus training from scratch across three encoder architectures for satellite image classification and OOD detection using Monte Carlo Dropout.

### Key Findings

- 🔥 **Lightweight CNNs trained from scratch outperform contrastively pretrained models** - achieving 98.13% accuracy and 0.9255 AUROC for OOD detection
- 📊 **Contrastive pretraining can degrade CNN performance** by 4.39-10.43% accuracy due to domain mismatch
- 🚀 **Vision Transformers require pretraining** - failing catastrophically from scratch (50.31% accuracy) but achieving competitive performance when pretrained (90.98% accuracy)
- 🎓 **Mutual Information outperforms other uncertainty metrics** for OOD detection (0.9255 vs 0.0994 AUROC for MSP)

## 📂 Repository Structure

```
.
├── README.md                              # This file
├── requirements.txt                       # Python dependencies (pip)
├── environment.yml                        # Conda environment file
├── LICENSE                                # MIT License
├── .gitignore                             # Git ignore patterns
│
├── Scratch_training_(No_pretraining)/    # Training from scratch (⭐ BEST RESULTS)
│   ├── README.md                         # Folder-specific documentation
│   ├── 01_train_custom_cnn_from_scratch.ipynb      # Custom CNN (98.13% acc)
│   ├── 02_train_resnet50_from_scratch.ipynb        # ResNet-50
│   ├── 03_train_vit_from_scratch.ipynb             # ViT (fails without pretraining)
│   ├── best_supervised_model_no_pretrain.pth
│   ├── models/                           # Trained model checkpoints
│   └── umap_visualizations/              # UMAP embedding visualizations
│
├── Pretraining+Finetuning/               # SimCLR pretraining + fine-tuning
│   ├── # Phase 1: SimCLR Pretraining
│   ├── 01_simclr_pretrain_custom_cnn_300epochs.ipynb
│   ├── 02_simclr_pretrain_custom_cnn_50epochs.ipynb
│   ├── 03_simclr_pretrain_resnet50.ipynb
│   ├── 04_simclr_pretrain_vit.ipynb
│   │
│   ├── # Phase 2: Fine-tuning
│   ├── 05_finetune_custom_cnn_after_simclr.ipynb
│   ├── 06_finetune_resnet50_after_simclr.ipynb
│   ├── 07_finetune_vit_after_simclr.ipynb
│   │
│   ├── best_supervised_model.pth
│   ├── models/                           # Pretrained and fine-tuned checkpoints
│   └── umap_visualizations/
│
└── Pretrained loaded weight/             # ImageNet pretrained weights
    ├── 01_finetune_densenet201_imagenet.ipynb
    ├── 02_finetune_efficientnet_b4_imagenet.ipynb
    ├── 03_finetune_resnet50_imagenet.ipynb
    ├── 04_finetune_vit_b16_imagenet.ipynb
    ├── best_supervised_model.pth
    ├── models/                           # Fine-tuned model checkpoints
    └── umap_visualizations/
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ood-detection-satellite-imagery.git
cd ood-detection-satellite-imagery
```

### 2. Setup Environment

```bash
# Create conda environment
conda create -n ood-satellite python=3.9
conda activate ood-satellite

# Install dependencies
pip install -r requirements.txt
```

See [SETUP.md](SETUP.md) for detailed installation instructions.

### 3. Download Datasets

**EuroSAT (In-Distribution):**
```bash
wget http://madm.dfki.de/files/sentinel/EuroSAT.zip
unzip EuroSAT.zip
```

**UC Merced (Out-of-Distribution):**
```bash
wget http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip
unzip UCMerced_LandUse.zip
```

**BigEarthNet-S2 (Pretraining - Optional):**
- Visit: https://bigearth.net/
- Download the Sentinel-2 dataset

### 4. Run Experiments

#### Option A: Training from Scratch (Best Performance)

```bash
# Custom CNN (Recommended - Best Results!)
jupyter notebook "Scratch_training_(No_pretraining)/01_train_custom_cnn_from_scratch.ipynb"

# ResNet-50
jupyter notebook "Scratch_training_(No_pretraining)/02_train_resnet50_from_scratch.ipynb"

# Vision Transformer (Note: Poor performance from scratch)
jupyter notebook "Scratch_training_(No_pretraining)/03_train_vit_from_scratch.ipynb"
```

#### Option B: Contrastive Pretraining + Fine-tuning

**Step 1: SimCLR Pretraining on BigEarthNet**
```bash
# Custom CNN (300 epochs)
jupyter notebook "Pretraining+Finetuning/01_simclr_pretrain_custom_cnn_300epochs.ipynb"

# ResNet-50
jupyter notebook "Pretraining+Finetuning/03_simclr_pretrain_resnet50.ipynb"

# Vision Transformer
jupyter notebook "Pretraining+Finetuning/04_simclr_pretrain_vit.ipynb"
```

**Step 2: Fine-tuning on EuroSAT**
```bash
# Custom CNN
jupyter notebook "Pretraining+Finetuning/05_finetune_custom_cnn_after_simclr.ipynb"

# ResNet-50
jupyter notebook "Pretraining+Finetuning/06_finetune_resnet50_after_simclr.ipynb"

# Vision Transformer
jupyter notebook "Pretraining+Finetuning/07_finetune_vit_after_simclr.ipynb"
```

#### Option C: Using Existing Pretrained Weights

```bash
# DenseNet-201
jupyter notebook "Pretrained loaded weight/01_finetune_densenet201_imagenet.ipynb"

# EfficientNet-B4
jupyter notebook "Pretrained loaded weight/02_finetune_efficientnet_b4_imagenet.ipynb"

# ResNet-50
jupyter notebook "Pretrained loaded weight/03_finetune_resnet50_imagenet.ipynb"

# Vision Transformer B-16
jupyter notebook "Pretrained loaded weight/04_finetune_vit_b16_imagenet.ipynb"
```


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Datasets**: EuroSAT, UC Merced, BigEarthNet-S2
- **Frameworks**: PyTorch, torchvision, scikit-learn
- **Inspiration**: SimCLR framework by Chen et al.

## 📧 Contact

For questions or collaboration opportunities, please open an issue or contact the authors.

## 🔗 Links

- [Paper PDF](paper/MCD_OOD.pdf)
- [WACV 2026 Workshop](https://wacv2026.thecvf.com/)
- [GeoCV Workshop](https://www.grss-ieee.org/)

---

