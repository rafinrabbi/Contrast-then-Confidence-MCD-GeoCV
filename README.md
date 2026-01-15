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
├── SETUP.md                               # Detailed setup instructions
├── requirements.txt                       # Python dependencies
├── LICENSE                                # License information
├── paper/                                 # Paper and documentation
│   ├── MCD_OOD.pdf                       # Final paper PDF
│   ├── COMPREHENSIVE_PUBLICATION_REPORT.md
│   └── WACV_RESULTS_SECTION.tex
│
├── Scratch_training_(No_pretraining)/    # Training from scratch experiments
│   ├── README.md                         # Folder-specific documentation
│   ├── D0.1_BS1024-scratch_training_custom_CNN.ipynb
│   ├── D0.1_BS1024-scratch_training_resnet50.ipynb
│   ├── D0.1_BS1024-scratch_training_ViTs.ipynb
│   ├── models/                           # Trained model checkpoints
│   └── umap_visualizations/              # UMAP embedding visualizations
│
├── Pretraining+Finetuning/               # SimCLR pretraining + fine-tuning
│   ├── README.md                         # Folder-specific documentation
│   ├── Final_Only_Pretrain100_D0.1_BS64_newaugm_300epoch.ipynb
│   ├── Final_Only_Pretrain100_D0.1_BS64_newaugm_50epoch.ipynb
│   ├── Final_Only_Pretrain100_D0.1_BS64_newaugm_restnet_enc.ipynb
│   ├── Final_Only_Pretrain100_D0.1_BS64_newaugm_Vit_SIMCLR.ipynb
│   ├── D0.1_BS1024 finetune-progessive-unf-v.20 gpsug.ipynb
│   ├── D0.1_BS1024 finetune-progessive-unf-v.20 gpsug_restnet_enc.ipynb
│   ├── D0.1_BS1024 finetune-progessive-unf-v.20 gpsug_vit_enc.ipynb
│   ├── models/                           # Pretrained and fine-tuned checkpoints
│   └── umap_visualizations/
│
└── Pretrained_loaded_weight/             # Using pre-existing pretrained weights
    ├── README.md                         # Folder-specific documentation
    ├── finetune-progessive-unf-v.20 gpsug_densenet201.ipynb
    ├── finetune-progessive-unf-v.20 gpsug_Efficientnet-b4.ipynb
    ├── finetune-progessive-unf-v.20 gpsug_restnet_enc.ipynb
    ├── finetune-progessive-unf-v.20 gpsug_ViT-B16.ipynb
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

## 📊 Results Summary

| Configuration | Architecture | Val Accuracy | OOD AUROC (MI) | OOD AUROC (MSP) | Parameters |
|--------------|--------------|--------------|----------------|-----------------|------------|
| **Scratch (Best)** | Custom CNN | **98.13%** | **0.9255** | 0.0994 | 1.14M |
| Scratch | ResNet-50 | 97.78% | 0.8954 | 0.0812 | 23.5M |
| Scratch | ViT-B/16 | 50.31% | 0.5231 | 0.5145 | 85.8M |
| Pretrained | Custom CNN | 93.74% | 0.8291 | 0.0745 | 1.14M |
| Pretrained | ResNet-50 | 93.22% | 0.8367 | 0.0689 | 23.5M |
| Pretrained | ViT-B/16 | 90.98% | 0.8725 | 0.0823 | 85.8M |

**Key Observations:**
- Mutual Information (MI) consistently outperforms Maximum Softmax Probability (MSP)
- Custom CNN trained from scratch achieves best overall performance
- Vision Transformers require pretraining to achieve competitive results
- Contrastive pretraining degrades CNN performance due to domain mismatch

## 🔬 Methodology

### Monte Carlo Dropout for Uncertainty Quantification

We use MC Dropout to estimate epistemic uncertainty by performing T=30 stochastic forward passes at test time:

- **Mutual Information (MI)**: `H[y|x] - E[H[y|x,θ]]`
- **Predictive Entropy**: `-Σ p(y|x) log p(y|x)`
- **Predictive Variance**: `Var[p(y|x)]`
- **Maximum Softmax Probability**: `max(p(y|x))`

### SimCLR Contrastive Pretraining

- Dataset: BigEarthNet-S2 (269,695 samples)
- Epochs: 50, 300
- Batch Size: 64
- Temperature: 0.5
- Augmentations: Random crops, color jitter, Gaussian blur

### Progressive Unfreezing Fine-tuning

1. Train classifier head (10 epochs)
2. Unfreeze last encoder layer (10 epochs)
3. Unfreeze all encoder layers (80 epochs)
4. Total: 100 fine-tuning epochs

## 📖 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{anonymous2026contrastive,
  title={Contrastive Self-Supervised Learning for Out-of-Distribution Detection in Satellite Imagery: When Simpler is Better},
  author={Anonymous Authors},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
  year={2026}
}
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

**Note**: Model checkpoints are stored in the respective `models/` folders within each experiment directory. Download links for pretrained models will be provided upon paper acceptance.
