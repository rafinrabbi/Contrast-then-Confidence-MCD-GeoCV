# Contrastive Self-Supervised Learning for OOD Detection in Satellite Imagery

[![WACV 2026](https://img.shields.io/badge/WACV-2026-blue)](https://wacv2026.thecvf.com/)
[![Workshop](https://img.shields.io/badge/Workshop-GeoCV-green)](https://www.grss-ieee.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Official implementation of "Contrastive Self-Supervised Learning for Out-of-Distribution Detection in Satellite Imagery: When Simpler is Better"**

Accepted at WACV 2026 GeoCV Workshop  
📄 *Paper will be available in the WACV 2026 proceedings*

## 📄 Abstract

Reliable deployment of Earth Observation (EO) image classifiers requires not only high in-distribution accuracy but also the ability to detect inputs that deviate from the training distribution. While recent work has explored self-supervised representation learning and Bayesian uncertainty estimation independently, their interaction in EO settings remains poorly understood. In this work, we propose **Contrast then Confidence (C2)**, a three-stage framework that combines contrastive pretraining with uncertainty-aware out-of-distribution (OOD) detection via Monte Carlo (MC) Dropout. The framework first learns task-agnostic representations from large-scale unlabeled EO imagery using SimCLR-style contrastive learning, then transfers these representations to a supervised land-use classification task with progressive fine-tuning, and finally estimates epistemic uncertainty at inference time using MC Dropout. We conduct a comprehensive evaluation across convolutional and transformer-based architectures using EuroSAT as the in-distribution dataset and UC Merced as the OOD benchmark. Our results reveal strong architectural dependencies: lightweight convolutional networks trained from scratch achieve superior uncertainty calibration, reaching up to 0.93 AUROC for OOD detection, while Vision Transformers fail to generalize without large-scale pretraining but improve substantially under contrastive pretraining, achieving 0.87 AUROC. Across all architectures and training regimes, mutual information derived from MC Dropout consistently outperforms predictive entropy, variance, and maximum softmax probability, highlighting the limitations of deterministic confidence for EO OOD detection. These findings demonstrate that reliable EO systems require jointly considering architecture choice, pretraining strategy, and principled uncertainty estimation.


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

- [WACV 2026 Workshop](https://wacv.thecvf.com/)
- [GeoCV Workshop](https://sites.google.com/view/geocv/home?authuser=0)

---

