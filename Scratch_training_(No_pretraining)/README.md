# Training from Scratch (No Pretraining)

This folder contains experiments for training models **from scratch** on the EuroSAT dataset without any pretraining. These experiments demonstrate that lightweight CNNs can achieve excellent performance without contrastive pretraining.

## 🎯 Key Finding

**Custom CNN trained from scratch achieves the best overall performance** across all experiments in the paper:
- **Validation Accuracy**: 98.13%
- **OOD AUROC (Mutual Information)**: 0.9255
- **Parameters**: 1.14M (smallest model)
- **Training Time**: 4-6 hours on RTX 3090

## 📁 Contents

### Notebooks

1. **`01_train_custom_cnn_from_scratch.ipynb`** ⭐ **RECOMMENDED**
   - Custom CNN architecture designed for satellite imagery
   - Best performing model in the entire study
   - Lightweight: Only 1.14M parameters
   - Training: 100 epochs, batch size 1024, dropout 0.1
   - Results: 98.13% accuracy, 0.9255 AUROC for OOD

2. **`02_train_resnet50_from_scratch.ipynb`**
   - ResNet-50 architecture trained from scratch
   - Good performance but slightly lower than custom CNN
   - Parameters: 23.5M
   - Results: 97.78% accuracy, 0.8954 AUROC for OOD

3. **`03_train_vit_from_scratch.ipynb`**
   - Vision Transformer (ViT-B/16) trained from scratch
   - ⚠️ **Poor performance** - demonstrates ViTs need pretraining
   - Parameters: 85.8M
   - Results: 50.31% accuracy (near random for 10 classes)

### Model Checkpoints

The `models/` folder contains saved model checkpoints:
- `final_supervised_model_progressive_unfr_acc97.02_*.pth`
- `final_supervised_model_progressive_unfr_acc97.76_*.pth`
- `final_supervised_model_progressive_unfr_acc98.07_*.pth` ⭐ Best model

### Visualizations

The `umap_visualizations/` folder contains:
- UMAP embeddings of learned representations
- Class separation visualizations
- Uncertainty distribution plots

## 🚀 Quick Start

```bash
# Activate your environment
conda activate ood-satellite

# Navigate to this folder
cd "Scratch_training_(No_pretraining)"

# Launch Jupyter
jupyter notebook

# Open and run: 01_train_custom_cnn_from_scratch.ipynb
```

## 🔧 Architecture Details

### Custom CNN Architecture

```
Input (64x64x3)
  ↓
Conv2d(3→64, 3x3) + BatchNorm + ReLU + MaxPool
  ↓
Conv2d(64→128, 3x3) + BatchNorm + ReLU + MaxPool
  ↓
Conv2d(128→256, 3x3) + BatchNorm + ReLU + MaxPool
  ↓
Conv2d(256→512, 3x3) + BatchNorm + ReLU + MaxPool
  ↓
Global Average Pooling
  ↓
Dropout(0.1)
  ↓
Linear(512→10)
```

**Total Parameters**: 1,146,890

### Training Configuration

```python
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
OPTIMIZER = Adam
SCHEDULER = ReduceLROnPlateau
EPOCHS = 100
DROPOUT_RATE = 0.1
```

### Data Augmentation

```python
RandomHorizontalFlip(p=0.5)
RandomVerticalFlip(p=0.5)
RandomRotation(degrees=15)
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

## 📊 Results Summary

| Model | Val Acc | OOD AUROC (MI) | OOD AUROC (Entropy) | OOD AUROC (MSP) | Params |
|-------|---------|----------------|---------------------|-----------------|--------|
| **Custom CNN** | **98.13%** | **0.9255** | 0.8891 | 0.0994 | 1.14M |
| ResNet-50 | 97.78% | 0.8954 | 0.8642 | 0.0812 | 23.5M |
| ViT-B/16 | 50.31% | 0.5231 | 0.5145 | 0.5145 | 85.8M |

### Uncertainty Quantification Methods

For each model, we evaluate OOD detection using:

1. **Mutual Information (MI)**: `H[y|x] - E[H[y|x,θ]]` ⭐ Best performer
2. **Predictive Entropy**: `-Σ p(y|x) log p(y|x)`
3. **Predictive Variance**: `Var[p(y|x)]`
4. **Maximum Softmax Probability**: `max(p(y|x))` (inverted for OOD)

Monte Carlo Dropout: **T=30** stochastic forward passes

## 💡 Key Insights

1. **Lightweight CNNs Outperform Heavy Models**
   - Custom CNN with 1.14M params beats ResNet-50 (23.5M) and ViT (85.8M)
   - Simpler architecture is more parameter-efficient

2. **Vision Transformers Need Pretraining**
   - ViT achieves only 50.31% accuracy from scratch (near random)
   - Demonstrates inductive bias importance for small datasets

3. **Mutual Information for OOD Detection**
   - MI consistently outperforms other uncertainty metrics
   - MSP (deterministic) fails for OOD detection (0.0994 AUROC)

4. **Domain-Specific Architectures**
   - Custom CNN designed for satellite imagery performs best
   - Standard architectures may not be optimal

## 🎓 Training Tips

### For Best Results (Custom CNN)

```python
# Use large batch size for stable training
BATCH_SIZE = 1024

# Moderate learning rate with scheduler
LEARNING_RATE = 0.001
scheduler = ReduceLROnPlateau(patience=5, factor=0.5)

# Light dropout for regularization
DROPOUT_RATE = 0.1

# Training epochs
EPOCHS = 100  # Usually converges around 80-90 epochs
```

### Memory Optimization

If you encounter GPU memory issues:

```python
# Option 1: Reduce batch size
BATCH_SIZE = 512  # or 256

# Option 2: Enable gradient accumulation
ACCUMULATION_STEPS = 2
effective_batch_size = BATCH_SIZE * ACCUMULATION_STEPS
```

## 📈 Expected Training Time

| Model | GPU | Training Time | Peak Memory |
|-------|-----|---------------|-------------|
| Custom CNN | RTX 3090 | 4-6 hours | 8GB |
| ResNet-50 | RTX 3090 | 8-10 hours | 12GB |
| ViT-B/16 | RTX 3090 | 12-15 hours | 16GB |

## 🔍 Reproducing Results

To reproduce the paper's best results:

1. Ensure datasets are properly setup (see main [SETUP.md](../SETUP.md))
2. Open `01_train_custom_cnn_from_scratch.ipynb`
3. Run all cells sequentially
4. Models will be saved in `models/` folder
5. UMAP visualizations will be generated in `umap_visualizations/`

## 📝 Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{anonymous2026contrastive,
  title={Contrastive Self-Supervised Learning for Out-of-Distribution Detection in Satellite Imagery: When Simpler is Better},
  author={Anonymous Authors},
  booktitle={WACV},
  year={2026}
}
```

## 🔗 Related Files

- Main README: [../README.md](../README.md)
- Setup Guide: [../SETUP.md](../SETUP.md)
- Pretraining Experiments: [../Pretraining+Finetuning/](../Pretraining+Finetuning/)
- Pretrained Weights: [../Pretrained_loaded_weight/](../Pretrained_loaded_weight/)
