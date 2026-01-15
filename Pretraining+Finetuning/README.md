# Contrastive Pretraining + Fine-tuning

This folder contains experiments for **SimCLR contrastive pretraining on BigEarthNet-S2** followed by **supervised fine-tuning on EuroSAT**. These experiments explore whether self-supervised pretraining improves downstream classification and OOD detection performance.

## 🎯 Key Finding

**Contrastive pretraining on BigEarthNet-S2 degrades CNN performance** due to domain mismatch between multispectral pretraining data and RGB-only downstream tasks:
- **Custom CNN**: Accuracy drops from 98.13% → 93.74% (−4.39%)
- **ResNet-50**: Accuracy drops from 97.78% → 93.22% (−4.56%)
- **OOD Detection**: AUROC decreases by 0.066-0.096

However, **Vision Transformers benefit from pretraining**:
- **ViT-B/16**: Accuracy improves from 50.31% → 90.98% (+40.67%)

## 📁 Contents

### Phase 1: SimCLR Contrastive Pretraining

These notebooks pretrain encoders on BigEarthNet-S2 using contrastive learning:

1. **`01_simclr_pretrain_custom_cnn_300epochs.ipynb`** ⭐ **RECOMMENDED**
   - Custom CNN pretraining
   - 300 epochs on BigEarthNet-S2 (269,695 samples)
   - Best pretraining configuration
   - Training time: ~40-50 hours on RTX 3090

2. **`02_simclr_pretrain_custom_cnn_50epochs.ipynb`**
   - Faster pretraining (50 epochs)
   - Slightly lower quality representations
   - Training time: ~8-10 hours

3. **`03_simclr_pretrain_resnet50.ipynb`**
   - ResNet-50 encoder pretraining
   - 300 epochs recommended
   - Training time: ~60-70 hours

4. **`04_simclr_pretrain_vit.ipynb`**
   - Vision Transformer pretraining
   - Essential for ViT to achieve competitive performance
   - Training time: ~80-100 hours

### Phase 2: Supervised Fine-tuning

After pretraining, these notebooks fine-tune the pretrained encoders on EuroSAT:

5. **`05_finetune_custom_cnn_after_simclr.ipynb`**
   - Fine-tuning Custom CNN on EuroSAT
   - Progressive unfreezing strategy
   - 100 epochs total fine-tuning

6. **`06_finetune_resnet50_after_simclr.ipynb`**
   - Fine-tuning ResNet-50 on EuroSAT
   - Progressive unfreezing strategy

7. **`07_finetune_vit_after_simclr.ipynb`**
   - Fine-tuning Vision Transformer on EuroSAT
   - Progressive unfreezing strategy

### Model Checkpoints

The `models/` folder contains:
- **Contrastive pretrained models**: `best_contrastive_model_*.pth`
- **Fine-tuned models**: `final_supervised_model_progressive_unfr_*.pth`

Key checkpoints:
- `best_contrastive_model_50Ep_32BS_Loss0_2010_20251215_061439.pth` (50 epochs)
- `best_contrastive_model_50Ep.pth`
- `final_supervised_model_progressive_unfr_acc90.81_20251215_122608.pth` (ViT)
- `final_supervised_model_progressive_unfr_acc93.22_20251220_010103.pth` (ResNet-50)

### Visualizations

The `umap_visualizations/` folder contains:
- UMAP embeddings before and after pretraining
- Comparison of learned representations
- Class separation analysis

## 🚀 Quick Start

### Step 1: SimCLR Pretraining

```bash
# Activate environment
conda activate ood-satellite

# Navigate to this folder
cd Pretraining+Finetuning

# Launch Jupyter
jupyter notebook

# Open and run pretraining notebook:
# 01_simclr_pretrain_custom_cnn_300epochs.ipynb
```

⚠️ **Note**: Pretraining requires BigEarthNet-S2 dataset (~100GB). See [SETUP.md](../SETUP.md) for download instructions.

### Step 2: Fine-tuning

```bash
# After pretraining completes, run fine-tuning:
jupyter notebook "05_finetune_custom_cnn_after_simclr.ipynb"
```

## 🔧 SimCLR Configuration

### Contrastive Learning Setup

```python
# Dataset
DATASET = "BigEarthNet-S2"
SAMPLES = 269_695  # 100% of dataset
EPOCHS = 300  # or 50 for faster training

# Training
BATCH_SIZE = 64
BASE_LEARNING_RATE = 0.001
TEMPERATURE = 0.5  # NT-Xent loss temperature

# Architecture
ENCODER = "Custom CNN" / "ResNet-50" / "ViT-B/16"
PROJECTION_HEAD = [512, 512, 128]  # 2-layer MLP
```

### Data Augmentations

SimCLR uses strong augmentations to create positive pairs:

```python
SimCLRAugmentation = transforms.Compose([
    transforms.RandomResizedCrop(64, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

### NT-Xent Loss

```python
def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    Normalized Temperature-scaled Cross Entropy Loss
    z_i, z_j: Embeddings of positive pairs
    """
    batch_size = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)  # 2N x D
    
    # Cosine similarity
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    sim = sim / temperature
    
    # Mask to remove self-similarity
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim = sim.masked_fill(mask, -9e15)
    
    # Positive pairs are at indices (i, i+N) and (i+N, i)
    pos_sim = torch.cat([sim[i, i+batch_size].unsqueeze(0) 
                         for i in range(batch_size)] +
                        [sim[i+batch_size, i].unsqueeze(0) 
                         for i in range(batch_size)])
    
    # Contrastive loss
    loss = -torch.log(torch.exp(pos_sim) / torch.exp(sim).sum(dim=1))
    return loss.mean()
```

## 🔄 Progressive Unfreezing Strategy

Fine-tuning uses a progressive unfreezing approach to prevent catastrophic forgetting:

### Phase 1: Freeze Encoder (10 epochs)
```python
# Only train classifier head
for param in encoder.parameters():
    param.requires_grad = False
for param in classifier.parameters():
    param.requires_grad = True
```

### Phase 2: Unfreeze Last Layer (10 epochs)
```python
# Unfreeze last encoder block
for param in encoder.layer4.parameters():  # ResNet
    param.requires_grad = True
```

### Phase 3: Full Fine-tuning (80 epochs)
```python
# Unfreeze all parameters
for param in model.parameters():
    param.requires_grad = True
```

**Total Fine-tuning**: 100 epochs

## 📊 Results Summary

### Custom CNN

| Configuration | Val Accuracy | OOD AUROC (MI) | OOD AUROC (MSP) | Training Time |
|--------------|--------------|----------------|-----------------|---------------|
| From Scratch | **98.13%** | **0.9255** | 0.0994 | 4-6 hours |
| Pretrained (50ep) | 92.86% | 0.8145 | 0.0701 | 8h + 2h |
| Pretrained (300ep) | 93.74% | 0.8291 | 0.0745 | 50h + 2h |

**Performance Drop**: −4.39% accuracy, −0.096 AUROC

### ResNet-50

| Configuration | Val Accuracy | OOD AUROC (MI) | OOD AUROC (MSP) | Training Time |
|--------------|--------------|----------------|-----------------|---------------|
| From Scratch | **97.78%** | **0.8954** | 0.0812 | 8-10 hours |
| Pretrained (300ep) | 93.22% | 0.8367 | 0.0689 | 70h + 3h |

**Performance Drop**: −4.56% accuracy, −0.059 AUROC

### Vision Transformer

| Configuration | Val Accuracy | OOD AUROC (MI) | OOD AUROC (MSP) | Training Time |
|--------------|--------------|----------------|-----------------|---------------|
| From Scratch | 50.31% | 0.5231 | 0.5145 | 12-15 hours |
| Pretrained (300ep) | **90.98%** | **0.8725** | 0.0823 | 100h + 4h |

**Performance Gain**: +40.67% accuracy, +0.349 AUROC ⭐

## 💡 Key Insights

### 1. Domain Mismatch Hurts CNNs

**Why does pretraining degrade CNN performance?**

- **Spectral Band Mismatch**: BigEarthNet-S2 uses 12 spectral bands, EuroSAT uses 3 RGB channels
- **Feature Distribution Shift**: Multispectral features don't transfer well to RGB
- **Overfitting to Pretraining**: CNNs learn BigEarthNet-specific features

### 2. Vision Transformers Need Pretraining

**Why do ViTs benefit from pretraining?**

- **Lack of Inductive Bias**: ViTs have no built-in spatial assumptions
- **Data Hungry**: Require large datasets to learn from scratch
- **Transfer Learning**: Pretrained features provide initialization

### 3. Dataset Scale vs. Domain Alignment

Our results suggest **domain alignment is more critical than dataset scale**:
- Large-scale pretraining (269K samples) hurts performance
- Training from scratch on small dataset (27K) achieves better results
- Domain-specific training outweighs quantity of pretraining data

## 🎓 Training Tips

### Pretraining Best Practices

```python
# Use large batch size for contrastive learning
BATCH_SIZE = 64  # Minimum; larger is better (128, 256)

# Strong augmentations
use_color_jitter = True
use_gaussian_blur = True

# Temperature tuning
TEMPERATURE = 0.5  # Lower = harder negatives

# Training epochs
EPOCHS = 300  # More epochs = better representations
```

### Fine-tuning Best Practices

```python
# Lower learning rate than training from scratch
LEARNING_RATE = 0.0001  # 10x smaller

# Progressive unfreezing
use_progressive_unfreezing = True

# Regularization
DROPOUT = 0.1
weight_decay = 0.0001
```

## 📈 Expected Training Time

| Task | Architecture | GPU | Time |
|------|-------------|-----|------|
| Pretrain (50ep) | Custom CNN | RTX 3090 | 8-10 hours |
| Pretrain (300ep) | Custom CNN | RTX 3090 | 40-50 hours |
| Pretrain (300ep) | ResNet-50 | RTX 3090 | 60-70 hours |
| Pretrain (300ep) | ViT-B/16 | RTX 3090 | 80-100 hours |
| Fine-tune | All | RTX 3090 | 2-4 hours |

## 🔍 Reproducing Results

To reproduce pretraining + fine-tuning results:

1. **Download BigEarthNet-S2** (see [SETUP.md](../SETUP.md))
2. **Run pretraining** (300 epochs recommended):
   ```bash
   jupyter notebook "01_simclr_pretrain_custom_cnn_300epochs.ipynb"
   ```
3. **Run fine-tuning**:
   ```bash
   jupyter notebook "05_finetune_custom_cnn_after_simclr.ipynb"
   ```

## ⚠️ Important Notes

1. **BigEarthNet Download**: Pretraining requires ~100GB of storage
2. **Long Training Time**: 300-epoch pretraining takes 2-4 days
3. **Negative Results**: CNNs perform worse after pretraining (expected!)
4. **ViT Success Story**: Only ViTs significantly benefit from this approach

## 📝 Citation

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
- Scratch Training: [../Scratch_training_(No_pretraining)/](../Scratch_training_(No_pretraining)/)
- Pretrained Weights: [../Pretrained_loaded_weight/](../Pretrained_loaded_weight/)
