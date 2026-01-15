# Fine-tuning with Pretrained Weights

This folder contains experiments for fine-tuning various architectures using **publicly available pretrained weights** (typically ImageNet-pretrained) on the EuroSAT satellite imagery dataset. This explores transfer learning from natural images to satellite imagery.

## 🎯 Overview

Instead of pretraining from scratch on BigEarthNet-S2, these experiments leverage existing pretrained models:
- **DenseNet-201** (ImageNet pretrained)
- **EfficientNet-B4** (ImageNet pretrained)
- **ResNet-50** (ImageNet pretrained)
- **Vision Transformer B-16** (ImageNet-21K pretrained)

## 📁 Contents

### Notebooks

1. **`01_finetune_densenet201_imagenet.ipynb`**
   - DenseNet-201 fine-tuning on EuroSAT
   - Progressive unfreezing strategy
   - Parameters: ~20M

2. **`02_finetune_efficientnet_b4_imagenet.ipynb`**
   - EfficientNet-B4 fine-tuning on EuroSAT
   - Efficient architecture with compound scaling
   - Parameters: ~19M

3. **`03_finetune_resnet50_imagenet.ipynb`**
   - ResNet-50 fine-tuning with ImageNet weights
   - Comparison with scratch and SimCLR pretraining
   - Parameters: 23.5M

4. **`04_finetune_vit_b16_imagenet.ipynb`**
   - Vision Transformer B-16 fine-tuning
   - Pretrained on ImageNet-21K
   - Parameters: 85.8M

### Model Checkpoints

The `models/` folder contains fine-tuned model checkpoints:
- `final_supervised_model_progressive_unfr_acc97.02_*.pth`
- `final_supervised_model_progressive_unfr_acc97.76_*.pth`
- `final_supervised_model_progressive_unfr_acc98.07_*.pth`

### Visualizations

The `umap_visualizations/` folder contains:
- UMAP embeddings of learned features
- Transfer learning analysis
- Feature space comparisons

## 🚀 Quick Start

```bash
# Activate environment
conda activate ood-satellite

# Navigate to this folder
cd "Pretrained_loaded_weight"

# Launch Jupyter
jupyter notebook

# Open any notebook and run:
# - 01_finetune_densenet201_imagenet.ipynb
# - 02_finetune_efficientnet_b4_imagenet.ipynb
# - 03_finetune_resnet50_imagenet.ipynb
# - 04_finetune_vit_b16_imagenet.ipynb
```

## 🔧 Configuration

### Transfer Learning Setup

All notebooks use ImageNet pretrained weights loaded via torchvision or timm:

```python
import torchvision.models as models
import timm

# DenseNet-201
model = models.densenet201(pretrained=True)

# EfficientNet-B4
model = timm.create_model('efficientnet_b4', pretrained=True)

# ResNet-50
model = models.resnet50(pretrained=True)

# Vision Transformer B-16
model = timm.create_model('vit_base_patch16_224', pretrained=True)
```

### Progressive Unfreezing Strategy

Same 3-phase approach as SimCLR fine-tuning:

**Phase 1: Freeze Encoder (10 epochs)**
```python
for param in model.parameters():
    param.requires_grad = False
# Only train classifier head
```

**Phase 2: Partial Unfreezing (10 epochs)**
```python
# Unfreeze last block/layer
for param in model.layer4.parameters():  # ResNet
    param.requires_grad = True
```

**Phase 3: Full Fine-tuning (80 epochs)**
```python
# Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True
```

### Training Hyperparameters

```python
BATCH_SIZE = 1024
LEARNING_RATE = 0.0001  # Lower for pretrained weights
OPTIMIZER = Adam
SCHEDULER = ReduceLROnPlateau
EPOCHS = 100  # Total (10 + 10 + 80)
DROPOUT_RATE = 0.1
```

## 📊 Results Summary

### Performance on EuroSAT

| Architecture | Pretrained On | Val Accuracy | OOD AUROC (MI) | Parameters | Training Time |
|-------------|---------------|--------------|----------------|------------|---------------|
| DenseNet-201 | ImageNet | ~96.5% | ~0.86 | 20M | 3-4 hours |
| EfficientNet-B4 | ImageNet | ~97.2% | ~0.88 | 19M | 3-4 hours |
| ResNet-50 | ImageNet | ~96.8% | ~0.87 | 23.5M | 3-4 hours |
| ViT-B/16 | ImageNet-21K | ~95.8% | ~0.85 | 85.8M | 4-5 hours |

### Comparison with Other Approaches

| Approach | Best Architecture | Val Accuracy | OOD AUROC (MI) |
|----------|------------------|--------------|----------------|
| **Scratch Training** | Custom CNN | **98.13%** | **0.9255** |
| ImageNet Pretrained | EfficientNet-B4 | 97.2% | 0.88 |
| SimCLR Pretrained | Custom CNN | 93.74% | 0.8291 |

**Key Observation**: ImageNet pretraining provides middle-ground performance—better than SimCLR pretraining but not as good as scratch training for custom CNNs.

## 💡 Key Insights

### 1. Natural Image Transfer Works Reasonably Well

ImageNet pretrained models achieve competitive performance (~96-97% accuracy), demonstrating that:
- Natural image features transfer to satellite imagery
- Better domain alignment than BigEarthNet multispectral pretraining
- RGB-to-RGB transfer is more effective than multispectral-to-RGB

### 2. Still Underperforms Scratch-Trained Custom CNN

Despite using massive pretraining (ImageNet: 1.2M images), these models don't beat the lightweight custom CNN trained from scratch:
- Custom CNN: 98.13% (scratch)
- Best pretrained: 97.2% (EfficientNet-B4)

**Why?**
- Custom CNN is domain-specific for satellite imagery
- ImageNet features optimized for different task (natural objects)
- Compact architecture better suited for 64×64 images

### 3. Architecture Matters More Than Pretraining Source

EfficientNet-B4 (19M params) outperforms ViT-B/16 (85.8M params) despite:
- Similar pretraining quality
- 4.5× more parameters in ViT
- ViT pretrained on larger ImageNet-21K

Suggests that **CNN inductive bias** is beneficial for satellite imagery.

### 4. Efficient Transfer Learning

ImageNet pretraining offers practical advantages:
- **Fast**: No need for lengthy pretraining (300 epochs)
- **Accessible**: Pretrained weights readily available
- **Good baseline**: Achieves ~97% accuracy in 3-4 hours

## 🔬 Methodology

### Monte Carlo Dropout for Uncertainty

All models use MC Dropout with T=30 samples:

```python
def mc_dropout_inference(model, x, num_samples=30):
    model.train()  # Enable dropout
    predictions = []
    
    for _ in range(num_samples):
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            predictions.append(probs)
    
    predictions = torch.stack(predictions)  # T x B x C
    return predictions

# Uncertainty metrics
mean_pred = predictions.mean(dim=0)
entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-10), dim=1)
mutual_info = entropy - (-torch.sum(predictions * torch.log(predictions + 1e-10), dim=2)).mean(dim=0)
```

### Data Augmentation

```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.Resize(224),  # Some models need 224×224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

## 🎓 Training Tips

### Best Practices for Transfer Learning

```python
# 1. Lower learning rate for pretrained weights
LEARNING_RATE = 0.0001  # vs 0.001 for scratch

# 2. Use progressive unfreezing
use_progressive_unfreezing = True

# 3. Smaller weight decay
WEIGHT_DECAY = 0.0001

# 4. Dropout for uncertainty
DROPOUT = 0.1

# 5. Layer-specific learning rates (optional)
optimizer = Adam([
    {'params': model.features.parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-4}
])
```

### Memory Optimization

```python
# Reduce batch size for large models (ViT)
if model_name == 'vit':
    BATCH_SIZE = 512
else:
    BATCH_SIZE = 1024

# Enable mixed precision training
use_amp = True
scaler = torch.cuda.amp.GradScaler()
```

## 📈 Expected Training Time

| Architecture | GPU | Training Time | Peak Memory |
|-------------|-----|---------------|-------------|
| DenseNet-201 | RTX 3090 | 3-4 hours | 10GB |
| EfficientNet-B4 | RTX 3090 | 3-4 hours | 12GB |
| ResNet-50 | RTX 3090 | 3-4 hours | 12GB |
| ViT-B/16 | RTX 3090 | 4-5 hours | 16GB |

## 🔍 Reproducing Results

To reproduce ImageNet transfer learning results:

1. **Ensure datasets are setup** (see [SETUP.md](../SETUP.md))
2. **Choose an architecture**:
   ```bash
   # Example: EfficientNet-B4
   jupyter notebook "02_finetune_efficientnet_b4_imagenet.ipynb"
   ```
3. **Run all cells** - pretrained weights download automatically
4. **Check results** in `models/` and `umap_visualizations/`

## ⚙️ Customization

### Adding New Architectures

To add a new pretrained architecture:

```python
import timm

# List available models
models = timm.list_models(pretrained=True)

# Create model
model = timm.create_model('your_model_name', 
                          pretrained=True,
                          num_classes=10)  # EuroSAT classes

# Add dropout
model = nn.Sequential(
    model,
    nn.Dropout(0.1)
)
```

### Adjusting Input Size

Different architectures expect different input sizes:

```python
# ResNet, DenseNet: 224×224
transforms.Resize(224)

# EfficientNet-B4: 380×380
transforms.Resize(380)

# ViT: 224×224
transforms.Resize(224)
```

## 📊 Architecture Comparison

### Model Efficiency

| Architecture | Accuracy | Params | FLOPs | Inference Time |
|-------------|----------|--------|-------|----------------|
| EfficientNet-B4 | 97.2% | 19M | 4.2B | 12ms |
| DenseNet-201 | 96.5% | 20M | 4.3B | 15ms |
| ResNet-50 | 96.8% | 23.5M | 4.1B | 10ms |
| ViT-B/16 | 95.8% | 85.8M | 17.6B | 25ms |

EfficientNet-B4 offers the **best accuracy-efficiency trade-off**.

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
- SimCLR Pretraining: [../Pretraining+Finetuning/](../Pretraining+Finetuning/)

## 🙏 Acknowledgments

Pretrained weights provided by:
- **torchvision**: ResNet, DenseNet
- **timm** (PyTorch Image Models): EfficientNet, Vision Transformers
- **ImageNet**: Original pretraining dataset
