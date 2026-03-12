# ViT with Pretrained Weights

This guide explains how to use the pretrained Vision Transformer models (ViT-B and ViT-L) with ImageNet weights.

## Installation

To use pretrained models, you need to install the `timm` library:

```bash
pip install timm
```

## Quick Start

### Load a pretrained model

```python
from vit import vitb, vitl

# ViT-Base with ImageNet-1k pretraining
model_b = vitb(pretrained=True, num_classes=1000)

# ViT-Large with ImageNet-1k pretraining
model_l = vitl(pretrained=True, num_classes=1000)
```

### Use for inference

```python
import torch
from vit import vitb

model = vitb(pretrained=True)
model.eval()

# Create sample input
x = torch.randn(1, 3, 224, 224)

# Forward pass
with torch.no_grad():
    output = model(x)

print(output.shape)  # torch.Size([1, 1000])
```

### Fine-tune on custom dataset

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from vit import vitb

# Load pretrained model for custom task
model = vitb(pretrained=True, num_classes=10)  # 10 classes for custom task
model.train()

# The classification head is automatically initialized for 10 classes
optimizer = AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop (simplified)
for images, labels in dataloader:
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

## Available Models

### ViT-Base (ViT-B)

- **Model Size**: ~86M parameters
- **Compute**: ~17.6 GFLOPs (224×224)
- **Use Cases**: Most common choice, good balance of accuracy and speed

#### Functions

1. **`vitb(pretrained=False, num_classes=1000, **kwargs)`**
   - ImageNet-1k pretrained weights
   - Best for general-purpose vision tasks

2. **`vitb_21k(pretrained=False, num_classes=21843, **kwargs)`**
   - ImageNet-21k pretrained weights
   - Better transfer learning for downstream tasks
   - Default output: 21,843 classes (ImageNet-21k)

### ViT-Large (ViT-L)

- **Model Size**: ~304M parameters (3.5× larger)
- **Compute**: ~61.6 GFLOPs (224×224)
- **Use Cases**: Large-scale datasets, transfer learning where accuracy is critical

#### Functions

1. **`vitl(pretrained=False, num_classes=1000, **kwargs)`**
   - ImageNet-1k pretrained weights
   - Best for general-purpose vision tasks with higher capacity

2. **`vitl_21k(pretrained=False, num_classes=21843, **kwargs)`**
   - ImageNet-21k pretrained weights
   - Recommended for transfer learning
   - Default output: 21,843 classes (ImageNet-21k)

## Examples

### Example 1: Classification with pretrained ViT-B

```python
import torch
from vit import vitb
from torchvision import transforms
from PIL import Image

# Load model
model = vitb(pretrained=True)
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image = Image.open('image.jpg')
x = transform(image).unsqueeze(0)

# Inference
with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=-1)
    pred_class = probs.argmax(dim=-1)

print(f"Predicted class: {pred_class.item()}")
print(f"Confidence: {probs.max().item():.2%}")
```

### Example 2: Feature extraction with ViT-L

```python
import torch
from vit import vitl

# Load pretrained model
model = vitl(pretrained=True)
model.eval()

# Remove classification head to use as feature extractor
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

# Extract features
x = torch.randn(8, 3, 224, 224)
with torch.no_grad():
    features = feature_extractor(x)  # (8, 1024) - embedding dimension

print(features.shape)  # torch.Size([8, 1024])
```

### Example 3: Transfer learning with ImageNet-21k pretraining

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from vit import vitl_21k

# Load ImageNet-21k pretrained weights
# Better starting point for transfer learning
model = vitl_21k(pretrained=True, num_classes=100)

# Freeze backbone, only train head
for param in model.parameters():
    param.requires_grad = False

# Only train the classification head
model.head.requires_grad = True

optimizer = AdamW(model.head.parameters(), lr=1e-3)

# Fine-tune on custom dataset
for epoch in range(10):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

## Customization

### Custom architecture parameters

```python
from vit import vitb

# Create pretrained model with custom patch size
model = vitb(
    pretrained=True,
    num_classes=1000,
    patch_size=14,  # Default is 16
    drop_rate=0.1,  # Add dropout
    attn_drop_rate=0.1  # Attention dropout
)
```

### Model creation without pretraining

```python
from vit import vitb, vitl

# Create scratch models (random initialization)
model_b = vitb(pretrained=False)
model_l = vitl(pretrained=False)

# This is also the default
model_b = vitb()  # Same as vitb(pretrained=False)
```

## Performance Notes

### ImageNet-1k Results

| Model | Top-1 Acc | Top-5 Acc | FLOPs | Params |
|-------|-----------|-----------|-------|--------|
| ViT-B/16 | 81.8% | 95.6% | 17.6G | 86M |
| ViT-L/16 | 85.2% | 97.6% | 61.6G | 304M |

### ImageNet-21k Pretraining Benefits

When using `vitb_21k()` or `vitl_21k()`, models are pretrained on the larger ImageNet-21k dataset (14M images, 21,843 classes). This typically provides:
- +2-3% accuracy improvement on ImageNet-1k when fine-tuned
- Better transfer learning performance on downstream tasks
- More robust feature representations

## Troubleshooting

### ImportError: timm not found

If you get an error about timm not being installed:
```bash
pip install timm
```

### CUDA out of memory

If you run out of VRAM with ViT-L:
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training

```python
from torch.cuda.amp import autocast

with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)
```

## References

- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929) - Dosovitskiy et al., ICLR 2021
- [ViT GitHub](https://github.com/google-research/vision_transformer)
- [timm Documentation](https://timm.fast.ai/)

