# PyTorch Model Architectures & Templates

A curated collection of clean, production-ready PyTorch implementations for modern machine learning. This repository contains model architectures, data loading templates, and training utilities designed as drop-in building blocks for computer vision, natural language processing, and multimodal learning tasks.

## Directory Structure

```
arch/
├── vit/              # Vision Transformer architectures
├── llava/            # Large Language and Vision Assistant (multimodal)
├── dataops/          # PyTorch Dataset templates
└── README.md         # This file
```

---

## 📁 Directories

### 1. **Vision Transformers (`vit/`)**

Pure PyTorch implementations of Vision Transformer (ViT) architectures with multiple position encoding strategies.

**Files:**
- `vitb.py` - ViT-B (Base) with learnable positional embeddings
- `vitl.py` - ViT-L (Large) with learnable positional embeddings
- `vitb_rope.py` - ViT-B with Rotary Position Embeddings (RoPE)
- `vitl_rope.py` - ViT-L with Rotary Position Embeddings (RoPE)

**Key Features:**
- 4 architecture variants for different use cases
- Learnable vs. rotary position embeddings
- Complete transformer blocks with attention and FFN
- Initialization strategies for stable training
- Configurable patch sizes, embedding dimensions, and depths

**When to Use:**
- Image classification tasks
- Feature extraction for downstream vision tasks
- Vision encoder in multimodal models (e.g., LLaVA)
- Experimentation with transformer architectures

**Quick Start:**
```python
from vit.vitb import ViTB

model = ViTB(
    img_size=224,
    patch_size=16,
    num_classes=1000,
    embed_dim=768,
    depth=12,
    n_heads=12
)

x = torch.randn(8, 3, 224, 224)
logits = model(x)  # (8, 1000)
```

📖 [Detailed Documentation](vit/doc.md)

---

### 2. **LLaVA: Large Language and Vision Assistant (`llava/`)**

Multimodal architecture combining vision encoders with language models for image-to-text understanding and generation.

**Components:**
- `VisionProjection` - Projects vision features to language space
- `MultimodalEmbedding` - Fuses text and image embeddings
- `LLaVA` - Complete multimodal model
- Helper functions for loading HuggingFace models

**Key Features:**
- Vision encoder integration (CLIP)
- Language model backbone (LLaMA, Mistral, etc.)
- Autoregressive generation with temperature and nucleus sampling
- Support for pretrained models from HuggingFace Hub
- Optional quantization (4-bit, 8-bit) for memory efficiency

**When to Use:**
- Image understanding and visual question answering
- Image captioning
- Vision-language instruction tuning
- Multimodal conversational AI

**Quick Start:**
```python
from llava.llava import LLaVA
from llava.llava import load_huggingface_vision_model
from llava.llava import load_huggingface_language_model

vision_model = load_huggingface_vision_model("openai/clip-vit-large-patch14")
language_model = load_huggingface_language_model("mistralai/Mistral-7B-Instruct-v0.1")

model = LLaVA(
    vision_encoder=vision_model,
    language_model=language_model,
    vision_hidden_size=1024,
    language_hidden_size=4096,
    vocab_size=32000
)

images = torch.randn(1, 3, 224, 224)
input_ids = torch.randint(0, 32000, (1, 128))
outputs = model(input_ids=input_ids, images=images)
```

📖 [Detailed Documentation](llava/doc.md)

---

### 3. **Data Loading Templates (`dataops/`)**

Production-ready PyTorch Dataset implementations for common data modalities.

**Files:**
- `vision_dataset.py` - VisionDataset for image classification
- `language_dataset.py` - LanguageDataset for NLP tasks
- `vl_dataset.py` - VisionLanguageDataset for multimodal tasks

**Key Features:**
- Support for multiple data formats (folder hierarchies, CSV, JSON, JSONL)
- Configurable preprocessing and transforms
- Task-specific outputs (classification, language modeling, masked LM)
- Optional in-memory caching
- Smoke tests with synthetic data (no real files required)

**When to Use:**
- Image classification with flexible data loading
- Text classification, language modeling, masked LM
- Image captioning, instruction tuning, multimodal tasks
- As templates for custom dataset implementations

**Quick Start:**
```python
from dataops.vision_dataset import VisionDataset
from dataops.language_dataset import LanguageDataset
from dataops.vl_dataset import VisionLanguageDataset

# Image classification
ds_vision = VisionDataset(root="data/imagenet", mode="train")
img, label = ds_vision[0]  # (C,H,W) tensor, scalar label

# Language modeling
ds_lang = LanguageDataset(
    data=texts,
    task="causal_lm",
    tokenizer=my_tokenizer
)
batch = ds_lang[0]  # Dict with input_ids, attention_mask, labels

# Vision-language instruction tuning
ds_vl = VisionLanguageDataset(
    data_path="conversations.jsonl",
    format="llava",
    tokenizer=my_tokenizer
)
batch = ds_vl[0]  # Dict with pixel_values, input_ids, labels
```

📖 [Detailed Documentation](dataops/doc.md)

---

## 🚀 Quick Examples

### Example 1: Image Classification Pipeline

```python
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from vit.vitb import ViTB
from dataops.vision_dataset import VisionDataset

# 1. Create dataset
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
ds = VisionDataset(root="data/imagenet", mode="train", train_transform=train_transform)

# 2. Create dataloader
loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)

# 3. Create model
model = ViTB(img_size=224, patch_size=16, num_classes=ds.num_classes)

# 4. Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    for images, labels in loader:
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Example 2: Vision-Language Instruction Tuning

```python
import torch
from torch.utils.data import DataLoader
from vl_dataset import VisionLanguageDataset
from llava.llava import LLaVA, load_huggingface_vision_model, load_huggingface_language_model

# 1. Create dataset
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
ds = VisionLanguageDataset(
    data_path="instructions.jsonl",
    format="llava",
    tokenizer=tokenizer,
    image_train_transform=train_aug
)

# 2. Create dataloader
loader = DataLoader(ds, batch_size=8, shuffle=True)

# 3. Create model
vision_model = load_huggingface_vision_model("openai/clip-vit-large-patch14")
language_model = load_huggingface_language_model(
    "mistralai/Mistral-7B-Instruct-v0.1",
    load_in_4bit=True  # Memory efficient
)
model = LLaVA(vision_encoder=vision_model, language_model=language_model, ...)

# 4. Training loop
for batch in loader:
    outputs = model(
        input_ids=batch["input_ids"],
        images=batch["pixel_values"],
        labels=batch["labels"]
    )
    loss = outputs.loss
    loss.backward()
    optimizer.step()
```

### Example 3: Text Classification

```python
from transformers import AutoTokenizer
from dataops.language_dataset import LanguageDataset
from torch.utils.data import DataLoader

# 1. Create dataset
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
texts = ["positive review...", "negative review..."]
labels = [1, 0]

ds = LanguageDataset(
    data=texts,
    task="classification",
    labels=labels,
    tokenizer=tokenizer,
    max_length=256
)

# 2. Create dataloader
loader = DataLoader(ds, batch_size=32)

# 3. Forward pass
for batch in loader:
    # batch: {"input_ids": Tensor, "attention_mask": Tensor, "labels": Tensor}
    logits = classifier_model(batch["input_ids"], batch["attention_mask"])
    loss = criterion(logits, batch["labels"])
```

---

## 📋 Feature Comparison

| Feature | ViT | LLaVA | DataOps |
|---------|-----|-------|---------|
| **Type** | Vision Encoder | Multimodal | Data Loading |
| **Input** | Images | Images + Text | Various (folder, CSV, JSON) |
| **Output** | Image Features | Text | Dict/Tuple of Tensors |
| **Pretrained** | Ready for ONNX export | HuggingFace integration | N/A (template) |
| **Trainable** | Yes | Vision frozen, LM trainable | N/A |
| **Use Case** | Feature extraction, classification | Image-to-text understanding | Data preprocessing |

---

## 🛠️ Installation & Setup

### Basic Requirements

```bash
pip install torch torchvision
```

### For LLaVA (Multimodal)

```bash
pip install transformers  # For HuggingFace models
pip install Pillow        # For image loading
pip install bitsandbytes  # For 4-bit/8-bit quantization (optional)
```

### For DataOps (Data Loading)

```bash
pip install Pillow  # For image loading
# For language tasks, install your preferred tokenizer:
pip install transformers  # HuggingFace tokenizers
```

### Full Development Setup

```bash
pip install torch torchvision transformers Pillow bitsandbytes
```

---

## 🧪 Testing

Each module includes smoke tests with synthetic data (no real files required):

```bash
# Test Vision Transformers
python vit/vitb.py

# Test LLaVA
python llava/llava.py

# Test Dataset Templates
python dataops/vision_dataset.py
python dataops/language_dataset.py
python dataops/vl_dataset.py
```

All tests skip gracefully if dependencies (PyTorch, Pillow) aren't installed.

---

## 📚 Code Style & Conventions

All code in this repository follows consistent patterns:

- **Docstrings:** Google-style with type hints
- **Type Hints:** Full typing from `typing` module
- **Tensor Shapes:** Inline comments showing dimensions
- **Deterministic:** Sorted loading for reproducibility
- **Configurable:** Sensible defaults with extensive optional parameters
- **Self-Contained:** Each file works independently

Example pattern:
```python
def forward(
    self,
    input_ids: torch.Tensor,
    images: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Forward pass description.

    Args:
        input_ids: [batch_size, seq_len] - text token IDs
        images: [batch_size, 3, height, width] - optional images

    Returns:
        Dict with keys: logits, loss (if labels provided)
    """
    # Implementation with inline shape comments
    # x: (B, N, D)
```

---

## 🔧 Common Tasks

### Loading a Pretrained ViT Model

```python
from vit.vitb import ViTB
import torch

model = ViTB(img_size=224, patch_size=16, num_classes=1000)
checkpoint = torch.load("vitb_imagenet.pt")
model.load_state_dict(checkpoint)
```

### Using LLaVA with Different Language Models

```python
from llava.llava import LLaVA, load_huggingface_language_model

# Mistral-7B (recommended)
lm = load_huggingface_language_model("mistralai/Mistral-7B-Instruct-v0.1")

# Or Llama-2
lm = load_huggingface_language_model("meta-llama/Llama-2-7b-hf")

# Or Falcon
lm = load_huggingface_language_model("tiiuae/falcon-7b")
```

### Building Custom Datasets

Use `dataops` templates as base classes:

```python
from dataops.vision_dataset import VisionDataset

class CustomImageDataset(VisionDataset):
    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        # Custom post-processing
        return {"image": img, "label": label, "metadata": {...}}
```

---

## 📖 Documentation

- **[Vision Transformers](vit/doc.md)** - Detailed ViT architecture guide with position embedding comparison
- **[LLaVA](llava/doc.md)** - Multimodal architecture, training, and real-world examples
- **[DataOps](dataops/doc.md)** - Dataset templates, formats, and usage patterns

---

## 🤝 Contributing

This repository is organized for clarity and reusability. When adding new components:

1. Follow the existing code style (Google docstrings, type hints)
2. Include smoke tests with synthetic data in `__main__`
3. Add comprehensive module docstrings
4. Document tensor shapes inline
5. Update relevant `doc.md` files

---

## 📝 License

Check individual files for license information.

---

## 🔗 Related Resources

- **PyTorch:** https://pytorch.org
- **Vision Transformer Paper:** https://arxiv.org/abs/2010.11929
- **RoPE (Rotary Position Embeddings):** https://arxiv.org/abs/2104.09864
- **LLaVA Paper:** https://arxiv.org/abs/2304.08485
- **CLIP:** https://github.com/openai/CLIP
- **HuggingFace Hub:** https://huggingface.co

---

**Last Updated:** 2026-03-12

A comprehensive toolkit for modern PyTorch modeling and training.
