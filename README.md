# PyTorch Model Architectures & Templates

A curated collection of clean, production-ready PyTorch implementations for modern machine learning. This repository contains model architectures, data loading templates, and training utilities designed as drop-in building blocks for computer vision, natural language processing, and multimodal learning tasks.

## Directory Structure

```
arch/
├── vit/              # Vision Transformer architectures
├── clip/             # CLIP: Contrastive Language-Image Pre-training
├── llava/            # Large Language and Vision Assistant (multimodal)
├── dataops/          # PyTorch Dataset templates
├── train/            # Training utilities and examples
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
- Architecture variants for different use cases
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

### 2. **CLIP: Contrastive Language-Image Pre-training (`clip/`)**

Foundation model that learns visual representations from natural language supervision by aligning image and text embeddings in a shared space.

**Files:**
- `clip.py` - CLIP model implementation with vision and text encoders
- `doc.md` - Comprehensive CLIP documentation

**Key Features:**
- Vision Encoder: Patch-based Vision Transformer for images
- Text Encoder: GPT-style transformer with causal attention masking
- Contrastive Learning: InfoNCE loss for image-text alignment
- L2 Normalization: Converts to cosine similarity in shared embedding space
- Learnable Temperature: Balances match difficulty during training
- HuggingFace Integration: Load pretrained OpenAI CLIP models

**Components:**
- `ResidualAttentionBlock` - Pre-LayerNorm transformer block
- `VisionEncoder` - Self-contained ViT for images (49 patches + class token)
- `TextEncoder` - GPT-style transformer with context_length=77
- `CLIP` - Main model combining both encoders with projections

**When to Use:**
- Zero-shot image classification
- Image-text similarity scoring
- Cross-modal retrieval (find images matching text)
- Vision encoder for multimodal models (e.g., LLaVA)
- Foundation model for downstream vision-language tasks

**Quick Start:**
```python
from clip.clip import CLIP
import torch

# Create model
model = CLIP(embed_dim=512, vision_width=768, transformer_width=512)
model.eval()

# Encode images and text
images = torch.randn(8, 3, 224, 224)
input_ids = torch.randint(0, 49408, (8, 77))

with torch.no_grad():
    image_features = model.encode_image(images)      # (8, 512), normalized
    text_features = model.encode_text(input_ids)     # (8, 512), normalized

    # Compute similarity scores
    similarities = image_features @ text_features.T  # (8, 8)
```

**Training Example:**
```python
# Full forward pass with loss
outputs = model(images, input_ids)
loss = outputs["loss"]

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss.backward()
optimizer.step()
```

📖 [Detailed Documentation](clip/doc.md)

---

### 3. **LLaVA: Large Language and Vision Assistant (`llava/`)**

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

### 4. **Data Loading Templates (`dataops/`)**

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

### 5. **Training Utilities (`train/`)**

Production-ready training loop template, model evaluation, and visualization tools.

**Files:**
- `train_template.py` - Reusable training loop with W&B integration
- `gradcam.py` - Attention map visualization using GradCAM
- `gradcam_example.py` - Complete example: load model, generate and visualize attention
- `vit-exp.py` - Example training and inference script

**Key Features:**
- Automatic device detection (CUDA, MPS, CPU)
- Weights & Biases integration for experiment tracking
- Learning rate scheduling (cosine annealing with warmup)
- Gradient accumulation and mixed precision training
- Checkpoint management and resumption
- Early stopping based on validation metrics
- Configurable via `TrainingConfig` dataclass

**Components:**
- `TrainingConfig` - Dataclass with all hyperparameters (reproducible, JSON serializable)
- `Trainer` - Main training class with full training pipeline

**When to Use:**
- Training any PyTorch model with standard supervised learning
- Organizing experiments and comparing runs (via W&B)
- Implementing best practices without starting from scratch
- Monitoring training progress in real-time

**Quick Start:**
```python
from train.train_template import TrainingConfig, Trainer
from vit.vitb import ViTB
from torch.utils.data import DataLoader

# Configure training
config = TrainingConfig(
    project_name="my_project",
    experiment_name="vitb_baseline",
    epochs=100,
    batch_size=32,
    learning_rate=1e-4,
    use_wandb=True,  # Track in Weights & Biases
)

# Create model and trainer
model = ViTB(num_classes=1000)
trainer = Trainer(model, config)

# Train!
results = trainer.train(train_loader, val_loader)
```

**Configuration Example:**
```python
config = TrainingConfig(
    # Project setup
    project_name="vision_experiments",
    experiment_name="vitb_imagenet_warmup10",
    seed=42,

    # Training
    epochs=300,
    batch_size=256,
    learning_rate=5e-4,
    warmup_epochs=20,

    # Optimization
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
    use_mixed_precision=True,

    # Checkpointing
    checkpoint_dir="checkpoints",
    save_every_n_epochs=10,
    patience=50,

    # Logging
    use_wandb=True,
    wandb_entity="my_team",
)
```

**Features in Detail:**

| Feature | Description |
|---------|-------------|
| **Device Management** | Auto-detects CUDA, MPS (Apple Silicon), CPU |
| **W&B Integration** | Real-time metrics, plots, hyperparameter comparison |
| **LR Scheduling** | Cosine annealing with linear warmup |
| **Gradient Accumulation** | Simulate larger batch sizes with limited memory |
| **Mixed Precision** | Faster training with AMP on CUDA |
| **Checkpoint Saving** | Save best models, resume interrupted training |
| **Early Stopping** | Stop training when validation loss plateaus |
| **Metric Tracking** | Loss, accuracy, learning rate history |

---

## 🚀 Quick Examples

### Example 1: Production Training with W&B (Recommended)

```python
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from vit.vitb import ViTB
from vit.device import get_device
from dataops.vision_dataset import VisionDataset
from train.train_template import TrainingConfig, Trainer

# 1. Create datasets
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
ds_train = VisionDataset(root="data/train", mode="train", train_transform=train_transform)
ds_val = VisionDataset(root="data/val", mode="val")

# 2. Create dataloaders
train_loader = DataLoader(ds_train, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(ds_val, batch_size=128, num_workers=4)

# 3. Configure training
config = TrainingConfig(
    project_name="imagenet_experiments",
    experiment_name="vitb_baseline",
    epochs=300,
    batch_size=64,
    learning_rate=1e-4,
    warmup_epochs=20,
    use_wandb=True,  # Enable W&B tracking
)

# 4. Create model and trainer
model = ViTB(img_size=224, patch_size=16, num_classes=ds_train.num_classes)
trainer = Trainer(model, config)

# 5. Train (full pipeline with checkpoints, early stopping, etc.)
results = trainer.train(train_loader, val_loader)

# 6. Resume from checkpoint if interrupted
# results = trainer.train(train_loader, val_loader, resume_from="checkpoints/checkpoint_epoch050_best.pt")
```

### Example 2: Simple Training Loop (Minimal)

```python
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from vit.vitb import ViTB
from vit.device import get_device
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

# 3. Create model on correct device
device = get_device()
model = ViTB(img_size=224, patch_size=16, num_classes=ds.num_classes).to(device)

# 4. Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    for images, labels in loader:
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)

        # Forward + backward
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

| Feature | ViT | CLIP | LLaVA | DataOps | Training |
|---------|-----|------|-------|---------|----------|
| **Type** | Vision Encoder | Image-Text Model | Multimodal | Data Loading | Training Loop |
| **Input** | Images | Images + Text | Images + Text | Various (folder, CSV, JSON) | Model + DataLoaders |
| **Output** | Image Features | Normalized Embeddings | Text | Dict/Tuple of Tensors | Checkpoints + Metrics |
| **Pretrained** | Ready for ONNX export | HuggingFace CLIP models | HuggingFace integration | N/A (template) | N/A |
| **Trainable** | Yes | Yes (contrastive loss) | Vision frozen, LM trainable | N/A | Any model |
| **Use Case** | Feature extraction, classification | Zero-shot, similarity, retrieval | Image-to-text understanding | Data preprocessing | Training & monitoring |
| **Key Feature** | Architecture | Contrastive learning | Multimodal fusion | Dataset abstraction | W&B + checkpointing |
| **Embedding Dim** | 768, 1024 | 512, 768 | Varies | N/A | N/A |

**GradCAM (Attention Visualization):**

Generic Gradient-weighted Class Activation Mapping that works with **any model, any image size, any input type**.

```bash
# Command-line usage (simplest!)
uv run utils/gradcam_viz.py \
    --checkpoint model.pt \
    --images data/test/ \
    --output visualizations/

# Works with ANY image size - no resizing needed!
uv run utils/gradcam_viz.py \
    --checkpoint model.pt \
    --images image_512x512.jpg \
    --num-classes 2 \
    --alpha 0.5
```

**Python API (fully generic):**

```python
from utils.gradcam_generic import GradCAM, VisualizationConfig
import torch

# Works with ANY model - auto-detects architecture
model = YourModel()
model.load_state_dict(torch.load("model.pt"))

# Create visualizer (fully automatic, no configuration needed)
viz = GradCAM(model, device="cuda")

# Works with ANY image size!
images_192 = torch.randn(4, 3, 192, 192)  # Any size!
images_512 = torch.randn(4, 3, 512, 512)  # Any size!

# Generate attention maps (same code for any size)
maps_192 = viz.generate_attention_maps(images_192)
maps_512 = viz.generate_attention_maps(images_512)

# Customize visualization
config = VisualizationConfig(
    cmap="hot",
    alpha=0.4,
    dpi=150,
)

# Save with custom config
viz.save_visualizations(
    images_512,
    maps_512,
    output_dir="results",
    config=config,
)
```

**Features:**
- ✅ **Model-agnostic** - Works with ViT, CNN, hybrid, custom architectures
- ✅ **Size-agnostic** - Any image size (192×192, 224×224, 512×512, etc.)
- ✅ **Input-agnostic** - RGB, grayscale, multi-channel, any format
- ✅ **Auto-detection** - No manual configuration needed
- ✅ **Flexible output** - Save individual maps, overlays, comparisons

**Outputs:**
- `original_*.png` - Input images
- `heatmap_*.png` - Pure attention maps
- `overlay_*.png` - Heatmap overlaid on original
- `comparison_*.png` - Side-by-side visualization with colorbar

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

# Test CLIP
python clip/clip.py

# Test LLaVA
python llava/llava.py

# Test Dataset Templates
python dataops/vision_dataset.py
python dataops/language_dataset.py
python dataops/vl_dataset.py

# Test Training Loop (with dummy data)
python train/train_template.py

# Test GradCAM Generic (with dummy model - works with any size!)
python utils/gradcam_generic.py
```

All tests skip gracefully if dependencies (PyTorch, Pillow) aren't installed.

### Generic GradCAM: Works with ANY Model & Image Size

The generic GradCAM is the recommended approach - it auto-detects your model architecture and works with any image size:

```bash
# Simple command-line usage
uv run utils/gradcam_viz.py \
    --checkpoint checkpoints/model_best.pt \
    --images data/test_images/ \
    --output visualizations/

# Works with ANY image size (no explicit size parameter needed!)
uv run utils/gradcam_viz.py \
    --checkpoint model.pt \
    --images image_512x512.jpg \
    --num-classes 2 \
    --alpha 0.5

# Custom colormap and output settings
uv run utils/gradcam_viz.py \
    --checkpoint model.pt \
    --images data/ \
    --output results/ \
    --cmap "hot" \
    --dpi 200 \
    --alpha 0.3
```

**Key advantages over old version:**
- ✅ No need to specify image size
- ✅ Works with any model (auto-detects ViT, CNN, etc.)
- ✅ Handles RGB, grayscale, multi-channel inputs
- ✅ Cleaner error messages
- ✅ Flexible visualization options

---

## 📚 Code Style & Conventions

All code in this repository follows consistent patterns:

- **Docstrings:** Google-style with comprehensive examples and parameter documentation
- **Type Hints:** Full typing from `typing` module for all function signatures
- **Tensor Shapes:** Inline comments showing dimensions (e.g., `# (B, N, D)`)
- **Deterministic:** Reproducible training with seed management and sorted loading
- **Configurable:** Sensible defaults with extensive optional parameters via dataclasses
- **Self-Contained:** Each module works independently with no hidden dependencies
- **Device-Aware:** Automatic device detection and handling throughout

Example patterns:

**Model Definition:**
```python
def forward(
    self,
    input_ids: torch.Tensor,
    images: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Forward pass with detailed tensor shapes.

    Args:
        input_ids: [batch_size, seq_len] - text token IDs
        images: [batch_size, 3, height, width] - optional images

    Returns:
        Dict with keys: logits, loss (if labels provided)
    """
    # Implementation with inline shape comments
    # x: (B, N, D)
```

**Training Configuration:**
```python
@dataclass
class TrainingConfig:
    """Dataclass-based config (JSON serializable, reproducible)."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    # ... all hyperparameters

config = TrainingConfig(epochs=300)
config.save("experiment.json")  # Reproducible
```

**Device Handling:**
```python
from vit.device import get_device

device = get_device()  # Auto-detects CUDA, MPS, CPU
model = MyModel().to(device)
x = torch.randn(...).to(device)
```

---

## 🔧 Common Tasks

### Visualizing Model Attention with Generic GradCAM

Works with **any image size, any model**:

```bash
# Simplest usage - just provide checkpoint and images!
uv run utils/gradcam_viz.py --checkpoint model.pt --images data/test/
```

**Python API:**

```python
from utils.gradcam_generic import GradCAM
import torch

# Load ANY model
model = YourModel()
model.load_state_dict(torch.load("model.pt"))

# Create visualizer (auto-detects architecture)
viz = GradCAM(model, device="cuda")

# Works with ANY image size (no need to specify!)
images = torch.randn(4, 3, 512, 512)  # Any size!

# Generate attention maps
attention_maps = viz.generate_attention_maps(images, target_class=0)

# Save visualizations
viz.save_visualizations(
    images,
    attention_maps,
    output_dir="visualizations",
    target_class=0,
)
```

**What this shows:**
- Which image regions the model focuses on for each class
- Attention intensity heatmaps overlaid on inputs
- Works with any image size without resizing
- Helps debug model behavior and identify failure cases
- Supports RGB, grayscale, and multi-channel inputs

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
- **[CLIP](clip/doc.md)** - Contrastive learning, zero-shot classification, multimodal alignment
- **[LLaVA](llava/doc.md)** - Multimodal architecture, training, and real-world examples
- **[DataOps](dataops/doc.md)** - Dataset templates, formats, and usage patterns
- **[Training Template](train/train_template.py)** - Production training loop with W&B integration (inline documented)
- **[GradCAM Visualization](utils/gradcam.md)** - Comprehensive guide to model attention visualization (any model, any size)

---

## 🤝 Contributing

This repository is organized for clarity and reusability. When adding new components:

1. **Follow the existing code style:**
   - Google-style docstrings with comprehensive examples
   - Full type hints on all function signatures
   - Inline comments for tensor shapes

2. **Test thoroughly:**
   - Include smoke tests with synthetic data in `__main__`
   - No real data files required for testing

3. **Document comprehensively:**
   - Module-level docstrings explaining purpose and usage
   - Inline shape comments for tensor operations
   - Update relevant `doc.md` or README sections

4. **Keep device handling in mind:**
   - Use `get_device()` for device detection
   - Move tensors and models to the same device
   - Test on multiple device types (CUDA, MPS, CPU) when possible

5. **Make it configurable:**
   - Use dataclasses for configs
   - Provide sensible defaults
   - Make configs JSON serializable for reproducibility

---

## 📝 License

Check individual files for license information.

---

## 🔗 Related Resources

- **PyTorch:** https://pytorch.org
- **Vision Transformer Paper:** https://arxiv.org/abs/2010.11929
- **CLIP Paper:** https://arxiv.org/abs/2103.00020
- **RoPE (Rotary Position Embeddings):** https://arxiv.org/abs/2104.09864
- **LLaVA Paper:** https://arxiv.org/abs/2304.08485
- **OpenAI CLIP GitHub:** https://github.com/openai/CLIP
- **HuggingFace Hub:** https://huggingface.co

---

**Last Updated:** 2026-03-12

A comprehensive toolkit for modern PyTorch modeling and training. Includes architecture implementations, training utilities, and dataset templates with W&B integration and production-ready best practices.
