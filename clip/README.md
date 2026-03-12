# CLIP: Contrastive Language-Image Pre-training

## Overview

CLIP (Contrastive Language-Image Pre-training) is a foundation model that learns visual representations from natural language supervision. Instead of being trained on a fixed set of object classes, CLIP learns from pairs of images and their textual descriptions, making it highly flexible for zero-shot and few-shot learning tasks.

**Core Insight:** By aligning image and text embeddings in a shared space using contrastive learning, CLIP learns transferable visual features without requiring labeled classification datasets.

**Key Capabilities:**
- Zero-shot image classification (classify images into unseen categories)
- Image-text similarity scoring
- Cross-modal retrieval (find images matching text descriptions)
- Transfer learning to downstream vision tasks

Reference: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (OpenAI, 2021) — https://arxiv.org/abs/2103.00020

---

## Architecture

### High-Level Flow

```
  Images              Text
    │                 │
    ├─[Patch Embed]   ├─[Token Embed]
    │                 │
    ├─[ViT Encoder]   ├─[GPT Encoder]
    │                 │
    ├─[Projection]    ├─[Projection]
    │                 │
    └─[L2 Normalize]  └─[L2 Normalize]
      │                 │
      image_features    text_features
         (B, 512)          (B, 512)
            │                 │
            └─────────────────┘
                    │
        logit_scale * (image @ text.T)
                    │
              Cosine Similarity Matrix
                    │
            InfoNCE Contrastive Loss
                    │
                  Loss
```

### Vision Encoder (VisionEncoder)

Patch-based Vision Transformer for encoding images:

1. **Patch Embedding** (`Conv2d`):
   - Splits 224×224 image into 7×7 grid of 32×32 patches
   - Each patch projected to 768-dim embedding via convolution
   - Output: (B, 49, 768)

2. **Class Token + Positional Embeddings:**
   - Learnable class token prepended (similar to ViT [CLS] token)
   - Learnable position embeddings for each patch + class token
   - Sequence: (B, 50, 768)

3. **Transformer Blocks (12 layers):**
   - Pre-LayerNorm + MultiheadAttention (12 heads)
   - Position-wise FFN (4× expansion)
   - Residual connections

4. **Output:**
   - LayerNorm applied to all tokens
   - Class token extracted: (B, 768)

**Configuration (ViT-B/32):**
- Input: 224×224 RGB images
- Patch size: 32×32
- Embedding dimension: 768
- Attention heads: 12
- Transformer depth: 12 layers
- Parameters: ~87M
- FLOPs: ~17.6G per image

### Text Encoder (TextEncoder)

GPT-style transformer for encoding text:

1. **Token + Positional Embeddings:**
   - Token embedding: vocab_size × 512
   - Learnable positional embeddings for context_length=77
   - Output: (B, 77, 512)

2. **Causal Attention Mask:**
   - Tokens only attend to previous tokens (not future)
   - Prevents information flow from right-to-left
   - Typical in autoregressive language models

3. **Transformer Blocks (12 layers):**
   - Pre-LayerNorm + MultiheadAttention (8 heads)
   - Causal mask built into attention: (77, 77) lower-triangular
   - Position-wise FFN (4× expansion)
   - Residual connections

4. **Output:**
   - LayerNorm applied
   - EOT (end-of-text) token extracted: (B, 512)
   - CLIP convention: take the last meaningful token or use special EOT token

**Configuration:**
- Vocabulary size: 49,408 (CLIP tokenizer)
- Context length: 77 tokens
- Embedding dimension: 512
- Attention heads: 8
- Transformer depth: 12 layers
- Parameters: ~63M

### Projection Layers & Temperature

**Projection:**
- Image features (B, 768) → projection matrix (768, 512) → (B, 512)
- Text features (B, 512) → projection matrix (512, 512) → (B, 512)
- Projects both modalities to shared embedding space

**L2 Normalization:**
- All embeddings normalized to unit length: ||x|| = 1
- Converts projection to cosine similarity: x₁ · x₂ = cos(θ)
- More stable than raw Euclidean distance

**Temperature Parameter (τ):**
- Learnable scalar: logit_scale = log(1/0.07) ≈ 2.66
- Scales similarity scores: scaled_logits = τ * (image_features @ text_features.T)
- Controls softness of probability distribution
- Higher τ → sharper probabilities, lower τ → smoother

---

## Components

### ResidualAttentionBlock

Pre-layer-normalized transformer block with attention and MLP:

```python
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

**Features:**
- Pre-LayerNorm (more stable than post-norm)
- Enables high learning rates
- Optional causal attention mask (for text encoder)

### VisionEncoder

Self-contained image encoder (does NOT import from vit/ directory):

```python
encoder = VisionEncoder(
    img_size=224,           # Input image size
    patch_size=32,          # Patch grid size
    width=768,              # Embedding dimension
    num_layers=12,          # Transformer depth
    num_heads=12,           # Attention heads
)
features = encoder(images)  # (B, 3, 224, 224) → (B, 768)
```

**Key Methods:**
- `forward(images)` → L2-normalized features ready for projection

### TextEncoder

GPT-style text encoder with causal masking:

```python
encoder = TextEncoder(
    vocab_size=49408,       # CLIP tokenizer vocabulary
    context_length=77,      # Max sequence length
    width=512,              # Embedding dimension
    num_layers=12,          # Transformer depth
    num_heads=8,            # Attention heads
)
features = encoder(input_ids)  # (B, 77) → (B, 512)
```

**Causal Mask:**
- Prevents attending to future tokens
- Built into ResidualAttentionBlock via attn_mask parameter

### CLIP Model

Main model combining both encoders:

```python
model = CLIP(
    embed_dim=512,              # Shared embedding dimension
    image_resolution=224,       # Image input size
    vision_layers=12,           # ViT depth
    vision_width=768,           # ViT embedding dim
    vision_patch_size=32,       # Patch size
    context_length=77,          # Text context length
    vocab_size=49408,           # Text vocabulary
    transformer_width=512,      # Text encoder embedding dim
    transformer_heads=8,        # Text attention heads
    transformer_layers=12,      # Text encoder depth
)

# Inference
image_features = model.encode_image(images)      # (B, embed_dim), normalized
text_features = model.encode_text(input_ids)    # (B, embed_dim), normalized

# Training with loss
outputs = model(images, input_ids)
loss = outputs["loss"]
```

---

## API Reference

### CLIP.encode_image(images)

Encode images to normalized embeddings.

**Args:**
- `images` (Tensor): Image batch of shape (B, 3, 224, 224)

**Returns:**
- (Tensor): Normalized embeddings of shape (B, 512)
- Each embedding has L2 norm = 1

**Example:**
```python
images = torch.randn(8, 3, 224, 224)
image_features = model.encode_image(images)
# image_features.shape = (8, 512)
# image_features.norm(dim=-1) ≈ 1.0
```

### CLIP.encode_text(input_ids)

Encode text to normalized embeddings.

**Args:**
- `input_ids` (Tensor): Token IDs of shape (B, 77)

**Returns:**
- (Tensor): Normalized embeddings of shape (B, 512)
- Each embedding has L2 norm = 1

**Example:**
```python
input_ids = torch.randint(0, 49408, (8, 77))
text_features = model.encode_text(input_ids)
# text_features.shape = (8, 512)
# text_features.norm(dim=-1) ≈ 1.0
```

### CLIP.forward(images, input_ids)

Full forward pass with loss computation.

**Args:**
- `images` (Tensor): Image batch of shape (B, 3, 224, 224)
- `input_ids` (Tensor): Token IDs of shape (B, 77)

**Returns:**
- Dict with keys:
  - `"image_features"` (B, 512): Normalized image embeddings
  - `"text_features"` (B, 512): Normalized text embeddings
  - `"logit_scale"` (scalar): Temperature parameter
  - `"loss"` (scalar): InfoNCE contrastive loss

**Example:**
```python
outputs = model(images, input_ids)
loss = outputs["loss"]
loss.backward()

# For inference:
with torch.no_grad():
    outputs = model(images, input_ids)
```

### InfoNCE Contrastive Loss

The loss function used during training:

```python
logits = logit_scale * (image_features @ text_features.T)  # (B, B)
labels = torch.arange(B)  # [0, 1, 2, ..., B-1]

# Image-to-text matching loss
loss_i2t = CrossEntropy(logits, labels)

# Text-to-image matching loss
loss_t2i = CrossEntropy(logits.T, labels)

# Combined loss
loss = (loss_i2t + loss_t2i) / 2
```

**Intuition:**
- For batch of B pairs, logits is (B, B) similarity matrix
- Diagonal: correct image-text pairs (should have high similarity)
- Off-diagonal: incorrect pairs (should have low similarity)
- Loss encourages correct pairs to rank highest

### HuggingFace Integration

Load pre-trained models from OpenAI:

```python
# Load CLIP model
model = load_huggingface_clip("openai/clip-vit-base-patch32")

# Load image processor and tokenizer
processor = load_huggingface_clip_processor("openai/clip-vit-base-patch32")

# Prepare inputs
from PIL import Image
image = Image.open("cat.jpg")
text = ["a photo of a cat", "a photo of a dog"]
inputs = processor(text=text, images=image, return_tensors="pt", padding=True)

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)
    image_features = outputs.image_embeds
    text_features = outputs.text_embeds
```

---

## Model Variants

### Officially Supported Models

| Variant | Vision Encoder | Image Res | Embed Dim | Parameters | Speed | Quality |
|---------|---------------|-----------|-----------|-----------|-------|---------|
| B/32    | ViT-B/32      | 224×224   | 512       | 150M      | ⭐⭐⭐⭐⭐ | ⭐⭐⭐   |
| B/16    | ViT-B/16      | 224×224   | 512       | 150M      | ⭐⭐⭐⭐  | ⭐⭐⭐⭐ |
| L/14    | ViT-L/14      | 224×224   | 768       | 428M      | ⭐⭐⭐   | ⭐⭐⭐⭐⭐|
| L/14@336| ViT-L/14      | 336×336   | 768       | 428M      | ⭐⭐    | ⭐⭐⭐⭐⭐|

**Notation:** ViT-B/32 = ViT-Base with 32×32 patches (7×7 grid per image)

---

## Quick Start

### Zero-Shot Image Classification

Use CLIP to classify images without fine-tuning:

```python
import torch
from PIL import Image
from clip import CLIP, load_huggingface_clip_processor

# Load model and processor
model = load_huggingface_clip("openai/clip-vit-base-patch32")
processor = load_huggingface_clip_processor("openai/clip-vit-base-patch32")
model.eval()

# Define class descriptions
classes = ["a photo of a dog", "a photo of a cat", "a photo of a bird"]

# Load and encode image
image = Image.open("pet.jpg")
inputs = processor(text=classes, images=image, return_tensors="pt", padding=True)

# Get similarity scores
with torch.no_grad():
    outputs = model(**inputs)
    image_emb = outputs.image_embeds
    text_embs = outputs.text_embeds

    # Compute cosine similarity
    similarities = (image_emb @ text_embs.T).squeeze(0)
    probs = torch.softmax(similarities, dim=-1)

    # Get prediction
    top_class_idx = probs.argmax().item()
    predicted_class = classes[top_class_idx]
    confidence = probs[top_class_idx].item()

    print(f"Prediction: {predicted_class} ({confidence:.1%})")
```

### Image-Text Similarity Scoring

Compute similarity between images and descriptions:

```python
from clip import CLIP, load_huggingface_clip_processor

model = load_huggingface_clip("openai/clip-vit-base-patch32")
processor = load_huggingface_clip_processor("openai/clip-vit-base-patch32")

# Load multiple images and texts
images = [Image.open(f"img_{i}.jpg") for i in range(3)]
texts = [
    "a photo of a landscape",
    "an indoor scene",
    "a portrait of a person",
]

# Prepare batch
inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)

# Compute similarities
with torch.no_grad():
    outputs = model(**inputs)
    image_embs = outputs.image_embeds
    text_embs = outputs.text_embeds

    # Similarity matrix: (num_images, num_texts)
    similarities = image_embs @ text_embs.T
    print(f"Similarity matrix shape: {similarities.shape}")
    print(similarities)
```

### Training CLIP from Scratch

Minimal training loop:

```python
import torch
from torch.optim import Adam
from clip import CLIP

# Initialize model
model = CLIP(embed_dim=512, vision_width=768, transformer_width=512)
model.train()

# Optimizer
optimizer = Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (images, text_ids) in enumerate(dataloader):
        optimizer.zero_grad()

        # Forward pass with loss
        outputs = model(images, text_ids)
        loss = outputs["loss"]

        # Backward pass
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
```

---

## Training Details

### Loss Formulation

**InfoNCE Contrastive Loss:**

Given a batch of B image-text pairs:

1. Encode all images: {I₁, I₂, ..., Iᵦ} → {f¹ᵢ, f²ᵢ, ..., fᵇᵢ} ∈ ℝᵈ
2. Encode all texts: {T₁, T₂, ..., Tᵦ} → {f¹ₜ, f²ₜ, ..., fᵇₜ} ∈ ℝᵈ
3. L2 normalize: fᵢ/||fᵢ||, fₜ/||fₜ||
4. Compute logits: L = τ * F · Fᵀ where F = [image_features; text_features]
5. Create similarity matrix: logits[i,j] = τ * (image_i · text_j)
6. Loss: symmetric cross-entropy

```
Loss = (CrossEntropy(logits, I) + CrossEntropy(logits.T, I)) / 2

where:
  logits ∈ ℝ^(B×B): image-text similarity matrix
  I = [0, 1, 2, ..., B-1]: ground truth labels (diagonal is positive pair)
  CrossEntropy(logits, I) = -log(exp(logits[i,i]) / Σⱼ exp(logits[i,j]))
```

**Properties:**
- Symmetric: encourages image→text and text→image matching
- Efficient: single B×B matrix instead of 2B×B
- Contrastive: harder negatives (full batch) improve representations
- Temperature: learnable parameter balances match difficulty

### Batch Size Considerations

Larger batches significantly improve CLIP training:

- **Batch Size 256:** ~45% better downstream performance vs B=32
- **Batch Size 1024:** ~52% better (diminishing returns)
- **Batch Size 32768:** ~62% better (OpenAI's original setting)

**Why larger batches help:**
- More negative examples per image-text pair
- Harder, more informative negatives improve representations
- Better gradient estimates

**Trade-off:**
- Larger batches need larger compute (TPU/GPU clusters)
- B=256 on modern GPUs (A100, H100) provides good results
- B=32-64 sufficient for research/fine-tuning

### Training Hyperparameters

Standard settings used in CLIP training:

| Hyperparameter | Value | Notes |
|---|---|---|
| Learning Rate | 5×10⁻⁴ | Cosine decay from 5×10⁻⁴ to 0 |
| Warmup Steps | 500 | Linear warmup over 500 steps |
| Batch Size | 32768 | On 256 V100 GPUs; adjust for your setup |
| Epochs | 32 | ~13M image-text pairs |
| Optimizer | AdamW | β₁=0.9, β₂=0.98, weight_decay=0.2 |
| Temperature Init | log(1/0.07) ≈ 2.66 | Learnable; typically increases during training |
| Weight Decay | 0.2 | Important for regularization |
| Gradient Clipping | 1.0 | Clip global norm |

### Mixed Precision Training

Important for memory efficiency:

```python
from torch.cuda.amp import autocast, GradScaler

model = CLIP()
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()

for images, text_ids in dataloader:
    optimizer.zero_grad()

    # Mixed precision forward pass
    with autocast():
        outputs = model(images, text_ids)
        loss = outputs["loss"]

    # Scaled backward pass
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    scaler.step(optimizer)
    scaler.update()
```

---

## References

**Primary Paper:**
- Radford et al. "Learning Transferable Visual Models From Natural Language Supervision" (2021)
  https://arxiv.org/abs/2103.00020

**Related Work:**
- Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (2020)
  https://arxiv.org/abs/2010.11929 [Vision Transformer architecture]

- Vaswani et al. "Attention is All You Need" (2017)
  https://arxiv.org/abs/1706.03762 [Transformer architecture]

- Chen et al. "A Simple Framework for Contrastive Learning of Visual Representations" (2020)
  https://arxiv.org/abs/2002.05709 [SimCLR contrastive learning]

**OpenAI Resources:**
- OpenAI CLIP GitHub: https://github.com/openai/CLIP
- CLIP Model Zoo: https://huggingface.co/models?search=clip
- CLIP Paper PDF: https://arxiv.org/pdf/2103.00020.pdf

**Tokenizer:**
CLIP uses a custom tokenizer with 49,408 vocabulary. Tokenization is handled by:
- transformers library: `CLIPTokenizer`
- Direct via OpenAI: available in official CLIP repository
