# Vision Transformer (ViT) Architectures Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture Variants](#architecture-variants)
3. [Core Components](#core-components)
4. [Position Embeddings](#position-embeddings)
5. [Implementation Details](#implementation-details)
6. [Comparisons](#comparisons)
7. [References](#references)

---

## Overview

Vision Transformers (ViT) represent a paradigm shift in computer vision, replacing convolutional neural networks with pure transformer architectures. The key insight is treating images as sequences of patches and applying standard transformer attention mechanisms.

### Key Innovation
Rather than using local convolutions, ViT divides an image into fixed-size patches, linearly embeds them, and processes them with transformer layers - similar to how transformers handle text tokens.

**Original Paper:** [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2020)

---

## Architecture Variants

### 1. ViT-B (Base) - Standard Positional Embeddings
**File:** `vitb.py`

Standard Vision Transformer Base variant using learnable absolute positional embeddings.

#### Configuration:
```
Image Size:        224×224
Patch Size:        16×16  (resulting in 14×14 = 196 patches)
Embedding Dim:     768
Attention Heads:   12
Transformer Depth: 12 layers
MLP Ratio:        4× (hidden = 3072)
Total Parameters:  ~86M
```

#### Architecture Flow:
```
Input Image (B, 3, 224, 224)
    ↓
Patch Embedding (B, 196, 768)
    ↓
[CLS] Token + Positional Embedding (B, 197, 768)
    ↓
12× TransformerBlock
    - LayerNorm + MultiHeadAttention + Residual
    - LayerNorm + MLP + Residual
    ↓
Classification Head (B, 1000)
```

#### Key Code Snippet - Positional Embedding:
```python
# Learnable positional embedding
self.pos_embed = nn.Parameter(
    torch.zeros(1, self.n_patches + 1, embed_dim)
)
nn.init.trunc_normal_(self.pos_embed, std=0.02)

# Applied in forward pass
x = x + self.pos_embed  # Absolute position encoding
```

**Characteristics:**
- Learnable positional embeddings
- Good for fixed image sizes (224×224)
- Parameters scale with sequence length
- Standard approach used in original ViT

---

### 2. ViT-L (Large) - Standard Positional Embeddings
**File:** `vitl.py`

Larger variant of Vision Transformer with expanded architecture.

#### Configuration:
```
Image Size:        224×224
Patch Size:        16×16
Embedding Dim:     1024
Attention Heads:   16
Transformer Depth: 24 layers
MLP Ratio:        4× (hidden = 4096)
Total Parameters:  ~304M
```

**Key Differences from ViT-B:**
- Larger embedding dimension (1024 vs 768)
- More transformer layers (24 vs 12)
- More attention heads (16 vs 12)
- ~3.5× more parameters

#### Use Cases:
- Better performance on large-scale datasets (ImageNet-21k, etc.)
- More capacity for complex visual understanding
- Improved transfer learning capabilities

---

### 3. ViT-B with RoPE (Rotary Position Embeddings)
**File:** `vitb_rope.py`

ViT-B enhanced with Rotary Position Embeddings instead of learnable absolute embeddings.

#### Configuration:
Same as ViT-B but with RoPE instead of learnable pos_embed

#### Key Innovation - RoPE:
Instead of learnable embeddings, positions are encoded through rotations applied to Q and K vectors.

**RoPE Paper:** [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (Su et al., 2021)

#### Mathematical Foundation:
```
For each dimension pair (2i, 2i+1) in the head:

Rotation angle: m * θ_i
where:
  m = position/token_index
  θ_i = 10000^(-2i/d_head)

Apply 2D rotation matrix to Q and K:
[cos(mθ)  -sin(mθ)] [q_2i  ]     [q'_2i  ]
[sin(mθ)   cos(mθ)] [q_2i+1]  =  [q'_2i+1]
```

#### Implementation - RoPE Application:
```python
def apply_rotary_emb(xq, xk, freqs_cis):
    """Apply rotary embeddings to Q and K."""
    # Convert to complex numbers for rotation
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Apply rotation via complex multiplication
    xq_rot = xq_ * freqs_cis  # e^(im*θ) * z
    xk_rot = xk_ * freqs_cis

    # Convert back to real
    xq_out = torch.view_as_real(xq_rot).reshape(*xq_rot.shape[:-1], -1)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

#### Advantages:
1. **No learnable parameters** - position info is implicit
2. **Better extrapolation** - handles sequences longer than training
3. **Memory efficient** - no pos_embed parameter storage
4. **Frequency basis** - similar to sinusoidal embeddings but with rotation
5. **Relative position awareness** - attention scores depend on relative positions

#### Code Example - RoPE in Attention:
```python
# Precompute rotation frequencies
freqs_cis = precompute_freqs_cis(self.head_dim, N, device)

# Apply to Q and K after projection
q, k, v = qkv[0], qkv[1], qkv[2]
freqs = self.freqs_cis[:N]
q, k = apply_rotary_emb(q, k, freqs)

# Standard attention with rotated queries/keys
attn = (q @ k.transpose(-2, -1)) * self.scale
```

---

### 4. ViT-L with RoPE
**File:** `vitl_rope.py`

Large Vision Transformer with Rotary Position Embeddings.

#### Configuration:
Same as ViT-L but with RoPE instead of learnable pos_embed

---

## Core Components

### 1. Patch Embedding
Converts an image into a sequence of patch embeddings.

#### How It Works:
```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        # Conv2d projects image patches to embedding dimension
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.proj(x)           # (B, 768, 14, 14)
        x = rearrange(x, 'b c h w -> b (h w) c')  # (B, 196, 768)
        return x
```

**Equivalent to:**
- Reshaping image: (224×224) → (14×14 patches of 16×16)
- Linear projection: (3×16×16=768) → (embed_dim=768)

#### Computational Efficiency:
Using Conv2d with kernel_size=patch_size and stride=patch_size is equivalent to:
1. Extracting patches: O(1) per patch
2. Reshaping: O(1)
3. Linear projection: O(P × D²) where P=patches, D=embedding dim

---

### 2. Multi-Head Attention
Standard scaled dot-product attention with multiple heads.

#### Architecture:
```python
class Attention(nn.Module):
    def forward(self, x):
        # x: (B, N, C) where N = num_patches + 1
        B, N, C = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x)  # (B, N, 3*C)
        qkv = rearrange(qkv, 'b n (h_count h c) -> h_count b h n c',
                       h_count=self.n_heads)
        # Each: (B, n_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, h, N, N)
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        x = attn @ v  # (B, h, N, head_dim)
        x = rearrange(x, 'b h n c -> b n (h c)')

        return self.proj(x)
```

#### Complexity:
- Computation: O(N² × D) where N=sequence length, D=embedding dim
- Memory: O(N²) for attention matrix storage

**Note:** For 196 patches + 1 CLS token = 197 tokens, attention is (197, 197)

---

### 3. Transformer Block
Combines attention and feedforward with residual connections and layer normalization.

#### Pre-LayerNorm Configuration:
```python
class TransformerBlock(nn.Module):
    def forward(self, x):
        # Pre-norm: norm before sublayer
        x = x + self.attn(self.norm1(x))      # Residual + attention
        x = x + self.mlp(self.norm2(x))       # Residual + FFN
        return x
```

**Design Choices:**
- Pre-LayerNorm (norm before sublayer) - better training stability
- Residual connections - easier gradient flow
- 4× MLP expansion - standard in transformers

#### Total Depth Comparison:
- ViT-B: 12 blocks × 2 sublayers = 24 transformer layers
- ViT-L: 24 blocks × 2 sublayers = 48 transformer layers

---

## Position Embeddings

### Comparison of Position Encoding Methods

| Aspect | Learnable (ViT-B/L) | RoPE (ViT-B/L-RoPE) |
|--------|------------------|-----------------|
| **Storage** | O(N×D) parameters | None (computed) |
| **Training** | Updated via backprop | Fixed formula |
| **Extrapolation** | Poor (fixed N) | Excellent (any N) |
| **Implementation** | Simple addition | Complex rotation |
| **Relative Positions** | Implicit | Explicit |
| **Memory Usage** | Higher | Lower |
| **Speed** | Fast addition | Complex ops |

### Absolute vs Relative Position Encoding

**Learnable Embeddings (Absolute):**
- Each position gets a unique learned embedding
- Applied as: `x = x + pos_embed`
- Assumes fixed sequence length during training
- Does not generalize to longer sequences

**RoPE (Relative):**
- Position encoded through rotation angles
- Attention weights naturally encode relative distances
- Generalizes to longer sequences
- Mathematical foundation in rotation matrices

#### Why RoPE Works Better for Extrapolation:

The rotation angle depends only on relative position difference (m - n):
```
When computing attention(m, n):
- Rotation at position m: angle = m * θ
- Rotation at position n: angle = n * θ
- Relative rotation = (m - n) * θ

The attention mechanism only sees relative rotation,
not absolute positions, enabling extrapolation.
```

---

## Implementation Details

### Forward Pass Flow

#### ViT Standard (vitb.py / vitl.py):
```
1. PatchEmbedding: Image → Sequence of patch embeddings
2. Add [CLS] token: Prepend learnable classification token
3. Add positional embeddings: Add absolute position info
4. Dropout: Regularization
5. Transformer blocks: Process through attention + FFN layers
6. LayerNorm: Normalize final representations
7. Classification head: Project [CLS] output to class logits
```

#### ViT with RoPE (vitb_rope.py / vitl_rope.py):
```
1. PatchEmbedding: Same
2. Add [CLS] token: Same
3. No positional embedding: RoPE is applied in attention
4. Dropout: Same
5. Transformer blocks: Attention applies RoPE internally
6. LayerNorm: Same
7. Classification head: Same
```

### Initialization Strategy

```python
def _init_weights(self, m):
    if isinstance(m, nn.Linear):
        # Truncated normal for weight matrices
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        # Identity initialization for LN
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
```

**Rationale:**
- **Trunc normal (σ=0.02)** - ViT is sensitive to initialization; prevents extreme values
- **Zero bias** - Simplifies learning dynamics
- **Identity LN** - Preserves initial activation distributions

---

## Comparisons

### Model Size Comparison

```
╔═══════════════════════════════════════════════════════════╗
║ Architecture Metrics Comparison                           ║
╠════════════════╦═════════╦═════════╦═════════╦══════════╣
║ Metric         ║ ViT-B   ║ ViT-B   ║ ViT-L   ║ ViT-L    ║
║                ║ (Std)   ║ (RoPE)  ║ (Std)   ║ (RoPE)   ║
╠════════════════╬═════════╬═════════╬═════════╬══════════╣
║ Embed Dim      ║ 768     ║ 768     ║ 1024    ║ 1024     ║
║ Heads          ║ 12      ║ 12      ║ 16      ║ 16       ║
║ Depth          ║ 12      ║ 12      ║ 24      ║ 24       ║
║ Params (M)     ║ 86      ║ 86      ║ 304     ║ 304      ║
║ Pos Embed Params║ 197×768║ 0       ║ 197×1024║ 0        ║
║ FLOPs (G)      ║ 17.6    ║ 17.6    ║ 61.6    ║ 61.6     ║
╚════════════════╩═════════╩═════════╩═════════╩══════════╝
```

**Key Observations:**
1. RoPE saves ~151KB (ViT-B) and ~201KB (ViT-L) of parameters
2. Computational complexity is identical (RoPE ops are cheap)
3. RoPE trades small speed overhead for better extrapolation

### Performance Characteristics

| Use Case | Recommended | Reason |
|----------|-------------|--------|
| Fixed 224×224 images | Standard ViT | Simpler, learnable embeddings |
| Variable input sizes | ViT-RoPE | Extrapolates to new sizes |
| Longer sequences/patches | ViT-RoPE | Better relative position awareness |
| Memory constrained | ViT-RoPE | No pos_embed parameter storage |
| Training from scratch | Standard ViT | More stable, tested thoroughly |
| Fine-tuning | Either | Both work well |

---

## Code Organization

### Files Structure:
```
arch/
├── vitb.py           # ViT-B with learnable positional embeddings
├── vitl.py           # ViT-L with learnable positional embeddings
├── vitb_rope.py      # ViT-B with Rotary Position Embeddings
├── vitl_rope.py      # ViT-L with Rotary Position Embeddings
└── doc.md           # This documentation
```

### Module Hierarchy:

**Shared Components (in all files):**
- `PatchEmbedding` - Converts images to sequences
- `MLP` - Feed-forward network (2-layer)
- `TransformerBlock` - Combines attention + FFN

**Different Components:**
- `Attention` vs `AttentionRoPE` - Position encoding method
- `ViTB` vs `ViTBRoPE` - Main model class
- `ViTL` vs `ViTLRoPE` - Main model class

---

## References

### Primary Papers:

1. **Vision Transformer (ViT)**
   - [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
   - Dosovitskiy et al., ICLR 2021
   - Introduces the core ViT architecture

2. **Rotary Position Embeddings (RoPE)**
   - [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
   - Su et al., 2021
   - Proposes RoPE as alternative to absolute position embeddings

3. **Attention Mechanism**
   - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
   - Vaswani et al., NeurIPS 2017
   - Foundation for transformer architecture

4. **LayerNorm and Initialization**
   - [Layer Normalization](https://arxiv.org/abs/1607.06450)
   - Ba et al., 2016

### Related Work:

- **DeiT** (Data-efficient Image Transformers): Distillation strategies for ViT
- **BERT**: Original transformer architecture adapted for vision
- **ALiBi** (Attention with Linear Biases): Alternative position encoding

---

## Usage Examples

### Using Standard ViT-B:
```python
from vitb import ViTB

model = ViTB(
    img_size=224,
    patch_size=16,
    in_channels=3,
    num_classes=1000,
    embed_dim=768,
    depth=12,
    n_heads=12,
    mlp_ratio=4.0,
)

x = torch.randn(8, 3, 224, 224)
output = model(x)  # (8, 1000)
```

### Using RoPE ViT-B:
```python
from vitb_rope import ViTBRoPE

model = ViTBRoPE(
    img_size=224,
    patch_size=16,
    in_channels=3,
    num_classes=1000,
    embed_dim=768,
    depth=12,
    n_heads=12,
)

x = torch.randn(8, 3, 224, 224)
output = model(x)  # (8, 1000)

# RoPE can also handle variable sequence lengths
# if patches are extracted differently
```

### Key Differences in Usage:
- **API is identical** - both have same `__init__` and `forward` signatures
- **Only difference:** RoPE handles variable-length sequences better
- **Drop-in replacement:** Can swap between implementations

---

## Troubleshooting

### Common Issues:

1. **Position embedding mismatch error**
   - Occurs when loading weights with different patch configurations
   - Solution: Ensure img_size and patch_size match training config

2. **Out of memory with ViT-L**
   - Attention is O(N²) in memory; 197 tokens = 38K attention weights per head
   - Solution: Use gradient checkpointing or reduce batch size

3. **RoPE numerical instability**
   - Rare with proper implementation; check dtype consistency
   - Ensure complex number conversions don't lose precision

---

## Performance Notes

### Training Speed (relative to ViT-B):
- Standard ViT-B: 1.0× baseline
- ViT-B-RoPE: ~1.02× (RoPE ops are cheap)
- ViT-L: ~0.5× throughput (larger, deeper model)
- ViT-L-RoPE: ~0.51× throughput

### Inference Speed:
Similar to training speed, as computation is dominated by attention and FFN layers, not position encoding.

### Memory Usage:
- Standard ViT-B: pos_embed adds ~600KB
- ViT-B-RoPE: Saves the pos_embed parameter
- Difference negligible compared to model weights

---

## Future Extensions

Potential improvements to these architectures:

1. **Sparse Attention** - Reduce O(N²) attention complexity
2. **Grouped Query Attention** - Reduce attention parameters
3. **Sliding Window Attention** - For longer sequences
4. **Mixed Precision Training** - Reduce memory usage
5. **Flash Attention** - Optimized attention kernel implementation

---

*Last Updated: 2026-03-12*
*Documentation for Vision Transformer architectures with multiple position encoding strategies*
