"""
Vision Transformer Base (ViT-B) with Standard Positional Embeddings.

This module implements ViT-B as described in "An Image is Worth 16x16 Words:
Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2020).

Reference: https://arxiv.org/abs/2010.11929

Architecture Highlights:
    - Splits images into 14×14 grid of 16×16 patches
    - Embeds patches to 768-dimensional vectors
    - Uses learnable absolute positional embeddings
    - 12 layers of transformer encoder (attention + feedforward)
    - Classifies using [CLS] token representation

Model Statistics:
    - Parameters: ~86M
    - FLOPs (224×224): ~17.6G
    - Throughput: ~700 images/sec on V100

Usage:
    >>> model = ViTB(num_classes=1000)
    >>> x = torch.randn(8, 3, 224, 224)
    >>> logits = model(x)  # (8, 1000)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    """
    Convert images to patch embeddings using convolutional projection.

    Splits an image into non-overlapping patches and projects each to embed_dim.
    More efficient than manual patch extraction + linear layer.

    For a 224×224 image with 16×16 patches:
        - Number of patches: 14×14 = 196
        - Each patch: 16×16×3 = 768 values → 768-dim embedding

    Implementation Details:
        Uses Conv2d with kernel_size=patch_size and stride=patch_size.
        This is mathematically equivalent to:
        1. Extract patches: (B, 3, 224, 224) → (B, 196, 768)  [flattened patches]
        2. Linear project: (B, 196, 768) → (B, 196, embed_dim)
        Using Conv2d is ~10× faster than manual extraction.

    Args:
        img_size (int): Input image size (height and width). Default: 224
        patch_size (int): Size of each patch (height and width). Default: 16
        in_channels (int): Number of input channels (3 for RGB). Default: 3
        embed_dim (int): Output embedding dimension. Default: 768

    Attributes:
        n_patches (int): Total number of patches = (img_size // patch_size)²
                        For 224÷16=14, so 14×14=196 patches
        proj (nn.Conv2d): Convolutional projection kernel_size=patch_size, stride=patch_size
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        """
        Project image to patch embeddings.

        Args:
            x (torch.Tensor): Image tensor of shape (B, C, H, W)
                Example: (8, 3, 224, 224) for batch of 8 RGB images

        Returns:
            torch.Tensor: Patch embeddings of shape (B, n_patches, embed_dim)
                Example: (8, 196, 768)

        Process:
            1. Conv2d projects patches: (B, 3, 224, 224) → (B, 768, 14, 14)
            2. Rearrange to sequence: (B, 768, 14, 14) → (B, 196, 768)
        """
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, n_patches_h, n_patches_w)
        x = rearrange(x, 'b c h w -> b (h w) c')  # (B, n_patches, embed_dim)
        return x


class Attention(nn.Module):
    """
    Multi-Head Self-Attention with scaled dot-product mechanism.

    Implements the scaled dot-product attention from "Attention is All You Need"
    (Vaswani et al., 2017). Each attention head independently computes attention
    over the sequence, then results are concatenated.

    Attention computation:
        Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V

    For multi-head:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W^O
        where head_i = Attention(Q @ W^Q_i, K @ W^K_i, V @ W^V_i)

    Args:
        dim (int): Input embedding dimension. Default: 768
        n_heads (int): Number of attention heads. Default: 12
                       Must divide dim evenly. head_dim = dim // n_heads
        attn_drop (float): Dropout rate on attention weights. Default: 0.0
        proj_drop (float): Dropout rate on output projection. Default: 0.0

    Attributes:
        n_heads (int): Number of attention heads
        head_dim (int): Dimension per head = dim // n_heads = 64 for ViT-B
        scale (float): Scaling factor = 1 / √head_dim ≈ 0.125 for head_dim=64
        qkv (nn.Linear): Projects input to Q, K, V (3×dim output)
        attn_drop (nn.Dropout): Dropout on attention weights
        proj (nn.Linear): Output projection from (n_heads × head_dim) → dim
        proj_drop (nn.Dropout): Dropout on projected output

    Complexity:
        - Time: O(N² × d) where N=seq_len, d=embedding_dim
        - Memory: O(N²) for attention matrix
        For ViT-B with N=197 (196 patches + 1 CLS): O(197² × 768) ≈ 30M ops

    Mathematical Detail:
        The scale factor √d_k prevents dot products from growing too large,
        which would push softmax into regions with tiny gradients.
    """

    def __init__(self, dim, n_heads=12, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Compute multi-head self-attention.

        Args:
            x (torch.Tensor): Input embeddings of shape (B, N, C)
                B: batch size
                N: sequence length (n_patches + 1 for ViT)
                C: embedding dimension (768 for ViT-B)

        Returns:
            torch.Tensor: Attention output of shape (B, N, C)

        Process:
            1. Project to Q, K, V: (B, N, C) → (B, N, 3C)
            2. Reshape to multi-head: (B, N, 3C) → (3, B, n_heads, N, head_dim)
            3. Compute attention: (B, h, N, N) = softmax(QK^T / √d)
            4. Apply to values: (B, h, N, head_dim) = attention @ V
            5. Concatenate heads: (B, N, C)
            6. Output projection: linear layer + dropout
        """
        # x: (B, N, C)
        B, N, C = x.shape

        # Project input to Q, K, V and reshape for multi-head
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = rearrange(qkv, 'b n h_count h c -> h_count b h n c')
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, n_heads, N, head_dim)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, n_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention weights to values
        x = (attn @ v)  # (B, n_heads, N, head_dim)
        x = rearrange(x, 'b h n c -> b n (h c)')  # (B, N, C)

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """
    Feed-Forward Network (Position-wise Feed-Forward Network).

    A 2-layer MLP with GELU activation between layers, applied independently
    to each sequence position. Expands feature dimension by mlp_ratio (default 4×).

    Architecture:
        Input (C) → Linear(C → hidden) → GELU → Dropout →
        Linear(hidden → C) → Dropout → Output (C)

    For ViT-B with embed_dim=768:
        Input (768) → Linear(768 → 3072) → GELU → Dropout →
        Linear(3072 → 768) → Dropout → Output (768)

    This bottleneck-style MLP increases model capacity and introduces
    non-linearity between attention layers.

    Args:
        in_features (int): Input feature dimension. Default: 768
        hidden_features (int): Hidden layer dimension. Default: in_features × 4
        out_features (int): Output feature dimension. Default: in_features
        drop (float): Dropout rate. Default: 0.0

    Attributes:
        fc1 (nn.Linear): First linear layer (expansion)
        act (nn.GELU): GELU activation function
        fc2 (nn.Linear): Second linear layer (contraction)
        drop (nn.Dropout): Dropout layer (applied after both fc layers)

    Complexity:
        - Parameters: 2 × in_features × hidden_features ≈ 8 × in_features²
        - For ViT-B: 768 × 3072 × 2 ≈ 4.7M parameters per block
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Apply position-wise feed-forward network.

        Args:
            x (torch.Tensor): Input of shape (B, N, C)

        Returns:
            torch.Tensor: Output of shape (B, N, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer Encoder Block combining attention and feed-forward networks.

    A single transformer layer with pre-layer normalization (pre-norm):
        1. LayerNorm → MultiHeadAttention → Residual Add
        2. LayerNorm → MLP → Residual Add

    Pre-LayerNorm Configuration:
        Pre-norm (norm before sublayer) is more stable than post-norm
        during training and enables residual connections with higher learning rates.

    Residual Connections:
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

        This design allows:
        - Stable gradient flow (residuals act as shortcuts)
        - Training of very deep networks (50+ layers)
        - Higher learning rates without divergence

    Args:
        dim (int): Embedding dimension (768 for ViT-B)
        n_heads (int): Number of attention heads (12 for ViT-B)
        mlp_ratio (float): Expansion ratio for MLP hidden dimension. Default: 4.0
        drop (float): Dropout rate for feedforward. Default: 0.0
        attn_drop (float): Attention weight dropout rate. Default: 0.0

    Attributes:
        norm1 (nn.LayerNorm): Layer normalization before attention
        attn (Attention): Multi-head self-attention module
        norm2 (nn.LayerNorm): Layer normalization before MLP
        mlp (MLP): Position-wise feed-forward network

    Mathematical Form:
        y1 = x + MultiHeadAttention(LayerNorm(x))
        y2 = y1 + MLP(LayerNorm(y1))
    """

    def __init__(self, dim, n_heads=12, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, n_heads=n_heads, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        """
        Apply transformer block with pre-norm residuals.

        Args:
            x (torch.Tensor): Input of shape (B, N, C)

        Returns:
            torch.Tensor: Output of shape (B, N, C)
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTB(nn.Module):
    """
    Vision Transformer Base (ViT-B) Model.

    Complete vision transformer implementation for image classification.
    Processes images by:
    1. Converting to patch embeddings
    2. Adding learnable positional embeddings
    3. Processing through 12 transformer layers
    4. Using the [CLS] token for classification

    Architecture Summary:
        PatchEmbedding(224×224 → 196×768)
        ↓
        [CLS] Token + Positional Embedding
        ↓
        12× TransformerBlock(attention + FFN)
        ↓
        LayerNorm + Classification Head (1000 classes)

    Configuration:
        - Image Size: 224×224
        - Patch Size: 16×16
        - Embedding Dimension: 768
        - Attention Heads: 12 (head_dim=64)
        - Transformer Depth: 12 layers
        - MLP Expansion: 4× (hidden=3072)
        - Positional Encoding: Learnable absolute embeddings

    Parameters: ~86M
    FLOPs (224×224): ~17.6G
    Memory (batch=8): ~15GB

    Training Details:
        - Initialization: Trunc-normal (σ=0.02) for weight matrices
        - Optimizer: AdamW with weight decay
        - Learning Rate: 1e-3 with cosine annealing
        - Warmup: First 20 epochs with linear warmup
        - Augmentation: RandAugment, Mixup, CutMix recommended

    Args:
        img_size (int): Input image size. Default: 224
        patch_size (int): Patch size. Default: 16
        in_channels (int): Number of input channels. Default: 3
        num_classes (int): Number of output classes. Default: 1000
        embed_dim (int): Embedding dimension. Default: 768
        depth (int): Number of transformer layers. Default: 12
        n_heads (int): Number of attention heads. Default: 12
        mlp_ratio (float): MLP expansion ratio. Default: 4.0
        drop_rate (float): Dropout rate. Default: 0.0
        attn_drop_rate (float): Attention dropout rate. Default: 0.0

    Reference:
        Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for
        Image Recognition at Scale" ICLR 2021. https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        n_heads=12,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding (includes cls token position)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(drop_rate)

        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for _ in range(depth)
        ])

        # Layer norm and classification head
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize weights according to ViT conventions.

        Uses truncated normal distribution for linear layers to prevent
        extreme values. ViT is sensitive to initialization - using the
        standard normal distribution often leads to divergence.

        Strategy:
            - Linear layers: Trunc-normal with σ=0.02
            - Biases: Constant 0
            - LayerNorm: Identity initialization (weight=1, bias=0)

        Args:
            m (nn.Module): Module to initialize
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Forward pass through ViT-B.

        Args:
            x (torch.Tensor): Input images of shape (B, 3, 224, 224)
                B: batch size
                3: RGB channels
                224: image height and width

        Returns:
            torch.Tensor: Classification logits of shape (B, num_classes)
                Contains unnormalized scores for each class
                Apply softmax for probabilities

        Process:
            1. Patch Embedding: (B, 3, 224, 224) → (B, 196, 768)
            2. Add [CLS] token: (B, 196, 768) → (B, 197, 768)
            3. Add Positional Embeddings: learnable position info
            4. Apply 12 Transformer Blocks:
               - Multi-head attention (self-attention)
               - Feed-forward network (MLP)
               - Residual connections and layer norm
            5. Final Classification:
               - LayerNorm on all tokens
               - Take [CLS] token representation (first token)
               - Linear projection to num_classes

        Example:
            >>> model = ViTB(num_classes=1000)
            >>> x = torch.randn(8, 3, 224, 224)
            >>> logits = model(x)  # (8, 1000)
            >>> probs = torch.softmax(logits, dim=-1)  # (8, 1000)
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)

        # Add class token
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=B)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches + 1, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer encoder
        for block in self.blocks:
            x = block(x)

        # Classification
        x = self.norm(x)
        x = x[:, 0]  # Take cls token
        x = self.head(x)

        return x


if __name__ == "__main__":
    # Test the model
    model = ViTB(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        n_heads=12,
    )

    # Create random input
    x = torch.randn(2, 3, 224, 224)

    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model created successfully!")
