"""
Vision Transformer Large (ViT-L) with Standard Positional Embeddings.

This module implements ViT-L as described in "An Image is Worth 16x16 Words:
Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2020).

Reference: https://arxiv.org/abs/2010.11929

ViT-L is a larger variant of the Vision Transformer with expanded capacity:
    - Larger embedding dimension (1024 vs 768)
    - Deeper transformer (24 vs 12 layers)
    - More attention heads (16 vs 12)
    - ~3.5× more parameters

Architecture Highlights:
    - Splits images into 14×14 grid of 16×16 patches
    - Embeds patches to 1024-dimensional vectors
    - Uses learnable absolute positional embeddings
    - 24 layers of transformer encoder (attention + feedforward)
    - Classifies using [CLS] token representation

Model Statistics:
    - Parameters: ~304M
    - FLOPs (224×224): ~61.6G
    - Throughput: ~350 images/sec on V100

Recommended Usage:
    - Large-scale training (ImageNet-21k pretraining)
    - Transfer learning with fine-tuning
    - Tasks requiring higher capacity
    - When compute budget allows

Usage:
    >>> model = ViTL(num_classes=1000)
    >>> x = torch.randn(8, 3, 224, 224)
    >>> logits = model(x)  # (8, 1000)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings."""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, n_patches_h, n_patches_w)
        x = rearrange(x, 'b c h w -> b (h w) c')  # (B, n_patches, embed_dim)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim, n_heads=16, attn_drop=0.0, proj_drop=0.0):
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
        # x: (B, N, C)
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = rearrange(qkv, 'b n h_count h c -> h_count b h n c')
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, n_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, n_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)  # (B, n_heads, N, head_dim)
        x = rearrange(x, 'b h n c -> b n (h c)')  # (B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Feed-forward network."""

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, dim, n_heads=16, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, n_heads=n_heads, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTL(nn.Module):
    """
    Vision Transformer Large (ViT-L) Model.

    Large-scale vision transformer for image classification with significantly
    more capacity than ViT-B. Suitable for large-scale datasets and transfer
    learning applications.

    Key Differences from ViT-B:
        ViT-B (Base):
            - Embedding Dim: 768
            - Heads: 12
            - Depth: 12 layers
            - Params: 86M
            - FLOPs: 17.6G

        ViT-L (Large):
            - Embedding Dim: 1024
            - Heads: 16 (head_dim=64)
            - Depth: 24 layers
            - Params: 304M  (3.5× larger)
            - FLOPs: 61.6G  (3.5× more computation)

    Architecture Summary:
        PatchEmbedding(224×224 → 196×1024)
        ↓
        [CLS] Token + Positional Embedding
        ↓
        24× TransformerBlock(attention + FFN)
        ↓
        LayerNorm + Classification Head (1000 classes)

    Training Details:
        - Initialization: Trunc-normal (σ=0.02)
        - Optimizer: AdamW with weight decay (typically 0.05)
        - Learning Rate: 1e-3 (lower than ViT-B for stability)
        - Warmup: 20 epochs linear warmup
        - Augmentation: RandAugment, Mixup, CutMix, Stochastic Depth
        - Best with ImageNet-21k pretraining + ImageNet fine-tuning

    Performance Characteristics:
        - Training: Slower convergence than ViT-B (more layers)
        - Inference: ~2× slower per image than ViT-B
        - Accuracy: +2-3% over ViT-B on ImageNet with same training
        - Memory: ~3.5× more than ViT-B for same batch size

    Args:
        img_size (int): Input image size. Default: 224
        patch_size (int): Patch size. Default: 16
        in_channels (int): Number of input channels. Default: 3
        num_classes (int): Number of output classes. Default: 1000
        embed_dim (int): Embedding dimension. Default: 1024
        depth (int): Number of transformer layers. Default: 24
        n_heads (int): Number of attention heads. Default: 16
        mlp_ratio (float): MLP expansion ratio. Default: 4.0
        drop_rate (float): Dropout rate. Default: 0.0
        attn_drop_rate (float): Attention dropout rate. Default: 0.0

    Attributes:
        patch_embed (PatchEmbedding): Patch embedding layer
        cls_token (nn.Parameter): Learnable classification token
        pos_embed (nn.Parameter): Learnable positional embeddings
        pos_drop (nn.Dropout): Dropout on embeddings
        blocks (nn.ModuleList): Stack of 24 TransformerBlocks
        norm (nn.LayerNorm): Final layer normalization
        head (nn.Linear): Classification head

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
        embed_dim=1024,
        depth=24,
        n_heads=16,
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
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
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
    model = ViTL(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=1024,
        depth=24,
        n_heads=16,
    )

    # Create random input
    x = torch.randn(2, 3, 224, 224)

    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model created successfully!")
