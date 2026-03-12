"""
Vision Transformer Base (ViT-B) with Pretrained Weights Support.

Provides factory functions to load ViT-B models with optional pretrained weights
from official sources (ImageNet-1k, ImageNet-21k).

Usage:
    >>> from vit import vitb
    >>> model = vitb(pretrained=True, num_classes=1000)
    >>> # or with fine-tuning for different task
    >>> model = vitb(pretrained=True, num_classes=10)  # 10 classes
"""

import torch
import torch.nn as nn
from typing import Optional
from .vitb import ViTB


def vitb(pretrained: bool = False, num_classes: int = 1000, **kwargs) -> ViTB:
    """
    Create a ViT-B model with optional pretrained weights.

    Args:
        pretrained (bool): Load pretrained ImageNet-1k weights. Default: False
        num_classes (int): Number of output classes. Default: 1000
        **kwargs: Additional arguments passed to ViTB constructor
            - img_size (int): Input image size. Default: 224
            - patch_size (int): Patch size. Default: 16
            - embed_dim (int): Embedding dimension. Default: 768
            - depth (int): Number of transformer layers. Default: 12
            - n_heads (int): Number of attention heads. Default: 12
            - mlp_ratio (float): MLP expansion ratio. Default: 4.0
            - drop_rate (float): Dropout rate. Default: 0.0
            - attn_drop_rate (float): Attention dropout rate. Default: 0.0

    Returns:
        ViTB: Initialized model, optionally with pretrained weights

    Example:
        >>> # Create model without pretraining
        >>> model = vitb(num_classes=1000)

        >>> # Create model with ImageNet-1k pretrained weights
        >>> model = vitb(pretrained=True, num_classes=1000)

        >>> # Fine-tune on a different task
        >>> model = vitb(pretrained=True, num_classes=10)
        >>> # Note: output head will be randomly initialized for 10 classes

    References:
        - Official pretrained weights from https://github.com/google-research/vision_transformer
        - Dosovitskiy et al. "An Image is Worth 16x16 Words" ICLR 2021
    """
    model = ViTB(num_classes=num_classes, **kwargs)

    if pretrained:
        try:
            import timm
            # Load from timm's official ViT-B/16 ImageNet-1k pretrained weights
            pretrained_model = timm.create_model('vit_base_patch16_224.augreg_in1k',
                                                  pretrained=True, num_classes=num_classes)
            # Transfer weights to our model
            state_dict = pretrained_model.state_dict()
            _load_pretrained_weights(model, state_dict, num_classes)
        except ImportError:
            raise ImportError(
                "timm library is required for loading pretrained weights. "
                "Install it with: pip install timm"
            )

    return model


def vitb_21k(pretrained: bool = False, num_classes: int = 21843, **kwargs) -> ViTB:
    """
    Create a ViT-B model pretrained on ImageNet-21k.

    ImageNet-21k pretraining provides better transfer learning performance
    for downstream tasks compared to ImageNet-1k.

    Args:
        pretrained (bool): Load pretrained ImageNet-21k weights. Default: False
        num_classes (int): Number of output classes. Default: 21843 (ImageNet-21k)
        **kwargs: Additional arguments passed to ViTB constructor

    Returns:
        ViTB: Model with ImageNet-21k pretrained weights

    Example:
        >>> # Load ImageNet-21k pretrained weights
        >>> model = vitb_21k(pretrained=True)
        >>> # Fine-tune on custom dataset
        >>> model = vitb_21k(pretrained=True, num_classes=10)
    """
    model = ViTB(num_classes=num_classes, **kwargs)

    if pretrained:
        try:
            import timm
            # Load from timm's ViT-B/16 ImageNet-21k pretrained weights
            pretrained_model = timm.create_model('vit_base_patch16_224_in21k',
                                                  pretrained=True, num_classes=num_classes)
            state_dict = pretrained_model.state_dict()
            _load_pretrained_weights(model, state_dict, num_classes)
        except ImportError:
            raise ImportError(
                "timm library is required for loading pretrained weights. "
                "Install it with: pip install timm"
            )

    return model


def _load_pretrained_weights(model: ViTB, pretrained_state_dict: dict, num_classes: int) -> None:
    """
    Load pretrained weights into ViT-B model, handling class-specific head differences.

    When the number of output classes differs from pretraining, the classification
    head is skipped and randomly initialized instead, allowing transfer learning
    to custom tasks.

    Args:
        model (ViTB): The target model to load weights into
        pretrained_state_dict (dict): State dict from pretrained model
        num_classes (int): Number of classes in target model
    """
    # Get model's state dict
    model_state_dict = model.state_dict()

    # Load all weights except the classification head
    for name, param in pretrained_state_dict.items():
        # Skip classification head if dimensions don't match
        if name.startswith('head') or name == 'norm.weight' or name == 'norm.bias':
            if num_classes != 1000:  # Different num_classes than pretraining
                continue

        if name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                model_state_dict[name] = param
            else:
                print(f"Skipping {name}: shape mismatch "
                      f"({param.shape} vs {model_state_dict[name].shape})")
        else:
            print(f"Skipping {name}: not found in target model")

    model.load_state_dict(model_state_dict)


if __name__ == "__main__":
    # Example: Create and test pretrained model
    print("Creating ViT-B without pretraining...")
    model_scratch = vitb(num_classes=1000)
    print(f"Model created: {type(model_scratch).__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model_scratch.parameters()):,}")

    print("\nCreating ViT-B with ImageNet-1k pretraining...")
    try:
        model_pretrained = vitb(pretrained=True, num_classes=1000)
        print(f"Pretrained model loaded successfully!")

        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = model_pretrained(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
    except ImportError as e:
        print(f"Note: {e}")
        print("Skipping pretrained test. Install timm to use pretrained weights.")
