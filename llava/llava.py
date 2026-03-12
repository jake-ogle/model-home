"""
PyTorch implementation of LLaVA (Large Language and Vision Assistant) architecture.

LLaVA combines a vision encoder (CLIP ViT) with a language model (LLaMA)
through a projection layer to enable multimodal understanding.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class VisionProjection(nn.Module):
    """
    Projects image features from the vision encoder to the language model's embedding space.

    This module takes the output from a vision encoder and projects it to match
    the hidden dimension of the language model, enabling fusion of visual and textual information.
    """

    def __init__(self, vision_hidden_size: int, language_hidden_size: int):
        """
        Args:
            vision_hidden_size: Hidden dimension of the vision encoder output
            language_hidden_size: Hidden dimension of the language model
        """
        super().__init__()
        self.linear = nn.Linear(vision_hidden_size, language_hidden_size)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Project image features to language model space.

        Args:
            image_features: [batch_size, num_patches, vision_hidden_size]

        Returns:
            Projected features: [batch_size, num_patches, language_hidden_size]
        """
        return self.linear(image_features)


class MultimodalEmbedding(nn.Module):
    """
    Combines text embeddings with projected image features.

    This module interleaves image tokens with text tokens to create a unified
    multimodal sequence that the language model can process.
    """

    def __init__(self, language_hidden_size: int, vocab_size: int):
        """
        Args:
            language_hidden_size: Hidden dimension of the language model
            vocab_size: Size of the text vocabulary
        """
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, language_hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        image_features: torch.Tensor,
        image_token_id: int = -100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Merge text and image embeddings.

        Args:
            input_ids: [batch_size, seq_len] - text token IDs
            image_features: [batch_size, num_patches, language_hidden_size] - projected image features
            image_token_id: Special token ID marking image positions (typically -100 for masking)

        Returns:
            embeddings: [batch_size, total_seq_len, language_hidden_size]
            attention_mask: [batch_size, total_seq_len] - mask for valid tokens
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Get text embeddings
        text_embeddings = self.text_embedding(input_ids)

        # Create attention mask for text
        text_attention_mask = (input_ids != -100).long()

        # Find positions of image tokens
        image_mask = (input_ids == image_token_id)
        num_image_tokens = image_mask.sum(dim=1)

        # Merge embeddings by replacing image tokens with actual image features
        merged_embeddings = text_embeddings.clone()
        merged_attention_mask = text_attention_mask.clone()

        # For simplicity, replace first image token position with image features
        # In practice, this would be more sophisticated batching
        for b in range(batch_size):
            img_positions = torch.where(image_mask[b])[0]
            if len(img_positions) > 0:
                start_idx = img_positions[0]
                num_patches = image_features.shape[1]

                # This is a simplified version - in production, handle variable image token counts
                if start_idx + num_patches <= seq_len:
                    merged_embeddings[b, start_idx:start_idx + num_patches] = image_features[b]
                    merged_attention_mask[b, start_idx:start_idx + num_patches] = 1

        return merged_embeddings, merged_attention_mask


class LLaVA(nn.Module):
    """
    LLaVA (Large Language and Vision Assistant) model.

    Combines a vision encoder with a language model through a projection layer.
    The architecture supports both image understanding and text generation.

    Typical architecture:
    - Vision Encoder (e.g., CLIP ViT-L): Encodes images to feature vectors
    - Projection Layer: Aligns vision features with language model's embedding space
    - Language Model (e.g., LLaMA): Processes merged text and image tokens
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        language_model: nn.Module,
        vision_hidden_size: int,
        language_hidden_size: int,
        vocab_size: int,
        image_token_id: int = 32000
    ):
        """
        Args:
            vision_encoder: Pre-trained vision encoder (e.g., CLIP ViT)
            language_model: Pre-trained language model (e.g., LLaMA)
            vision_hidden_size: Hidden dimension of vision encoder
            language_hidden_size: Hidden dimension of language model
            vocab_size: Vocabulary size of language model
            image_token_id: Special token ID for images in the language model
        """
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.projection = VisionProjection(vision_hidden_size, language_hidden_size)
        self.multimodal_embedding = MultimodalEmbedding(language_hidden_size, vocab_size)
        self.image_token_id = image_token_id

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images using the vision encoder.

        Args:
            images: [batch_size, 3, height, width] - images in RGB format

        Returns:
            image_features: [batch_size, num_patches, vision_hidden_size]
        """
        with torch.no_grad():
            image_features = self.vision_encoder(images)
        return image_features

    def project_image_features(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Project image features to language model space.

        Args:
            image_features: [batch_size, num_patches, vision_hidden_size]

        Returns:
            projected_features: [batch_size, num_patches, language_hidden_size]
        """
        return self.projection(image_features)

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for LLaVA model.

        Args:
            input_ids: [batch_size, seq_len] - text token IDs, use image_token_id for image positions
            images: [batch_size, 3, height, width] - optional raw images to encode
            image_features: [batch_size, num_patches, language_hidden_size] - pre-computed and projected image features
            attention_mask: [batch_size, seq_len] - attention mask for text tokens
            labels: [batch_size, seq_len] - target token IDs for training (optional)

        Returns:
            logits: [batch_size, seq_len, vocab_size] or loss if labels provided
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Encode images if provided
        if images is not None:
            image_features = self.encode_image(images)
            image_features = self.project_image_features(image_features)
        elif image_features is not None:
            # Ensure image features are projected
            if image_features.shape[-1] != self.language_model.config.hidden_size:
                image_features = self.project_image_features(image_features)

        # Merge text and image embeddings
        embeddings, merged_attention_mask = self.multimodal_embedding(
            input_ids,
            image_features,
            self.image_token_id
        )

        # Use provided attention mask or merged one
        if attention_mask is None:
            attention_mask = merged_attention_mask

        # Forward through language model
        outputs = self.language_model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            labels=labels
        )

        return outputs

    def generate(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """
        Generate text conditioned on images.

        Args:
            input_ids: [batch_size, seq_len] - initial text prompt
            images: [batch_size, 3, height, width] - input images
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            generated_ids: [batch_size, seq_len + max_new_tokens]
        """
        # Encode images once
        image_features = self.encode_image(images)
        projected_features = self.project_image_features(image_features)

        # Generate tokens autoregressively
        for _ in range(max_new_tokens):
            # Get embeddings
            embeddings, _ = self.multimodal_embedding(
                input_ids,
                projected_features,
                self.image_token_id
            )

            # Forward through language model
            outputs = self.language_model(inputs_embeds=embeddings)
            logits = outputs.logits[:, -1, :]

            # Apply temperature
            logits = logits / temperature

            # Sample next token
            if top_p < 1.0:
                # Nucleus sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumsum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumsum > top_p
                sorted_indices_to_remove[..., 0] = False  # Keep at least one token
                logits[sorted_indices[sorted_indices_to_remove]] = -float('inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids


# ============================================================================
# Hugging Face Integration Helpers
# ============================================================================

def load_huggingface_vision_model(model_name: str = "openai/clip-vit-large-patch14") -> nn.Module:
    """
    Load a vision encoder from Hugging Face.

    Args:
        model_name: Model identifier on Hugging Face Hub
            Popular options:
            - "openai/clip-vit-large-patch14" (1024-dim, recommended)
            - "openai/clip-vit-base-patch32" (512-dim, faster)
            - "openai/clip-vit-large-patch14-336" (larger images)

    Returns:
        Vision model for encoding images

    Example:
        vision_model = load_huggingface_vision_model("openai/clip-vit-large-patch14")
    """
    try:
        from transformers import CLIPVisionModel
    except ImportError:
        raise ImportError("transformers package required. Install with: pip install transformers")

    model = CLIPVisionModel.from_pretrained(model_name)
    # Freeze vision encoder (typically not fine-tuned)
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_huggingface_language_model(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
    device: str = "cuda",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False
) -> nn.Module:
    """
    Load a language model from Hugging Face with optional quantization.

    Args:
        model_name: Model identifier on Hugging Face Hub
            Popular open-source options:
            - "mistralai/Mistral-7B-Instruct-v0.1" (7B, fast, good quality)
            - "meta-llama/Llama-2-7b-hf" (7B, needs approval)
            - "meta-llama/Llama-2-13b-hf" (13B)
            - "tiiuae/falcon-7b" (7B, trained on permissive license)
            - "bigcode/starcoder" (15B, code-focused)
            - "teknium/OpenHermes-2.5-Mistral-7B" (7B, fine-tuned)

            Note: GPT-OSS 20B or similar can be loaded similarly if available
            on Hugging Face Hub or loaded from local path

        device: Device to load model on ("cuda", "cpu", "mps")
        load_in_8bit: Use 8-bit quantization (requires bitsandbytes)
        load_in_4bit: Use 4-bit quantization (requires bitsandbytes)

    Returns:
        Language model for text generation

    Example:
        # Standard loading
        lm = load_huggingface_language_model("mistralai/Mistral-7B-Instruct-v0.1")

        # With 8-bit quantization to save memory
        lm = load_huggingface_language_model(
            "mistralai/Mistral-7B-Instruct-v0.1",
            load_in_8bit=True
        )

        # For larger models with 4-bit
        lm = load_huggingface_language_model(
            "meta-llama/Llama-2-13b-hf",
            load_in_4bit=True
        )
    """
    try:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError:
        raise ImportError("transformers package required. Install with: pip install transformers")

    # Quantization config
    quantization_config = None
    if load_in_8bit or load_in_4bit:
        try:
            if load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            elif load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
        except ImportError:
            raise ImportError(
                "bitsandbytes required for quantization. "
                "Install with: pip install bitsandbytes"
            )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        quantization_config=quantization_config,
        trust_remote_code=True  # Some models need this
    )

    return model


def load_huggingface_tokenizer(model_name: str):
    """
    Load a tokenizer matching the language model.

    Args:
        model_name: Same model identifier as language model

    Returns:
        Tokenizer for encoding/decoding text

    Example:
        tokenizer = load_huggingface_tokenizer("mistralai/Mistral-7B-Instruct-v0.1")
        tokens = tokenizer("Hello, how are you?")
        text = tokenizer.decode(tokens["input_ids"])
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("transformers package required. Install with: pip install transformers")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_model_dimensions(model_name: str) -> dict:
    """
    Get dimension information for known models.

    Args:
        model_name: Model identifier

    Returns:
        Dictionary with hidden_size and vocab_size
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    return {
        "language_hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "vision_hidden_size": 1024,  # CLIP ViT-L standard
    }


# ============================================================================
# Example utility functions
# ============================================================================

def create_dummy_vision_encoder(hidden_size: int = 768, num_patches: int = 256) -> nn.Module:
    """Create a dummy vision encoder for testing."""
    class DummyVisionEncoder(nn.Module):
        def __init__(self, hidden_size, num_patches):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_patches = num_patches
            self.projection = nn.Linear(768, hidden_size)  # Assume input is 768-dim

        def forward(self, x):
            # x: [batch_size, 3, height, width]
            batch_size = x.shape[0]
            # In practice, this would extract patches and encode them
            # For demo, return dummy features
            return torch.randn(batch_size, self.num_patches, self.hidden_size, device=x.device)

    return DummyVisionEncoder(hidden_size, num_patches)


def create_dummy_language_model(hidden_size: int = 768, vocab_size: int = 32000) -> nn.Module:
    """Create a dummy language model for testing."""
    class DummyLanguageModel(nn.Module):
        def __init__(self, hidden_size, vocab_size):
            super().__init__()
            self.config = type('Config', (), {'hidden_size': hidden_size})()
            self.lm_head = nn.Linear(hidden_size, vocab_size)
            self.num_layers = 12

        def forward(self, inputs_embeds, attention_mask=None, labels=None):
            # In practice, this would be a full transformer model
            batch_size, seq_len, _ = inputs_embeds.shape

            # Dummy forward pass (in reality, this is the transformer)
            hidden_states = inputs_embeds

            logits = self.lm_head(hidden_states)

            class Output:
                def __init__(self, logits):
                    self.logits = logits

            return Output(logits)

    return DummyLanguageModel(hidden_size, vocab_size)
