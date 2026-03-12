"""
Practical examples of using LLaVA with Hugging Face models.

This module demonstrates:
1. Loading real models from Hugging Face
2. Inference with different model sizes
3. Using quantization for memory efficiency
4. Batch processing and generation
5. Fine-tuning with LoRA
"""

import torch
from typing import Optional, List
from pathlib import Path


def example_basic_inference():
    """
    Basic example: Load models and perform inference.

    Requirements:
        pip install torch transformers pillow
    """
    from llava import (
        LLaVA,
        load_huggingface_vision_model,
        load_huggingface_language_model,
        load_huggingface_tokenizer,
        get_model_dimensions
    )
    from PIL import Image
    from torchvision import transforms

    print("=== Basic LLaVA Inference ===\n")

    # Load models
    print("Loading vision encoder...")
    vision_encoder = load_huggingface_vision_model("openai/clip-vit-large-patch14")

    print("Loading language model...")
    language_model = load_huggingface_language_model("mistralai/Mistral-7B-Instruct-v0.1")

    print("Loading tokenizer...")
    tokenizer = load_huggingface_tokenizer("mistralai/Mistral-7B-Instruct-v0.1")

    # Get model dimensions
    dims = get_model_dimensions("mistralai/Mistral-7B-Instruct-v0.1")
    print(f"Model dimensions: hidden_size={dims['language_hidden_size']}, vocab_size={dims['vocab_size']}\n")

    # Create LLaVA model
    model = LLaVA(
        vision_encoder=vision_encoder,
        language_model=language_model,
        vision_hidden_size=dims["vision_hidden_size"],
        language_hidden_size=dims["language_hidden_size"],
        vocab_size=dims["vocab_size"]
    )

    model.eval()

    # Prepare image
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # Create dummy image for example
    dummy_image = torch.randn(1, 3, 224, 224)  # Replace with real image

    # Prepare prompt
    prompt = "What can you see in this image?"
    tokens = tokenizer(prompt, return_tensors="pt")

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(
            input_ids=tokens["input_ids"],
            images=dummy_image,
            attention_mask=tokens["attention_mask"]
        )

    print(f"Output shape: {outputs.logits.shape}")
    print("Inference complete!\n")


def example_with_quantization():
    """
    Example: Load large model with 4-bit quantization for memory efficiency.

    Requirements:
        pip install torch transformers bitsandbytes

    Memory requirements:
        - Full precision: ~80GB
        - 8-bit: ~20GB
        - 4-bit: ~10GB
    """
    from llava import (
        LLaVA,
        load_huggingface_vision_model,
        load_huggingface_language_model,
        get_model_dimensions
    )

    print("=== LLaVA with 4-bit Quantization (GPT-OSS 20B style) ===\n")

    # Load vision encoder (always FP32 or FP16)
    vision_encoder = load_huggingface_vision_model("openai/clip-vit-large-patch14")

    # Load 20B model with 4-bit quantization
    print("Loading 20B model with 4-bit quantization...")
    language_model = load_huggingface_language_model(
        "mistralai/Mistral-7B-Instruct-v0.1",  # Example - use actual 20B model
        device="cuda",
        load_in_4bit=True
    )

    dims = get_model_dimensions("mistralai/Mistral-7B-Instruct-v0.1")

    # Create model
    model = LLaVA(
        vision_encoder=vision_encoder,
        language_model=language_model,
        vision_hidden_size=dims["vision_hidden_size"],
        language_hidden_size=dims["language_hidden_size"],
        vocab_size=dims["vocab_size"]
    )

    model.eval()

    # Check memory usage
    if torch.cuda.is_available():
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB\n")

    # Run inference
    images = torch.randn(1, 3, 224, 224).cuda()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]]).cuda()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, images=images)

    print(f"Output shape: {outputs.logits.shape}")
    print("Inference with quantization complete!\n")


def example_batch_processing():
    """
    Example: Process multiple images in batches.

    Requirements:
        pip install torch transformers
    """
    from llava import (
        LLaVA,
        load_huggingface_vision_model,
        load_huggingface_language_model,
        load_huggingface_tokenizer,
        get_model_dimensions
    )

    print("=== Batch Processing Example ===\n")

    # Load models
    vision_encoder = load_huggingface_vision_model("openai/clip-vit-large-patch14")
    language_model = load_huggingface_language_model("mistralai/Mistral-7B-Instruct-v0.1")
    tokenizer = load_huggingface_tokenizer("mistralai/Mistral-7B-Instruct-v0.1")
    dims = get_model_dimensions("mistralai/Mistral-7B-Instruct-v0.1")

    model = LLaVA(
        vision_encoder=vision_encoder,
        language_model=language_model,
        vision_hidden_size=dims["vision_hidden_size"],
        language_hidden_size=dims["language_hidden_size"],
        vocab_size=dims["vocab_size"]
    )

    model.eval()

    # Batch of 4 images
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)

    prompts = [
        "What is in this image?",
        "Describe the scene.",
        "Analyze the content.",
        "What can you tell me about this image?"
    ]

    # Tokenize prompts
    tokens = tokenizer(prompts, padding=True, return_tensors="pt")

    print(f"Processing batch of {batch_size} images...")
    print(f"Input IDs shape: {tokens['input_ids'].shape}")
    print(f"Images shape: {images.shape}\n")

    # Run inference
    with torch.no_grad():
        outputs = model(
            input_ids=tokens["input_ids"],
            images=images,
            attention_mask=tokens["attention_mask"]
        )

    print(f"Output logits shape: {outputs.logits.shape}")
    print("Batch processing complete!\n")


def example_text_generation():
    """
    Example: Generate text from image and prompt.

    Requirements:
        pip install torch transformers
    """
    from llava import (
        LLaVA,
        load_huggingface_vision_model,
        load_huggingface_language_model,
        load_huggingface_tokenizer,
        get_model_dimensions
    )

    print("=== Text Generation Example ===\n")

    # Load models
    vision_encoder = load_huggingface_vision_model("openai/clip-vit-large-patch14")
    language_model = load_huggingface_language_model("mistralai/Mistral-7B-Instruct-v0.1")
    tokenizer = load_huggingface_tokenizer("mistralai/Mistral-7B-Instruct-v0.1")
    dims = get_model_dimensions("mistralai/Mistral-7B-Instruct-v0.1")

    model = LLaVA(
        vision_encoder=vision_encoder,
        language_model=language_model,
        vision_hidden_size=dims["vision_hidden_size"],
        language_hidden_size=dims["language_hidden_size"],
        vocab_size=dims["vocab_size"]
    )

    model.eval()

    # Image and prompt
    image = torch.randn(1, 3, 224, 224)
    prompt = "Describe this image in detail:"

    # Tokenize prompt
    tokens = tokenizer(prompt, return_tensors="pt")

    print(f"Prompt: {prompt}")
    print(f"Generating up to 50 tokens...\n")

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=tokens["input_ids"],
            images=image,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9
        )

    # Decode
    generated_text = tokenizer.decode(generated_ids[0])
    print(f"Generated text: {generated_text}\n")


def example_lora_finetuning():
    """
    Example: Fine-tune with LoRA for parameter efficiency.

    Requirements:
        pip install torch transformers peft

    This reduces trainable parameters from billions to millions.
    """
    from llava import (
        LLaVA,
        load_huggingface_vision_model,
        load_huggingface_language_model,
        load_huggingface_tokenizer,
        get_model_dimensions
    )

    try:
        from peft import get_peft_model, LoraConfig, TaskType
    except ImportError:
        print("peft not installed. Install with: pip install peft")
        return

    print("=== LoRA Fine-tuning Example ===\n")

    # Load models
    vision_encoder = load_huggingface_vision_model("openai/clip-vit-large-patch14")
    language_model = load_huggingface_language_model("mistralai/Mistral-7B-Instruct-v0.1")
    tokenizer = load_huggingface_tokenizer("mistralai/Mistral-7B-Instruct-v0.1")
    dims = get_model_dimensions("mistralai/Mistral-7B-Instruct-v0.1")

    model = LLaVA(
        vision_encoder=vision_encoder,
        language_model=language_model,
        vision_hidden_size=dims["vision_hidden_size"],
        language_hidden_size=dims["language_hidden_size"],
        vocab_size=dims["vocab_size"]
    )

    # Freeze vision encoder
    for param in model.vision_encoder.parameters():
        param.requires_grad = False

    # Apply LoRA to language model
    lora_config = LoraConfig(
        r=8,  # LoRA rank
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # Adapt attention projections
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model.language_model = get_peft_model(
        model.language_model,
        lora_config
    )

    # Show trainable parameters
    model.language_model.print_trainable_parameters()

    # Example training step
    model.train()

    images = torch.randn(2, 3, 224, 224)
    tokens = tokenizer(
        ["Image description", "Another description"],
        padding=True,
        return_tensors="pt"
    )
    labels = tokens["input_ids"].clone()

    # Forward pass
    outputs = model(
        input_ids=tokens["input_ids"],
        images=images,
        attention_mask=tokens["attention_mask"],
        labels=labels
    )

    # Backward would happen here with optimizer
    print(f"Output shape: {outputs.logits.shape}")
    print("LoRA fine-tuning setup complete!\n")


def example_multi_gpu_inference():
    """
    Example: Distribute model across multiple GPUs.

    Requirements:
        pip install torch transformers
        Multiple GPUs available
    """
    from llava import (
        LLaVA,
        load_huggingface_vision_model,
        load_huggingface_language_model,
        get_model_dimensions
    )

    if not torch.cuda.is_available():
        print("CUDA not available. This example requires GPUs.")
        return

    print("=== Multi-GPU Inference Example ===\n")
    print(f"Number of GPUs: {torch.cuda.device_count()}\n")

    # Load models with automatic device mapping
    vision_encoder = load_huggingface_vision_model("openai/clip-vit-large-patch14")
    language_model = load_huggingface_language_model(
        "mistralai/Mistral-7B-Instruct-v0.1",
        device="cuda"
    )
    dims = get_model_dimensions("mistralai/Mistral-7B-Instruct-v0.1")

    model = LLaVA(
        vision_encoder=vision_encoder,
        language_model=language_model,
        vision_hidden_size=dims["vision_hidden_size"],
        language_hidden_size=dims["language_hidden_size"],
        vocab_size=dims["vocab_size"]
    )

    model.eval()

    # Check device mapping
    if hasattr(language_model, "hf_device_map"):
        print("Language model device mapping:")
        for layer, device in language_model.hf_device_map.items():
            if isinstance(device, torch.device):
                print(f"  {layer}: {device}")

    # Run inference
    images = torch.randn(1, 3, 224, 224).cuda()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]]).cuda()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, images=images)

    print(f"\nOutput shape: {outputs.logits.shape}")
    print("Multi-GPU inference complete!\n")


if __name__ == "__main__":
    """
    Run examples by uncommenting the desired one.

    Note: These require downloading large models from Hugging Face.
    First run may take time to download models.
    """

    print("LLaVA Examples with Hugging Face Models\n")
    print("=" * 50 + "\n")

    # Uncomment to run examples:

    # example_basic_inference()
    # example_with_quantization()
    # example_batch_processing()
    # example_text_generation()
    # example_lora_finetuning()
    # example_multi_gpu_inference()

    print("Examples ready to run! Uncomment in __main__ to execute.")
