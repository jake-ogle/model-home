# LLaVA: Large Language and Vision Assistant

A complete PyTorch implementation of the LLaVA architecture for multimodal AI, with integrated Hugging Face model support.

## 📁 Files Overview

### Core Implementation
- **`llava.py`** - Complete PyTorch implementation
  - `VisionProjection`: Aligns vision encoder to language model space
  - `MultimodalEmbedding`: Fuses text and image tokens
  - `LLaVA`: Main model with forward, generate, and inference methods
  - Hugging Face integration helpers for loading real models

### Documentation
- **`doc.md`** - Comprehensive architecture documentation
  - Architecture overview and component descriptions
  - Code walkthroughs with examples
  - Training considerations and optimization tips
  - Real-world integration with Hugging Face models

- **`MODELS.md`** - Model selection and usage guide
  - Comparison of 10+ language models
  - Quick-start templates for different scenarios
  - Performance benchmarks and memory requirements
  - Troubleshooting guide

- **`examples.py`** - Practical working examples
  - Basic inference
  - Quantization (4-bit, 8-bit)
  - Batch processing
  - Text generation
  - LoRA fine-tuning
  - Multi-GPU inference

## 🚀 Quick Start

### Installation
```bash
pip install torch transformers pillow
```

### Basic Usage
```python
from llava import (
    LLaVA,
    load_huggingface_vision_model,
    load_huggingface_language_model,
    load_huggingface_tokenizer,
    get_model_dimensions
)

# Load models
vision_encoder = load_huggingface_vision_model("openai/clip-vit-large-patch14")
language_model = load_huggingface_language_model("mistralai/Mistral-7B-Instruct-v0.1")
tokenizer = load_huggingface_tokenizer("mistralai/Mistral-7B-Instruct-v0.1")
dims = get_model_dimensions("mistralai/Mistral-7B-Instruct-v0.1")

# Create model
model = LLaVA(
    vision_encoder=vision_encoder,
    language_model=language_model,
    vision_hidden_size=dims["vision_hidden_size"],
    language_hidden_size=dims["language_hidden_size"],
    vocab_size=dims["vocab_size"]
)

# Inference
images = torch.randn(1, 3, 224, 224)
tokens = tokenizer("What is in this image?", return_tensors="pt")

with torch.no_grad():
    outputs = model(input_ids=tokens["input_ids"], images=images)
```

## 🎯 Features

✅ **Complete Implementation**
- Vision encoder (CLIP ViT)
- Projection layer
- Multimodal embedding fusion
- Language model integration
- Autoregressive text generation

✅ **Hugging Face Integration**
- Load models directly from Hub
- Support for quantization (4-bit, 8-bit)
- Automatic dimension detection
- Multi-GPU support

✅ **Production Ready**
- Well-documented code
- Error handling
- Memory efficiency
- Multiple optimization strategies

✅ **Flexible Architecture**
- Supports various vision encoders
- Works with any Hugging Face language model
- Configurable projection dimensions
- Trainable and inference modes

## 📊 Supported Models

### Language Models (Examples)
- **7B Models**: Mistral-7B, Llama-2-7B, Falcon-7B (fast, memory efficient)
- **13B Models**: Llama-2-13B, OpenHermes-2.5 (balanced)
- **20B Models**: GPT-OSS 20B, Falcon-40B (high quality)
- **Code Models**: StarCoder, CodeLlama (specialized)

### Vision Encoders
- CLIP ViT-L (1024-dim, recommended)
- CLIP ViT-B (512-dim, faster)
- CLIP ViT-L-336 (larger images)

See `MODELS.md` for detailed comparison and selection guide.

## 💾 Memory Requirements

| Model | Full Precision | 8-bit | 4-bit |
|-------|---|---|---|
| Mistral-7B | 28 GB | 7 GB | 3.5 GB |
| Llama-2-13B | 52 GB | 13 GB | 7 GB |
| GPT-OSS-20B | 80 GB | 20 GB | 10 GB |
| Vision (CLIP) | 0.5 GB | 0.5 GB | 0.5 GB |

## 🔧 Configuration Examples

### Memory-Constrained Setup
```python
model = load_huggingface_language_model(
    "mistralai/Mistral-7B-Instruct-v0.1",
    load_in_4bit=True  # Fits in 8GB VRAM
)
```

### High-Quality Setup
```python
model = load_huggingface_language_model(
    "gpt-oss/20b",
    load_in_4bit=True  # 4-bit GPT-OSS 20B
)
```

### Fine-tuning with LoRA
```python
from peft import get_peft_model, LoraConfig, TaskType

lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM
)
model.language_model = get_peft_model(model.language_model, lora_config)
```

## 📖 Documentation Structure

1. **Start here**: `README.md` (this file)
2. **Learn architecture**: `doc.md` - detailed explanation
3. **Choose model**: `MODELS.md` - model comparison and selection
4. **Run examples**: `examples.py` - working code samples
5. **Implement**: `llava.py` - source code

## 🎓 Learning Path

1. **Understand the architecture**
   - Read "Architecture" section in `doc.md`
   - Look at component diagrams

2. **Explore the code**
   - Read `llava.py` docstrings
   - Understand `VisionProjection`, `MultimodalEmbedding`, `LLaVA` classes

3. **Choose a model**
   - See decision tree in `MODELS.md`
   - Check memory requirements for your hardware

4. **Run examples**
   - Uncomment examples in `examples.py`
   - Start with `example_basic_inference()`

5. **Customize**
   - Modify vision encoder
   - Change language model
   - Adjust projection dimensions

## 🔬 Advanced Features

### Gradient Checkpointing
```python
model.language_model.gradient_checkpointing_enable()  # Lower memory, slower
```

### Flash Attention
```python
model.language_model.config.use_flash_attention_2 = True  # Faster inference
```

### Batch Processing
```python
batch_images = torch.randn(4, 3, 224, 224)
batch_tokens = tokenizer(["prompt1", "prompt2", "prompt3", "prompt4"],
                        padding=True, return_tensors="pt")
outputs = model(input_ids=batch_tokens["input_ids"], images=batch_images)
```

### Multi-GPU Inference
```python
model = load_huggingface_language_model("gpt-oss/20b", device="cuda")
# Automatically distributed across available GPUs
```

## 📦 Dependencies

**Core**
- `torch`: Deep learning framework
- `transformers`: Hugging Face models

**Optional**
- `bitsandbytes`: 4-bit and 8-bit quantization
- `peft`: LoRA fine-tuning
- `flash-attn`: Faster attention
- `pillow`: Image processing
- `datasets`: Data loading

## 🤝 Contributing

This is a reference implementation. Feel free to:
- Modify components for your needs
- Add new vision encoders
- Integrate different language models
- Implement new training strategies

## 📚 References

- **LLaVA Paper**: [Liu et al., 2023](https://arxiv.org/abs/2304.08485)
- **CLIP**: [Radford et al., 2021](https://arxiv.org/abs/2103.14030)
- **Vision Transformer**: [Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929)
- **Llama**: [Touvron et al., 2023](https://arxiv.org/abs/2302.13971)

## 📝 License

This implementation is provided as-is for educational and research purposes.

## ❓ FAQ

**Q: Which model should I start with?**
A: Start with Mistral-7B-Instruct. It's fast, memory-efficient, and has excellent quality.

**Q: How do I reduce memory usage?**
A: Use 4-bit quantization. Reduce batch size. Freeze vision encoder. Use gradient checkpointing.

**Q: Can I use a different vision encoder?**
A: Yes. Replace `load_huggingface_vision_model()` with any CLIP variant or custom encoder.

**Q: How do I fine-tune the model?**
A: Use LoRA (see `examples.py`). Keeps trainable parameters to 1-2% of total.

**Q: Does this support inference on CPU?**
A: Yes, but slow. Recommended: GPU with at least 8GB VRAM.

**Q: Can I use custom images?**
A: Yes. Load images with PIL/torchvision and apply standard normalization.
