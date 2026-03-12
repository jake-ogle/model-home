# Using LLaVA with Different Language Models

Quick reference guide for loading and using various language models with LLaVA.

## Model Comparison

### Compact Models (Fast, Low Memory)

#### Mistral-7B-Instruct (Recommended for most use cases)
```python
from llava import (
    load_huggingface_language_model,
    load_huggingface_tokenizer,
    get_model_dimensions,
    LLaVA
)

model = load_huggingface_language_model("mistralai/Mistral-7B-Instruct-v0.1")
tokenizer = load_huggingface_tokenizer("mistralai/Mistral-7B-Instruct-v0.1")
dims = get_model_dimensions("mistralai/Mistral-7B-Instruct-v0.1")
```

**Stats:**
- Size: 7B parameters
- VRAM: 28 GB (FP32), 7 GB (8-bit), 3.5 GB (4-bit)
- Speed: Very fast
- Quality: Excellent instruction following
- License: Apache 2.0
- Best for: Most applications, production deployment

#### Falcon-7B
```python
model = load_huggingface_language_model("tiiuae/falcon-7b")
tokenizer = load_huggingface_tokenizer("tiiuae/falcon-7b")
```

**Stats:**
- Size: 7B parameters
- VRAM: 28 GB (FP32), 7 GB (8-bit)
- Speed: Very fast
- Quality: Good general purpose
- License: Apache 2.0 (commercial friendly)
- Best for: Applications where permissive license is critical

#### OpenHermes-2.5-Mistral-7B (Best reasoning)
```python
model = load_huggingface_language_model("teknium/OpenHermes-2.5-Mistral-7B")
tokenizer = load_huggingface_tokenizer("teknium/OpenHermes-2.5-Mistral-7B")
```

**Stats:**
- Size: 7B parameters
- Base model: Mistral-7B fine-tuned
- VRAM: 28 GB (FP32), 7 GB (8-bit)
- Quality: Excellent reasoning and complex tasks
- License: MIT
- Best for: Tasks requiring detailed reasoning

### Mid-Size Models (Balanced)

#### Llama-2-13B
```python
# Note: Requires Meta's model approval
model = load_huggingface_language_model("meta-llama/Llama-2-13b-hf")
tokenizer = load_huggingface_tokenizer("meta-llama/Llama-2-13b-hf")
```

**Stats:**
- Size: 13B parameters
- VRAM: 52 GB (FP32), 13 GB (8-bit), 7 GB (4-bit)
- Speed: Medium
- Quality: Better than 7B, good reasoning
- License: Llama Community License (requires approval)
- Best for: Complex tasks, commercial applications (with approval)

#### Llama-2-13B-Chat
```python
# Instruction-tuned version of Llama-2-13B
model = load_huggingface_language_model("meta-llama/Llama-2-13b-chat-hf")
tokenizer = load_huggingface_tokenizer("meta-llama/Llama-2-13b-chat-hf")
```

**Stats:**
- Size: 13B parameters
- Quality: Better for conversational tasks
- Best for: Chat and instruction-following scenarios

### Large Models (High Quality, High Resource)

#### GPT-OSS 20B (and similar 20B models)
```python
# With 4-bit quantization (recommended for 20B)
model = load_huggingface_language_model(
    "gpt-oss/20b",  # Or your specific GPT-OSS model identifier
    device="cuda",
    load_in_4bit=True
)
tokenizer = load_huggingface_tokenizer("gpt-oss/20b")
dims = get_model_dimensions("gpt-oss/20b")
```

**Stats:**
- Size: 20B parameters
- VRAM: 80 GB (FP32), 20 GB (8-bit), 10 GB (4-bit)
- Speed: Medium (but high quality output)
- Quality: Very high, complex reasoning
- License: Varies (check specific model)
- Best for: High-quality outputs when resources available

**Memory-Efficient Loading:**
```python
# 4-bit quantization (best for limited VRAM)
model = load_huggingface_language_model(
    "gpt-oss/20b",
    load_in_4bit=True
)

# 8-bit quantization (medium compression)
model = load_huggingface_language_model(
    "gpt-oss/20b",
    load_in_8bit=True
)

# Multi-GPU distribution
model = load_huggingface_language_model(
    "gpt-oss/20b",
    device="cuda"  # Automatically distributes across GPUs
)
```

#### Code-Focused Models

##### StarCoder-15B
```python
model = load_huggingface_language_model("bigcode/starcoder")
tokenizer = load_huggingface_tokenizer("bigcode/starcoder")
```

**Stats:**
- Size: 15B parameters
- Specialty: Code generation
- License: BigCode OpenRAIL
- Best for: Vision-based code understanding

## Choosing the Right Model

### Decision Tree

```
Do you have limited resources? (< 8GB VRAM)
├─ YES → Quantized Mistral-7B (4-bit)
│         load_in_4bit=True
│
└─ NO
    └─ Do you need production performance?
        ├─ YES → Mistral-7B-Instruct
        │         (Fast, reliable, good quality)
        │
        └─ NO
            └─ Do you need complex reasoning?
                ├─ YES → OpenHermes-2.5-Mistral-7B
                │         (Better reasoning)
                │
                └─ NO
                    └─ Do you need maximum quality?
                        ├─ YES → GPT-OSS 20B (4-bit)
                        │         (Best output, slower)
                        │
                        └─ NO → Mistral-7B-Instruct
                                (Balanced choice)
```

## Quick Start Templates

### Template 1: Memory-Constrained (< 10GB VRAM)
```python
from llava import (
    LLaVA,
    load_huggingface_vision_model,
    load_huggingface_language_model,
    load_huggingface_tokenizer,
    get_model_dimensions
)

# Load with 4-bit quantization
vision_encoder = load_huggingface_vision_model("openai/clip-vit-large-patch14")
language_model = load_huggingface_language_model(
    "mistralai/Mistral-7B-Instruct-v0.1",
    device="cuda",
    load_in_4bit=True
)
tokenizer = load_huggingface_tokenizer("mistralai/Mistral-7B-Instruct-v0.1")
dims = get_model_dimensions("mistralai/Mistral-7B-Instruct-v0.1")

model = LLaVA(
    vision_encoder=vision_encoder,
    language_model=language_model,
    vision_hidden_size=dims["vision_hidden_size"],
    language_hidden_size=dims["language_hidden_size"],
    vocab_size=dims["vocab_size"]
)
```

### Template 2: Production (Balanced)
```python
from llava import (
    LLaVA,
    load_huggingface_vision_model,
    load_huggingface_language_model,
    load_huggingface_tokenizer,
    get_model_dimensions
)

vision_encoder = load_huggingface_vision_model("openai/clip-vit-large-patch14")
language_model = load_huggingface_language_model(
    "mistralai/Mistral-7B-Instruct-v0.1"
)
tokenizer = load_huggingface_tokenizer("mistralai/Mistral-7B-Instruct-v0.1")
dims = get_model_dimensions("mistralai/Mistral-7B-Instruct-v0.1")

model = LLaVA(
    vision_encoder=vision_encoder,
    language_model=language_model,
    vision_hidden_size=dims["vision_hidden_size"],
    language_hidden_size=dims["language_hidden_size"],
    vocab_size=dims["vocab_size"]
).eval()

# Inference
with torch.no_grad():
    outputs = model(input_ids=tokens, images=images)
```

### Template 3: High Quality (Max Resources)
```python
from llava import (
    LLaVA,
    load_huggingface_vision_model,
    load_huggingface_language_model,
    load_huggingface_tokenizer,
    get_model_dimensions
)

vision_encoder = load_huggingface_vision_model("openai/clip-vit-large-patch14")
language_model = load_huggingface_language_model(
    "gpt-oss/20b",
    device="cuda",
    load_in_4bit=True  # Use quantization even with 20B
)
tokenizer = load_huggingface_tokenizer("gpt-oss/20b")
dims = get_model_dimensions("gpt-oss/20b")

model = LLaVA(
    vision_encoder=vision_encoder,
    language_model=language_model,
    vision_hidden_size=dims["vision_hidden_size"],
    language_hidden_size=dims["language_hidden_size"],
    vocab_size=dims["vocab_size"]
).eval()
```

### Template 4: Fine-tuning
```python
from llava import (
    LLaVA,
    load_huggingface_vision_model,
    load_huggingface_language_model,
    load_huggingface_tokenizer,
    get_model_dimensions
)
from peft import get_peft_model, LoraConfig, TaskType

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

# Apply LoRA
lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM
)
model.language_model = get_peft_model(model.language_model, lora_config)
model.language_model.print_trainable_parameters()
```

## Installation

### Minimal Installation
```bash
pip install torch transformers pillow
```

### Full Installation (recommended)
```bash
# Core
pip install torch transformers pillow

# Hugging Face utilities
pip install huggingface-hub

# Quantization support
pip install bitsandbytes

# Fine-tuning
pip install peft

# Optimization
pip install flash-attn

# Data handling
pip install datasets

# Utilities
pip install tqdm scikit-learn
```

## Troubleshooting

### Model Download Issues
```python
# Set cache directory
import os
os.environ['HF_HOME'] = '/path/to/cache'

# Or use huggingface_hub
from huggingface_hub import login
login()  # Authenticate if needed
```

### Out of Memory
```python
# 1. Try 4-bit quantization
load_in_4bit=True

# 2. Use smaller model
"mistralai/Mistral-7B-Instruct-v0.1"  # Instead of larger models

# 3. Use gradient checkpointing
model.language_model.gradient_checkpointing_enable()

# 4. Reduce batch size
batch_size = 1  # Instead of larger batches
```

### Slow Inference
```python
# 1. Use Flash Attention
model.language_model.config.use_flash_attention_2 = True

# 2. Use smaller model
# 3. Use quantization
load_in_4bit=True

# 4. Enable evaluation mode
model.eval()

# 5. Use no_grad context
with torch.no_grad():
    outputs = model(...)
```

## Performance Benchmarks

Approximate inference time (single GPU, batch_size=1):

| Model | Mode | Time/Token | VRAM |
|-------|------|-----------|------|
| Mistral-7B | FP16 | 50ms | 15GB |
| Mistral-7B | 8-bit | 55ms | 7GB |
| Mistral-7B | 4-bit | 60ms | 3.5GB |
| Llama-2-13B | FP16 | 90ms | 26GB |
| Llama-2-13B | 8-bit | 100ms | 13GB |
| GPT-OSS-20B | FP16 | 150ms | 40GB |
| GPT-OSS-20B | 4-bit | 180ms | 10GB |

*Times vary based on hardware, batch size, and sequence length*

## References

- [Hugging Face Model Hub](https://huggingface.co/models)
- [Mistral AI](https://www.mistral.ai/)
- [Meta Llama](https://www.meta.com/llama/)
- [Falcon Model](https://falconllm.tii.ae/)
- [Open Hermes](https://github.com/teknium1/OpenHermes)
- [BigCode](https://www.bigcode.org/)
