# LLaVA: Large Language and Vision Assistant

## Overview

LLaVA (Large Language and Vision Assistant) is a multimodal AI architecture that combines a pre-trained vision encoder with a large language model. It enables the system to understand both images and text, answering questions about images, describing them, and performing vision-language understanding tasks.

The core insight of LLaVA is that you can leverage existing powerful pre-trained models (CLIP for vision, LLaMA for language) and connect them with a simple projection layer, avoiding the need to train both components from scratch.

## Architecture

### High-Level Design

```
Input Image ──→ Vision Encoder ──→ Image Features ──→ Projection ──→ Merged Embeddings ──→ Language Model ──→ Output Text
                    (CLIP ViT)      [B, N, 768]      [768→768]    [Text + Images]      (LLaMA)

Input Text ──────────────────────────────────────→ Text Embeddings ──→ Merged Embeddings ──→ Language Model ──→ Output Text
                                                     [B, L, 768]      [Text + Images]      (LLaMA)
```

### Components

#### 1. Vision Encoder
- **Role**: Converts raw images into semantic feature representations
- **Typical Implementation**: CLIP Vision Transformer (ViT-L/14)
- **Input**: Images of shape `[batch_size, 3, height, width]` (RGB)
- **Output**: Patch embeddings of shape `[batch_size, num_patches, vision_hidden_size]`
  - For ViT-L: typically 256 patches of dimension 1024

**Why CLIP?**
- Pre-trained on 400M image-text pairs (robust features)
- Open-source and well-tested
- Produces aligned image and text embeddings

#### 2. Vision Projection Layer
- **Role**: Bridges the dimensional gap between vision and language models
- **Implementation**: Simple linear transformation
- **Input**: Vision features `[B, N, 1024]` (example)
- **Output**: Projected features `[B, N, 4096]` to match language model dimension
- **Why Linear?** A simple linear layer is sufficient because both representations are already well-aligned through CLIP pre-training

```python
projection = nn.Linear(vision_hidden_size, language_hidden_size)
```

#### 3. Multimodal Embedding Fusion
- **Role**: Combines text tokens and image patches into a unified sequence
- **Strategy**: Replace special `<image>` tokens in text with actual image patch embeddings
- **Sequence Construction**:
  ```
  Original: "What is in the image?" + <image>
  After:    [What, is, in, the, image, ?, img_patch_1, img_patch_2, ..., img_patch_N]
  ```
- **Attention**: The language model attends over both text and image tokens jointly

#### 4. Language Model
- **Role**: Processes the merged multimodal sequence and generates output
- **Typical Implementation**: LLaMA (7B, 13B, or larger)
- **Input**: Merged embeddings of shape `[batch_size, seq_len_total, hidden_size]`
- **Output**: Logits for next token prediction `[batch_size, seq_len, vocab_size]`

## Code Components

### `VisionProjection`
Simple linear layer that projects vision encoder outputs to language model space.

```python
class VisionProjection(nn.Module):
    def __init__(self, vision_hidden_size: int, language_hidden_size: int):
        self.linear = nn.Linear(vision_hidden_size, language_hidden_size)

    def forward(self, image_features):
        # [B, N, vision_dim] → [B, N, language_dim]
        return self.linear(image_features)
```

### `MultimodalEmbedding`
Handles the fusion of text and image embeddings.

**Key Methods:**
- `forward(input_ids, image_features, image_token_id=-100)`: Merges text and image embeddings
  - Looks for positions marked with `image_token_id` in the input sequence
  - Replaces these positions with actual image patch embeddings
  - Returns merged embeddings and attention mask

**Design Decisions:**
- Uses -100 as default masking token (standard in PyTorch for ignoring positions)
- Handles variable batch sizes
- Creates proper attention masks to prevent attention to padding

### `LLaVA` (Main Model)
The complete multimodal model.

**Architecture:**
```python
class LLaVA(nn.Module):
    def __init__(self, vision_encoder, language_model, ...):
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.projection = VisionProjection(...)
        self.multimodal_embedding = MultimodalEmbedding(...)
```

**Key Methods:**

1. **`encode_image(images)`**: Passes images through vision encoder
   - Input: `[B, 3, H, W]`
   - Output: `[B, num_patches, vision_hidden_size]`

2. **`project_image_features(image_features)`**: Projects vision features to language space
   - Input: `[B, num_patches, vision_hidden_size]`
   - Output: `[B, num_patches, language_hidden_size]`

3. **`forward(...)`**: Complete forward pass
   - Encodes images (if provided)
   - Projects image features
   - Merges with text embeddings
   - Runs through language model
   - Optional: Computes loss if labels provided

4. **`generate(...)`**: Autoregressive text generation
   - Takes images and initial prompt
   - Iteratively generates tokens with temperature and nucleus sampling
   - Supports diverse generation strategies

## Usage Examples

### Training Setup
```python
import torch
from llava import LLaVA, create_dummy_vision_encoder, create_dummy_language_model

# Initialize components
vision_encoder = create_dummy_vision_encoder(hidden_size=768)
language_model = create_dummy_language_model(hidden_size=768, vocab_size=32000)

# Create LLaVA model
model = LLaVA(
    vision_encoder=vision_encoder,
    language_model=language_model,
    vision_hidden_size=768,
    language_hidden_size=768,
    vocab_size=32000,
    image_token_id=32000  # Special token marking image positions
)

# Prepare data
batch_size = 4
height, width = 224, 224
images = torch.randn(batch_size, 3, height, width)
input_ids = torch.randint(0, 32000, (batch_size, 128))
labels = torch.randint(0, 32000, (batch_size, 128))

# Forward pass with loss computation
outputs = model(
    input_ids=input_ids,
    images=images,
    labels=labels
)

loss = outputs.logits  # In real implementation, compute cross-entropy
```

### Inference with Generation
```python
# Generate text from image + prompt
prompt_ids = torch.tensor([[32000, 11, 4998, 338, 297, 1339, 29973]])  # Example token IDs
images = torch.randn(1, 3, 224, 224)

generated_ids = model.generate(
    input_ids=prompt_ids,
    images=images,
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9
)

# Decode tokens to text (requires tokenizer)
# output_text = tokenizer.decode(generated_ids[0])
```

## Training Considerations

### Data Format
- **Images**: RGB format, typically normalized with ImageNet statistics
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]
- **Text**: Tokenized input with special image tokens
  - Example: "What is shown? <image>" → token IDs with image_token_id at image position
- **Labels**: Same length as input_ids, with -100 at positions to ignore (image positions, etc.)

### Optimization Tips
1. **Projection Layer Only**: In early training, keep vision encoder and language model frozen, only train the projection layer
2. **Multi-stage Training**:
   - Stage 1: Train projection only (1-2 epochs)
   - Stage 2: Unfreeze language model layers near input (lower layers)
   - Stage 3: Fine-tune entire model with low learning rate
3. **Learning Rate**: Use lower LR for freezing pre-trained components (typically 2e-5 for projection-only)
4. **Batch Size**: Larger batches (32-64) help with multimodal training
5. **Mixed Precision**: Use fp16 to reduce memory and speed up training

## Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| vision_hidden_size | 1024 | CLIP ViT-L output dimension |
| language_hidden_size | 4096 | LLaMA hidden dimension (depends on model size) |
| vocab_size | 32000 | LLaMA vocabulary size |
| image_token_id | 32000 | Special token marking image positions |
| max_num_patches | 576 | Maximum image patches in sequence |

## Performance Notes

### Computational Complexity
- **Vision Encoding**: O(N²) where N = number of patches (typically ~256, fast)
- **Projection**: O(N × D) where D = hidden dimension
- **Language Model**: O(L² × D) where L = sequence length (bottleneck)

### Memory Usage
- Vision Encoder: ~1-2 GB (frozen, no gradients)
- Language Model: Varies by size (7B model ~14-28 GB depending on precision)
- Projected Features: ~256 × 4096 × 4 bytes ≈ 4 MB per image

### Optimization Strategies
1. **Gradient Checkpointing**: Reduces memory at cost of compute
2. **LoRA**: Low-rank adaptation for efficient fine-tuning
3. **Flash Attention**: Faster attention computation
4. **Quantization**: 8-bit or 4-bit language model for inference

## Real-World Implementation Details

### Using Hugging Face Models

The `llava.py` module includes helper functions to load real models from Hugging Face Hub. Here are practical examples:

#### Quick Start with Hugging Face

```python
from llava import (
    LLaVA,
    load_huggingface_vision_model,
    load_huggingface_language_model,
    load_huggingface_tokenizer,
    get_model_dimensions
)
import torch

# Load models
vision_encoder = load_huggingface_vision_model("openai/clip-vit-large-patch14")
language_model = load_huggingface_language_model("mistralai/Mistral-7B-Instruct-v0.1")
tokenizer = load_huggingface_tokenizer("mistralai/Mistral-7B-Instruct-v0.1")

# Get dimensions
dims = get_model_dimensions("mistralai/Mistral-7B-Instruct-v0.1")

# Create LLaVA model
model = LLaVA(
    vision_encoder=vision_encoder,
    language_model=language_model,
    vision_hidden_size=dims["vision_hidden_size"],
    language_hidden_size=dims["language_hidden_size"],
    vocab_size=dims["vocab_size"]
)

# Use for inference
images = torch.randn(1, 3, 224, 224)
prompt = "What is in this image?"
tokens = tokenizer(prompt, return_tensors="pt")

outputs = model(
    input_ids=tokens["input_ids"],
    images=images,
    attention_mask=tokens["attention_mask"]
)
```

#### Popular Language Models Available on Hugging Face

| Model | Size | License | Speed | Quality | Use Case |
|-------|------|---------|-------|---------|----------|
| Mistral-7B-Instruct | 7B | Apache 2.0 | Fast | Good | General purpose, recommended |
| Falcon-7B | 7B | Apache 2.0 | Fast | Good | Licensed for commercial use |
| OpenHermes-2.5-Mistral-7B | 7B | MIT | Fast | Very Good | Instruction-following, better reasoning |
| Llama-2-7B | 7B | Llama License | Fast | Good | Meta's model (needs approval) |
| Llama-2-13B | 13B | Llama License | Medium | Very Good | Better performance, more VRAM needed |
| StarCoder-15B | 15B | BigCode License | Medium | Good | Code generation focus |
| GPT-OSS models | Varies | Varies | Depends | Depends | Check specific model docs |

#### Loading GPT-OSS 20B (or similar larger models)

If you're using GPT-OSS 20B or similar models:

```python
from llava import (
    LLaVA,
    load_huggingface_vision_model,
    load_huggingface_language_model,
    load_huggingface_tokenizer,
    get_model_dimensions
)

# Option 1: Load from Hugging Face Hub (if available)
vision_encoder = load_huggingface_vision_model("openai/clip-vit-large-patch14")
language_model = load_huggingface_language_model(
    "gpt-oss/20b",  # Model name on HF Hub
    device="cuda",
    load_in_4bit=True  # 4-bit quantization to fit in memory
)
tokenizer = load_huggingface_tokenizer("gpt-oss/20b")

# Option 2: Load from local path
language_model = load_huggingface_language_model(
    "/path/to/gpt-oss-20b",
    load_in_4bit=True
)

# Get dimensions - for 20B models typically 4096-5120 hidden size
dims = get_model_dimensions("gpt-oss/20b")

# Create model
model = LLaVA(
    vision_encoder=vision_encoder,
    language_model=language_model,
    vision_hidden_size=dims["vision_hidden_size"],
    language_hidden_size=dims["language_hidden_size"],
    vocab_size=dims["vocab_size"]
)
```

#### Memory Requirements for Different Models

| Model | Full Precision | 8-bit | 4-bit |
|-------|----------------|-------|-------|
| 7B | 28 GB | 7 GB | 3.5 GB |
| 13B | 52 GB | 13 GB | 7 GB |
| 20B | 80 GB | 20 GB | 10 GB |
| Vision Encoder (CLIP-L) | 0.5 GB | 0.5 GB | 0.5 GB |

**Note**: These are approximate. Actual usage depends on batch size and sequence length.

#### Example: Using 4-bit Quantization for Large Models

```python
from llava import load_huggingface_language_model, LLaVA
import torch

# Load 20B model in 4-bit to fit in 24GB VRAM
language_model = load_huggingface_language_model(
    "gpt-oss/20b",
    device="cuda",
    load_in_4bit=True
)

# For batch inference, you might need gradient checkpointing
model = LLaVA(
    vision_encoder=vision_encoder,
    language_model=language_model,
    vision_hidden_size=1024,
    language_hidden_size=5120,  # Typical for 20B models
    vocab_size=50256
)

# Enable gradient checkpointing for lower memory
if hasattr(language_model, "gradient_checkpointing_enable"):
    language_model.gradient_checkpointing_enable()
```

#### Inference with Different Models

```python
import torch
from transformers import pipeline

# Create a text generation pipeline for efficient inference
generator = pipeline(
    "text-generation",
    model=language_model,
    tokenizer=tokenizer,
    device=0
)

# Or use the model directly for more control
with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        images=images,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9
    )
```

#### Installing Required Packages

```bash
# Core dependencies
pip install torch torchvision transformers

# For Hugging Face models
pip install huggingface-hub

# For quantization (4-bit and 8-bit)
pip install bitsandbytes

# For Flash Attention (faster inference)
pip install flash-attn

# For optional: DPO, LoRA fine-tuning
pip install peft

# For image processing
pip install Pillow

# For tokenizer (if not included)
pip install tokenizers
```

### From This Implementation to Production

1. **Vision Encoder**: Use helper function
   ```python
   from llava import load_huggingface_vision_model
   vision_encoder = load_huggingface_vision_model("openai/clip-vit-large-patch14")
   ```

2. **Language Model**: Use helper function with optional quantization
   ```python
   from llava import load_huggingface_language_model
   language_model = load_huggingface_language_model(
       "mistralai/Mistral-7B-Instruct-v0.1",
       load_in_4bit=True  # Optional: for memory efficiency
   )
   ```

3. **Tokenizer**: Use helper function
   ```python
   from llava import load_huggingface_tokenizer
   tokenizer = load_huggingface_tokenizer("mistralai/Mistral-7B-Instruct-v0.1")
   ```

4. **Image Processing**: Use standard preprocessing
   ```python
   from torchvision import transforms
   image_transform = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
   ])
   ```

### Working with Large Models: GPT-OSS 20B and Similar

GPT-OSS 20B is a large open-source model that requires careful management. Here are best practices:

#### Model Specifics

**GPT-OSS 20B Characteristics:**
- Parameters: 20 billion
- Hidden size: Typically 5120
- Vocabulary: 50257 (similar to GPT-2)
- Training: Trained on diverse open-source datasets
- License: Check the specific model's license on Hugging Face

#### Loading GPT-OSS 20B

```python
from llava import load_huggingface_language_model, LLaVA
import torch

# Standard loading (requires ~80GB VRAM for FP32)
model = load_huggingface_language_model("gpt-oss/20b")

# Recommended: 4-bit quantization (requires ~10GB VRAM)
model = load_huggingface_language_model(
    "gpt-oss/20b",
    device="cuda:0",
    load_in_4bit=True
)

# Alternative: 8-bit quantization (requires ~20GB VRAM)
model = load_huggingface_language_model(
    "gpt-oss/20b",
    device="cuda:0",
    load_in_8bit=True
)
```

#### Multi-GPU Inference

For distributed inference across multiple GPUs:

```python
from llava import load_huggingface_language_model

# Automatically distribute across available GPUs
model = load_huggingface_language_model(
    "gpt-oss/20b",
    device="cuda"  # Uses device_map="auto"
)

# Check model placement
print(model.hf_device_map)
```

#### Inference Optimization

```python
import torch
from llava import LLaVA

# Create model with optimizations
model = LLaVA(
    vision_encoder=vision_encoder,
    language_model=language_model,
    vision_hidden_size=1024,
    language_hidden_size=5120,  # GPT-OSS 20B
    vocab_size=50257
)

# Use Flash Attention for faster inference (if available)
try:
    from flash_attn import flash_attn_2_cuda_packed_ref
    model.language_model.config.use_flash_attention_2 = True
except ImportError:
    print("Flash Attention not available, using standard attention")

# Set to evaluation mode
model.eval()

# Inference
with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
    outputs = model.generate(
        input_ids=input_ids,
        images=images,
        max_new_tokens=100,
        temperature=0.7
    )
```

#### Fine-tuning Large Models

For efficient fine-tuning of 20B models, use Parameter-Efficient Fine-Tuning (PEFT):

```python
from peft import get_peft_model, LoraConfig, TaskType
from llava import LLaVA

# Create base model
model = LLaVA(...)

# Apply LoRA (Low-Rank Adaptation)
lora_config = LoraConfig(
    r=8,  # LoRA rank
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model.language_model = get_peft_model(
    model.language_model,
    lora_config
)

# Now only ~1-2% of parameters are trainable
print(model.language_model.print_trainable_parameters())
```

#### Batch Processing with Large Models

```python
import torch
from torch.utils.data import DataLoader

# Use smaller batch sizes with large models
batch_size = 2  # Reduced batch size for 20B model

# Consider gradient accumulation for larger effective batch size
accumulation_steps = 8

for batch_idx, batch in enumerate(data_loader):
    images = batch['images']
    input_ids = batch['input_ids']
    labels = batch['labels']

    outputs = model(
        input_ids=input_ids,
        images=images,
        labels=labels
    )

    loss = outputs.logits.mean()  # Simplified loss computation
    loss.backward()

    # Update weights every accumulation_steps batches
    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Common Modifications

**Dynamic Image Token Handling:**
The current implementation assumes a fixed number of image tokens. For variable-length images:
- Pad images to fixed size, or
- Use multiple image tokens and handle variable counts in `MultimodalEmbedding`

**Positional Embeddings:**
Consider adding learnable positional embeddings for image patches:
```python
self.image_pos_embed = nn.Parameter(torch.randn(1, num_patches, hidden_size))
```

**Multiple Images:**
Extend sequence construction to handle multiple images per prompt:
```
"Image 1 <image1> What about Image 2? <image2>"
```

## Related Work

- **CLIP** (Radford et al., 2021): Vision-language pre-training foundation
- **LLaMA** (Touvron et al., 2023): Efficient large language model
- **LLaVA Paper** (Liu et al., 2023): Original architecture and training details
- **BLIP** (Li et al., 2022): Alternative vision-language architecture
- **GPT-4V**: Commercial multimodal model

## References

1. OpenAI CLIP: https://github.com/openai/CLIP
2. LLaVA GitHub: https://github.com/haotian-liu/LLaVA
3. LLaVA Paper: https://arxiv.org/abs/2304.08485
4. LLaMA: https://arxiv.org/abs/2302.13971
5. Vision Transformer: https://arxiv.org/abs/2010.11929
