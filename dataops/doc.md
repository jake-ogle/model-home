# DataOps: PyTorch Dataset Templates

## Overview

The `dataops` directory contains production-ready PyTorch Dataset templates for common machine learning tasks. Each template is designed to be configurable, reusable, and serving as a drop-in solution for loading and preprocessing different data modalities:

- **Vision Data** - Image classification and computer vision tasks
- **Language Data** - Text classification and language modeling
- **Vision-Language Data** - Multimodal tasks combining images and text

All datasets follow PyTorch's `Dataset` API, include comprehensive docstrings with type hints, support multiple data formats, and come with synthetic smoke tests for verification.

## Files Overview

### 1. `vision_dataset.py` - VisionDataset

**Purpose:** Image classification with flexible data loading from disk or memory.

**Key Features:**
- **Multiple data source patterns:** Folder hierarchies (ImageNet-style) or manifest files (CSV/JSON/JSONL)
- **Automatic label assignment:** Infers class names from directory structure or manifest
- **Mode-specific transforms:** Different preprocessing for train/val/test splits
- **In-memory caching:** Optional pre-loading for small datasets
- **Deterministic ordering:** Sorted loading for reproducible label mappings

**Data Format Support:**
```
Folder hierarchy:        Manifest (CSV):        Manifest (JSON):
root/                    filepath,label         [{
  cat/                   path/to/img1.jpg,cat     "filepath": "...",
    image1.jpg           path/to/img2.jpg,dog     "label": "cat"
  dog/                                          }, ...]
    image2.jpg
```

**Core API:**
```python
ds = VisionDataset(
    root="data/imagenet",  # OR manifest="data/splits.csv"
    mode="train",          # train, val, test
    train_transform=...,   # Transform for training
    val_transform=...,     # Transform for validation
    in_memory=False        # Preload all images
)

img, label = ds[0]  # (C, H, W) tensor, scalar LongTensor
print(ds.num_classes, ds.classes)
```

**Common Use Cases:**
- ImageNet-style folder structures
- CSV annotations with file paths
- Custom train/val split transforms
- Small dataset in-memory caching

---

### 2. `language_dataset.py` - LanguageDataset

**Purpose:** Text data loading and preprocessing for various NLP tasks.

**Supported Tasks:**
- **Classification** - Text classification with class labels
- **Causal LM** - Next token prediction (language modeling)
- **Masked LM** - Masked language modeling (BERT-style)

**Key Features:**
- **Custom tokenizer support:** Any `Callable[[str], List[int]]` tokenizer
- **Automatic padding/truncation:** To fixed sequence length
- **Task-specific preprocessing:** Different label handling per task
- **BERT-style masking:** 80/10/10 strategy (80% mask, 10% random, 10% keep)
- **Attention masks:** Track valid tokens vs. padding

**Data Format Support:**
```
Python list:           JSON file:           Plain text file:
["text1", "text2"]    [{"text": "..."}, ...]   Line 1
                      [{"text": "..."}]        Line 2
                                               ...
```

**Core API:**
```python
# Text Classification
ds = LanguageDataset(
    data=["positive review", "negative review"],
    task="classification",
    labels=[1, 0],
    tokenizer=tokenizer,
    max_length=512
)
batch = ds[0]
# {input_ids, attention_mask, labels}

# Causal Language Modeling
ds = LanguageDataset(
    filepath="data/corpus.txt",
    task="causal_lm",
    tokenizer=tokenizer,
    max_length=512
)

# Masked Language Modeling
ds = LanguageDataset(
    data=texts,
    task="masked_lm",
    tokenizer=tokenizer,
    mask_prob=0.15,
    max_length=512
)
```

**Task-Specific Outputs:**

| Task | Output Keys | Label Handling |
|------|-------------|-----------------|
| classification | input_ids, attention_mask, labels | Single class index |
| causal_lm | input_ids, attention_mask, labels | Shifted by 1 for next-token prediction |
| masked_lm | input_ids, attention_mask, masked_input_ids, labels | Masked positions marked |

**Common Use Cases:**
- Sentiment analysis classification
- Text generation and language modeling
- Fine-tuning pretrained models (BERT, GPT)
- Multi-task NLP experiments

---

### 3. `vl_dataset.py` - VisionLanguageDataset

**Purpose:** Multimodal data loading for vision-language tasks like image captioning and instruction-tuning.

**Supported Formats:**
- **LLaVA** - Conversation format with image-instruction-response pairs
- **COCO** - Standard annotation format with image descriptions
- **CSV** - Custom format with image paths and text descriptions

**Key Features:**
- **Conversation flattening:** Handles multi-turn dialog with proper token tracking
- **Role-based label masking:** Selectively mask certain conversation roles (e.g., human queries)
- **Image token placement:** Special token positioning for multimodal alignment
- **Multiple format parsers:** Flexible input format support
- **Mode-specific image transforms:** Different augmentation for train/val

**Data Format Examples:**

```json
// LLaVA format (JSONL - one per line)
{"image": "image1.jpg", "conversations": [
  {"from": "human", "value": "What is in this image?"},
  {"from": "assistant", "value": "The image shows..."}
]}

// COCO format (JSON)
{
  "images": [{"id": 1, "file_name": "img1.jpg"}],
  "annotations": [{"image_id": 1, "caption": "A cat sits..."}]
}

// CSV format
image_path,text
path/to/img1.jpg,This image shows...
path/to/img2.jpg,Another description...
```

**Core API:**
```python
# LLaVA format - conversation-based instruction tuning
ds = VisionLanguageDataset(
    data_path="data/llava_conversations.jsonl",
    format="llava",
    image_root="data/images",
    tokenizer=tokenizer,
    image_train_transform=train_aug,
    mode="train",
    label_mask_roles=["human"]  # Don't train on human queries
)

# COCO format - image captioning
ds = VisionLanguageDataset(
    data_path="data/coco_annotations.json",
    format="coco",
    image_root="data/coco_images",
    tokenizer=tokenizer
)

# CSV format - custom
ds = VisionLanguageDataset(
    data_path="data/image_text_pairs.csv",
    format="csv",
    image_root="data/images",
    text_field="caption"
)

batch = ds[0]
# {pixel_values (C,H,W), input_ids, attention_mask, labels}
```

**Label Masking:**
- Positions with roles in `label_mask_roles` (default `["human"]`) are set to -100 in labels
- Image token position is always -100 (don't train on image features)
- Only assistant/response positions are trained on (loss computed)

**Common Use Cases:**
- Image captioning training
- Vision-language instruction tuning (LLaVA-style)
- CLIP-style image-text matching
- Multimodal dialog systems

---

## Comparison: When to Use Each

| Dataset | Input Type | Output Format | Best For |
|---------|-----------|---------------|-----------|
| **VisionDataset** | Images | Tensors | Image classification, supervised vision tasks |
| **LanguageDataset** | Text | Dict[str, Tensor] | Text classification, language modeling |
| **VisionLanguageDataset** | Images + Text | Dict[str, Tensor] | Image captioning, multimodal instruction tuning |

---

## Common Patterns

### 1. Using Custom Tokenizers

All datasets accept any tokenizer as a callable:

```python
from transformers import AutoTokenizer

# HuggingFace tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
ds = LanguageDataset(data=texts, tokenizer=tokenizer)

# Custom function
def custom_tokenizer(text):
    return [ord(c) for c in text]  # Simple byte-level

ds = LanguageDataset(data=texts, tokenizer=custom_tokenizer)
```

### 2. Mode-Specific Preprocessing

For train/val/test splits with different transforms:

```python
from torchvision import transforms

train_aug = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_aug = transforms.Compose([
    transforms.ToTensor(),
])

ds_train = VisionDataset(root="data", mode="train", train_transform=train_aug)
ds_val = VisionDataset(root="data", mode="val", val_transform=val_aug)
```

### 3. Building DataLoaders

```python
from torch.utils.data import DataLoader

ds = VisionDataset(root="data/imagenet", mode="train")

loader = DataLoader(
    ds,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=custom_collate  # If needed
)

for batch_images, batch_labels in loader:
    # batch_images: (B, C, H, W)
    # batch_labels: (B,)
    pass
```

---

## Implementation Details

### Tensor Shapes and Dtypes

**VisionDataset:**
- `pixel_values`: `(C, H, W)`, dtype `float32`
- `label`: scalar, dtype `int64` (LongTensor)

**LanguageDataset:**
- `input_ids`: `(max_length,)`, dtype `int64`
- `attention_mask`: `(max_length,)`, dtype `int64`
- `labels`: `(max_length,)`, dtype `int64` (with -100 for padding in MLM)
- `masked_input_ids` (MLM only): `(max_length,)` with mask tokens

**VisionLanguageDataset:**
- `pixel_values`: `(C, H, W)`, dtype `float32`
- `input_ids`: `(max_length,)`, dtype `int64`
- `attention_mask`: `(max_length,)`, dtype `int64`
- `labels`: `(max_length,)`, dtype `int64` (with -100 for masked positions)

### Label Masking Strategy

In MLM and vision-language tasks, certain positions are masked during training:
- `-100` values are ignored by standard PyTorch loss functions
- Image tokens: always -100 (don't compute loss on images)
- Human queries (LLaVA): -100 (only train on assistant responses)
- Padding tokens: -100 (ignore padding in loss)

### Deterministic Ordering

All datasets sort their inputs for reproducibility:
- **VisionDataset:** Folder names and image files sorted
- **LanguageDataset:** Consistent ordering for reproducible splits
- **VisionLanguageDataset:** Conversation order preserved

---

## Testing

Each module includes smoke tests with synthetic data (no real files required):

```bash
python dataops/vision_dataset.py      # Tests VisionDataset with 2×2 synthetic images
python dataops/language_dataset.py    # Tests all three task modes with synthetic text
python dataops/vl_dataset.py          # Tests all three format parsers with synthetic data
```

All smoke tests are dependency-aware and skip gracefully if PyTorch/PIL aren't installed.

---

## Performance Considerations

### Memory Usage

| Dataset | In-Memory Cache | Size |
|---------|----------------|------|
| VisionDataset (100 images) | No | Minimal (load on demand) |
| VisionDataset (100 images) | Yes | ~50-200 MB (depends on image size) |
| LanguageDataset (10k texts) | No | Minimal |
| VisionLanguageDataset (1k) | No | Minimal (load images on demand) |

### Loading Speed

- **VisionDataset with `in_memory=True`:** Faster after initial load
- **VisionDataset with `in_memory=False`:** Slower but memory-efficient
- **LanguageDataset:** Tokenization can be bottleneck; consider pre-tokenizing
- **VisionLanguageDataset:** Image loading is typically the bottleneck

### Optimization Tips

1. Use `num_workers > 0` in DataLoader for parallel loading
2. Pin memory for GPU training: `pin_memory=True` in DataLoader
3. For large datasets, avoid `in_memory=True`
4. Pre-tokenize text data if using slow custom tokenizers
5. Use image transforms in Dataset, not in DataLoader

---

## References

- PyTorch Dataset API: https://pytorch.org/docs/stable/data.html
- Torchvision transforms: https://pytorch.org/vision/stable/transforms.html
- HuggingFace Datasets: https://huggingface.co/docs/datasets/
- HuggingFace Tokenizers: https://huggingface.co/docs/tokenizers/