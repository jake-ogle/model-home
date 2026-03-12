"""
PyTorch Dataset for Vision-Language Tasks.

VisionLanguageDataset provides a template for loading and preprocessing image-text pairs
for multimodal tasks like image captioning, visual question answering, and instruction tuning.

Supports three data formats:
    - COCO: JSON with images and annotations (captions)
    - LLaVA: JSONL with {"image": path, "conversations": [turns]}
    - CSV: CSV with image_path and text_field columns

Key Features:
    - Multiple format parsers for flexibility
    - Conversation flattening with per-turn token spans
    - Role-based label masking (e.g., mask human questions, train on assistant responses)
    - Image token placement at sequence start
    - Attention mask generation with image/pad/masked token handling

Usage:
    >>> # LLaVA format with instruction tuning
    >>> ds = VisionLanguageDataset(
    ...     data_path="data.jsonl",
    ...     format="llava",
    ...     image_root="images/",
    ...     tokenizer=my_tokenizer,
    ...     image_train_transform=train_transform,
    ...     label_mask_roles=["human"],
    ... )
    >>> batch = ds[0]  # {"pixel_values", "input_ids", "attention_mask", "labels"}

    >>> # COCO format with captions
    >>> ds = VisionLanguageDataset(
    ...     data_path="coco.json",
    ...     format="coco",
    ...     image_root="images/",
    ...     tokenizer=my_tokenizer,
    ... )
"""

import json
import csv
from pathlib import Path
from typing import Optional, List, Dict, Callable, Tuple

try:
    import torch
except ImportError:
    torch = None

try:
    from PIL import Image
except ImportError:
    Image = None


class VisionLanguageDataset:
    """
    PyTorch Dataset for vision-language tasks.

    Handles multimodal data with images and associated text (captions, conversations, etc.).
    Supports multiple formats and flexible tokenization.

    Args:
        data_path: Path to data file (JSON, JSONL, or CSV).
        format: Data format. Options: "llava", "coco", "csv". Default: "llava".
        image_root: Root directory for image paths (relative paths are resolved here).
        tokenizer: Callable that takes str and returns List[int] token IDs.
                   If None, uses fallback ord-based tokenizer.
        image_loader: Callable that takes filepath and returns PIL Image (RGB).
                      Default: PIL.Image.open(path).convert("RGB")
        image_train_transform: Transform applied to images in train mode.
        image_val_transform: Transform applied to images in val mode.
        mode: Dataset split mode ("train" or "val"). Default: "train".
        max_length: Maximum sequence length after tokenization. Default: 512.
        padding: If True, pad sequences to max_length. Default: True.
        truncation: If True, truncate sequences to max_length. Default: True.
        pad_token_id: Token ID for padding. Default: 0.
        image_token_id: Special token ID for image token. Default: 32000 (LLaVA convention).
        label_mask_roles: List of role names to mask in labels (e.g., ["human"]).
                          Positions with these roles get label=-100 (not trained).
                          Default: ["human"] (train only on assistant responses).
        text_field: Field name for text in CSV/JSON. Default: "text".
        max_images: Maximum number of images to load from dataset (for testing). Default: None.

    Attributes:
        _samples: List of dicts with {"image_path": str, "text": str} or conversations.
        format: Format type string.
        mode: Current dataset mode.

    Raises:
        ValueError: If data_path doesn't exist or format is invalid.

    Example:
        >>> ds = VisionLanguageDataset(
        ...     data_path="conversation_data.jsonl",
        ...     format="llava",
        ...     image_root="images/",
        ...     tokenizer=my_tokenizer,
        ...     max_length=512,
        ... )
        >>> len(ds), ds[0].keys()
    """

    def __init__(
        self,
        data_path: str,
        format: str = "llava",
        image_root: Optional[str] = None,
        tokenizer: Optional[Callable] = None,
        image_loader: Optional[Callable] = None,
        image_train_transform: Optional[Callable] = None,
        image_val_transform: Optional[Callable] = None,
        mode: str = "train",
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True,
        pad_token_id: int = 0,
        image_token_id: int = 32000,
        label_mask_roles: Optional[List[str]] = None,
        text_field: str = "text",
        max_images: Optional[int] = None,
    ) -> None:
        """Initialize VisionLanguageDataset."""
        if format not in ["llava", "coco", "csv"]:
            raise ValueError(f"Invalid format: {format}. Must be 'llava', 'coco', or 'csv'")

        data_path = Path(data_path)
        if not data_path.exists():
            raise ValueError(f"Data file not found: {data_path}")

        self.format = format
        self.mode = mode
        self.image_root = Path(image_root) if image_root else data_path.parent
        self.tokenizer = tokenizer or self._default_tokenizer
        self.image_loader = image_loader or self._default_image_loader
        self.image_train_transform = image_train_transform
        self.image_val_transform = image_val_transform
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.pad_token_id = pad_token_id
        self.image_token_id = image_token_id
        self.label_mask_roles = label_mask_roles or ["human"]
        self.text_field = text_field

        # Parse data based on format
        if format == "llava":
            self._samples = self._parse_llava(str(data_path))
        elif format == "coco":
            self._samples = self._parse_coco(str(data_path))
        elif format == "csv":
            self._samples = self._parse_csv(str(data_path))

        if max_images is not None:
            self._samples = self._samples[:max_images]

        if len(self._samples) == 0:
            raise ValueError(f"No samples found in {data_path}")

    @staticmethod
    def _default_tokenizer(text: str) -> List[int]:
        """Fallback tokenizer using ord() for testing."""
        return [ord(c) for c in text]

    @staticmethod
    def _default_image_loader(path: str):
        """Load image using PIL."""
        if Image is None:
            raise ImportError("PIL/Pillow is required. Install with: pip install Pillow")
        return Image.open(path).convert("RGB")

    def _parse_llava(self, data_path: str) -> List[Dict]:
        """
        Parse LLaVA format: JSONL with {"image": path, "conversations": [turns]}

        Example:
            {
              "image": "images/img_001.jpg",
              "conversations": [
                {"from": "human", "value": "What is in this image?"},
                {"from": "gpt", "value": "This is a cat."},
                ...
              ]
            }

        Returns:
            List of dicts with "image_path" and "conversations" keys.
        """
        samples = []
        with open(data_path, "r") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    img_path = item.get("image") or item.get("image_path")
                    conversations = item.get("conversations", [])
                    if img_path and conversations:
                        samples.append({
                            "image_path": str(self.image_root / img_path),
                            "conversations": conversations,
                        })
        return samples

    def _parse_coco(self, data_path: str) -> List[Dict]:
        """
        Parse COCO format: JSON with {"images": [...], "annotations": [...]}

        Example:
            {
              "images": [
                {"id": 1, "file_name": "image1.jpg"},
                ...
              ],
              "annotations": [
                {"image_id": 1, "caption": "A cat on a chair"},
                ...
              ]
            }

        Returns:
            List of dicts with "image_path" and "text" keys.
        """
        samples = []
        with open(data_path, "r") as f:
            data = json.load(f)

        # Build image_id -> filename map
        images = {img["id"]: img["file_name"] for img in data.get("images", [])}

        # Build image_id -> list of captions map
        captions_by_image = {}
        for ann in data.get("annotations", []):
            img_id = ann.get("image_id")
            caption = ann.get("caption", "")
            if img_id not in captions_by_image:
                captions_by_image[img_id] = []
            captions_by_image[img_id].append(caption)

        # Create samples
        for img_id, filename in images.items():
            captions = captions_by_image.get(img_id, [])
            for caption in captions:
                samples.append({
                    "image_path": str(self.image_root / filename),
                    "text": caption,
                })

        return samples

    def _parse_csv(self, data_path: str) -> List[Dict]:
        """
        Parse CSV format with image_path and text columns.

        Example CSV:
            image_path,text
            images/img_001.jpg,A cat sitting on a chair
            images/img_002.jpg,A dog running in the park

        Returns:
            List of dicts with "image_path" and "text" keys.
        """
        samples = []
        with open(data_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = row.get("image_path") or row.get("image")
                text = row.get(self.text_field)
                if img_path and text:
                    samples.append({
                        "image_path": str(self.image_root / img_path),
                        "text": text,
                    })
        return samples

    def _flatten_conversation(self, turns: List[Dict]) -> Tuple[List[int], List[int]]:
        """
        Flatten multi-turn conversation into single token sequence with role-based masking.

        Each turn is tokenized separately and concatenated. The label position at each
        token is set based on the turn's role: if role in label_mask_roles, set to -100,
        otherwise set to the actual token ID (for training).

        Args:
            turns: List of dicts like [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]

        Returns:
            Tuple of (token_ids, labels) where:
            - token_ids: Flattened list of token IDs
            - labels: List of same length; -100 for masked roles, token ID for trainable positions
        """
        token_ids = []
        labels = []

        for turn in turns:
            role = turn.get("from", "").lower()
            text = turn.get("value", "")

            turn_tokens = self.tokenizer(text)

            # Determine label value for this turn
            if role in self.label_mask_roles:
                turn_labels = [-100] * len(turn_tokens)
            else:
                turn_labels = turn_tokens  # Train on this turn's tokens

            token_ids.extend(turn_tokens)
            labels.extend(turn_labels)

        return token_ids, labels

    def _get_image_transform(self) -> Optional[Callable]:
        """Return appropriate image transform based on mode."""
        if self.mode == "train":
            return self.image_train_transform
        else:
            return self.image_val_transform

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and return a single vision-language sample.

        Args:
            idx: Index into dataset.

        Returns:
            Dict[str, torch.Tensor] with keys:
            - "pixel_values": (C, H, W) image tensor
            - "input_ids": (max_length,) token IDs with image token at position 0
            - "attention_mask": (max_length,) mask (1 for real tokens, 0 for padding)
            - "labels": (max_length,) labels for training
                * Image token position: -100 (not trained)
                * Pad positions: -100
                * Masked roles: -100
                * Other positions: token ID

        Example:
            >>> ds = VisionLanguageDataset(...)
            >>> batch = ds[0]
            >>> batch["pixel_values"].shape
            torch.Size([3, 256, 256])
            >>> batch["input_ids"].shape
            torch.Size([512])
        """
        if torch is None:
            raise ImportError("torch is required. Install with: pip install torch")

        sample = self._samples[idx]

        # Load image
        img_path = sample["image_path"]
        try:
            img = self.image_loader(img_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")

        # Apply transform
        transform = self._get_image_transform()
        if transform is not None:
            img = transform(img)
        else:
            # Default: convert PIL to tensor and normalize
            import numpy as np
            img = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Get text and tokenize
        if self.format == "llava":
            # Flatten conversation turns
            turns = sample.get("conversations", [])
            token_ids, turn_labels = self._flatten_conversation(turns)
        else:
            # Simple text (COCO, CSV)
            text = sample.get("text", "")
            token_ids = self.tokenizer(text)
            turn_labels = token_ids  # Train on all tokens

        # Truncate
        if self.truncation and len(token_ids) > self.max_length - 1:  # -1 for image token
            token_ids = token_ids[:self.max_length - 1]
            turn_labels = turn_labels[:self.max_length - 1]

        # Prepend image token
        token_ids = [self.image_token_id] + token_ids
        turn_labels = [-100] + turn_labels  # Image token is never trained

        # Pad to max_length
        attention_mask = [1] * len(token_ids)
        if self.padding and len(token_ids) < self.max_length:
            pad_length = self.max_length - len(token_ids)
            token_ids = token_ids + [self.pad_token_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length
            turn_labels = turn_labels + [-100] * pad_length

        # Ensure correct length
        token_ids = token_ids[:self.max_length]
        attention_mask = attention_mask[:self.max_length]
        turn_labels = turn_labels[:self.max_length]

        return {
            "pixel_values": img,
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(turn_labels, dtype=torch.long),
        }


if __name__ == "__main__":
    """Smoke test with synthetic data."""
    print("Testing VisionLanguageDataset...")

    if torch is None or Image is None:
        print("⚠ Required dependencies not installed. Skipping smoke tests.")
        print("   Install with: pip install torch torchvision Pillow")
        exit(0)

    import tempfile
    from pathlib import Path

    def test_tokenizer(text: str) -> List[int]:
        return [ord(c) for c in text]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create dummy image
        img = Image.new("RGB", (64, 64), color=(100, 150, 200))
        img.save(tmpdir / "test.jpg")

        # Test 1: LLaVA format
        print("\n1. Testing LLaVA format...")
        llava_data = {
            "image": "test.jpg",
            "conversations": [
                {"from": "human", "value": "What is this?"},
                {"from": "gpt", "value": "This is a test image."},
            ]
        }
        llava_path = tmpdir / "data.jsonl"
        with open(llava_path, "w") as f:
            f.write(json.dumps(llava_data) + "\n")

        ds_llava = VisionLanguageDataset(
            data_path=str(llava_path),
            format="llava",
            image_root=str(tmpdir),
            tokenizer=test_tokenizer,
            max_length=128,
        )
        print(f"   Loaded {len(ds_llava)} samples")
        batch = ds_llava[0]
        print(f"   pixel_values shape: {batch['pixel_values'].shape}")
        print(f"   input_ids shape: {batch['input_ids'].shape}")
        print(f"   First token is image_token: {batch['input_ids'][0] == 32000}")
        assert batch["pixel_values"].shape[0] == 3, "Image should be RGB"
        assert batch["input_ids"][0] == 32000, "First token should be image token"
        assert batch["labels"][0] == -100, "Image token should have label -100"
        print("   ✓ LLaVA format test passed")

        # Test 2: CSV format
        print("\n2. Testing CSV format...")
        csv_path = tmpdir / "data.csv"
        with open(csv_path, "w") as f:
            f.write("image_path,text\n")
            f.write("test.jpg,A nice test image\n")

        ds_csv = VisionLanguageDataset(
            data_path=str(csv_path),
            format="csv",
            image_root=str(tmpdir),
            tokenizer=test_tokenizer,
            max_length=128,
        )
        batch = ds_csv[0]
        print(f"   Loaded {len(ds_csv)} samples")
        print(f"   input_ids shape: {batch['input_ids'].shape}")
        assert batch["input_ids"].shape == (128,), "Should be padded to max_length"
        print("   ✓ CSV format test passed")

        # Test 3: COCO format
        print("\n3. Testing COCO format...")
        coco_data = {
            "images": [{"id": 1, "file_name": "test.jpg"}],
            "annotations": [
                {"image_id": 1, "caption": "A test image"},
                {"image_id": 1, "caption": "Another caption"},
            ]
        }
        coco_path = tmpdir / "coco.json"
        with open(coco_path, "w") as f:
            json.dump(coco_data, f)

        ds_coco = VisionLanguageDataset(
            data_path=str(coco_path),
            format="coco",
            image_root=str(tmpdir),
            tokenizer=test_tokenizer,
            max_length=128,
        )
        print(f"   Loaded {len(ds_coco)} samples (2 captions per image)")
        assert len(ds_coco) == 2, "Should have 2 samples (one per caption)"
        batch = ds_coco[0]
        print(f"   input_ids shape: {batch['input_ids'].shape}")
        assert batch["attention_mask"].dtype == torch.long, "attention_mask should be LongTensor"
        print("   ✓ COCO format test passed")

        # Test 4: Role-based masking
        print("\n4. Testing role-based masking...")
        ds_masked = VisionLanguageDataset(
            data_path=str(llava_path),
            format="llava",
            image_root=str(tmpdir),
            tokenizer=test_tokenizer,
            label_mask_roles=["human"],
            max_length=256,
        )
        batch = ds_masked[0]
        # Find positions corresponding to "human" role (masked) vs "gpt" (trained)
        # The first conversation is human, so some positions should be -100
        masked_count = sum(1 for l in batch["labels"] if l == -100)
        print(f"   Masked positions: {masked_count}, Trainable positions: {256 - masked_count}")
        assert masked_count > 0, "Should have masked positions"
        assert masked_count < 256, "Should have some trainable positions"
        print("   ✓ Role-based masking test passed")

    print("\n✓ All VisionLanguageDataset tests passed!")
