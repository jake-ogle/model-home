"""
PyTorch Dataset for Language Modeling Tasks.

LanguageDataset provides a flexible, configurable template for loading and preprocessing
text data for multiple task types: text classification, causal language modeling (CLM),
and masked language modeling (MLM). Supports custom tokenizers and automatic padding/truncation.

Key Features:
    - Multiple task modes: classification, causal_lm, masked_lm
    - Custom tokenizer support (any callable that returns List[int])
    - Automatic padding and truncation to fixed sequence length
    - BERT-style masking for masked_lm tasks (80/10/10 strategy)
    - Attention mask generation for padding tokens
    - Flexible data loading from lists or files (JSON/JSONL)

Usage:
    >>> # Classification task
    >>> ds = LanguageDataset(
    ...     data=["text1", "text2"],
    ...     task="classification",
    ...     labels=[0, 1],
    ...     tokenizer=my_tokenizer,
    ... )
    >>> batch = ds[0]  # {"input_ids", "attention_mask", "labels"}

    >>> # Causal LM (next token prediction)
    >>> ds = LanguageDataset(
    ...     filepath="data.jsonl",
    ...     task="causal_lm",
    ...     tokenizer=my_tokenizer,
    ... )
    >>> batch = ds[0]  # {"input_ids", "attention_mask", "labels"}

    >>> # Masked LM (BERT-style pre-training)
    >>> ds = LanguageDataset(
    ...     data=["text1", "text2"],
    ...     task="masked_lm",
    ...     tokenizer=my_tokenizer,
    ... )
    >>> batch = ds[0]  # {"input_ids", "attention_mask", "masked_input_ids", "labels"}
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Callable
import random

try:
    import torch
except ImportError:
    torch = None


class LanguageDataset:
    """
    PyTorch Dataset for language tasks with flexible tokenization and masking.

    Supports three task modes:
    - "classification": Text classification with explicit labels
    - "causal_lm": Causal language modeling (next token prediction)
    - "masked_lm": Masked language modeling (BERT-style pre-training)

    Args:
        data: List of text strings. Exactly one of data or filepath must be provided.
        filepath: Path to file with text data (JSON, JSONL, or plain text lines).
                  Exactly one of data or filepath must be provided.
        task: Task type. Options: "classification", "causal_lm", "masked_lm". Default: "classification".
        tokenizer: Callable that takes str and returns List[int] token IDs.
                   If None, uses fallback ord()-based tokenizer (for smoke tests only).
        max_length: Maximum sequence length after tokenization. Default: 512.
        padding: If True, pad sequences to max_length. Default: True.
        truncation: If True, truncate sequences to max_length. Default: True.
        pad_token_id: Token ID for padding tokens. Default: 0.
        mask_token_id: Token ID for masked tokens (masked_lm only). Default: 103 (BERT [MASK]).
        mask_prob: Probability of masking each token in masked_lm. Default: 0.15.
        labels: List of labels for classification task. Must be same length as data.
        text_field: Field name when loading from JSON/JSONL. Default: "text".
        label_field: Field name for labels when loading from JSON/JSONL. Default: "label".

    Attributes:
        _texts: List[str] - preprocessed text samples.
        _labels: List[int] or None - integer labels (classification only).
        task: Task type string.

    Raises:
        ValueError: If neither data nor filepath provided, or both provided.
                    If task not in ["classification", "causal_lm", "masked_lm"].
                    If labels provided but doesn't match data length.

    Example:
        >>> def my_tokenizer(text: str) -> List[int]:
        ...     return [ord(c) for c in text]

        >>> ds = LanguageDataset(
        ...     data=["hello", "world"],
        ...     task="classification",
        ...     labels=[0, 1],
        ...     tokenizer=my_tokenizer,
        ...     max_length=512,
        ... )
        >>> batch = ds[0]
        >>> batch["input_ids"].shape  # (512,)
    """

    def __init__(
        self,
        data: Optional[List[str]] = None,
        filepath: Optional[str] = None,
        task: str = "classification",
        tokenizer: Optional[Callable] = None,
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True,
        pad_token_id: int = 0,
        mask_token_id: int = 103,
        mask_prob: float = 0.15,
        labels: Optional[List[int]] = None,
        text_field: str = "text",
        label_field: str = "label",
    ) -> None:
        """Initialize LanguageDataset."""
        # Validate inputs
        if (data is None and filepath is None) or (data is not None and filepath is not None):
            raise ValueError("Exactly one of 'data' or 'filepath' must be provided")

        if task not in ["classification", "causal_lm", "masked_lm"]:
            raise ValueError(f"Invalid task: {task}. Must be 'classification', 'causal_lm', or 'masked_lm'")

        self.task = task
        self.tokenizer = tokenizer or self._default_tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.mask_prob = mask_prob
        self.text_field = text_field
        self.label_field = label_field

        # Load texts
        if data is not None:
            self._texts = data
            self._labels = labels
        else:
            self._texts, self._labels = self._load_from_file(filepath)

        # Validate labels
        if self.task == "classification":
            if self._labels is None:
                raise ValueError("labels must be provided for classification task")
            if len(self._labels) != len(self._texts):
                raise ValueError(f"labels length ({len(self._labels)}) != data length ({len(self._texts)})")

    @staticmethod
    def _default_tokenizer(text: str) -> List[int]:
        """
        Fallback tokenizer using ord() for character-level encoding.
        For testing only; use a real tokenizer (e.g., from transformers) in practice.
        """
        return [ord(c) for c in text]

    def _load_from_file(self, filepath: str):
        """
        Load text data from file (JSON, JSONL, or plain text lines).

        Returns:
            Tuple of (texts, labels) where labels is None if not in file.
        """
        file_path = Path(filepath)
        if not file_path.exists():
            raise ValueError(f"File not found: {filepath}")

        texts = []
        labels = None

        if filepath.endswith(".json"):
            with open(filepath, "r") as f:
                data = json.load(f)
                # Handle both list of dicts and list of strings
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            texts.append(item.get(self.text_field, ""))
                            if self.label_field in item:
                                if labels is None:
                                    labels = []
                                labels.append(item[self.label_field])
                        else:
                            texts.append(item)

        elif filepath.endswith(".jsonl"):
            with open(filepath, "r") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if isinstance(item, dict):
                            texts.append(item.get(self.text_field, ""))
                            if self.label_field in item:
                                if labels is None:
                                    labels = []
                                labels.append(item[self.label_field])
                        else:
                            texts.append(item)

        else:
            # Plain text file (one line per sample)
            with open(filepath, "r") as f:
                texts = [line.rstrip("\n") for line in f if line.strip()]

        if len(texts) == 0:
            raise ValueError(f"No text data found in {filepath}")

        return texts, labels

    def _tokenize_and_pad(self, text: str) -> Dict[str, List[int]]:
        """
        Tokenize text and apply padding/truncation.

        Returns:
            Dict with "input_ids" and "attention_mask" keys (both List[int]).
            input_ids: token IDs, padded/truncated to max_length
            attention_mask: 1 for real tokens, 0 for padding
        """
        token_ids = self.tokenizer(text)

        # Truncate
        if self.truncation and len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]

        # Pad
        attention_mask = [1] * len(token_ids)
        if self.padding and len(token_ids) < self.max_length:
            pad_length = self.max_length - len(token_ids)
            token_ids = token_ids + [self.pad_token_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length

        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
        }

    def _apply_masking(self, token_ids: List[int], attention_mask: List[int]) -> Dict[str, List[int]]:
        """
        Apply BERT-style masking for masked_lm task.

        Masking strategy (80/10/10):
        - 80% of the time: replace token with [MASK] token
        - 10% of the time: replace token with random token
        - 10% of the time: keep token unchanged

        Args:
            token_ids: List of token IDs (max_length)
            attention_mask: List of 0s and 1s (1 for real tokens)

        Returns:
            Dict with:
            - "input_ids": original token IDs
            - "masked_input_ids": token IDs with masking applied
            - "labels": original token IDs at masked positions, -100 elsewhere
        """
        masked_input_ids = token_ids.copy()
        labels = [-100] * len(token_ids)

        # Only mask real tokens (attention_mask == 1)
        real_token_positions = [i for i in range(len(token_ids)) if attention_mask[i] == 1]

        for pos in real_token_positions:
            if random.random() < self.mask_prob:
                labels[pos] = token_ids[pos]

                rand = random.random()
                if rand < 0.8:
                    # 80%: replace with [MASK]
                    masked_input_ids[pos] = self.mask_token_id
                elif rand < 0.9:
                    # 10%: replace with random token
                    masked_input_ids[pos] = random.randint(0, 1000)  # Assume vocab size > 1000
                # else: 10% keep unchanged

        return {
            "input_ids": token_ids,
            "masked_input_ids": masked_input_ids,
            "labels": labels,
        }

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self._texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and return a single text sample with appropriate outputs for task.

        Args:
            idx: Index into dataset.

        Returns:
            Dict[str, torch.Tensor] with the following keys:

            - "input_ids": (max_length,) LongTensor - token IDs
            - "attention_mask": (max_length,) LongTensor - 1 for real, 0 for pad
            - "labels": (max_length,) LongTensor
                * Classification: single class index, rest are -100
                * Causal LM: shifted by 1 (next token), pad positions are -100
                * Masked LM: original IDs at masked positions, -100 elsewhere

            Optional:
            - "masked_input_ids": (max_length,) LongTensor (masked_lm only)

        Example:
            >>> ds = LanguageDataset(data=["hello"], task="classification", labels=[0])
            >>> batch = ds[0]
            >>> batch["input_ids"].shape
            torch.Size([512])
            >>> batch["labels"].dtype
            torch.int64
        """
        if torch is None:
            raise ImportError("torch is required. Install with: pip install torch")

        text = self._texts[idx]

        # Tokenize and pad
        encoded = self._tokenize_and_pad(text)
        input_ids = encoded["input_ids"]  # (max_length,)
        attention_mask = encoded["attention_mask"]  # (max_length,)

        # Build labels based on task
        if self.task == "classification":
            # Label: class index as first position, rest are -100
            label_value = self._labels[idx]
            labels = [-100] * len(input_ids)
            labels[0] = label_value
            output = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

        elif self.task == "causal_lm":
            # Shift labels by 1 (predict next token)
            # Last position has no target, pad positions are -100
            labels = [-100] + input_ids[:-1]
            for i in range(len(labels)):
                if attention_mask[i] == 0:  # Padding position
                    labels[i] = -100
            output = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

        elif self.task == "masked_lm":
            # Apply masking and return masked input + labels
            masked = self._apply_masking(input_ids, attention_mask)
            labels = masked["labels"]
            # Ensure pad positions are -100
            for i in range(len(labels)):
                if attention_mask[i] == 0:
                    labels[i] = -100
            output = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "masked_input_ids": torch.tensor(masked["masked_input_ids"], dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

        else:
            raise ValueError(f"Unknown task: {self.task}")

        return output


if __name__ == "__main__":
    """Smoke test with synthetic in-memory data."""
    print("Testing LanguageDataset...")

    if torch is None:
        print("⚠ PyTorch not installed. Skipping smoke tests.")
        print("   Install with: pip install torch")
        exit(0)

    # Simple ord-based tokenizer for testing
    def test_tokenizer(text: str) -> List[int]:
        return [ord(c) for c in text]

    print("\n1. Testing classification task...")
    ds_cls = LanguageDataset(
        data=["hello world", "goodbye world"],
        task="classification",
        labels=[0, 1],
        tokenizer=test_tokenizer,
        max_length=64,
    )
    print(f"   Loaded {len(ds_cls)} samples")
    batch = ds_cls[0]
    print(f"   input_ids shape: {batch['input_ids'].shape}")
    print(f"   labels shape: {batch['labels'].shape}")
    assert batch["input_ids"].shape == (64,), "input_ids should be (max_length,)"
    assert batch["attention_mask"].shape == (64,), "attention_mask should be (max_length,)"
    assert batch["labels"].dtype == torch.long, "labels should be LongTensor"
    # Check that only first position has non-(-100) label in classification
    assert batch["labels"][0] == 0, "First label should be class index"
    assert all(batch["labels"][i] == -100 for i in range(1, 64)), "Other positions should be -100"
    print("   ✓ Classification test passed")

    print("\n2. Testing causal_lm task...")
    ds_clm = LanguageDataset(
        data=["hello world"],
        task="causal_lm",
        tokenizer=test_tokenizer,
        max_length=64,
    )
    batch = ds_clm[0]
    print(f"   input_ids shape: {batch['input_ids'].shape}")
    print(f"   labels shape: {batch['labels'].shape}")
    # First position label should be -100 (no target for position -1)
    assert batch["labels"][0] == -100, "First label should be -100 for CLM"
    assert batch["labels"][1] == batch["input_ids"][0], "Position i label should be input[i-1]"
    print("   ✓ Causal LM test passed")

    print("\n3. Testing masked_lm task...")
    random.seed(42)  # For reproducibility
    ds_mlm = LanguageDataset(
        data=["hello world"],
        task="masked_lm",
        tokenizer=test_tokenizer,
        max_length=64,
        mask_prob=0.15,
    )
    batch = ds_mlm[0]
    print(f"   input_ids shape: {batch['input_ids'].shape}")
    print(f"   masked_input_ids shape: {batch['masked_input_ids'].shape}")
    print(f"   labels shape: {batch['labels'].shape}")
    assert "masked_input_ids" in batch, "masked_lm should have masked_input_ids"
    assert batch["labels"].dtype == torch.long, "labels should be LongTensor"
    # Check that pad positions have label -100
    assert all(batch["labels"][i] == -100 for i in range(11, 64) if batch["attention_mask"][i] == 0), "Pad positions should have label -100"
    print("   ✓ Masked LM test passed")

    print("\n4. Testing padding and truncation...")
    ds_pad = LanguageDataset(
        data=["short"],
        task="classification",
        labels=[0],
        tokenizer=test_tokenizer,
        max_length=32,
        padding=True,
        truncation=True,
    )
    batch = ds_pad[0]
    # Count non-padding tokens
    real_tokens = sum(1 for x in batch["attention_mask"] if x == 1)
    print(f"   Real tokens: {real_tokens}, Padded tokens: {32 - real_tokens}")
    assert batch["input_ids"].shape == (32,), "Should be padded to max_length"
    assert batch["attention_mask"].sum() == real_tokens, "attention_mask should match real tokens"
    print("   ✓ Padding/truncation test passed")

    print("\n✓ All LanguageDataset tests passed!")
