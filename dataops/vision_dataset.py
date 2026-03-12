"""
PyTorch Dataset for Image Classification Tasks.

VisionDataset provides a flexible, configurable template for loading and preprocessing
image data from disk or in-memory storage. Supports both folder hierarchies (where
class labels are inferred from directory names) and manifest files (CSV/JSON/JSONL).

Key Features:
    - Multiple data source patterns: folder hierarchy or explicit manifest
    - Automatic label assignment from folder structure or manifest
    - Deterministic sorting for reproducible label mappings
    - Optional in-memory caching for fast iteration (useful for small datasets)
    - Configurable image transformations per mode (train/val/test)
    - Extensible image loader (defaults to PIL.Image)

Usage:
    >>> # From folder hierarchy (imagenet-style: root/class_name/*.jpg)
    >>> ds = VisionDataset(
    ...     root="data/imagenet_mini",
    ...     mode="train",
    ...     train_transform=transforms.Compose([...])
    ... )
    >>> pixel_values, label = ds[0]  # (C, H, W), scalar

    >>> # From manifest file (CSV with filepath,label columns)
    >>> ds = VisionDataset(
    ...     manifest="data/splits.csv",
    ...     mode="val",
    ...     val_transform=transforms.Compose([...])
    ... )
    >>> len(ds), ds.num_classes
"""

import os
import csv
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Callable

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object  # type: ignore

try:
    from PIL import Image
except ImportError:
    Image = None


class VisionDataset(Dataset):
    """
    PyTorch Dataset for image classification tasks.

    Loads images from disk and applies configurable transformations. Supports
    two data source patterns: folder hierarchy (class labels from directory names)
    or explicit manifest file (CSV/JSON/JSONL with filepath and label columns).

    Args:
        root: Path to root directory with folder structure root/class_name/*.ext.
              Exactly one of root or manifest must be provided.
        manifest: Path to CSV, JSON, or JSONL file with image paths and labels.
                  CSV format: filepath,label (header required)
                  JSON format: [{"filepath": "...", "label": 0}, ...]
                  JSONL format: one object per line with "filepath" and "label"
        mode: Dataset split mode. Options: "train", "val", "test". Default: "train".
              Used to select appropriate transform from train_transform/val_transform.
        train_transform: Callable applied to images when mode="train". None means no transform.
        val_transform: Callable applied to images when mode="val" or "test". None means no transform.
        label_map: Optional dict mapping string label names to integer indices.
                   If not provided, generated automatically from data.
        image_loader: Callable that takes filepath and returns PIL Image (RGB).
                      Default: PIL.Image.open(path).convert("RGB")
        in_memory: If True, preload all images in __init__. Useful for small datasets
                   to avoid disk I/O during training. Default: False.
        extensions: Tuple of valid image file extensions (case-insensitive).
                    Default: (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    Attributes:
        classes: List of class names in sorted order.
        num_classes: Number of unique classes.
        _samples: List[Tuple[filepath, label_int]] - file paths and numeric labels.
        _images_cache: Dict[filepath, Image] or None - in-memory image cache.

    Raises:
        ValueError: If neither root nor manifest provided, or both provided.
                    If folder hierarchy has no images, or manifest is malformed.

    Example:
        >>> ds = VisionDataset(root="data/imagenet", mode="train")
        >>> print(ds.num_classes, ds.classes)
        >>> img, label = ds[0]  # img: (C, H, W) tensor, label: scalar tensor
    """

    def __init__(
        self,
        root: Optional[str] = None,
        manifest: Optional[str] = None,
        mode: str = "train",
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        label_map: Optional[Dict[str, int]] = None,
        image_loader: Optional[Callable] = None,
        in_memory: bool = False,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
    ) -> None:
        """Initialize VisionDataset."""
        # Validate inputs
        if (root is None and manifest is None) or (root is not None and manifest is not None):
            raise ValueError("Exactly one of 'root' or 'manifest' must be provided")

        self.root = root
        self.manifest = manifest
        self.mode = mode
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.image_loader = image_loader or self._default_image_loader
        self.in_memory = in_memory
        self.extensions = tuple(ext.lower() for ext in extensions)

        # Load samples and build label map
        if root is not None:
            self._samples, unique_classes = self._scan_folder_hierarchy()
        else:
            self._samples, unique_classes = self._load_manifest()

        # Build label map if not provided
        if label_map is not None:
            self.label_map = label_map
            self.classes = sorted(label_map.keys(), key=lambda x: label_map[x])
        else:
            self.classes = sorted(unique_classes)
            self.label_map = {cls: idx for idx, cls in enumerate(self.classes)}

        # Remap labels in samples using label_map
        self._samples = [(path, self.label_map[label]) for path, label in self._samples]

        # Optional: preload all images into memory
        self._images_cache = None
        if in_memory:
            self._preload_images()

    @staticmethod
    def _default_image_loader(path: str):
        """Load image from disk using PIL and convert to RGB."""
        if Image is None:
            raise ImportError("PIL/Pillow is required to load images. Install with: pip install Pillow")
        return Image.open(path).convert("RGB")

    def _scan_folder_hierarchy(self) -> Tuple[List[Tuple[str, str]], set]:
        """
        Scan root directory for class folders and images.

        Expected structure:
            root/
            ├── class_1/
            │   ├── image1.jpg
            │   └── image2.png
            └── class_2/
                └── image3.jpg

        Returns:
            Tuple of (samples, unique_classes) where:
            - samples: List[(image_path, class_name_str)]
            - unique_classes: Set of discovered class names
        """
        samples = []
        unique_classes = set()

        root_path = Path(self.root)
        if not root_path.exists():
            raise ValueError(f"Root directory not found: {self.root}")

        # Iterate over class directories (sorted for determinism)
        for class_dir in sorted(root_path.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            unique_classes.add(class_name)

            # Find all images in this class directory (sorted for determinism)
            for img_file in sorted(class_dir.iterdir()):
                if img_file.is_file() and img_file.suffix.lower() in self.extensions:
                    samples.append((str(img_file), class_name))

        if len(samples) == 0:
            raise ValueError(f"No images found in {self.root} with extensions {self.extensions}")

        return samples, unique_classes

    def _load_manifest(self) -> Tuple[List[Tuple[str, str]], set]:
        """
        Load samples from manifest file (CSV, JSON, or JSONL).

        CSV format (with header):
            filepath,label
            path/to/image1.jpg,cat
            path/to/image2.jpg,dog

        JSON format:
            [
              {"filepath": "path/to/image1.jpg", "label": "cat"},
              {"filepath": "path/to/image2.jpg", "label": "dog"}
            ]

        JSONL format (one object per line):
            {"filepath": "path/to/image1.jpg", "label": "cat"}
            {"filepath": "path/to/image2.jpg", "label": "dog"}

        Returns:
            Tuple of (samples, unique_classes)
        """
        samples = []
        unique_classes = set()
        manifest_path = Path(self.manifest)

        if not manifest_path.exists():
            raise ValueError(f"Manifest file not found: {self.manifest}")

        if manifest_path.suffix.lower() == ".csv":
            with open(manifest_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filepath = row.get("filepath") or row.get("path")
                    label = row.get("label")
                    if filepath is None or label is None:
                        raise ValueError("CSV must have 'filepath' and 'label' columns")
                    samples.append((filepath, label))
                    unique_classes.add(label)

        elif manifest_path.suffix.lower() == ".json":
            with open(manifest_path, "r") as f:
                data = json.load(f)
                for item in data:
                    filepath = item.get("filepath") or item.get("path")
                    label = item.get("label")
                    if filepath is None or label is None:
                        raise ValueError("JSON items must have 'filepath' and 'label' keys")
                    samples.append((filepath, label))
                    unique_classes.add(label)

        elif manifest_path.suffix.lower() == ".jsonl":
            with open(manifest_path, "r") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        filepath = item.get("filepath") or item.get("path")
                        label = item.get("label")
                        if filepath is None or label is None:
                            raise ValueError("JSONL items must have 'filepath' and 'label' keys")
                        samples.append((filepath, label))
                        unique_classes.add(label)

        else:
            raise ValueError(f"Unsupported manifest format: {manifest_path.suffix}. Use .csv, .json, or .jsonl")

        if len(samples) == 0:
            raise ValueError(f"No samples found in manifest: {self.manifest}")

        return samples, unique_classes

    def _preload_images(self) -> None:
        """Load all images into memory. Called if in_memory=True."""
        self._images_cache = {}
        for img_path, _ in self._samples:
            try:
                img = self.image_loader(img_path)
                self._images_cache[img_path] = img
            except Exception as e:
                print(f"Warning: Failed to load {img_path}: {e}")

    def _get_active_transform(self) -> Optional[Callable]:
        """Return appropriate transform based on current mode."""
        if self.mode == "train":
            return self.train_transform
        else:  # val, test
            return self.val_transform

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return a single image with its label.

        Args:
            idx: Index into dataset.

        Returns:
            Tuple of (pixel_values, label) where:
            - pixel_values: torch.Tensor of shape (C, H, W), dtype float32
            - label: torch.Tensor scalar, dtype int64 (LongTensor)

        Example:
            >>> ds = VisionDataset(root="data")
            >>> img, lbl = ds[0]
            >>> img.shape, lbl.shape
            (torch.Size([3, 224, 224]), torch.Size([]))
        """
        img_path, label_idx = self._samples[idx]

        # Load image from cache or disk
        if self._images_cache is not None:
            img = self._images_cache[img_path]
        else:
            img = self.image_loader(img_path)

        # Apply transform if available
        transform = self._get_active_transform()
        if transform is not None:
            img = transform(img)
        else:
            # Default: convert PIL image to tensor
            img = torch.tensor(list(img.getdata()), dtype=torch.float32).reshape(
                1, img.height, img.width, -1
            ).permute(3, 1, 2, 0).squeeze(-1)  # (H, W, C) -> (C, H, W)

        # Ensure img is a tensor
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"Transform must return torch.Tensor, got {type(img)}")

        # Label as LongTensor scalar
        label = torch.tensor(label_idx, dtype=torch.long)

        return img, label

    @property
    def num_classes(self) -> int:
        """Return number of unique classes."""
        return len(self.classes)


if __name__ == "__main__":
    """Smoke test with synthetic in-memory data."""
    print("Testing VisionDataset...")

    if torch is None or Image is None:
        print("⚠ Required dependencies not installed. Skipping smoke tests.")
        print("   Install with: pip install torch torchvision Pillow")
        exit(0)

    # Create synthetic data in memory
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create folder structure: tmpdir/cat/*.jpg, tmpdir/dog/*.jpg
        for class_name in ["cat", "dog"]:
            class_dir = Path(tmpdir) / class_name
            class_dir.mkdir()

            # Create 2 dummy images per class
            for i in range(2):
                # Create a simple RGB image
                img = Image.new("RGB", (64, 64), color=(i*50, i*100, i*150))
                img.save(class_dir / f"image_{i}.jpg")

        # Test 1: Load from folder hierarchy
        print("\n1. Testing folder hierarchy loading...")
        ds = VisionDataset(
            root=tmpdir,
            mode="train",
            train_transform=None,  # No transform for testing
        )
        print(f"   Loaded {len(ds)} samples")
        print(f"   Classes: {ds.classes}")
        print(f"   Num classes: {ds.num_classes}")

        img, label = ds[0]
        print(f"   Sample 0 - img shape: {img.shape}, label: {label}")
        assert img.shape[0] == 3, "Image should have 3 channels"
        assert label.dtype == torch.long, "Label should be LongTensor"
        print("   ✓ Folder hierarchy test passed")

        # Test 2: Load from manifest (CSV)
        print("\n2. Testing manifest (CSV) loading...")
        manifest_path = Path(tmpdir) / "manifest.csv"
        with open(manifest_path, "w") as f:
            f.write("filepath,label\n")
            f.write(f"{tmpdir}/cat/image_0.jpg,cat\n")
            f.write(f"{tmpdir}/cat/image_1.jpg,cat\n")
            f.write(f"{tmpdir}/dog/image_0.jpg,dog\n")
            f.write(f"{tmpdir}/dog/image_1.jpg,dog\n")

        ds_manifest = VisionDataset(
            manifest=str(manifest_path),
            mode="val",
            val_transform=None,
        )
        print(f"   Loaded {len(ds_manifest)} samples from manifest")
        print(f"   Classes: {ds_manifest.classes}")
        assert ds_manifest.num_classes == 2, "Should have 2 classes"
        print("   ✓ CSV manifest test passed")

        # Test 3: Mode switching
        print("\n3. Testing mode switching...")
        ds_train = VisionDataset(root=tmpdir, mode="train")
        ds_val = VisionDataset(root=tmpdir, mode="val")
        print(f"   Train mode transform: {ds_train._get_active_transform()}")
        print(f"   Val mode transform: {ds_val._get_active_transform()}")
        print("   ✓ Mode switching test passed")

        # Test 4: In-memory loading
        print("\n4. Testing in-memory loading...")
        ds_inmem = VisionDataset(root=tmpdir, in_memory=True)
        print(f"   Cache size: {len(ds_inmem._images_cache) if ds_inmem._images_cache else 0}")
        assert ds_inmem._images_cache is not None, "Cache should be populated"
        assert len(ds_inmem._images_cache) == 4, "Should cache 4 images"
        img, label = ds_inmem[0]
        print(f"   Retrieved cached image - shape: {img.shape}")
        print("   ✓ In-memory test passed")

    print("\n✓ All VisionDataset tests passed!")
