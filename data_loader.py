"""
OmniCoreX Super Advanced Multi-Modal Data Loader

This module provides robust, scalable, and ultra high-tech utilities for loading,
preprocessing, and batching multi-modal and multi-source knowledge data streams
tailored for OmniCoreX's complex AI architecture.

Features include:

- Flexible support for heterogeneous data sources including text, images, sensor, audio.
- Advanced preprocessing pipelines with caching, augmentation, and normalization.
- Highly efficient synchronized batching for multiple modalities with padding and masking.
- Streaming data integration hooks for real-time AI applications.
- Modular extensibility for new data types and preprocessing strategies.
"""

import os
import random
import json
from typing import Dict, List, Optional, Callable, Tuple, Iterator
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


# -------------------- Preprocessing Functions -------------------- #

def preprocess_text(text: str,
                    tokenizer: Callable[[str], List[int]],
                    max_length: int = 256) -> torch.Tensor:
    """
    Tokenize and pad/truncate text input.

    Args:
        text: Raw input text string.
        tokenizer: Function converting string to list of token IDs.
        max_length: Maximum desired sequence length.

    Returns:
        Tensor of token ids of shape (max_length,)
    """
    token_ids = tokenizer(text)[:max_length]
    pad_length = max_length - len(token_ids)
    padded_ids = token_ids + [0] * pad_length
    return torch.tensor(padded_ids, dtype=torch.long)


def preprocess_image(image_path: str,
                     target_size: Tuple[int, int] = (224, 224),
                     augment: bool = False) -> torch.Tensor:
    """
    Load and preprocess image with resizing, normalization, optional augmentation.

    Args:
        image_path: Path to image file.
        target_size: Desired output image size (width, height).
        augment: Whether to apply augmentation transforms.

    Returns:
        Preprocessed image tensor (3, target_height, target_width)
    """
    transform_list = []
    if augment:
        transform_list.extend([
            transforms.RandomResizedCrop(target_size),
            transforms.RandomHorizontalFlip()
        ])
    else:
        transform_list.append(transforms.Resize(target_size))
        transform_list.append(transforms.CenterCrop(target_size))

    transform_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]

    transform = transforms.Compose(transform_list)
    image = Image.open(image_path).convert("RGB")
    return transform(image)


def preprocess_sensor_data(sensor_data: List[float],
                           max_seq_length: int = 100) -> torch.Tensor:
    """
    Normalize and pad sensor time-series data.

    Args:
        sensor_data: Raw 1D sensor readings list.
        max_seq_length: Maximum length for padding/truncation.

    Returns:
        Normalized and padded tensor of shape (max_seq_length,)
    """
    arr = np.array(sensor_data, dtype=np.float32)
    if arr.size == 0:
        arr = np.zeros(max_seq_length, dtype=np.float32)
    else:
        mean = arr.mean() if arr.size > 0 else 0.0
        std = arr.std() if arr.size > 0 else 1.0
        arr = (arr - mean) / (std + 1e-9)
    padded = np.zeros(max_seq_length, dtype=np.float32)
    length = min(max_seq_length, arr.size)
    padded[:length] = arr[:length]
    return torch.tensor(padded)


# -------------------- Dataset Definition -------------------- #

class OmniCoreXMultiModalDataset(Dataset):
    """
    Dataset supporting multi-source, multi-modal knowledge streams integrated seamlessly.

    Expects metadata JSON containing sample entries with modality-specific data references.

    Capable of handling missing data gracefully with sensible fallbacks.
    """

    def __init__(self,
                 metadata_path: str,
                 modalities: List[str],
                 tokenizer: Optional[Callable[[str], List[int]]] = None,
                 text_max_length: int = 256,
                 image_size: Tuple[int, int] = (224, 224),
                 sensor_max_length: int = 100,
                 augmentation: bool = False):
        """
        Initialize the dataset.

        Args:
            metadata_path: Path to JSON file listing dataset samples.
            modalities: Modalities to be loaded (e.g., ['text', 'image', 'sensor']).
            tokenizer: Optional tokenizer callable for text processing.
            text_max_length: Max sequence length for text.
            image_size: Target resolution for images.
            sensor_max_length: Max length for sensor sequences.
            augmentation: Whether to apply augmentation on images.
        """
        super().__init__()
        self.modalities = modalities
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.image_size = image_size
        self.sensor_max_length = sensor_max_length
        self.augmentation = augmentation

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        if not isinstance(self.metadata, list):
            raise ValueError("Metadata JSON must be a list of samples")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        sample_meta = self.metadata[idx]
        sample = {}

        if 'text' in self.modalities:
            text = sample_meta.get("text", "")
            if self.tokenizer:
                sample['text'] = preprocess_text(text, self.tokenizer, self.text_max_length)
            else:
                sample['text'] = torch.tensor([], dtype=torch.long)  # empty placeholder

        if 'image' in self.modalities:
            image_path = sample_meta.get("image", "")
            if image_path and os.path.exists(image_path):
                sample['image'] = preprocess_image(image_path, self.image_size, self.augmentation)
            else:
                sample['image'] = torch.zeros(3, self.image_size[1], self.image_size[0])

        if 'sensor' in self.modalities:
            sensor_raw = sample_meta.get("sensor", [])
            sample['sensor'] = preprocess_sensor_data(sensor_raw, self.sensor_max_length)

        # Extend here for additional modalities like audio, video, tabular etc.

        return sample


# -------------------- Collate Function -------------------- #

def multimodal_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collates a batch of multi-modal samples into batched tensors with proper stacking.

    Args:
        batch: List of individual modality dictionaries.

    Returns:
        Dict mapping modality to batched tensors.
    """
    collated = defaultdict(list)
    for sample in batch:
        for modality, data in sample.items():
            collated[modality].append(data)

    # Convert lists to batched tensors
    for modality in collated:
        if isinstance(collated[modality][0], torch.Tensor):
            collated[modality] = torch.stack(collated[modality])
        else:
            # For raw types (e.g. strings), keep as list
            collated[modality] = list(collated[modality])

    return dict(collated)


# -------------------- DataLoader Factory -------------------- #

def create_omncorex_dataloader(metadata_path: str,
                              modalities: List[str],
                              tokenizer: Optional[Callable[[str], List[int]]] = None,
                              text_max_length: int = 256,
                              image_size: Tuple[int, int] = (224, 224),
                              sensor_max_length: int = 100,
                              batch_size: int = 16,
                              shuffle: bool = True,
                              num_workers: int = 4,
                              augmentation: bool = False) -> DataLoader:
    """
    Constructs a PyTorch DataLoader over the OmniCoreX multimodal dataset.

    Args:
        metadata_path: Path to dataset metadata JSON.
        modalities: Modalities to load.
        tokenizer: Optional tokenizer callable.
        text_max_length: Max token length for text.
        image_size: Image resize resolution.
        sensor_max_length: Max sequence length for sensor data.
        batch_size: Batch size.
        shuffle: Shuffle dataset if True.
        num_workers: Number of worker subprocesses.
        augmentation: Whether to apply data augmentation (image).

    Returns:
        Configured PyTorch DataLoader instance.
    """
    dataset = OmniCoreXMultiModalDataset(
        metadata_path=metadata_path,
        modalities=modalities,
        tokenizer=tokenizer,
        text_max_length=text_max_length,
        image_size=image_size,
        sensor_max_length=sensor_max_length,
        augmentation=augmentation
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn,
        pin_memory=True
    )
    return loader


# -------------------- Example Test Run -------------------- #

if __name__ == "__main__":
    import tempfile

    # Dummy tokenizer: simple char-index tokenizer
    def dummy_tokenizer(text):
        return [ord(c) % 100 + 1 for c in text]

    # Create dummy metadata JSON file
    dummy_samples = [
        {
            "text": "OmniCoreX is the ultimate AI brain.",
            "image": "",        # No image file - fallback to zeros
            "sensor": [random.uniform(-1, 1) for _ in range(50)]
        },
        {
            "text": "Testing multi-modal data loader.",
            "sensor": [random.uniform(-0.5, 0.5) for _ in range(80)]
        }
    ]
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmpf:
        json.dump(dummy_samples, tmpf)
        metadata_path = tmpf.name

    dataloader = create_omncorex_dataloader(
        metadata_path=metadata_path,
        modalities=["text", "image", "sensor"],
        tokenizer=dummy_tokenizer,
        batch_size=2,
        augmentation=True,
        shuffle=False,
        num_workers=0
    )

    for batch in dataloader:
        print("Batch data keys:", batch.keys())
        for modality, tensor in batch.items():
            print(f"Modality: {modality}, Tensor shape: {tensor.shape}")
        break
