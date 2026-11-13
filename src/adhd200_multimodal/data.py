"""Dataset adapters for pre-aligned ADHD-200 multimodal arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class MultimodalNpzDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Read subject-aligned phenotype, volume, and label arrays from a NPZ file."""

    required_keys = frozenset({"phenotypes", "images", "labels", "subject_ids"})

    def __init__(self, dataset_path: str | Path) -> None:
        path = Path(dataset_path)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")

        with np.load(path, allow_pickle=False) as archive:
            missing = self.required_keys.difference(archive.files)
            if missing:
                raise ValueError(f"Dataset missing required arrays: {sorted(missing)}")
            self.phenotypes = torch.as_tensor(archive["phenotypes"], dtype=torch.float32)
            self.images = torch.as_tensor(archive["images"], dtype=torch.float32)
            self.labels = torch.as_tensor(archive["labels"], dtype=torch.long)
            self.subject_ids = archive["subject_ids"]

        sample_count = len(self.labels)
        if (
            len(self.phenotypes) != sample_count
            or len(self.images) != sample_count
            or len(self.subject_ids) != sample_count
        ):
            raise ValueError("All arrays must have identical sample counts")
        if self.phenotypes.ndim != 2 or self.images.ndim != 5 or self.labels.ndim != 1:
            raise ValueError("Expected phenotypes (N,P), images (N,C,D,H,W), and labels (N,)")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.phenotypes[index], self.images[index], self.labels[index]
