"""Modality-specific encoders used before middle fusion."""

from __future__ import annotations

from torch import Tensor, nn


class PhenotypeEncoder(nn.Module):
    """Encode already imputed and scaled phenotypic features."""

    def __init__(self, input_dim: int, embedding_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        hidden_dim = max(embedding_dim, input_dim * 2)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
        )

    def forward(self, phenotypes: Tensor) -> Tensor:
        if phenotypes.ndim != 2:
            raise ValueError("phenotypes must have shape (batch, features)")
        return self.network(phenotypes)


class VolumetricImageEncoder(nn.Module):
    """Compact 3D CNN encoder for preprocessed structural/functional volumes."""

    def __init__(self, in_channels: int, embedding_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.GELU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 5:
            raise ValueError("images must have shape (batch, channels, depth, height, width)")
        return self.projection(self.features(images))
