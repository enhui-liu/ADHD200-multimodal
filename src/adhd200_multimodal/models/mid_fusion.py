"""Middle-fusion classifier for phenotypic and volumetric imaging inputs."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .encoders import PhenotypeEncoder, VolumetricImageEncoder


class MidFusionClassifier(nn.Module):
    """Fuse independent phenotype and image embeddings for binary classification."""

    def __init__(
        self,
        phenotype_dim: int,
        image_channels: int,
        phenotype_embedding_dim: int = 64,
        imaging_embedding_dim: int = 128,
        fusion_hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.phenotype_encoder = PhenotypeEncoder(
            phenotype_dim, phenotype_embedding_dim, dropout
        )
        self.image_encoder = VolumetricImageEncoder(image_channels, imaging_embedding_dim, dropout)
        fusion_dim = phenotype_embedding_dim + imaging_embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def encode(self, phenotypes: Tensor, images: Tensor) -> tuple[Tensor, Tensor]:
        """Return modality-specific representations before fusion."""
        return self.phenotype_encoder(phenotypes), self.image_encoder(images)

    def forward(self, phenotypes: Tensor, images: Tensor) -> Tensor:
        phenotype_embedding, image_embedding = self.encode(phenotypes, images)
        fused_embedding = torch.cat((phenotype_embedding, image_embedding), dim=1)
        return self.classifier(fused_embedding)
