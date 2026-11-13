"""Minimal configuration-driven training entry point."""

from __future__ import annotations

import json
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from .data import MultimodalNpzDataset
from .models import MidFusionClassifier


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(setting: str) -> torch.device:
    if setting != "auto":
        return torch.device(setting)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for phenotypes, images, labels in loader:
            logits = model(phenotypes.to(device), images.to(device))
            correct += (logits.argmax(dim=1).cpu() == labels).sum().item()
            total += len(labels)
    return {"accuracy": correct / total if total else 0.0}


@hydra.main(version_base=None, config_path="../../configs", config_name="experiment")
def main(config: DictConfig) -> None:
    seed_everything(config.seed)
    dataset = MultimodalNpzDataset(config.data.dataset_path)
    validation_size = max(1, round(len(dataset) * config.training.validation_fraction))
    if validation_size >= len(dataset):
        raise ValueError("Dataset must contain more samples than the validation split")
    if len(np.unique(dataset.subject_ids)) < 2:
        raise ValueError(
            "At least two distinct subject_ids are required for subject-level splitting"
        )
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=config.training.validation_fraction,
        random_state=config.seed,
    )
    train_indices, validation_indices = next(
        splitter.split(np.zeros(len(dataset)), dataset.labels.numpy(), groups=dataset.subject_ids)
    )
    train_set = Subset(dataset, train_indices.tolist())
    validation_set = Subset(dataset, validation_indices.tolist())
    train_loader = DataLoader(train_set, batch_size=config.training.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=config.training.batch_size)
    device = choose_device(config.training.device)
    model = MidFusionClassifier(
        phenotype_dim=config.data.phenotype_dim,
        image_channels=config.data.image_channels,
        num_classes=config.data.num_classes,
        **config.model,
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    loss_function = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    checkpoint_path = Path(config.output.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for _ in range(config.training.epochs):
        model.train()
        for phenotypes, images, labels in train_loader:
            optimizer.zero_grad()
            logits = model(phenotypes.to(device), images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, validation_loader, device)
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            torch.save({"model_state": model.state_dict(), "config": dict(config)}, checkpoint_path)

    metrics_path = Path(config.output.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({"validation_accuracy": best_accuracy}, indent=2))
    print(f"Best validation accuracy: {best_accuracy:.3f}")


if __name__ == "__main__":
    main()
