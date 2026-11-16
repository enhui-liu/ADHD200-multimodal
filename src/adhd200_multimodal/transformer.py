"""Transformer baseline on preprocessed CC200 fMRI time-series (sfnwmrda)."""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from adhd200_multimodal.phenotype_xgb import (
    bootstrap_ci,
    choose_threshold,
    compute_metrics,
    plot_calibration,
)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "transformer.yaml"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def load_time_series(file_path: str | Path, standardize: bool = True) -> np.ndarray:
    """Load a CC200 .1D time-series file as float32 array of shape (T, R)."""
    df = pd.read_csv(file_path, sep="\t")
    roi_cols = [col for col in df.columns if col not in ("File", "Sub-brick")]
    arr = df[roi_cols].to_numpy(dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"Invalid time-series shape {arr.shape} in {file_path}")
    if standardize:
        mean = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        arr = (arr - mean) / std
    return arr


def load_processed_dataset(
    manifest_path: str | Path,
    processed_dir: str | Path,
    run_selection: str = "first",
    standardize: bool = True,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Load one preprocessed time-series sample per eligible subject."""
    manifest = pd.read_csv(manifest_path)
    copied = manifest[manifest["status"] == "copied"].copy().reset_index(drop=True)
    if copied["subject_id"].duplicated().any():
        raise ValueError("subject_id must be unique in manifest")

    root = Path(processed_dir)
    sequences: list[np.ndarray] = []
    for files_str in copied["files"]:
        rel_paths = [p for p in str(files_str).split(";") if p]
        if not rel_paths:
            raise ValueError("Copied subject entry has no files listed")
        if run_selection == "first":
            chosen = rel_paths[0]
        else:
            raise ValueError(f"Unsupported run_selection: {run_selection}")
        sequences.append(load_time_series(root / chosen, standardize=standardize))

    labels = copied["label"].to_numpy(dtype=np.int64)
    subject_ids = copied["subject_id"].to_numpy(dtype=np.int64)
    site_names = copied["site_name"].to_numpy()
    return sequences, labels, subject_ids, site_names


class TimeSeriesDataset(Dataset[tuple[Tensor, Tensor]]):
    """Subject-level variable-length ROI time-series dataset."""

    def __init__(self, sequences: list[np.ndarray], labels: np.ndarray) -> None:
        if len(sequences) != len(labels):
            raise ValueError("sequences and labels must have identical lengths")
        self.sequences = [torch.as_tensor(seq, dtype=torch.float32) for seq in sequences]
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.sequences[index], self.labels[index]


def collate_time_series(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor, Tensor]:
    """Pad variable-length (T_i, R) sequences to (B, T_max, R) and return padding mask."""
    batch_size = len(batch)
    max_len = max(seq.shape[0] for seq, _ in batch)
    feature_dim = batch[0][0].shape[1]

    padded = torch.zeros((batch_size, max_len, feature_dim), dtype=torch.float32)
    padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)
    labels = torch.zeros((batch_size,), dtype=torch.float32)

    for idx, (seq, label) in enumerate(batch):
        seq_len = seq.shape[0]
        padded[idx, :seq_len, :] = seq
        padding_mask[idx, :seq_len] = False
        labels[idx] = label

    return padded, padding_mask, labels


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for batch-first (B, T, D) tensors."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1), :]


class TimeSeriesTransformer(nn.Module):
    """Transformer encoder with masked mean pooling for ROI time-series classification."""

    def __init__(
        self,
        input_dim: int = 190,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.3,
        activation: str = "gelu",
        norm_first: bool = True,
        batch_first: bool = True,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if not batch_first:
            raise ValueError("TimeSeriesTransformer requires batch_first=True")
        if pooling != "mean":
            raise ValueError(f"Unsupported pooling: {pooling}")

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.pos_encoder = SinusoidalPositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, seq_len, features)")
        if padding_mask is not None and padding_mask.shape != x.shape[:2]:
            raise ValueError("padding_mask must have shape (batch, seq_len)")

        hidden = self.pos_encoder(self.input_proj(x))
        encoded = self.encoder(hidden, src_key_padding_mask=padding_mask)

        if padding_mask is not None:
            valid_mask = (~padding_mask).unsqueeze(-1).to(encoded.dtype)
            pooled = (encoded * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = encoded.mean(dim=1)

        return self.classifier(pooled).squeeze(-1)


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    """Evaluate model on a DataLoader, returning mean loss and predicted probabilities."""
    model.eval()
    loss_sum = 0.0
    count = 0
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for x, padding_mask, y in loader:
            x = x.to(device)
            padding_mask = padding_mask.to(device)
            y = y.to(device)
            logits = model(x, padding_mask=padding_mask)
            loss = criterion(logits, y)
            batch_n = y.shape[0]
            loss_sum += float(loss.item()) * batch_n
            count += batch_n
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return loss_sum / max(count, 1), np.concatenate(probs, axis=0)


def _build_model_candidates(config: DictConfig) -> list[tuple[str, DictConfig]]:
    """Build candidate model configs by merging optional architecture_candidates onto model."""
    base_model_cfg = config.model
    raw_candidates = config.get("architecture_candidates", None)
    if not raw_candidates:
        return [("default", base_model_cfg)]

    candidates: list[tuple[str, DictConfig]] = []
    for idx, cand in enumerate(raw_candidates):
        cand_dict = OmegaConf.to_container(cand, resolve=True)
        name = str(cand_dict.pop("name", f"candidate_{idx + 1}"))
        merged = OmegaConf.merge(base_model_cfg, OmegaConf.create(cand_dict))
        candidates.append((name, merged))
    return candidates


def _load_pretrained(config: DictConfig) -> tuple[DictConfig, dict[str, Tensor]] | None:
    """Load architecture and weights from `pretrained_checkpoint`, if configured."""
    checkpoint_path = config.get("pretrained_checkpoint", None)
    if not checkpoint_path:
        return None
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return OmegaConf.create(payload["model_config"]), payload["model_state_dict"]


def _train_single_candidate(
    model_cfg: DictConfig,
    inner_train_loader: DataLoader[tuple[Tensor, Tensor]],
    inner_val_loader: DataLoader[tuple[Tensor, Tensor]],
    inner_val_labels: np.ndarray,
    config: DictConfig,
    device: torch.device,
    candidate_seed: int,
    init_state_dict: dict[str, Tensor] | None = None,
) -> dict[str, object]:
    """Train one candidate architecture on inner train/val split with early stopping."""
    set_seed(candidate_seed)
    patience = int(config.training.early_stopping_patience)
    min_delta = float(config.training.early_stopping_min_delta)

    model = TimeSeriesTransformer(
        input_dim=model_cfg.input_dim,
        d_model=model_cfg.d_model,
        nhead=model_cfg.nhead,
        num_layers=model_cfg.num_layers,
        dim_feedforward=model_cfg.dim_feedforward,
        dropout=model_cfg.dropout,
        activation=model_cfg.activation,
        norm_first=model_cfg.norm_first,
        batch_first=model_cfg.batch_first,
        pooling=model_cfg.pooling,
    ).to(device)

    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    train_losses: list[float] = []
    inner_valid_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, int(config.training.epochs) + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for x, padding_mask, y in inner_train_loader:
            x = x.to(device)
            padding_mask = padding_mask.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x, padding_mask=padding_mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            batch_n = y.shape[0]
            train_loss_sum += float(loss.item()) * batch_n
            train_count += batch_n

        train_loss = train_loss_sum / max(train_count, 1)
        inner_val_loss, _ = evaluate_loader(model, inner_val_loader, criterion, device)
        train_losses.append(train_loss)
        inner_valid_losses.append(inner_val_loss)

        if inner_val_loss < best_val_loss - min_delta:
            best_val_loss = inner_val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state_dict)
    _, inner_val_probs = evaluate_loader(model, inner_val_loader, criterion, device)
    inner_val_auroc = float(roc_auc_score(inner_val_labels, inner_val_probs))

    return {
        "model_cfg": model_cfg,
        "model_state_dict": best_state_dict,
        "n_params": int(n_params),
        "train_losses": train_losses,
        "inner_valid_losses": inner_valid_losses,
        "best_epoch": best_epoch,
        "stopped_epoch": len(train_losses),
        "best_inner_valid_loss": float(best_val_loss),
        "best_inner_valid_auroc": inner_val_auroc,
    }


def train_one_fold(
    train_sequences: list[np.ndarray],
    train_labels: np.ndarray,
    valid_sequences: list[np.ndarray],
    valid_labels: np.ndarray,
    config: DictConfig,
    device: torch.device,
    fold_seed: int | None = None,
) -> dict[str, object]:
    """Select architecture & stop early strictly inside the training fold, then evaluate."""
    inner_val_fraction = float(config.training.inner_val_fraction)
    split_seed = int(config.seed) if fold_seed is None else int(fold_seed)

    indices = np.arange(len(train_labels))
    inner_train_idx, inner_val_idx = train_test_split(
        indices,
        test_size=inner_val_fraction,
        stratify=train_labels,
        random_state=split_seed,
    )

    inner_train_loader = DataLoader(
        TimeSeriesDataset(
            [train_sequences[i] for i in inner_train_idx],
            train_labels[inner_train_idx],
        ),
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_time_series,
    )
    inner_val_loader = DataLoader(
        TimeSeriesDataset(
            [train_sequences[i] for i in inner_val_idx],
            train_labels[inner_val_idx],
        ),
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_time_series,
    )
    outer_valid_loader = DataLoader(
        TimeSeriesDataset(valid_sequences, valid_labels),
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_time_series,
    )

    pretrained = _load_pretrained(config)
    if pretrained is None:
        candidates = _build_model_candidates(config)
        init_state_dict = None
    else:
        candidates = [("pretrained", pretrained[0])]
        init_state_dict = pretrained[1]
    candidate_runs: list[tuple[str, dict[str, object]]] = []
    candidate_summaries: list[dict[str, object]] = []

    for name, model_cfg in candidates:
        cand_out = _train_single_candidate(
            model_cfg=model_cfg,
            inner_train_loader=inner_train_loader,
            inner_val_loader=inner_val_loader,
            inner_val_labels=train_labels[inner_val_idx],
            config=config,
            device=device,
            candidate_seed=split_seed,
            init_state_dict=init_state_dict,
        )
        candidate_runs.append((name, cand_out))
        candidate_summaries.append(
            {
                "name": name,
                "n_params": cand_out["n_params"],
                "d_model": int(model_cfg.d_model),
                "nhead": int(model_cfg.nhead),
                "num_layers": int(model_cfg.num_layers),
                "dim_feedforward": int(model_cfg.dim_feedforward),
                "best_epoch": cand_out["best_epoch"],
                "stopped_epoch": cand_out["stopped_epoch"],
                "best_inner_valid_loss": cand_out["best_inner_valid_loss"],
                "best_inner_valid_auroc": cand_out["best_inner_valid_auroc"],
            }
        )

    best_name, best_out = min(candidate_runs, key=lambda item: item[1]["best_inner_valid_loss"])
    best_model_cfg = best_out["model_cfg"]

    model = TimeSeriesTransformer(
        input_dim=best_model_cfg.input_dim,
        d_model=best_model_cfg.d_model,
        nhead=best_model_cfg.nhead,
        num_layers=best_model_cfg.num_layers,
        dim_feedforward=best_model_cfg.dim_feedforward,
        dropout=best_model_cfg.dropout,
        activation=best_model_cfg.activation,
        norm_first=best_model_cfg.norm_first,
        batch_first=best_model_cfg.batch_first,
        pooling=best_model_cfg.pooling,
    ).to(device)
    model.load_state_dict(best_out["model_state_dict"])

    criterion = nn.BCEWithLogitsLoss()
    outer_valid_loss, probs = evaluate_loader(model, outer_valid_loader, criterion, device)

    return {
        "probs": probs,
        "selected_architecture": best_name,
        "selected_n_params": best_out["n_params"],
        "candidate_results": candidate_summaries,
        "train_losses": best_out["train_losses"],
        "inner_valid_losses": best_out["inner_valid_losses"],
        "best_epoch": best_out["best_epoch"],
        "stopped_epoch": best_out["stopped_epoch"],
        "best_inner_valid_loss": best_out["best_inner_valid_loss"],
        "outer_valid_loss": outer_valid_loss,
        "n_inner_train": int(len(inner_train_idx)),
        "n_inner_valid": int(len(inner_val_idx)),
    }


def summarize_calibration_bins(
    y: np.ndarray, p: np.ndarray, n_bins: int
) -> list[dict[str, object]]:
    """Summarize quantile calibration bins matching sklearn's calibration_curve."""
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(p, quantiles)
    edges[-1] += 1e-8
    bin_ids = np.searchsorted(edges[1:-1], p, side="right")

    summaries: list[dict[str, object]] = []
    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        count = int(mask.sum())
        if count == 0:
            continue
        y_bin = y[mask]
        p_bin = p[mask]
        summaries.append(
            {
                "bin": bin_idx + 1,
                "n_samples": count,
                "n_adhd": int((y_bin == 1).sum()),
                "n_control": int((y_bin == 0).sum()),
                "prob_min": float(p_bin.min()),
                "prob_max": float(p_bin.max()),
                "mean_predicted_prob": float(p_bin.mean()),
                "observed_proportion": float(y_bin.mean()),
            }
        )
    return summaries


def _pad_loss_curves(curves: list[list[float]]) -> np.ndarray:
    max_len = max(len(c) for c in curves)
    padded = np.full((len(curves), max_len), np.nan, dtype=np.float64)
    for idx, curve in enumerate(curves):
        padded[idx, : len(curve)] = curve
    return padded


def plot_loss_curves(
    train_losses_per_fold: list[list[float]],
    valid_losses_per_fold: list[list[float]],
    path: Path,
) -> None:
    """Plot per-fold and mean train/internal-validation loss curves across epochs."""
    plt.switch_backend("Agg")
    train_arr = _pad_loss_curves(train_losses_per_fold)
    valid_arr = _pad_loss_curves(valid_losses_per_fold)
    epochs = np.arange(1, train_arr.shape[1] + 1)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i in range(train_arr.shape[0]):
        ax.plot(epochs, train_arr[i], color="#1f77b4", alpha=0.25, linewidth=1.0)
        ax.plot(epochs, valid_arr[i], color="#d62728", alpha=0.25, linewidth=1.0)

    ax.plot(
        epochs,
        np.nanmean(train_arr, axis=0),
        color="#1f77b4",
        linewidth=2.2,
        label="Inner train loss (mean)",
    )
    ax.plot(
        epochs,
        np.nanmean(valid_arr, axis=0),
        color="#d62728",
        linewidth=2.2,
        label="Inner valid loss (mean)",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCEWithLogitsLoss")
    ax.set_title("5-Fold Inner Train vs Inner Validation Loss")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_probability_histogram(y: np.ndarray, p: np.ndarray, path: Path) -> None:
    """Plot predicted probability histogram split by ADHD vs Control."""
    plt.switch_backend("Agg")
    bins = np.linspace(0.0, 1.0, 21)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(
        p[y == 0],
        bins=bins,
        alpha=0.6,
        color="#1f77b4",
        edgecolor="white",
        label=f"Control (n={int((y == 0).sum())})",
    )
    ax.hist(
        p[y == 1],
        bins=bins,
        alpha=0.6,
        color="#d62728",
        edgecolor="white",
        label=f"ADHD (n={int((y == 1).sum())})",
    )
    ax.set_xlabel("Predicted probability of ADHD")
    ax.set_ylabel("Number of subjects")
    ax.set_title("Out-of-Fold Predicted Probability Distribution")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run(config: DictConfig) -> dict[str, object]:
    set_seed(config.seed)
    device = resolve_device(config.training.device)

    sequences, labels, subject_ids, site_names = load_processed_dataset(
        manifest_path=config.data.manifest_path,
        processed_dir=config.data.processed_dir,
        run_selection=config.data.run_selection,
        standardize=config.data.standardize_per_subject,
    )

    oof = np.zeros(len(labels), dtype=np.float64)
    train_losses_per_fold: list[list[float]] = []
    valid_losses_per_fold: list[list[float]] = []
    fold_metrics: list[dict[str, object]] = []

    folds = StratifiedKFold(config.cv.n_splits, shuffle=True, random_state=config.seed)
    for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(subject_ids, labels), start=1):
        train_seqs = [sequences[i] for i in train_idx]
        valid_seqs = [sequences[i] for i in valid_idx]
        fold_out = train_one_fold(
            train_seqs,
            labels[train_idx],
            valid_seqs,
            labels[valid_idx],
            config,
            device,
            fold_seed=int(config.seed) + fold_idx,
        )
        fold_probs = fold_out["probs"]
        train_losses = fold_out["train_losses"]
        inner_valid_losses = fold_out["inner_valid_losses"]

        oof[valid_idx] = fold_probs
        train_losses_per_fold.append(train_losses)
        valid_losses_per_fold.append(inner_valid_losses)

        y_val = labels[valid_idx]
        fold_metrics.append(
            {
                "fold": fold_idx,
                "selected_architecture": fold_out["selected_architecture"],
                "selected_n_params": fold_out["selected_n_params"],
                "candidate_results": fold_out["candidate_results"],
                "n_inner_train": fold_out["n_inner_train"],
                "n_inner_valid": fold_out["n_inner_valid"],
                "n_outer_valid": int(len(valid_idx)),
                "auroc": float(roc_auc_score(y_val, fold_probs)),
                "brier": float(brier_score_loss(y_val, fold_probs)),
                "prob_min": float(fold_probs.min()),
                "prob_max": float(fold_probs.max()),
                "prob_median": float(np.median(fold_probs)),
                "prob_mean": float(fold_probs.mean()),
                "best_epoch": fold_out["best_epoch"],
                "stopped_epoch": fold_out["stopped_epoch"],
                "best_inner_valid_loss": fold_out["best_inner_valid_loss"],
                "outer_valid_loss": fold_out["outer_valid_loss"],
                "final_train_loss": float(train_losses[-1]),
                "final_inner_valid_loss": float(inner_valid_losses[-1]),
            }
        )

    bins = config.calibration_bins
    threshold = choose_threshold(labels, oof, config.threshold)
    calibration_bins_summary = summarize_calibration_bins(labels, oof, bins)
    prob_distribution = {
        "n_le_0.1": int((oof <= 0.1).sum()),
        "n_ge_0.9": int((oof >= 0.9).sum()),
        "n_between_0.1_and_0.9": int(((oof > 0.1) & (oof < 0.9)).sum()),
        "adhd_le_0.1": int(((oof <= 0.1) & (labels == 1)).sum()),
        "adhd_ge_0.9": int(((oof >= 0.9) & (labels == 1)).sum()),
        "control_le_0.1": int(((oof <= 0.1) & (labels == 0)).sum()),
        "control_ge_0.9": int(((oof >= 0.9) & (labels == 0)).sum()),
    }

    results: dict[str, object] = {
        "n_subjects": int(len(labels)),
        "n_adhd": int(labels.sum()),
        "architecture_candidates": (
            OmegaConf.to_container(config.architecture_candidates)
            if config.get("architecture_candidates", None) is not None
            else None
        ),
        "selected_architecture_counts": {
            str(k): int(v)
            for k, v in pd.Series([m["selected_architecture"] for m in fold_metrics])
            .value_counts()
            .items()
        },
        "n_control": int((labels == 0).sum()),
        "site_counts": {str(k): int(v) for k, v in pd.Series(site_names).value_counts().items()},
        "model": OmegaConf.to_container(config.model),
        "threshold": threshold,
        "cv_metrics": compute_metrics(labels, oof, threshold, bins),
        "cv_ci95": bootstrap_ci(
            labels, oof, threshold, bins, config.bootstrap.n_resamples, config.seed
        ),
        "fold_metrics": fold_metrics,
        "probability_distribution": prob_distribution,
        "calibration_bins_summary": calibration_bins_summary,
        "mean_train_loss_curve": np.nanmean(
            _pad_loss_curves(train_losses_per_fold), axis=0
        ).tolist(),
        "mean_inner_valid_loss_curve": np.nanmean(
            _pad_loss_curves(valid_losses_per_fold), axis=0
        ).tolist(),
    }

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    plot_calibration(labels, {"cross-validated": oof}, bins, output_dir / "calibration_curve.png")
    plot_loss_curves(train_losses_per_fold, valid_losses_per_fold, output_dir / "loss_curve.png")
    plot_probability_histogram(labels, oof, output_dir / "prob_histogram.png")
    print(json.dumps(results, indent=2))
    return results


def main() -> None:
    cli = OmegaConf.from_cli()
    config_path = cli.pop("config", str(CONFIG_PATH))
    run(OmegaConf.merge(OmegaConf.load(config_path), cli))


if __name__ == "__main__":
    main()
