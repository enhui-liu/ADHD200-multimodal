"""Intermediate fusion (fMRI Transformer + Phenotype MLP) on the 292-subject ADHD-200 cohort."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from adhd200_multimodal.phenotype_xgb import (
    ID_COLUMN,
    MISSING_CODE,
    choose_threshold,
    expected_calibration_error,
    fit_logistic,
    plot_calibration,
)
from adhd200_multimodal.transformer import (
    SinusoidalPositionalEncoding,
    _pad_loss_curves,
    load_processed_dataset,
    plot_loss_curves,
    plot_probability_histogram,
    resolve_device,
    set_seed,
    summarize_calibration_bins,
)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "intermediate_fusion.yaml"


def load_aligned_phenotypes(
    phenotype_path: str | Path,
    subject_ids: np.ndarray,
    continuous_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    """Load and strictly align phenotypic features to the given subject_ids order."""
    raw = pd.read_csv(phenotype_path, sep="\t", na_values=["N/A", "pending"])
    raw[ID_COLUMN] = pd.to_numeric(raw[ID_COLUMN], errors="coerce")
    if raw[ID_COLUMN].isna().any():
        raise ValueError(f"{ID_COLUMN} contains non-numeric or missing values")
    if raw[ID_COLUMN].duplicated().any():
        raise ValueError(f"{ID_COLUMN} must be unique so that alignment stays 1-to-1")

    raw[ID_COLUMN] = raw[ID_COLUMN].astype(np.int64)
    indexed = raw.set_index(ID_COLUMN)

    target_ids = [int(sid) for sid in subject_ids]
    missing_ids = [sid for sid in target_ids if sid not in indexed.index]
    if missing_ids:
        raise ValueError(f"Subjects missing from phenotype table: {missing_ids[:5]}")

    aligned = indexed.loc[target_ids]
    all_features = list(continuous_features) + list(categorical_features)
    features_frame = aligned[all_features].apply(pd.to_numeric, errors="coerce")
    return features_frame.replace(MISSING_CODE, np.nan).reset_index(drop=True)


def get_outer_folds(
    subject_ids: np.ndarray,
    labels: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Reproduce the exact outer StratifiedKFold split used by the Transformer baseline."""
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(folds.split(subject_ids, labels))


class PhenotypePreprocessor:
    """Fold-safe preprocessor handling continuous and categorical phenotypes separately."""

    def __init__(
        self,
        continuous_features: list[str],
        categorical_features: list[str],
    ) -> None:
        self.continuous_features = list(continuous_features)
        self.categorical_features = list(categorical_features)
        self.dropped_all_missing_: list[str] = []
        self.active_continuous_: list[str] = []
        self.active_categorical_: list[str] = []
        self.transformer_: ColumnTransformer | None = None

    def fit(self, df: pd.DataFrame) -> PhenotypePreprocessor:
        all_cols = self.continuous_features + self.categorical_features
        self.dropped_all_missing_ = [col for col in all_cols if df[col].isna().all()]
        self.active_continuous_ = [
            col for col in self.continuous_features if col not in self.dropped_all_missing_
        ]
        self.active_categorical_ = [
            col for col in self.categorical_features if col not in self.dropped_all_missing_
        ]

        transformers: list[tuple[str, Pipeline, list[str]]] = []
        if self.active_continuous_:
            cont_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("continuous", cont_pipe, self.active_continuous_))

        if self.active_categorical_:
            cat_ohe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            )
            transformers.append(("categorical", cat_ohe, self.active_categorical_))
            transformers.append(
                ("categorical_missing", MissingIndicator(features="all"), self.active_categorical_)
            )

        if not transformers:
            raise ValueError("All phenotypic features are completely missing in training fold")

        self.transformer_ = ColumnTransformer(transformers=transformers, remainder="drop")
        self.transformer_.fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.transformer_ is None:
            raise RuntimeError("PhenotypePreprocessor must be fitted before transform")
        arr = self.transformer_.transform(df)
        return np.asarray(arr, dtype=np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


class MultimodalTimeSeriesDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Subject-level dataset pairing variable-length ROI time-series and phenotypic vectors."""

    def __init__(
        self,
        sequences: list[np.ndarray],
        phenotypes: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        if not (len(sequences) == len(phenotypes) == len(labels)):
            raise ValueError("sequences, phenotypes, and labels must have identical lengths")
        self.sequences = [torch.as_tensor(seq, dtype=torch.float32) for seq in sequences]
        self.phenotypes = torch.as_tensor(phenotypes, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.sequences[index], self.phenotypes[index], self.labels[index]


def collate_multimodal(
    batch: list[tuple[Tensor, Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pad variable-length (T_i, R) sequences and stack phenotype vectors and labels."""
    batch_size = len(batch)
    max_len = max(seq.shape[0] for seq, _, _ in batch)
    fmri_dim = batch[0][0].shape[1]
    pheno_dim = batch[0][1].shape[0]

    padded = torch.zeros((batch_size, max_len, fmri_dim), dtype=torch.float32)
    padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)
    phenotypes = torch.zeros((batch_size, pheno_dim), dtype=torch.float32)
    labels = torch.zeros((batch_size,), dtype=torch.float32)

    for idx, (seq, pheno, label) in enumerate(batch):
        seq_len = seq.shape[0]
        padded[idx, :seq_len, :] = seq
        padding_mask[idx, :seq_len] = False
        phenotypes[idx] = pheno
        labels[idx] = label

    return padded, padding_mask, phenotypes, labels


class FMRITransformerEncoder(nn.Module):
    """Transformer encoder with attention padding mask and masked mean pooling."""

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
            raise ValueError("FMRITransformerEncoder requires batch_first=True")
        if pooling != "mean":
            raise ValueError(f"Unsupported pooling: {pooling}")

        self.d_model = d_model
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
        return pooled


class PhenotypeMLP(nn.Module):
    """MLP encoder for fold-preprocessed phenotypic features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        embedding_dim: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

    def forward(self, phenotypes: Tensor) -> Tensor:
        if phenotypes.ndim != 2:
            raise ValueError("phenotypes must have shape (batch, features)")
        return self.network(phenotypes)


class IntermediateFusionModel(nn.Module):
    """Intermediate fusion: independent fMRI Transformer + Phenotype MLP -> Concat + Fusion MLP."""

    def __init__(
        self,
        fmri_cfg: DictConfig,
        phenotype_dim: int,
        phenotype_hidden_dim: int = 32,
        phenotype_embedding_dim: int = 32,
        phenotype_dropout: float = 0.3,
        fusion_hidden_dim: int = 64,
        fusion_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.fmri_encoder = FMRITransformerEncoder(
            input_dim=int(fmri_cfg.input_dim),
            d_model=int(fmri_cfg.d_model),
            nhead=int(fmri_cfg.nhead),
            num_layers=int(fmri_cfg.num_layers),
            dim_feedforward=int(fmri_cfg.dim_feedforward),
            dropout=float(fmri_cfg.dropout),
            activation=str(fmri_cfg.activation),
            norm_first=bool(fmri_cfg.norm_first),
            batch_first=bool(fmri_cfg.batch_first),
            pooling=str(fmri_cfg.pooling),
        )
        self.phenotype_encoder = PhenotypeMLP(
            input_dim=phenotype_dim,
            hidden_dim=phenotype_hidden_dim,
            embedding_dim=phenotype_embedding_dim,
            dropout=phenotype_dropout,
        )
        fusion_in_dim = int(fmri_cfg.d_model) + phenotype_embedding_dim
        self.fusion_mlp = nn.Sequential(
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_in_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def encode(
        self,
        fmri: Tensor,
        padding_mask: Tensor | None,
        phenotypes: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self.fmri_encoder(fmri, padding_mask=padding_mask), self.phenotype_encoder(
            phenotypes
        )

    def forward(
        self,
        fmri: Tensor,
        padding_mask: Tensor | None,
        phenotypes: Tensor,
    ) -> Tensor:
        fmri_emb, pheno_emb = self.encode(fmri, padding_mask, phenotypes)
        fused = torch.cat((fmri_emb, pheno_emb), dim=1)
        return self.fusion_mlp(fused).squeeze(-1)


def _build_fmri_candidates(config: DictConfig) -> list[tuple[str, DictConfig]]:
    base_cfg = config.fmri_encoder
    raw_candidates = config.get("architecture_candidates", None)
    if not raw_candidates:
        return [("default", base_cfg)]

    candidates: list[tuple[str, DictConfig]] = []
    for idx, cand in enumerate(raw_candidates):
        cand_dict = OmegaConf.to_container(cand, resolve=True)
        name = str(cand_dict.pop("name", f"candidate_{idx + 1}"))
        merged = OmegaConf.merge(base_cfg, OmegaConf.create(cand_dict))
        candidates.append((name, merged))
    return candidates


def _build_model(
    fmri_cfg: DictConfig, phenotype_dim: int, config: DictConfig
) -> IntermediateFusionModel:
    return IntermediateFusionModel(
        fmri_cfg=fmri_cfg,
        phenotype_dim=phenotype_dim,
        phenotype_hidden_dim=int(config.phenotype_mlp.hidden_dim),
        phenotype_embedding_dim=int(config.phenotype_mlp.embedding_dim),
        phenotype_dropout=float(config.phenotype_mlp.dropout),
        fusion_hidden_dim=int(config.fusion_mlp.hidden_dim),
        fusion_dropout=float(config.fusion_mlp.dropout),
    )


def evaluate_multimodal_loader(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    model.eval()
    loss_sum = 0.0
    count = 0
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for fmri, padding_mask, pheno, y in loader:
            fmri = fmri.to(device)
            padding_mask = padding_mask.to(device)
            pheno = pheno.to(device)
            y = y.to(device)
            logits = model(fmri, padding_mask=padding_mask, phenotypes=pheno)
            loss = criterion(logits, y)
            batch_n = y.shape[0]
            loss_sum += float(loss.item()) * batch_n
            count += batch_n
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return loss_sum / max(count, 1), np.concatenate(probs, axis=0)


def _train_single_candidate(
    fmri_cfg: DictConfig,
    phenotype_dim: int,
    inner_train_loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    inner_val_loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    inner_val_labels: np.ndarray,
    config: DictConfig,
    device: torch.device,
    candidate_seed: int,
) -> dict[str, object]:
    set_seed(candidate_seed)
    patience = int(config.training.early_stopping_patience)
    min_delta = float(config.training.early_stopping_min_delta)

    model = _build_model(fmri_cfg, phenotype_dim, config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training.learning_rate),
        weight_decay=float(config.training.weight_decay),
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
        for fmri, padding_mask, pheno, y in inner_train_loader:
            fmri = fmri.to(device)
            padding_mask = padding_mask.to(device)
            pheno = pheno.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(fmri, padding_mask=padding_mask, phenotypes=pheno)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            batch_n = y.shape[0]
            train_loss_sum += float(loss.item()) * batch_n
            train_count += batch_n

        train_loss = train_loss_sum / max(train_count, 1)
        inner_val_loss, _ = evaluate_multimodal_loader(model, inner_val_loader, criterion, device)
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
    _, inner_val_probs = evaluate_multimodal_loader(model, inner_val_loader, criterion, device)
    inner_val_auroc = float(roc_auc_score(inner_val_labels, inner_val_probs))

    return {
        "fmri_cfg": fmri_cfg,
        "model_state_dict": best_state_dict,
        "n_params": int(n_params),
        "train_losses": train_losses,
        "inner_valid_losses": inner_valid_losses,
        "best_epoch": best_epoch,
        "stopped_epoch": len(train_losses),
        "best_inner_valid_loss": float(best_val_loss),
        "best_inner_valid_auroc": inner_val_auroc,
        "inner_val_probs": inner_val_probs,
    }


def train_one_fold(
    train_sequences: list[np.ndarray],
    train_phenotypes_df: pd.DataFrame,
    train_labels: np.ndarray,
    valid_sequences: list[np.ndarray],
    valid_phenotypes_df: pd.DataFrame,
    valid_labels: np.ndarray,
    config: DictConfig,
    device: torch.device,
    fold_seed: int | None = None,
) -> dict[str, object]:
    """Fit preprocessor, select model, early-stop, and pick threshold strictly inside train fold."""
    inner_val_fraction = float(config.training.inner_val_fraction)
    split_seed = int(config.seed) if fold_seed is None else int(fold_seed)

    indices = np.arange(len(train_labels))
    inner_train_idx, inner_val_idx = train_test_split(
        indices,
        test_size=inner_val_fraction,
        stratify=train_labels,
        random_state=split_seed,
    )

    preprocessor = PhenotypePreprocessor(
        continuous_features=list(config.data.continuous_features),
        categorical_features=list(config.data.categorical_features),
    )
    inner_train_pheno = preprocessor.fit_transform(
        train_phenotypes_df.iloc[inner_train_idx].reset_index(drop=True)
    )
    inner_val_pheno = preprocessor.transform(
        train_phenotypes_df.iloc[inner_val_idx].reset_index(drop=True)
    )
    outer_valid_pheno = preprocessor.transform(valid_phenotypes_df.reset_index(drop=True))
    phenotype_dim = int(inner_train_pheno.shape[1])

    inner_train_loader = DataLoader(
        MultimodalTimeSeriesDataset(
            [train_sequences[i] for i in inner_train_idx],
            inner_train_pheno,
            train_labels[inner_train_idx],
        ),
        batch_size=int(config.training.batch_size),
        shuffle=True,
        collate_fn=collate_multimodal,
    )
    inner_val_loader = DataLoader(
        MultimodalTimeSeriesDataset(
            [train_sequences[i] for i in inner_val_idx],
            inner_val_pheno,
            train_labels[inner_val_idx],
        ),
        batch_size=int(config.training.batch_size),
        shuffle=False,
        collate_fn=collate_multimodal,
    )
    outer_valid_loader = DataLoader(
        MultimodalTimeSeriesDataset(valid_sequences, outer_valid_pheno, valid_labels),
        batch_size=int(config.training.batch_size),
        shuffle=False,
        collate_fn=collate_multimodal,
    )

    candidates = _build_fmri_candidates(config)
    candidate_runs: list[tuple[str, dict[str, object]]] = []
    candidate_summaries: list[dict[str, object]] = []
    inner_val_labels = train_labels[inner_val_idx]

    for name, fmri_cfg in candidates:
        cand_out = _train_single_candidate(
            fmri_cfg=fmri_cfg,
            phenotype_dim=phenotype_dim,
            inner_train_loader=inner_train_loader,
            inner_val_loader=inner_val_loader,
            inner_val_labels=inner_val_labels,
            config=config,
            device=device,
            candidate_seed=split_seed,
        )
        candidate_runs.append((name, cand_out))
        candidate_summaries.append(
            {
                "name": name,
                "n_params": cand_out["n_params"],
                "d_model": int(fmri_cfg.d_model),
                "nhead": int(fmri_cfg.nhead),
                "num_layers": int(fmri_cfg.num_layers),
                "dim_feedforward": int(fmri_cfg.dim_feedforward),
                "best_epoch": cand_out["best_epoch"],
                "stopped_epoch": cand_out["stopped_epoch"],
                "best_inner_valid_loss": cand_out["best_inner_valid_loss"],
                "best_inner_valid_auroc": cand_out["best_inner_valid_auroc"],
            }
        )

    best_name, best_out = min(candidate_runs, key=lambda item: item[1]["best_inner_valid_loss"])
    best_fmri_cfg = best_out["fmri_cfg"]

    # 阈值仅在当前折的内部验证集上确定，绝不使用外层测试集标签。
    fold_threshold = choose_threshold(
        inner_val_labels,
        best_out["inner_val_probs"],
        config.threshold,
    )

    model = _build_model(best_fmri_cfg, phenotype_dim, config).to(device)
    model.load_state_dict(best_out["model_state_dict"])
    criterion = nn.BCEWithLogitsLoss()
    outer_valid_loss, probs = evaluate_multimodal_loader(
        model, outer_valid_loader, criterion, device
    )
    preds = (probs >= fold_threshold).astype(np.int64)

    return {
        "probs": probs,
        "preds": preds,
        "fold_threshold": float(fold_threshold),
        "phenotype_dim": phenotype_dim,
        "dropped_phenotype_features": preprocessor.dropped_all_missing_,
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


def compute_oof_metrics(
    y: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    n_bins: int,
) -> dict[str, float]:
    """Compute discrimination, thresholded, and calibration metrics from OOF outputs."""
    predicted = preds.astype(bool)
    z = np.log(np.clip(probs, 1e-6, 1 - 1e-6) / (1.0 - np.clip(probs, 1e-6, 1 - 1e-6)))
    return {
        "auc": float(roc_auc_score(y, probs)),
        "sensitivity": float(predicted[y == 1].mean()),
        "specificity": float((~predicted[y == 0]).mean()),
        "f1": float(f1_score(y, predicted)),
        "brier": float(brier_score_loss(y, probs)),
        "calibration_slope": fit_logistic(z, y, fit_slope=True)[1],
        "calibration_intercept": fit_logistic(z, y, fit_slope=False)[0],
        "ece": expected_calibration_error(y, probs, n_bins),
    }


def bootstrap_oof_ci(
    y: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    n_bins: int,
    n_resamples: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        index = rng.integers(0, len(y), len(y))
        if y[index].min() == y[index].max():
            continue
        draws.append(compute_oof_metrics(y[index], probs[index], preds[index], n_bins))
    frame = pd.DataFrame(draws)
    return {
        name: [float(frame[name].quantile(0.025)), float(frame[name].quantile(0.975))]
        for name in frame
    }


def run(config: DictConfig) -> dict[str, object]:
    set_seed(int(config.seed))
    device = resolve_device(str(config.training.device))

    sequences, labels, subject_ids, site_names = load_processed_dataset(
        manifest_path=config.data.manifest_path,
        processed_dir=config.data.processed_dir,
        run_selection=config.data.run_selection,
        standardize=config.data.standardize_per_subject,
    )
    phenotypes_df = load_aligned_phenotypes(
        phenotype_path=config.data.phenotype_path,
        subject_ids=subject_ids,
        continuous_features=list(config.data.continuous_features),
        categorical_features=list(config.data.categorical_features),
    )

    n_subjects = len(labels)
    oof_probs = np.zeros(n_subjects, dtype=np.float64)
    oof_preds = np.zeros(n_subjects, dtype=np.int64)
    oof_folds = np.zeros(n_subjects, dtype=np.int64)
    oof_thresholds = np.zeros(n_subjects, dtype=np.float64)

    train_losses_per_fold: list[list[float]] = []
    valid_losses_per_fold: list[list[float]] = []
    fold_metrics: list[dict[str, object]] = []

    outer_folds = get_outer_folds(subject_ids, labels, int(config.cv.n_splits), int(config.seed))
    for fold_idx, (train_idx, valid_idx) in enumerate(outer_folds, start=1):
        train_seqs = [sequences[i] for i in train_idx]
        valid_seqs = [sequences[i] for i in valid_idx]
        train_pheno = phenotypes_df.iloc[train_idx].reset_index(drop=True)
        valid_pheno = phenotypes_df.iloc[valid_idx].reset_index(drop=True)

        fold_out = train_one_fold(
            train_seqs,
            train_pheno,
            labels[train_idx],
            valid_seqs,
            valid_pheno,
            labels[valid_idx],
            config,
            device,
            fold_seed=int(config.seed) + fold_idx,
        )
        fold_probs = fold_out["probs"]
        fold_preds = fold_out["preds"]
        fold_threshold = float(fold_out["fold_threshold"])
        train_losses = fold_out["train_losses"]
        inner_valid_losses = fold_out["inner_valid_losses"]

        oof_probs[valid_idx] = fold_probs
        oof_preds[valid_idx] = fold_preds
        oof_folds[valid_idx] = fold_idx
        oof_thresholds[valid_idx] = fold_threshold
        train_losses_per_fold.append(train_losses)
        valid_losses_per_fold.append(inner_valid_losses)

        y_val = labels[valid_idx]
        fold_metrics.append(
            {
                "fold": fold_idx,
                "selected_architecture": fold_out["selected_architecture"],
                "selected_n_params": fold_out["selected_n_params"],
                "phenotype_dim": fold_out["phenotype_dim"],
                "dropped_phenotype_features": fold_out["dropped_phenotype_features"],
                "threshold": fold_threshold,
                "candidate_results": fold_out["candidate_results"],
                "n_inner_train": fold_out["n_inner_train"],
                "n_inner_valid": fold_out["n_inner_valid"],
                "n_outer_valid": int(len(valid_idx)),
                "auroc": float(roc_auc_score(y_val, fold_probs)),
                "brier": float(brier_score_loss(y_val, fold_probs)),
                "f1": float(f1_score(y_val, fold_preds)),
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

    bins = int(config.calibration_bins)
    calibration_bins_summary = summarize_calibration_bins(labels, oof_probs, bins)
    prob_distribution = {
        "n_le_0.1": int((oof_probs <= 0.1).sum()),
        "n_ge_0.9": int((oof_probs >= 0.9).sum()),
        "n_between_0.1_and_0.9": int(((oof_probs > 0.1) & (oof_probs < 0.9)).sum()),
        "adhd_le_0.1": int(((oof_probs <= 0.1) & (labels == 1)).sum()),
        "adhd_ge_0.9": int(((oof_probs >= 0.9) & (labels == 1)).sum()),
        "control_le_0.1": int(((oof_probs <= 0.1) & (labels == 0)).sum()),
        "control_ge_0.9": int(((oof_probs >= 0.9) & (labels == 0)).sum()),
    }

    fold_thresholds = [float(m["threshold"]) for m in fold_metrics]
    results: dict[str, object] = {
        "n_subjects": int(n_subjects),
        "n_adhd": int(labels.sum()),
        "n_control": int((labels == 0).sum()),
        "site_counts": {str(k): int(v) for k, v in pd.Series(site_names).value_counts().items()},
        "continuous_features": list(config.data.continuous_features),
        "categorical_features": list(config.data.categorical_features),
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
        "fmri_encoder": OmegaConf.to_container(config.fmri_encoder),
        "phenotype_mlp": OmegaConf.to_container(config.phenotype_mlp),
        "fusion_mlp": OmegaConf.to_container(config.fusion_mlp),
        "threshold_rule": str(config.threshold),
        "fold_thresholds": fold_thresholds,
        "mean_fold_threshold": float(np.mean(fold_thresholds)),
        "cv_metrics": compute_oof_metrics(labels, oof_probs, oof_preds, bins),
        "cv_ci95": bootstrap_oof_ci(
            labels,
            oof_probs,
            oof_preds,
            bins,
            int(config.bootstrap.n_resamples),
            int(config.seed),
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

    oof_df = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "fold": oof_folds,
            "site_name": site_names,
            "label": labels,
            "prob": oof_probs,
            "pred": oof_preds,
            "threshold": oof_thresholds,
        }
    )
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)

    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    plot_calibration(
        labels, {"cross-validated": oof_probs}, bins, output_dir / "calibration_curve.png"
    )
    plot_loss_curves(train_losses_per_fold, valid_losses_per_fold, output_dir / "loss_curve.png")
    plot_probability_histogram(labels, oof_probs, output_dir / "prob_histogram.png")
    print(json.dumps(results, indent=2))
    return results


def main() -> None:
    cli = OmegaConf.from_cli()
    config_path = cli.pop("config", str(CONFIG_PATH))
    run(OmegaConf.merge(OmegaConf.load(config_path), cli))


if __name__ == "__main__":
    main()
