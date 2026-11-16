"""Pretrain Transformer weights on valid subjects from data/raw/additional ADHD200."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from adhd200_multimodal.phenotype_xgb import (
    bootstrap_ci,
    choose_threshold,
    compute_metrics,
    plot_calibration,
)
from adhd200_multimodal.transformer import (
    TimeSeriesDataset,
    TimeSeriesTransformer,
    _build_model_candidates,
    _train_single_candidate,
    collate_time_series,
    evaluate_loader,
    load_time_series,
    plot_loss_curves,
    plot_probability_histogram,
    resolve_device,
    set_seed,
)

ID_COLUMN = "ScanDir ID"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "transformer_pretrain.yaml"


def collect_additional_dataset(
    phenotype_path: str | Path,
    additional_tcs_dir: str | Path,
    file_pattern: str = "sfnwmrda*",
    run_selection: str = "first",
    standardize: bool = True,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Load valid sfnwmrda time-series and labels from additional ADHD-200 sites."""
    raw = pd.read_csv(phenotype_path, sep="\t", na_values=["N/A", "pending"])
    if raw[ID_COLUMN].duplicated().any():
        raise ValueError(f"{ID_COLUMN} must be unique so that splits stay subject-level")

    valid_pheno = raw.dropna(subset=["DX"]).copy()
    valid_pheno[ID_COLUMN] = pd.to_numeric(valid_pheno[ID_COLUMN], errors="raise").astype(int)
    valid_pheno["label"] = (pd.to_numeric(valid_pheno["DX"], errors="raise") > 0).astype(int)
    pheno_by_id = valid_pheno.set_index(ID_COLUMN)

    root = Path(additional_tcs_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Additional ADHD-200 directory not found: {root}")

    sequences: list[np.ndarray] = []
    labels: list[int] = []
    subject_ids: list[int] = []
    site_names: list[str] = []
    skipped_missing_dx: list[int] = []
    skipped_missing_files: list[int] = []
    seen_ids: set[int] = set()

    for site_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for sub_dir in sorted(d for d in site_dir.iterdir() if d.is_dir()):
            try:
                subject_id = int(sub_dir.name)
            except ValueError:
                continue
            if subject_id in seen_ids:
                raise ValueError(f"Duplicate subject_id found across directories: {subject_id}")
            seen_ids.add(subject_id)

            if subject_id not in pheno_by_id.index:
                skipped_missing_dx.append(subject_id)
                continue

            matched_files = sorted(p for p in sub_dir.glob(file_pattern) if p.is_file())
            if not matched_files:
                skipped_missing_files.append(subject_id)
                continue

            if run_selection == "first":
                chosen_file = matched_files[0]
            else:
                raise ValueError(f"Unsupported run_selection: {run_selection}")

            arr = load_time_series(chosen_file, standardize=standardize)
            sequences.append(arr)
            labels.append(int(pheno_by_id.loc[subject_id, "label"]))
            subject_ids.append(subject_id)
            site_names.append(site_dir.name)

    if not sequences:
        raise ValueError(f"No valid subjects found under {root}")

    labels_arr = np.asarray(labels, dtype=np.int64)
    subject_ids_arr = np.asarray(subject_ids, dtype=np.int64)
    site_names_arr = np.asarray(site_names)

    summary: dict[str, object] = {
        "n_valid_subjects": int(len(labels_arr)),
        "n_adhd": int(labels_arr.sum()),
        "n_control": int((labels_arr == 0).sum()),
        "n_skipped_missing_dx": int(len(skipped_missing_dx)),
        "n_skipped_missing_files": int(len(skipped_missing_files)),
        "site_counts": {
            str(k): int(v) for k, v in pd.Series(site_names_arr).value_counts().items()
        },
    }
    return sequences, labels_arr, subject_ids_arr, site_names_arr, summary


def run(config: DictConfig) -> dict[str, object]:
    set_seed(config.seed)
    device = resolve_device(config.training.device)

    sequences, labels, subject_ids, _, data_summary = collect_additional_dataset(
        phenotype_path=config.data.phenotype_path,
        additional_tcs_dir=config.data.additional_tcs_dir,
        file_pattern=config.data.file_pattern,
        run_selection=config.data.run_selection,
        standardize=config.data.standardize_per_subject,
    )

    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=float(config.training.val_fraction),
        stratify=labels,
        random_state=int(config.seed),
    )

    train_loader = DataLoader(
        TimeSeriesDataset([sequences[i] for i in train_idx], labels[train_idx]),
        batch_size=int(config.training.batch_size),
        shuffle=True,
        collate_fn=collate_time_series,
    )
    val_loader = DataLoader(
        TimeSeriesDataset([sequences[i] for i in val_idx], labels[val_idx]),
        batch_size=int(config.training.batch_size),
        shuffle=False,
        collate_fn=collate_time_series,
    )

    # Compatibility adapter for _train_single_candidate which reads inner_val_* keys.
    train_cfg = OmegaConf.create(
        {
            "seed": int(config.seed),
            "model": OmegaConf.to_container(config.model, resolve=True),
            "training": {
                "batch_size": int(config.training.batch_size),
                "epochs": int(config.training.epochs),
                "learning_rate": float(config.training.learning_rate),
                "weight_decay": float(config.training.weight_decay),
                "early_stopping_patience": int(config.training.early_stopping_patience),
                "early_stopping_min_delta": float(config.training.early_stopping_min_delta),
            },
        }
    )

    candidates = _build_model_candidates(config)
    candidate_runs: list[tuple[str, dict[str, object]]] = []
    candidate_summaries: list[dict[str, object]] = []

    for name, model_cfg in candidates:
        cand_out = _train_single_candidate(
            model_cfg=model_cfg,
            inner_train_loader=train_loader,
            inner_val_loader=val_loader,
            inner_val_labels=labels[val_idx],
            config=train_cfg,
            device=device,
            candidate_seed=int(config.seed),
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
                "best_val_loss": cand_out["best_inner_valid_loss"],
                "best_val_auroc": cand_out["best_inner_valid_auroc"],
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

    criterion = torch.nn.BCEWithLogitsLoss()
    val_loss, val_probs = evaluate_loader(model, val_loader, criterion, device)

    y_val = labels[val_idx]
    bins = int(config.calibration_bins)
    threshold = choose_threshold(y_val, val_probs, config.threshold)
    val_metrics = compute_metrics(y_val, val_probs, threshold, bins)
    val_ci95 = bootstrap_ci(
        y_val, val_probs, threshold, bins, int(config.bootstrap.n_resamples), int(config.seed)
    )

    checkpoint_path = Path(config.output.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_out["model_state_dict"],
            "model_config": OmegaConf.to_container(best_model_cfg, resolve=True),
            "selected_architecture": best_name,
            "n_params": best_out["n_params"],
            "best_epoch": best_out["best_epoch"],
            "best_val_loss": float(val_loss),
            "val_auroc": val_metrics["auc"],
            "seed": int(config.seed),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
        },
        checkpoint_path,
    )

    report_dir = Path(config.output.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_loss_curves(
        [best_out["train_losses"]],
        [best_out["inner_valid_losses"]],
        report_dir / "loss_curve.png",
    )
    plot_calibration(
        y_val,
        {"pretrain-validation": val_probs},
        bins,
        report_dir / "calibration_curve.png",
    )
    plot_probability_histogram(y_val, val_probs, report_dir / "prob_histogram.png")

    results: dict[str, object] = {
        **data_summary,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "selected_architecture": best_name,
        "selected_n_params": best_out["n_params"],
        "selected_model_config": OmegaConf.to_container(best_model_cfg, resolve=True),
        "best_epoch": best_out["best_epoch"],
        "stopped_epoch": best_out["stopped_epoch"],
        "best_val_loss": float(val_loss),
        "threshold": threshold,
        "val_metrics": val_metrics,
        "val_ci95": val_ci95,
        "candidate_results": candidate_summaries,
        "checkpoint_path": str(checkpoint_path),
        "train_losses": best_out["train_losses"],
        "val_losses": best_out["inner_valid_losses"],
    }
    (report_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return results


def main() -> None:
    run(OmegaConf.merge(OmegaConf.load(CONFIG_PATH), OmegaConf.from_cli()))


if __name__ == "__main__":
    main()
