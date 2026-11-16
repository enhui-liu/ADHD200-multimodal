from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from adhd200_multimodal.transformer import (
    TimeSeriesTransformer,
    collate_time_series,
    load_processed_dataset,
    plot_loss_curves,
    plot_probability_histogram,
    summarize_calibration_bins,
    train_one_fold,
)


def test_transformer_config_and_masked_mean_pooling() -> None:
    model = TimeSeriesTransformer(
        input_dim=190,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.3,
        activation="gelu",
        norm_first=True,
        batch_first=True,
        pooling="mean",
    )
    model.eval()

    layer0 = model.encoder.layers[0]
    assert layer0.self_attn.num_heads == 4
    assert layer0.self_attn.embed_dim == 64
    assert layer0.linear1.out_features == 128
    assert layer0.norm_first is True
    assert layer0.self_attn.batch_first is True

    x1 = torch.randn(2, 10, 190)
    mask = torch.zeros((2, 10), dtype=torch.bool)
    mask[0, 7:] = True
    mask[1, 5:] = True

    x2 = x1.clone()
    x2[0, 7:, :] = 999.0
    x2[1, 5:, :] = -999.0

    with torch.no_grad():
        out1 = model(x1, padding_mask=mask)
        out2 = model(x2, padding_mask=mask)

    assert out1.shape == (2,)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_collate_time_series_pads_and_creates_mask() -> None:
    batch = [
        (torch.ones((4, 190)), torch.tensor(0.0)),
        (torch.ones((7, 190)), torch.tensor(1.0)),
    ]
    padded, mask, labels = collate_time_series(batch)

    assert padded.shape == (2, 7, 190)
    assert mask.shape == (2, 7)
    assert mask[0].tolist() == [False, False, False, False, True, True, True]
    assert mask[1].tolist() == [False] * 7
    assert labels.tolist() == [0.0, 1.0]


def test_load_processed_dataset(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    sub_dir = processed / "KKI" / "1018959"
    sub_dir.mkdir(parents=True)

    cols = ["File", "Sub-brick"] + [f"Mean_{i}" for i in range(1, 191)]
    rows = []
    for t in range(5):
        vals = [f"{float(t + i):.4f}" for i in range(190)]
        rows.append("\t".join(["f.nii.gz", f"{t}[?]"] + vals))
    (sub_dir / "sfnwmrda1018959_session_1_rest_1_cc200_TCs.1D").write_text(
        "\t".join(cols) + "\n" + "\n".join(rows) + "\n"
    )

    manifest = processed / "transformer_pre_manifest.csv"
    manifest.write_text(
        "subject_id,site_id,site_name,subject_dir,dx,label,adhd_index,n_files,files,status\n"
        "1018959,3,KKI,1018959,0.0,0,45.0,1,"
        "KKI/1018959/sfnwmrda1018959_session_1_rest_1_cc200_TCs.1D,copied\n"
        "10016,5,NYU,0010016,1.0,1,65.0,0,,missing_sfnwmrda\n"
    )

    seqs, labels, subject_ids, site_names = load_processed_dataset(manifest, processed)
    assert len(seqs) == 1
    assert seqs[0].shape == (5, 190)
    assert np.allclose(seqs[0].mean(axis=0), 0.0, atol=1e-5)
    assert labels.tolist() == [0]
    assert subject_ids.tolist() == [1018959]
    assert site_names.tolist() == ["KKI"]


def test_diagnostics_helpers(tmp_path: Path) -> None:
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    p = np.linspace(0.05, 0.95, 10)
    bins_summary = summarize_calibration_bins(y, p, n_bins=5)
    assert len(bins_summary) == 5
    assert sum(item["n_samples"] for item in bins_summary) == 10

    loss_path = tmp_path / "loss.png"
    hist_path = tmp_path / "hist.png"
    plot_loss_curves([[0.7, 0.5, 0.4], [0.68, 0.48]], [[0.72, 0.75, 0.8], [0.71, 0.74]], loss_path)
    plot_probability_histogram(y, p, hist_path)
    assert loss_path.is_file()
    assert hist_path.is_file()


def test_train_one_fold_early_stopping_and_restores_best_model() -> None:
    rng = np.random.default_rng(7)
    train_seqs = [rng.normal(size=(6, 16)).astype(np.float32) for _ in range(20)]
    train_labels = np.array([0, 1] * 10, dtype=np.int64)
    valid_seqs = [rng.normal(size=(6, 16)).astype(np.float32) for _ in range(6)]
    valid_labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)

    cfg = OmegaConf.create(
        {
            "seed": 42,
            "model": {
                "input_dim": 16,
                "d_model": 16,
                "nhead": 4,
                "num_layers": 1,
                "dim_feedforward": 32,
                "dropout": 0.1,
                "activation": "gelu",
                "norm_first": True,
                "batch_first": True,
                "pooling": "mean",
            },
            "training": {
                "batch_size": 4,
                "epochs": 25,
                "learning_rate": 0.01,
                "weight_decay": 0.01,
                "inner_val_fraction": 0.2,
                "early_stopping_patience": 2,
                "early_stopping_min_delta": 0.0,
            },
        }
    )

    out = train_one_fold(
        train_seqs,
        train_labels,
        valid_seqs,
        valid_labels,
        cfg,
        torch.device("cpu"),
        fold_seed=43,
    )

    assert out["probs"].shape == (6,)
    assert out["selected_architecture"] == "default"
    assert 1 <= out["best_epoch"] <= out["stopped_epoch"] <= 25
    assert out["stopped_epoch"] <= out["best_epoch"] + 2
    assert out["best_inner_valid_loss"] == min(out["inner_valid_losses"])


def test_train_one_fold_selects_architecture_inside_training_fold() -> None:
    rng = np.random.default_rng(11)
    train_seqs = [rng.normal(size=(6, 16)).astype(np.float32) for _ in range(24)]
    train_labels = np.array([0, 1] * 12, dtype=np.int64)
    valid_seqs = [rng.normal(size=(6, 16)).astype(np.float32) for _ in range(6)]
    valid_labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)

    cfg = OmegaConf.create(
        {
            "seed": 42,
            "model": {
                "input_dim": 16,
                "d_model": 32,
                "nhead": 2,
                "num_layers": 2,
                "dim_feedforward": 64,
                "dropout": 0.3,
                "activation": "gelu",
                "norm_first": True,
                "batch_first": True,
                "pooling": "mean",
            },
            "architecture_candidates": [
                {
                    "name": "base",
                    "d_model": 32,
                    "nhead": 2,
                    "num_layers": 2,
                    "dim_feedforward": 64,
                },
                {
                    "name": "lightweight",
                    "d_model": 16,
                    "nhead": 2,
                    "num_layers": 1,
                    "dim_feedforward": 32,
                },
            ],
            "training": {
                "batch_size": 4,
                "epochs": 10,
                "learning_rate": 0.005,
                "weight_decay": 0.01,
                "inner_val_fraction": 0.25,
                "early_stopping_patience": 2,
                "early_stopping_min_delta": 0.0,
            },
        }
    )

    out = train_one_fold(
        train_seqs,
        train_labels,
        valid_seqs,
        valid_labels,
        cfg,
        torch.device("cpu"),
        fold_seed=100,
    )

    candidates = out["candidate_results"]
    assert len(candidates) == 2
    assert {c["name"] for c in candidates} == {"base", "lightweight"}
    best_by_inner = min(candidates, key=lambda c: c["best_inner_valid_loss"])
    assert out["selected_architecture"] == best_by_inner["name"]
    assert out["probs"].shape == (6,)


def test_train_one_fold_initializes_from_pretrained_checkpoint(tmp_path: Path) -> None:
    model_cfg = {
        "input_dim": 16,
        "d_model": 16,
        "nhead": 2,
        "num_layers": 1,
        "dim_feedforward": 32,
        "dropout": 0.1,
        "activation": "gelu",
        "norm_first": True,
        "batch_first": True,
        "pooling": "mean",
    }
    torch.manual_seed(0)
    source = TimeSeriesTransformer(**model_cfg)
    checkpoint = tmp_path / "pretrained.pt"
    torch.save({"model_config": model_cfg, "model_state_dict": source.state_dict()}, checkpoint)

    rng = np.random.default_rng(3)
    train_seqs = [rng.normal(size=(6, 16)).astype(np.float32) for _ in range(20)]
    train_labels = np.array([0, 1] * 10, dtype=np.int64)
    valid_seqs = [rng.normal(size=(6, 16)).astype(np.float32) for _ in range(6)]
    valid_labels = np.array([0, 1] * 3, dtype=np.int64)

    cfg = OmegaConf.create(
        {
            "seed": 42,
            "pretrained_checkpoint": str(checkpoint),
            "model": model_cfg,
            "training": {
                "batch_size": 4,
                "epochs": 3,
                "learning_rate": 0.0,
                "weight_decay": 0.0,
                "inner_val_fraction": 0.25,
                "early_stopping_patience": 2,
                "early_stopping_min_delta": 0.0,
            },
        }
    )

    out = train_one_fold(
        train_seqs, train_labels, valid_seqs, valid_labels, cfg, torch.device("cpu"), fold_seed=1
    )

    source.eval()
    with torch.no_grad():
        x, mask, _ = collate_time_series(
            [(torch.as_tensor(s), torch.tensor(float(y))) for s, y in zip(valid_seqs, valid_labels)]
        )
        expected = torch.sigmoid(source(x, padding_mask=mask)).numpy()

    assert out["selected_architecture"] == "pretrained"
    assert np.allclose(out["probs"], expected, atol=1e-5)
