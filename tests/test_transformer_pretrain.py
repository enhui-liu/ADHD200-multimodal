from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from adhd200_multimodal.transformer_pretrain import collect_additional_dataset, run


def _write_tc_file(path: Path, n_timepoints: int = 8, n_rois: int = 16, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    cols = ["File", "Sub-brick"] + [f"Mean_{i}" for i in range(1, n_rois + 1)]
    rows = []
    for t in range(n_timepoints):
        vals = [f"{v:.4f}" for v in rng.normal(size=n_rois)]
        rows.append("\t".join(["f.nii.gz", f"{t}[?]"] + vals))
    path.write_text("\t".join(cols) + "\n" + "\n".join(rows) + "\n")


def test_collect_additional_dataset_filters_missing_dx_and_files(tmp_path: Path) -> None:
    tsv = tmp_path / "phenotypics.tsv"
    tsv.write_text(
        "ScanDir ID\tDX\n"
        "1001\t0\n"
        "1002\t3\n"
        "1003\tpending\n"
        "1004\t1\n"
    )

    add_dir = tmp_path / "additional"
    (add_dir / "OHSU" / "1001").mkdir(parents=True)
    _write_tc_file(add_dir / "OHSU" / "1001" / "sfnwmrda1001_session_1_rest_1_cc200_TCs.1D", seed=1)

    (add_dir / "Peking_1" / "1002").mkdir(parents=True)
    _write_tc_file(
        add_dir / "Peking_1" / "1002" / "sfnwmrda1002_session_1_rest_1_cc200_TCs.1D", seed=2
    )

    (add_dir / "WashU" / "1003").mkdir(parents=True)
    _write_tc_file(
        add_dir / "WashU" / "1003" / "sfnwmrda1003_session_1_rest_1_cc200_TCs.1D", seed=3
    )

    (add_dir / "WashU" / "1004").mkdir(parents=True)
    (add_dir / "WashU" / "1004" / "snwmrda1004_session_1_rest_1_cc200_TCs.1D").write_text("other")

    seqs, labels, subject_ids, sites, summary = collect_additional_dataset(tsv, add_dir)

    assert len(seqs) == 2
    assert labels.tolist() == [0, 1]
    assert subject_ids.tolist() == [1001, 1002]
    assert sites.tolist() == ["OHSU", "Peking_1"]
    assert summary["n_valid_subjects"] == 2
    assert summary["n_skipped_missing_dx"] == 1
    assert summary["n_skipped_missing_files"] == 1


def test_pretrain_run_saves_checkpoint_and_metrics(tmp_path: Path) -> None:
    tsv = tmp_path / "phenotypics.tsv"
    rows = ["ScanDir ID\tDX"]
    add_dir = tmp_path / "additional"
    for idx in range(12):
        sid = 2000 + idx
        dx = 0 if idx % 2 == 0 else 1
        rows.append(f"{sid}\t{dx}")
        sub_dir = add_dir / "OHSU" / str(sid)
        sub_dir.mkdir(parents=True)
        _write_tc_file(
            sub_dir / f"sfnwmrda{sid}_session_1_rest_1_cc200_TCs.1D",
            n_timepoints=6,
            n_rois=16,
            seed=idx,
        )
    tsv.write_text("\n".join(rows) + "\n")

    ckpt_path = tmp_path / "models" / "transformer_additional_best.pt"
    report_dir = tmp_path / "reports" / "transformer_pretrain"

    cfg = OmegaConf.create(
        {
            "seed": 42,
            "data": {
                "phenotype_path": str(tsv),
                "additional_tcs_dir": str(add_dir),
                "file_pattern": "sfnwmrda*",
                "run_selection": "first",
                "standardize_per_subject": True,
            },
            "model": {
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
            },
            "training": {
                "batch_size": 4,
                "epochs": 6,
                "learning_rate": 0.005,
                "weight_decay": 0.01,
                "val_fraction": 0.25,
                "early_stopping_patience": 2,
                "early_stopping_min_delta": 0.0,
                "device": "cpu",
            },
            "threshold": "f1",
            "calibration_bins": 5,
            "bootstrap": {"n_resamples": 100},
            "output": {
                "checkpoint_path": str(ckpt_path),
                "report_dir": str(report_dir),
            },
        }
    )

    summary = run(cfg)
    assert summary["n_valid_subjects"] == 12
    assert ckpt_path.is_file()
    assert (report_dir / "metrics.json").is_file()
    assert (report_dir / "loss_curve.png").is_file()

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in payload
    assert "model_config" in payload
    assert payload["model_config"]["input_dim"] == 16
