from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedKFold

from adhd200_multimodal.intermediate_fusion import (
    IntermediateFusionModel,
    PhenotypePreprocessor,
    collate_multimodal,
    get_outer_folds,
    run,
)


def test_intermediate_fusion_model_shapes_and_padding_invariance() -> None:
    fmri_cfg = OmegaConf.create(
        {
            "input_dim": 190,
            "d_model": 32,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 64,
            "dropout": 0.2,
            "activation": "gelu",
            "norm_first": True,
            "batch_first": True,
            "pooling": "mean",
        }
    )
    model = IntermediateFusionModel(
        fmri_cfg=fmri_cfg,
        phenotype_dim=10,
        phenotype_hidden_dim=16,
        phenotype_embedding_dim=12,
        phenotype_dropout=0.2,
        fusion_hidden_dim=20,
        fusion_dropout=0.2,
    )
    model.eval()

    x1 = torch.randn(2, 9, 190)
    mask = torch.zeros((2, 9), dtype=torch.bool)
    mask[0, 6:] = True
    mask[1, 4:] = True

    x2 = x1.clone()
    x2[0, 6:, :] = 1000.0
    x2[1, 4:, :] = -1000.0

    pheno = torch.randn(2, 10)
    with torch.no_grad():
        fmri_emb1, pheno_emb1 = model.encode(x1, mask, pheno)
        fmri_emb2, pheno_emb2 = model.encode(x2, mask, pheno)
        logits1 = model(x1, mask, pheno)
        logits2 = model(x2, mask, pheno)

    assert fmri_emb1.shape == (2, 32)
    assert pheno_emb1.shape == (2, 12)
    assert logits1.shape == (2,)
    assert torch.allclose(fmri_emb1, fmri_emb2, atol=1e-5)
    assert torch.allclose(pheno_emb1, pheno_emb2, atol=1e-5)
    assert torch.allclose(logits1, logits2, atol=1e-5)


def test_collate_multimodal_pads_and_stacks() -> None:
    batch = [
        (torch.ones((3, 16)), torch.tensor([1.0, 2.0]), torch.tensor(0.0)),
        (torch.ones((5, 16)), torch.tensor([3.0, 4.0]), torch.tensor(1.0)),
    ]
    padded, mask, pheno, labels = collate_multimodal(batch)
    assert padded.shape == (2, 5, 16)
    assert mask.shape == (2, 5)
    assert mask[0].tolist() == [False, False, False, True, True]
    assert mask[1].tolist() == [False] * 5
    assert pheno.shape == (2, 2)
    assert labels.tolist() == [0.0, 1.0]


def test_phenotype_preprocessor_fits_only_on_training_fold() -> None:
    train_df = pd.DataFrame(
        {
            "Age": [10.0, 12.0, np.nan, 14.0],
            "Full2 IQ": [np.nan, np.nan, np.nan, np.nan],
            "Gender": [0.0, 1.0, 1.0, np.nan],
        }
    )
    test_df_a = pd.DataFrame(
        {
            "Age": [11.0, 13.0],
            "Full2 IQ": [100.0, 110.0],
            "Gender": [0.0, 1.0],
        }
    )
    test_df_b = pd.DataFrame(
        {
            "Age": [999.0, -999.0],
            "Full2 IQ": [100.0, 110.0],
            "Gender": [0.0, 1.0],
        }
    )

    prep = PhenotypePreprocessor(
        continuous_features=["Age", "Full2 IQ"],
        categorical_features=["Gender"],
    )
    train_arr = prep.fit_transform(train_df)
    assert prep.dropped_all_missing_ == ["Full2 IQ"]
    assert train_arr.shape[0] == 4
    # Age(1) + Age_missing(1) + Gender_0/Gender_1(2) + Gender_missing(1) = 5
    assert train_arr.shape[1] == 5

    # 仅测试集改变极端值，不应影响训练集输出的均值/方差或已拟合的变换器
    test_a = prep.transform(test_df_a)
    test_b = prep.transform(test_df_b)
    assert test_a.shape == (2, 5)
    assert test_b.shape == (2, 5)
    assert np.allclose(train_arr[:, 0].mean(), 0.0, atol=1e-5)


def test_outer_folds_match_transformer_baseline_exactly() -> None:
    manifest = Path("data/processed/transformer_pre_manifest.csv")
    if not manifest.is_file():
        return
    df = pd.read_csv(manifest)
    copied = df[df["status"] == "copied"].reset_index(drop=True)
    subject_ids = copied["subject_id"].to_numpy(dtype=np.int64)
    labels = copied["label"].to_numpy(dtype=np.int64)

    fusion_folds = get_outer_folds(subject_ids, labels, n_splits=5, seed=42)
    baseline_folds = list(
        StratifiedKFold(5, shuffle=True, random_state=42).split(subject_ids, labels)
    )

    for (tr1, va1), (tr2, va2) in zip(fusion_folds, baseline_folds, strict=True):
        assert np.array_equal(subject_ids[tr1], subject_ids[tr2])
        assert np.array_equal(subject_ids[va1], subject_ids[va2])


def test_train_one_fold_and_end_to_end_run(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    pheno_rows = [
        "ScanDir ID\tDX\tAge\tGender\tHandedness\t"
        "Verbal IQ\tPerformance IQ\tFull2 IQ\tFull4 IQ"
    ]
    manifest_rows = [
        "subject_id,site_id,site_name,subject_dir,dx,"
        "label,adhd_index,n_files,files,status"
    ]

    rng = np.random.default_rng(123)
    cols = ["File", "Sub-brick"] + [f"Mean_{i}" for i in range(1, 17)]
    for idx in range(30):
        sid = 1000 + idx
        label = idx % 2
        site = "KKI" if idx < 15 else "NYU"
        sub_dir = processed / site / str(sid)
        sub_dir.mkdir(parents=True, exist_ok=True)
        fname = f"sfnwmrda{sid}_session_1_rest_1_cc200_TCs.1D"
        ts_rows = []
        for t in range(6):
            vals = [f"{float(v):.4f}" for v in rng.normal(loc=0.2 * label, size=16)]
            ts_rows.append("\t".join(["f.nii.gz", f"{t}[?]"] + vals))
        (sub_dir / fname).write_text("\t".join(cols) + "\n" + "\n".join(ts_rows) + "\n")
        manifest_rows.append(
            f"{sid},3,{site},{sid},{float(label)},{label},50.0,1,{site}/{sid}/{fname},copied"
        )

        hand = "-999" if idx % 7 == 0 else f"{0.5 + 0.01 * idx:.2f}"
        viq = "N/A" if idx % 9 == 0 else str(95 + idx)
        pheno_rows.append(
            f"{sid}\t{label}\t{10.0 + 0.2 * idx:.1f}\t{idx % 2}\t"
            f"{hand}\t{viq}\t{100 + idx}\t-999\t{98 + idx}"
        )

    manifest_path = processed / "manifest.csv"
    manifest_path.write_text("\n".join(manifest_rows) + "\n")
    pheno_path = tmp_path / "pheno.tsv"
    pheno_path.write_text("\n".join(pheno_rows) + "\n")

    out_dir = tmp_path / "reports"
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "data": {
                "manifest_path": str(manifest_path),
                "processed_dir": str(processed),
                "phenotype_path": str(pheno_path),
                "run_selection": "first",
                "standardize_per_subject": True,
                "continuous_features": [
                    "Age",
                    "Handedness",
                    "Verbal IQ",
                    "Performance IQ",
                    "Full2 IQ",
                    "Full4 IQ",
                ],
                "categorical_features": ["Gender"],
            },
            "fmri_encoder": {
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
            "phenotype_mlp": {"hidden_dim": 16, "embedding_dim": 8, "dropout": 0.1},
            "fusion_mlp": {"hidden_dim": 16, "dropout": 0.1},
            "cv": {"n_splits": 5},
            "training": {
                "batch_size": 8,
                "epochs": 4,
                "learning_rate": 0.01,
                "weight_decay": 0.01,
                "inner_val_fraction": 0.25,
                "early_stopping_patience": 2,
                "early_stopping_min_delta": 0.0,
                "device": "cpu",
            },
            "threshold": "f1",
            "calibration_bins": 5,
            "bootstrap": {"n_resamples": 20},
            "output_dir": str(out_dir),
        }
    )

    results = run(cfg)
    assert results["n_subjects"] == 30
    assert len(results["fold_thresholds"]) == 5
    assert (out_dir / "metrics.json").is_file()
    assert (out_dir / "oof_predictions.csv").is_file()
    assert (out_dir / "calibration_curve.png").is_file()
    assert (out_dir / "loss_curve.png").is_file()
    assert (out_dir / "prob_histogram.png").is_file()

    oof_df = pd.read_csv(out_dir / "oof_predictions.csv")
    assert list(oof_df.columns) == [
        "subject_id",
        "fold",
        "site_name",
        "label",
        "prob",
        "pred",
        "threshold",
    ]
    assert len(oof_df) == 30
    assert set(oof_df["fold"].unique()) == {1, 2, 3, 4, 5}
