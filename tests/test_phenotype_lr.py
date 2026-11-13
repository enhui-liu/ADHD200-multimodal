import json

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from adhd200_multimodal.phenotype_lr import run


def test_run_uses_same_cohort_and_writes_metrics(tmp_path) -> None:
    rng = np.random.default_rng(0)
    n = 60
    frame = pd.DataFrame(
        {
            "ScanDir ID": np.arange(n),
            "DX": np.tile([0, 1], n // 2),
            "Age": rng.uniform(8, 13, n),
            "Verbal IQ": rng.normal(100, 10, n),
            "ADHD Measure": np.where(np.arange(n) < 50, 2, 1),
            "ADHD Index": np.where(np.arange(n) % 10 == 0, -999, 60),
        }
    )
    frame.loc[3, "Verbal IQ"] = np.nan
    data_path = tmp_path / "phenotypics.tsv"
    frame.to_csv(data_path, sep="\t", index=False)

    run(
        OmegaConf.create(
            {
                "seed": 0,
                "data": {
                    "path": str(data_path),
                    "adhd_measure": 2,
                    "require_positive_adhd_index": True,
                    "features": ["Age", "Verbal IQ"],
                },
                "cv": {"n_splits": 3},
                "model": {"C": 1.0, "max_iter": 200},
                "threshold": "youden",
                "calibration_bins": 3,
                "bootstrap": {"n_resamples": 5},
                "output_dir": str(tmp_path / "out"),
            }
        )
    )

    results = json.loads((tmp_path / "out" / "metrics.json").read_text())
    # 前 50 人中 ADHD Index 为 -999 的有 5 人，应被剔除。
    assert results["n_subjects"] == 45
    assert 0.0 <= results["cv_metrics"]["auc"] <= 1.0
    assert set(results["coefficients_standardized"]) >= {"Age", "Verbal IQ"}
