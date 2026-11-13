import numpy as np

from adhd200_multimodal.phenotype_xgb import (
    _logit,
    _sigmoid,
    compute_metrics,
    fit_logistic,
    load_dataset,
)


def test_load_dataset_drops_pending_and_maps_labels(tmp_path) -> None:
    path = tmp_path / "phenotypics.tsv"
    path.write_text(
        "ScanDir ID\tDX\tAge\tVerbal IQ\n1\t0\t10.5\t100\n2\t3\t9.0\t-999\n3\tpending\t11.0\t95\n"
    )

    features, labels = load_dataset(path, ["Age", "Verbal IQ"])

    assert labels.tolist() == [0, 1]
    assert features.shape == (2, 2)
    assert np.isnan(features.loc[1, "Verbal IQ"])


def test_load_dataset_filters_adhd_measure_and_index(tmp_path) -> None:
    path = tmp_path / "phenotypics.tsv"
    path.write_text(
        "ScanDir ID\tDX\tAge\tADHD Measure\tADHD Index\n"
        "1\t0\t10\t2\t45\n"
        "2\t1\t11\t2\t-999\n"
        "3\t1\t12\t2\tN/A\n"
        "4\t1\t9\t1\t60\n"
        "5\t3\t8\t2\t70\n"
    )

    features, labels = load_dataset(
        path, ["Age"], adhd_measure=2, require_positive_adhd_index=True
    )

    assert features["Age"].tolist() == [10.0, 8.0]
    assert labels.tolist() == [0, 1]


def test_fit_logistic_recovers_perfect_calibration() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.uniform(size=p.size) < p).astype(int)

    intercept, slope = fit_logistic(_logit(p), y, fit_slope=True)

    assert abs(intercept) < 0.1
    assert abs(slope - 1.0) < 0.1
    assert np.allclose(_sigmoid(_logit(p)), p, atol=1e-5)


def test_compute_metrics_perfect_separation() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = compute_metrics(y, p, threshold=0.5, n_bins=2)

    assert metrics["auc"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["f1"] == 1.0
