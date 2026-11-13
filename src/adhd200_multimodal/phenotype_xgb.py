"""XGBoost on ADHD-200 phenotypes: discrimination, sensitivity, F1 and calibration."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import minimize
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

ID_COLUMN = "ScanDir ID"
MISSING_CODE = -999
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "phenotype_xgboost.yaml"


def load_dataset(
    path: str | Path,
    features: list[str],
    adhd_measure: int | None = None,
    require_positive_adhd_index: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return phenotype features and ADHD labels (DX 1/2/3 -> 1, DX 0 -> 0)."""
    raw = pd.read_csv(path, sep="\t", na_values=["N/A", "pending"])
    if raw[ID_COLUMN].duplicated().any():
        raise ValueError(f"{ID_COLUMN} must be unique so that splits stay subject-level")
    # 尚未公开诊断的受试者(DX=pending)没有标签，直接剔除。
    raw = raw.dropna(subset=["DX"])
    if adhd_measure is not None:
        raw = raw[pd.to_numeric(raw["ADHD Measure"], errors="coerce") == adhd_measure]
    if require_positive_adhd_index:
        # -999 与缺失值都不是有效量表分数。
        raw = raw[pd.to_numeric(raw["ADHD Index"], errors="coerce") > 0]
    labels = (raw["DX"].astype(float) > 0).astype(int).to_numpy()
    features_frame = raw[features].apply(pd.to_numeric, errors="coerce")
    return features_frame.replace(MISSING_CODE, np.nan).reset_index(drop=True), labels


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(z: np.ndarray, y: np.ndarray, fit_slope: bool) -> tuple[float, float]:
    """Fit y ~ intercept + slope * z by maximum likelihood; slope is fixed at 1 if not fitted."""

    def negative_log_likelihood(params: np.ndarray) -> float:
        slope = params[1] if fit_slope else 1.0
        eta = params[0] + slope * z
        return float(np.mean(np.logaddexp(0.0, eta) - y * eta))

    result = minimize(negative_log_likelihood, np.array([0.0, 1.0][: 1 + fit_slope]), method="BFGS")
    return float(result.x[0]), float(result.x[1]) if fit_slope else 1.0


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int) -> float:
    """Mean |observed - predicted| over equal-frequency bins, weighted by bin size."""
    chunks = np.array_split(np.argsort(p), n_bins)
    return float(sum(len(c) / len(p) * abs(y[c].mean() - p[c].mean()) for c in chunks))


def choose_threshold(y: np.ndarray, p: np.ndarray, rule: str | float) -> float:
    if not isinstance(rule, str):
        return float(rule)
    if rule == "youden":
        fpr, tpr, thresholds = roc_curve(y, p)
        return float(thresholds[np.argmax(tpr - fpr)])
    if rule == "f1":
        grid = np.linspace(0.01, 0.99, 99)
        return float(grid[np.argmax([f1_score(y, p >= t) for t in grid])])
    raise ValueError(f"Unknown threshold rule: {rule}")


def compute_metrics(
    y: np.ndarray, p: np.ndarray, threshold: float, n_bins: int
) -> dict[str, float]:
    predicted = p >= threshold
    z = _logit(p)
    return {
        "auc": float(roc_auc_score(y, p)),
        "sensitivity": float(predicted[y == 1].mean()),
        "specificity": float((~predicted[y == 0]).mean()),
        "f1": float(f1_score(y, predicted)),
        "brier": float(brier_score_loss(y, p)),
        "calibration_slope": fit_logistic(z, y, fit_slope=True)[1],
        "calibration_intercept": fit_logistic(z, y, fit_slope=False)[0],
        "ece": expected_calibration_error(y, p, n_bins),
    }


def bootstrap_ci(
    y: np.ndarray, p: np.ndarray, threshold: float, n_bins: int, n_resamples: int, seed: int
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        index = rng.integers(0, len(y), len(y))
        if y[index].min() == y[index].max():
            continue
        draws.append(compute_metrics(y[index], p[index], threshold, n_bins))
    frame = pd.DataFrame(draws)
    return {
        name: [float(frame[name].quantile(0.025)), float(frame[name].quantile(0.975))]
        for name in frame
    }


def plot_calibration(y: np.ndarray, probs: dict[str, np.ndarray], n_bins: int, path: Path) -> None:
    plt.switch_backend("Agg")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for name, p in probs.items():
        observed, predicted = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        ax.plot(predicted, observed, marker="o", label=name)
    ax.set_xlabel("Predicted probability of ADHD")
    ax.set_ylabel("Observed proportion of ADHD")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    # 命令行可用 key=value 覆盖配置，例如 threshold=youden。
    run(OmegaConf.merge(OmegaConf.load(CONFIG_PATH), OmegaConf.from_cli()))


def run(config: DictConfig) -> None:
    features, labels = load_dataset(
        config.data.path,
        list(config.data.features),
        adhd_measure=config.data.adhd_measure,
        require_positive_adhd_index=config.data.require_positive_adhd_index,
    )
    model_params = OmegaConf.to_container(config.model)

    def new_model() -> XGBClassifier:
        return XGBClassifier(**model_params, random_state=config.seed, eval_metric="logloss")

    x_dev, x_test, y_dev, y_test = train_test_split(
        features,
        labels,
        test_size=config.split.test_fraction,
        stratify=labels,
        random_state=config.seed,
    )
    x_dev = x_dev.reset_index(drop=True)

    # 开发集的折外预测同时用于确定阈值与重校准参数，测试集只评估一次。
    oof = np.zeros(len(y_dev))
    folds = StratifiedKFold(config.cv.n_splits, shuffle=True, random_state=config.seed)
    for train_index, valid_index in folds.split(x_dev, y_dev):
        fold_model = new_model().fit(x_dev.iloc[train_index], y_dev[train_index])
        oof[valid_index] = fold_model.predict_proba(x_dev.iloc[valid_index])[:, 1]
    test_raw = new_model().fit(x_dev, y_dev).predict_proba(x_test)[:, 1]

    intercept, slope = fit_logistic(_logit(oof), y_dev, fit_slope=True)
    variants = {
        "uncalibrated": (oof, test_raw),
        "recalibrated": (
            _sigmoid(intercept + slope * _logit(oof)),
            _sigmoid(intercept + slope * _logit(test_raw)),
        ),
    }

    bins = config.calibration_bins
    results: dict[str, dict] = {
        "n_development": len(y_dev),
        "n_test": len(y_test),
        "adhd_prevalence_test": float(y_test.mean()),
    }
    for name, (p_dev, p_test) in variants.items():
        threshold = choose_threshold(y_dev, p_dev, config.threshold)
        results[name] = {
            "threshold": threshold,
            "development_cv": compute_metrics(y_dev, p_dev, threshold, bins),
            "test": compute_metrics(y_test, p_test, threshold, bins),
            "test_ci95": bootstrap_ci(
                y_test, p_test, threshold, bins, config.bootstrap.n_resamples, config.seed
            ),
        }

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    plot_calibration(
        y_test,
        {name: p_test for name, (_, p_test) in variants.items()},
        bins,
        output_dir / "calibration_curve.png",
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
