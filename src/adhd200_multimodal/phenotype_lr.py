"""Logistic regression on ADHD-200 phenotypes, same cohort and CV folds as the XGBoost baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from adhd200_multimodal.phenotype_xgb import (
    bootstrap_ci,
    choose_threshold,
    compute_metrics,
    load_dataset,
    plot_calibration,
)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "phenotype_lr.yaml"


def main() -> None:
    # 命令行可用 key=value 覆盖配置，例如 model.C=0.1。
    run(OmegaConf.merge(OmegaConf.load(CONFIG_PATH), OmegaConf.from_cli()))


def run(config: DictConfig) -> None:
    features, labels = load_dataset(
        config.data.path,
        list(config.data.features),
        adhd_measure=config.data.adhd_measure,
        require_positive_adhd_index=config.data.require_positive_adhd_index,
    )
    model_params = OmegaConf.to_container(config.model)
    # 全部缺失的列无法填补，与标签无关，直接剔除并记录。
    all_missing = [name for name in features.columns if features[name].isna().all()]
    features = features.drop(columns=all_missing)

    def new_model():
        # 填补与标准化放进流水线，每折只用训练折拟合，避免泄漏。
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(**model_params, random_state=config.seed),
        )

    # 与 XGBoost 使用相同的种子和标签，折划分一致，结果可直接比较。
    oof = np.zeros(len(labels))
    folds = StratifiedKFold(config.cv.n_splits, shuffle=True, random_state=config.seed)
    for train_index, valid_index in folds.split(features, labels):
        fold_model = new_model().fit(features.iloc[train_index], labels[train_index])
        oof[valid_index] = fold_model.predict_proba(features.iloc[valid_index])[:, 1]

    bins = config.calibration_bins
    # 阈值在同一批折外预测上选取，敏感度/特异度/F1 因此略偏乐观。
    threshold = choose_threshold(labels, oof, config.threshold)
    results = {
        "n_subjects": len(labels),
        "n_adhd": int(labels.sum()),
        "n_control": int((labels == 0).sum()),
        "dropped_all_missing_features": all_missing,
        "threshold": threshold,
        "cv_metrics": compute_metrics(labels, oof, threshold, bins),
        "cv_ci95": bootstrap_ci(
            labels, oof, threshold, bins, config.bootstrap.n_resamples, config.seed
        ),
    }

    # 在全部数据上拟合一次，仅用于报告标准化系数（缺失指示列名带 missingindicator_ 前缀）。
    final = new_model().fit(features, labels)
    coefficients = final[-1].coef_[0]
    names = final[0].get_feature_names_out(list(features.columns))
    results["coefficients_standardized"] = dict(zip(names, map(float, coefficients), strict=True))

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    plot_calibration(labels, {"cross-validated": oof}, bins, output_dir / "calibration_curve.png")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
