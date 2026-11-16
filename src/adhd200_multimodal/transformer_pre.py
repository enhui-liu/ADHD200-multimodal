"""Prepare sfnwmrda CC200 time-course files for the Transformer branch."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig, OmegaConf

ID_COLUMN = "ScanDir ID"
DEFAULT_SITE_MAP: dict[int, str] = {3: "KKI", 5: "NYU"}
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "transformer_pre.yaml"


def select_eligible_subjects(
    phenotype_path: str | Path,
    adhd_measure: int | None = 2,
    require_positive_adhd_index: bool = True,
    site_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Filter subjects using the same inclusion criteria as XGBoost."""
    if site_map is None:
        site_map = DEFAULT_SITE_MAP

    raw = pd.read_csv(phenotype_path, sep="\t", na_values=["N/A", "pending"])
    if raw[ID_COLUMN].duplicated().any():
        raise ValueError(f"{ID_COLUMN} must be unique so that splits stay subject-level")

    filtered = raw.dropna(subset=["DX"]).copy()
    if adhd_measure is not None:
        measure = pd.to_numeric(filtered["ADHD Measure"], errors="coerce")
        filtered = filtered[measure == adhd_measure]
    if require_positive_adhd_index:
        filtered = filtered[pd.to_numeric(filtered["ADHD Index"], errors="coerce") > 0]

    filtered["Site"] = pd.to_numeric(filtered["Site"], errors="coerce").astype("Int64")
    if site_map is not None:
        normalized_map = {int(k): str(v) for k, v in site_map.items()}
        filtered = filtered[filtered["Site"].isin(normalized_map)].copy()
        filtered["site_name"] = filtered["Site"].map(normalized_map)
    else:
        filtered["site_name"] = filtered["Site"].astype(str)

    filtered[ID_COLUMN] = pd.to_numeric(filtered[ID_COLUMN], errors="raise").astype(int)
    filtered["label"] = (pd.to_numeric(filtered["DX"], errors="raise") > 0).astype(int)
    return filtered.reset_index(drop=True)


def find_subject_dir(site_dir: Path, subject_id: int) -> Path | None:
    """Locate a subject directory, supporting both unpadded and 7-digit zero-padded IDs."""
    raw_id = str(int(subject_id))
    for candidate_name in (raw_id, raw_id.zfill(7)):
        candidate = site_dir / candidate_name
        if candidate.is_dir():
            return candidate
    return None


def prepare_sfnwmrda(
    phenotype_path: str | Path,
    raw_tcs_dir: str | Path,
    processed_dir: str | Path,
    adhd_measure: int | None = 2,
    require_positive_adhd_index: bool = True,
    site_map: dict[int, str] | None = None,
    file_pattern: str = "sfnwmrda*",
    manifest_filename: str = "transformer_pre_manifest.csv",
) -> dict[str, object]:
    """Copy eligible subjects' sfnwmrda files into processed/<SITE>/<subject_dir>/."""
    if site_map is None:
        site_map = {3: "KKI", 5: "NYU"}

    subjects = select_eligible_subjects(
        phenotype_path=phenotype_path,
        adhd_measure=adhd_measure,
        require_positive_adhd_index=require_positive_adhd_index,
        site_map=site_map,
    )

    raw_root = Path(raw_tcs_dir)
    out_root = Path(processed_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    n_copied_files = 0
    missing_subject_ids: list[int] = []

    for row in subjects.itertuples(index=False):
        subject_id = int(getattr(row, "ScanDir_ID", row[subjects.columns.get_loc(ID_COLUMN)]))
        site_id = int(row.Site)
        site_name = str(row.site_name)
        dx = float(row.DX)
        label = int(row.label)
        adhd_index = float(getattr(row, "ADHD_Index", row[subjects.columns.get_loc("ADHD Index")]))

        subject_dir = find_subject_dir(raw_root / site_name, subject_id)
        if subject_dir is None:
            missing_subject_ids.append(subject_id)
            manifest_rows.append(
                {
                    "subject_id": subject_id,
                    "site_id": site_id,
                    "site_name": site_name,
                    "subject_dir": "",
                    "dx": dx,
                    "label": label,
                    "adhd_index": adhd_index,
                    "n_files": 0,
                    "files": "",
                    "status": "missing_dir",
                }
            )
            continue

        matched_files = sorted(p for p in subject_dir.glob(file_pattern) if p.is_file())
        if not matched_files:
            missing_subject_ids.append(subject_id)
            manifest_rows.append(
                {
                    "subject_id": subject_id,
                    "site_id": site_id,
                    "site_name": site_name,
                    "subject_dir": subject_dir.name,
                    "dx": dx,
                    "label": label,
                    "adhd_index": adhd_index,
                    "n_files": 0,
                    "files": "",
                    "status": "missing_sfnwmrda",
                }
            )
            continue

        target_dir = out_root / site_name / subject_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        copied_rel_paths: list[str] = []
        for src_file in matched_files:
            dst_file = target_dir / src_file.name
            shutil.copy2(src_file, dst_file)
            copied_rel_paths.append(str(Path(site_name) / subject_dir.name / src_file.name))
            n_copied_files += 1

        manifest_rows.append(
            {
                "subject_id": subject_id,
                "site_id": site_id,
                "site_name": site_name,
                "subject_dir": subject_dir.name,
                "dx": dx,
                "label": label,
                "adhd_index": adhd_index,
                "n_files": len(copied_rel_paths),
                "files": ";".join(copied_rel_paths),
                "status": "copied",
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = out_root / manifest_filename
    manifest_df.to_csv(manifest_path, index=False)

    by_site: dict[str, dict[str, int]] = {}
    for site_name in sorted(set(site_map.values())):
        site_rows = manifest_df[manifest_df["site_name"] == site_name]
        copied_rows = site_rows[site_rows["status"] == "copied"]
        by_site[site_name] = {
            "n_eligible": int(len(site_rows)),
            "n_copied_subjects": int(len(copied_rows)),
            "n_missing_subjects": int(len(site_rows) - len(copied_rows)),
            "n_copied_files": int(copied_rows["n_files"].sum()) if len(copied_rows) else 0,
        }

    return {
        "n_eligible": int(len(subjects)),
        "n_copied_subjects": int((manifest_df["status"] == "copied").sum()),
        "n_missing_subjects": int((manifest_df["status"] != "copied").sum()),
        "n_copied_files": int(n_copied_files),
        "by_site": by_site,
        "missing_subject_ids": missing_subject_ids,
        "manifest_path": str(manifest_path),
    }


def run(config: DictConfig) -> dict[str, object]:
    site_map = {int(k): str(v) for k, v in OmegaConf.to_container(config.data.site_map).items()}
    summary = prepare_sfnwmrda(
        phenotype_path=config.data.phenotype_path,
        raw_tcs_dir=config.data.raw_tcs_dir,
        processed_dir=config.data.processed_dir,
        adhd_measure=config.data.adhd_measure,
        require_positive_adhd_index=config.data.require_positive_adhd_index,
        site_map=site_map,
        file_pattern=config.data.file_pattern,
        manifest_filename=config.data.manifest_filename,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    run(OmegaConf.merge(OmegaConf.load(CONFIG_PATH), OmegaConf.from_cli()))


if __name__ == "__main__":
    main()
