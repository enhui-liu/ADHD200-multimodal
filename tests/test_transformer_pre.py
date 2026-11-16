from pathlib import Path

import pandas as pd

from adhd200_multimodal.transformer_pre import prepare_sfnwmrda, select_eligible_subjects


def test_select_eligible_subjects_filters_measure_index_and_site(tmp_path: Path) -> None:
    tsv = tmp_path / "phenotypics.tsv"
    tsv.write_text(
        "ScanDir ID\tSite\tDX\tADHD Measure\tADHD Index\n"
        "1018959\t3\t0\t2\t45\n"
        "10001\t5\t1\t2\t60\n"
        "10002\t5\t1\t2\t-999\n"
        "10003\t5\tpending\t2\t55\n"
        "20001\t1\t1\t2\t50\n"
        "30001\t3\t1\t1\t50\n"
    )

    df = select_eligible_subjects(tsv)

    assert df["ScanDir ID"].tolist() == [1018959, 10001]
    assert df["site_name"].tolist() == ["KKI", "NYU"]
    assert df["label"].tolist() == [0, 1]


def test_prepare_sfnwmrda_copies_matching_files_and_records_missing(tmp_path: Path) -> None:
    tsv = tmp_path / "phenotypics.tsv"
    tsv.write_text(
        "ScanDir ID\tSite\tDX\tADHD Measure\tADHD Index\n"
        "1018959\t3\t0\t2\t45\n"
        "10001\t5\t3\t2\t70\n"
        "10016\t5\t1\t2\t65\n"
    )

    raw_tcs = tmp_path / "raw_tcs"
    kki_sub = raw_tcs / "KKI" / "1018959"
    kki_sub.mkdir(parents=True)
    (kki_sub / "sfnwmrda1018959_session_1_rest_1_cc200_TCs.1D").write_text("kki_sfn")
    (kki_sub / "snwmrda1018959_session_1_rest_1_cc200_TCs.1D").write_text("kki_snw")

    nyu_sub = raw_tcs / "NYU" / "0010001"
    nyu_sub.mkdir(parents=True)
    (nyu_sub / "sfnwmrda0010001_session_1_rest_1_cc200_TCs.1D").write_text("nyu_sfn_1")
    (nyu_sub / "sfnwmrda0010001_session_1_rest_2_cc200_TCs.1D").write_text("nyu_sfn_2")
    (nyu_sub / "snwmrda0010001_session_1_rest_1_cc200_TCs.1D").write_text("nyu_snw_1")

    nyu_missing = raw_tcs / "NYU" / "0010016"
    nyu_missing.mkdir(parents=True)
    (nyu_missing / "s*rest*_cc200_TCs.1D").write_text("placeholder")

    processed = tmp_path / "processed"
    summary = prepare_sfnwmrda(tsv, raw_tcs, processed)

    assert summary["n_eligible"] == 3
    assert summary["n_copied_subjects"] == 2
    assert summary["n_missing_subjects"] == 1
    assert summary["n_copied_files"] == 3
    assert summary["missing_subject_ids"] == [10016]

    kki_out = processed / "KKI" / "1018959"
    nyu_out = processed / "NYU" / "0010001"
    assert (kki_out / "sfnwmrda1018959_session_1_rest_1_cc200_TCs.1D").is_file()
    assert not (kki_out / "snwmrda1018959_session_1_rest_1_cc200_TCs.1D").exists()
    assert (nyu_out / "sfnwmrda0010001_session_1_rest_1_cc200_TCs.1D").is_file()
    assert (nyu_out / "sfnwmrda0010001_session_1_rest_2_cc200_TCs.1D").is_file()
    assert not (processed / "NYU" / "0010016").exists()

    manifest = pd.read_csv(processed / "transformer_pre_manifest.csv")
    assert manifest["status"].tolist() == ["copied", "copied", "missing_sfnwmrda"]
