from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "benchmark_staging" / "raw"
OUT = ROOT / "benchmark_staging" / "output"
OUT.mkdir(parents=True, exist_ok=True)


def save_csv(frame: pd.DataFrame, filename: str) -> Path:
    path = OUT / filename
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    print(f"WROTE {filename}: rows={len(frame):,}, cols={len(frame.columns):,}")
    return path


def build_nhanes() -> Path:
    demo = pd.read_sas(RAW / "DEMO_J.XPT", format="xport")
    bmx = pd.read_sas(RAW / "BMX_J.XPT", format="xport")
    glu = pd.read_sas(RAW / "GLU_J.XPT", format="xport")
    tchol = pd.read_sas(RAW / "TCHOL_J.XPT", format="xport")

    demo_columns = [
        "SEQN", "SDDSRVYR", "RIDSTATR", "RIAGENDR", "RIDAGEYR", "RIDRETH1",
        "RIDRETH3", "RIDEXMON", "DMQMILIZ", "DMQADFC", "DMDBORN4", "DMDCITZN",
        "DMDYRSUS", "DMDEDUC3", "DMDEDUC2", "DMDMARTL", "RIDEXPRG", "DMDHHSIZ",
        "DMDFMSIZ", "DMDHHSZA", "DMDHHSZB", "DMDHHSZE", "DMDHRGND", "DMDHRAGZ",
        "DMDHREDZ", "DMDHRMAZ", "WTINT2YR", "WTMEC2YR", "SDMVPSU", "SDMVSTRA",
        "INDHHIN2", "INDFMIN2", "INDFMPIR",
    ]
    bmx_columns = [
        "SEQN", "BMDSTATS", "BMXWT", "BMIWT", "BMXHT", "BMIHT", "BMXBMI",
        "BMXLEG", "BMXARML", "BMXARMC", "BMXWAIST", "BMXHIP",
    ]
    glu_columns = ["SEQN", "WTSAF2YR", "LBXGLU", "LBDGLUSI"]
    tchol_columns = ["SEQN", "LBXTC", "LBDTCSI"]

    result = (
        bmx[bmx_columns]
        .merge(demo[demo_columns], on="SEQN", how="left", validate="one_to_one")
        .merge(glu[glu_columns], on="SEQN", how="left", validate="one_to_one")
        .merge(tchol[tchol_columns], on="SEQN", how="left", validate="one_to_one")
    )
    ordered = (
        ["SEQN"]
        + [column for column in demo_columns if column != "SEQN"]
        + [column for column in bmx_columns if column != "SEQN"]
        + [column for column in glu_columns if column != "SEQN"]
        + [column for column in tchol_columns if column != "SEQN"]
    )
    result = result[ordered].sort_values("SEQN").reset_index(drop=True)
    assert result.shape == (8704, 49), result.shape
    return save_csv(result, "NHANES_statbench.csv")


def read_metabric_clinical(
    archive: tarfile.TarFile, member_name: str
) -> pd.DataFrame:
    member = archive.extractfile(member_name)
    if member is None:
        raise FileNotFoundError(member_name)
    return pd.read_csv(member, sep="\t", comment="#", low_memory=False)


def build_metabric() -> Path:
    genes = [
        "ESR1", "PGR", "ERBB2", "MKI67", "BRCA1", "BRCA2",
        "TP53", "EGFR", "GATA3", "FOXA1", "CCND1", "MYC",
    ]
    expression_member = "brca_metabric/data_mrna_illumina_microarray.txt"
    with tarfile.open(RAW / "brca_metabric.tar.gz", "r:gz") as archive:
        patient = read_metabric_clinical(
            archive, "brca_metabric/data_clinical_patient.txt"
        )
        sample = read_metabric_clinical(
            archive, "brca_metabric/data_clinical_sample.txt"
        )
        member = archive.extractfile(expression_member)
        if member is None:
            raise FileNotFoundError(expression_member)
        header = member.readline().decode("utf-8").rstrip("\r\n").split("\t")
        sample_ids = header[2:]
        selected: dict[str, list[float]] = {}
        wanted = set(genes)
        for raw_line in member:
            first = raw_line.find(b"\t")
            if first < 0:
                continue
            symbol = raw_line[:first].decode("utf-8", errors="replace")
            if symbol not in wanted or symbol in selected:
                continue
            second = raw_line.find(b"\t", first + 1)
            values = raw_line[second + 1 :].decode("utf-8").rstrip("\r\n").split("\t")
            selected[symbol] = [
                float(value) if value not in {"", "NA", "NaN"} else np.nan
                for value in values
            ]
            if len(selected) == len(wanted):
                break

    missing_genes = [gene for gene in genes if gene not in selected]
    if missing_genes:
        raise RuntimeError(f"Missing METABRIC genes: {missing_genes}")

    expression = pd.DataFrame({"SAMPLE_ID": sample_ids})
    for gene in genes:
        expression[f"EXPR_{gene}"] = selected[gene]

    clinical = sample.merge(
        patient, on="PATIENT_ID", how="left", validate="many_to_one"
    )
    result = clinical.merge(
        expression, on="SAMPLE_ID", how="inner", validate="one_to_one"
    )
    numeric_columns = [
        "AGE_AT_DIAGNOSIS", "TUMOR_SIZE", "GRADE",
        "LYMPH_NODES_EXAMINED_POSITIVE", "NPI", "OS_MONTHS", "RFS_MONTHS",
        "TUMOR_STAGE", "TMB_NONSYNONYMOUS",
    ]
    for column in numeric_columns:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    result = result.sort_values("SAMPLE_ID").reset_index(drop=True)
    assert 1800 <= len(result) <= 2100, result.shape
    assert 40 <= len(result.columns) <= 50, result.shape
    return save_csv(result, "METABRIC_statbench.csv")


def build_airbnb() -> Path:
    source = pd.read_csv(RAW / "inside_airbnb_nyc_2019_mirror.csv")
    last_review = pd.to_datetime(source["last_review"], errors="coerce")
    reference_date = last_review.max()

    result = source.copy()
    result["price_log1p"] = np.log1p(result["price"].clip(lower=0))
    result["minimum_nights_log1p"] = np.log1p(
        result["minimum_nights"].clip(lower=0)
    )
    result["number_of_reviews_log1p"] = np.log1p(
        result["number_of_reviews"].clip(lower=0)
    )
    result["availability_rate_365"] = result["availability_365"] / 365.0
    result["reviews_per_year_est"] = result["reviews_per_month"] * 12.0
    result["has_reviews"] = (result["number_of_reviews"] > 0).astype("int8")
    result["days_since_last_review"] = (reference_date - last_review).dt.days
    result["last_review_year"] = last_review.dt.year.astype("Int64")
    result["last_review_month"] = last_review.dt.month.astype("Int64")
    result["is_entire_home"] = (
        result["room_type"] == "Entire home/apt"
    ).astype("int8")
    result["is_private_room"] = (
        result["room_type"] == "Private room"
    ).astype("int8")
    result["is_shared_room"] = (
        result["room_type"] == "Shared room"
    ).astype("int8")
    result["host_listing_count_log1p"] = np.log1p(
        result["calculated_host_listings_count"].clip(lower=0)
    )
    denominator = result["minimum_nights"].replace(0, np.nan)
    result["price_per_minimum_night"] = result["price"] / denominator
    result["last_review"] = last_review.dt.strftime("%Y-%m-%d")

    result = (
        result.sample(n=10000, random_state=20260723, replace=False)
        .sort_values("id")
        .reset_index(drop=True)
    )
    assert result.shape == (10000, 30), result.shape
    return save_csv(result, "Inside_Airbnb_NYC_statbench.csv")


def build_bike() -> Path:
    with zipfile.ZipFile(RAW / "bike_sharing_dataset.zip") as archive:
        with archive.open("hour.csv") as handle:
            source = pd.read_csv(handle)
    source["datetime"] = pd.to_datetime(source["dteday"]) + pd.to_timedelta(
        source["hr"], unit="h"
    )
    result = source.drop(columns=["instant"]).rename(columns={"hum": "humidity"})
    result["datetime"] = result["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    result = result[
        ["datetime"]
        + [column for column in result.columns if column != "datetime"]
    ]
    assert result.shape == (17379, 17), result.shape
    return save_csv(result, "Bike_Sharing_statbench.csv")


def read_gzip_csv(filename: str, **kwargs) -> pd.DataFrame:
    with gzip.open(RAW / filename, "rt", encoding="utf-8") as handle:
        return pd.read_csv(handle, **kwargs)


def aggregate_metric(
    frame: pd.DataFrame,
    group_key: str,
    metric_col: str,
    value_col: str,
    time_col: str,
) -> pd.DataFrame:
    filtered = frame.dropna(subset=[group_key, metric_col, value_col]).copy()
    filtered = filtered.sort_values([group_key, metric_col, time_col])
    grouped = filtered.groupby([group_key, metric_col], sort=False)[value_col]
    aggregated = grouped.agg(["first", "mean", "min", "max"]).unstack(metric_col)
    aggregated.columns = [
        f"{metric}_{stat}" for stat, metric in aggregated.columns.to_flat_index()
    ]
    return aggregated.reset_index()


def build_mimic_demo() -> Path:
    patients = read_gzip_csv("mimic_patients.csv.gz")
    admissions = read_gzip_csv(
        "mimic_admissions.csv.gz",
        parse_dates=["admittime", "dischtime", "deathtime", "edregtime", "edouttime"],
    )
    icustays = read_gzip_csv(
        "mimic_icustays.csv.gz", parse_dates=["intime", "outtime"]
    )
    diagnoses = read_gzip_csv("mimic_diagnoses_icd.csv.gz")
    d_items = read_gzip_csv("mimic_d_items.csv.gz")
    d_labitems = read_gzip_csv("mimic_d_labitems.csv.gz")
    chart = read_gzip_csv(
        "mimic_chartevents.csv.gz", parse_dates=["charttime"], low_memory=False
    )
    labs = read_gzip_csv(
        "mimic_labevents.csv.gz", parse_dates=["charttime"], low_memory=False
    )

    item_labels = d_items.set_index("itemid")["label"].fillna("").astype(str)
    chart = chart[chart["valuenum"].notna()].copy()
    chart["label"] = chart["itemid"].map(item_labels).fillna("")
    lower = chart["label"].str.lower()
    chart["metric"] = pd.NA
    chart.loc[lower.eq("heart rate"), "metric"] = "heart_rate_bpm"
    chart.loc[
        lower.str.contains("blood pressure", regex=False)
        & lower.str.contains("systolic", regex=False),
        "metric",
    ] = "sbp_mmhg"
    chart.loc[
        lower.str.contains("blood pressure", regex=False)
        & lower.str.contains("diastolic", regex=False),
        "metric",
    ] = "dbp_mmhg"
    chart.loc[
        lower.str.contains("blood pressure", regex=False)
        & lower.str.contains("mean", regex=False),
        "metric",
    ] = "map_mmhg"
    chart.loc[
        lower.isin(["respiratory rate", "respiratory rate (total)"]),
        "metric",
    ] = "resp_rate_bpm"
    chart.loc[
        lower.str.contains("o2 saturation pulseoxymetry", regex=False),
        "metric",
    ] = "spo2_percent"
    chart.loc[
        lower.str.contains("temperature celsius", regex=False),
        "metric",
    ] = "temperature_c"
    fahrenheit = lower.str.contains("temperature fahrenheit", regex=False)
    chart.loc[fahrenheit, "metric"] = "temperature_c"
    chart.loc[fahrenheit, "valuenum"] = (
        chart.loc[fahrenheit, "valuenum"] - 32.0
    ) * (5.0 / 9.0)

    vitals = aggregate_metric(
        chart, "stay_id", "metric", "valuenum", "charttime"
    )

    lab_labels = d_labitems.set_index("itemid")["label"].fillna("").astype(str)
    labs = labs[labs["valuenum"].notna()].copy()
    labs["label"] = labs["itemid"].map(lab_labels).fillna("")
    lab_map = {
        "Creatinine": "creatinine_mg_dl",
        "Glucose": "glucose_mg_dl",
        "Hemoglobin": "hemoglobin_g_dl",
        "White Blood Cells": "wbc_k_ul",
        "Platelet Count": "platelet_k_ul",
        "Sodium": "sodium_meq_l",
        "Potassium": "potassium_meq_l",
        "Urea Nitrogen": "bun_mg_dl",
    }
    labs["metric"] = labs["label"].map(lab_map)
    lab_summary = aggregate_metric(
        labs, "hadm_id", "metric", "valuenum", "charttime"
    )

    admissions["hospital_los_days"] = (
        admissions["dischtime"] - admissions["admittime"]
    ).dt.total_seconds() / 86400.0
    patients["dod"] = pd.to_datetime(patients["dod"], errors="coerce")
    result = (
        icustays.merge(
            admissions, on=["subject_id", "hadm_id"], how="left", validate="many_to_one"
        )
        .merge(patients, on="subject_id", how="left", validate="many_to_one")
        .merge(vitals, on="stay_id", how="left", validate="one_to_one")
        .merge(lab_summary, on="hadm_id", how="left", validate="many_to_one")
    )
    result["age_at_admission"] = (
        result["anchor_age"]
        + result["admittime"].dt.year
        - result["anchor_year"]
    )

    diagnoses["icd_code"] = diagnoses["icd_code"].astype(str)
    hypertension = (
        ((diagnoses["icd_version"] == 9) & diagnoses["icd_code"].str.startswith("401"))
        | (
            (diagnoses["icd_version"] == 10)
            & diagnoses["icd_code"].str.startswith(("I10", "I11", "I12", "I13", "I15"))
        )
    )
    hypertension_by_admission = (
        diagnoses.assign(hypertension_dx=hypertension.astype("int8"))
        .groupby("hadm_id", as_index=False)["hypertension_dx"]
        .max()
    )
    diagnosis_count = (
        diagnoses.groupby("hadm_id", as_index=False)
        .size()
        .rename(columns={"size": "diagnosis_count"})
    )
    primary_diagnosis = (
        diagnoses.sort_values(["hadm_id", "seq_num"])
        .drop_duplicates("hadm_id")
        [["hadm_id", "icd_code", "icd_version"]]
        .rename(
            columns={
                "icd_code": "primary_icd_code",
                "icd_version": "primary_icd_version",
            }
        )
    )
    result = (
        result.merge(hypertension_by_admission, on="hadm_id", how="left")
        .merge(diagnosis_count, on="hadm_id", how="left")
        .merge(primary_diagnosis, on="hadm_id", how="left")
    )
    result["hypertension_dx"] = result["hypertension_dx"].fillna(0).astype("int8")

    date_columns = result.select_dtypes(include=["datetime64[ns]"]).columns
    for column in date_columns:
        result[column] = result[column].dt.strftime("%Y-%m-%d %H:%M:%S")

    result = result.sort_values("stay_id").reset_index(drop=True)
    assert len(result) == 140, result.shape
    assert 70 <= len(result.columns) <= 110, result.shape
    return save_csv(result, "MIMIC_IV_demo_statbench.csv")


def write_manifest(paths: list[Path]) -> None:
    manifest = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.append(
            {
                "filename": path.name,
                "rows": len(frame),
                "columns": len(frame.columns),
                "bytes": path.stat().st_size,
                "sha256": digest,
                "duplicate_column_names": bool(frame.columns.duplicated().any()),
                "all_empty_columns": [
                    column for column in frame.columns if frame[column].isna().all()
                ],
            }
        )
    (OUT / "qa_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nQA MANIFEST")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    paths = [
        build_nhanes(),
        build_metabric(),
        build_mimic_demo(),
        build_airbnb(),
        build_bike(),
    ]
    write_manifest(paths)


if __name__ == "__main__":
    main()
