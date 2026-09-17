from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "benchmark_staging" / "output"

CHECKS = {
    "NHANES_statbench.csv": {
        "key": "SEQN",
        "critical": ["BMXBMI", "BMXHT", "BMXWT", "LBXGLU", "LBXTC"],
    },
    "METABRIC_statbench.csv": {
        "key": "SAMPLE_ID",
        "critical": [
            "TUMOR_SIZE", "GRADE", "CLAUDIN_SUBTYPE",
            "LYMPH_NODES_EXAMINED_POSITIVE", "EXPR_ESR1",
        ],
    },
    "MIMIC_IV_demo_statbench.csv": {
        "key": "stay_id",
        "critical": [
            "gender", "age_at_admission", "los", "heart_rate_bpm_mean",
            "sbp_mmhg_mean", "creatinine_mg_dl_mean", "glucose_mg_dl_mean",
        ],
    },
    "Inside_Airbnb_NYC_statbench.csv": {
        "key": "id",
        "critical": [
            "price", "room_type", "number_of_reviews",
            "availability_365", "minimum_nights",
        ],
    },
    "Bike_Sharing_statbench.csv": {
        "key": "datetime",
        "critical": ["temp", "humidity", "windspeed", "cnt"],
    },
}


def main() -> None:
    summary = {}
    for filename, specification in CHECKS.items():
        frame = pd.read_csv(OUT / filename, low_memory=False)
        key = specification["key"]
        critical = specification["critical"]
        absent = [column for column in critical if column not in frame.columns]
        if absent:
            raise AssertionError(f"{filename} missing critical columns: {absent}")
        if frame[key].isna().any() or not frame[key].is_unique:
            raise AssertionError(f"{filename} key is not unique/non-null: {key}")
        summary[filename] = {
            "rows": len(frame),
            "columns": len(frame.columns),
            "key": key,
            "key_unique": True,
            "critical_nonmissing": {
                column: int(frame[column].notna().sum()) for column in critical
            },
        }

    airbnb = pd.read_csv(OUT / "Inside_Airbnb_NYC_statbench.csv")
    summary["Inside_Airbnb_NYC_statbench.csv"]["price_skew"] = float(
        airbnb["price"].skew()
    )
    summary["Inside_Airbnb_NYC_statbench.csv"]["room_type_counts"] = (
        airbnb["room_type"].value_counts().to_dict()
    )

    metabric = pd.read_csv(OUT / "METABRIC_statbench.csv", low_memory=False)
    summary["METABRIC_statbench.csv"]["subtype_counts"] = (
        metabric["CLAUDIN_SUBTYPE"].value_counts(dropna=False).to_dict()
    )

    mimic = pd.read_csv(OUT / "MIMIC_IV_demo_statbench.csv", low_memory=False)
    summary["MIMIC_IV_demo_statbench.csv"]["gender_counts"] = (
        mimic["gender"].value_counts(dropna=False).to_dict()
    )
    summary["MIMIC_IV_demo_statbench.csv"]["hypertension_dx_counts"] = (
        mimic["hypertension_dx"].value_counts(dropna=False).to_dict()
    )

    (OUT / "qa_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
