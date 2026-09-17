from __future__ import annotations

import gzip
import tarfile
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "benchmark_staging" / "raw"


def show_frame(label: str, frame: pd.DataFrame) -> None:
    print(f"\n[{label}] shape={frame.shape}")
    print(", ".join(map(str, frame.columns)))


for filename in ["DEMO_J.XPT", "BMX_J.XPT", "GLU_J.XPT", "TCHOL_J.XPT"]:
    show_frame(filename, pd.read_sas(RAW / filename, format="xport"))

with tarfile.open(RAW / "brca_metabric.tar.gz", "r:gz") as archive:
    for member_name in [
        "brca_metabric/data_clinical_patient.txt",
        "brca_metabric/data_clinical_sample.txt",
        "brca_metabric/data_mrna_illumina_microarray.txt",
    ]:
        member = archive.extractfile(member_name)
        assert member is not None
        print(f"\n[{member_name}]")
        for _ in range(6):
            print(member.readline().decode("utf-8", errors="replace").rstrip()[:1200])

with zipfile.ZipFile(RAW / "bike_sharing_dataset.zip") as archive:
    with archive.open("hour.csv") as handle:
        show_frame("Bike hour.csv", pd.read_csv(handle))

show_frame(
    "Inside Airbnb mirror",
    pd.read_csv(RAW / "inside_airbnb_nyc_2019_mirror.csv"),
)

for filename in [
    "mimic_admissions.csv.gz",
    "mimic_d_labitems.csv.gz",
    "mimic_diagnoses_icd.csv.gz",
    "mimic_labevents.csv.gz",
    "mimic_patients.csv.gz",
    "mimic_chartevents.csv.gz",
    "mimic_d_items.csv.gz",
    "mimic_icustays.csv.gz",
]:
    with gzip.open(RAW / filename, "rt", encoding="utf-8") as handle:
        show_frame(filename, pd.read_csv(handle))
