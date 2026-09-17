from __future__ import annotations

import re
import shutil
from pathlib import Path
from urllib.parse import urljoin
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "benchmark_staging" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "DEMO_J.XPT": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.XPT",
    "BMX_J.XPT": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BMX_J.XPT",
    "GLU_J.XPT": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/GLU_J.XPT",
    "TCHOL_J.XPT": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/TCHOL_J.XPT",
    "brca_metabric.tar.gz": "https://datahub.assets.cbioportal.org/brca_metabric.tar.gz",
    "bike_sharing_dataset.zip": "https://archive.ics.uci.edu/static/public/275/bike%2Bsharing%2Bdataset.zip",
    "mimic_admissions.csv.gz": "https://physionet.org/files/mimic-iv-demo/2.2/hosp/admissions.csv.gz",
    "mimic_d_labitems.csv.gz": "https://physionet.org/files/mimic-iv-demo/2.2/hosp/d_labitems.csv.gz",
    "mimic_diagnoses_icd.csv.gz": "https://physionet.org/files/mimic-iv-demo/2.2/hosp/diagnoses_icd.csv.gz",
    "mimic_labevents.csv.gz": "https://physionet.org/files/mimic-iv-demo/2.2/hosp/labevents.csv.gz",
    "mimic_patients.csv.gz": "https://physionet.org/files/mimic-iv-demo/2.2/hosp/patients.csv.gz",
    "mimic_chartevents.csv.gz": "https://physionet.org/files/mimic-iv-demo/2.2/icu/chartevents.csv.gz",
    "mimic_d_items.csv.gz": "https://physionet.org/files/mimic-iv-demo/2.2/icu/d_items.csv.gz",
    "mimic_icustays.csv.gz": "https://physionet.org/files/mimic-iv-demo/2.2/icu/icustays.csv.gz",
}


def download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        print(f"SKIP {destination.name} ({destination.stat().st_size:,} bytes)")
        return
    temp = destination.with_suffix(destination.suffix + ".part")
    existing = temp.stat().st_size if temp.exists() else 0
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0 Safari/537.36"
        ),
        "Accept": "text/csv,application/gzip,application/octet-stream,*/*",
        "Referer": "https://beta.insideairbnb.com/",
    }
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=180) as response:
        resumed = existing > 0 and getattr(response, "status", None) == 206
        mode = "ab" if resumed else "wb"
        if existing and not resumed:
            existing = 0
        with temp.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    temp.replace(destination)
    print(f"OK   {destination.name} ({destination.stat().st_size:,} bytes)")


def find_nyc_airbnb_url() -> str:
    page_url = "https://beta.insideairbnb.com/get-the-data/"
    request = Request(page_url, headers={"User-Agent": "Mozilla/5.0 CodexDatasetBuilder/1.0"})
    with urlopen(request, timeout=120) as response:
        html = response.read().decode("utf-8", errors="replace")
    city_pos = html.find("New York City")
    if city_pos >= 0:
        local = html[city_pos : city_pos + 6000]
        match = re.search(
            r'href=["\']([^"\']*listings\.csv\.gz[^"\']*)["\']',
            local,
            flags=re.IGNORECASE,
        )
        if match:
            return urljoin(page_url, match.group(1))
    match = re.search(
        r'href=["\']([^"\']+/united-states/ny/new-york-city/[^"\']+/data/listings\.csv\.gz)["\']',
        html,
        flags=re.IGNORECASE,
    )
    if match:
        return urljoin(page_url, match.group(1))
    raise RuntimeError("Could not locate the New York City listings.csv.gz link.")


def main() -> None:
    print(f"Staging directory: {RAW}")
    for filename, url in SOURCES.items():
        download(url, RAW / filename)

    airbnb_url = find_nyc_airbnb_url()
    try:
        download(airbnb_url, RAW / "inside_airbnb_nyc_listings.csv.gz")
        source_note = airbnb_url
    except HTTPError as error:
        if error.code != 403:
            raise
        mirror_url = (
            "https://huggingface.co/datasets/gradio/NYC-Airbnb-Open-Data/"
            "resolve/main/AB_NYC_2019.csv?download=true"
        )
        download(mirror_url, RAW / "inside_airbnb_nyc_2019_mirror.csv")
        source_note = (
            f"Official file blocked with HTTP 403: {airbnb_url}\n"
            f"Mirror used: {mirror_url}"
        )
    (RAW / "inside_airbnb_nyc_source_url.txt").write_text(
        source_note + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
