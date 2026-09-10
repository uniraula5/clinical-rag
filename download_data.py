"""
Downloads the 12 MedQuAD CSV files into data/raw/.

MedQuAD (https://github.com/abachaa/MedQuAD) is published as XML files.
These CSV versions come from https://github.com/avery-lockwood/MedQuAD-CSVs.
Files that are already downloaded are skipped.

    python download_data.py
"""

import urllib.request
from pathlib import Path

# pinned to one commit of the CSV repo so the data can't change under our results
REPO_COMMIT = "c9ef89909f7e7a3c4deec3aef263cb6b563c8701"
BASE_URL = f"https://raw.githubusercontent.com/avery-lockwood/MedQuAD-CSVs/{REPO_COMMIT}/_data_as_csvs/"

RAW_DIR = Path(__file__).parent / "data" / "raw"

FILES = [
    "1_CancerGov_QA.csv",
    "2_GARD_QA.csv",
    "3_GHR_QA.csv",
    "4_MPlus_Health_Topics_QA.csv",
    "5_NIDDK_QA.csv",
    "6_NINDS_QA.csv",
    "7_SeniorHealth_QA.csv",
    "8_NHLBI_QA_XML.csv",
    "9_CDC_QA.csv",
    "10_MPlus_ADAM_QA.csv",
    "11_MPlusDrugs_QA.csv",
    "12_MPlusHerbsSupplements_QA.csv",
]


def download_all(raw_dir=RAW_DIR, files=FILES):
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        path = raw_dir / name
        if path.exists():
            print(f"  already have {name}")
            continue
        print(f"  downloading {name} ...")
        # save under a temporary name first, so a failed download never
        # leaves a half-written file that looks finished
        temp_path = path.with_suffix(".part")
        urllib.request.urlretrieve(BASE_URL + name, temp_path)
        temp_path.rename(path)
    print(f"Done: {len(list(raw_dir.glob('*.csv')))} CSV files in {raw_dir}")


if __name__ == "__main__":
    download_all()
