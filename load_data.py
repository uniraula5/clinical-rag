"""
Loads the MedQuAD CSV files into one pandas DataFrame and prints a
quick profile of the data (rows per source, empty answers, answer lengths).

MedQuAD is a public medical Q&A dataset built from NIH websites:
https://github.com/abachaa/MedQuAD
"""

import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parent / "data" / "raw"
EXPECTED_FILES = 12


def source_from_filename(path):
    # "5_NIDDK_QA.csv" -> "NIDDK", "8_NHLBI_QA_XML.csv" -> "NHLBI",
    # "10_MPlus_ADAM_QA.csv" -> "MPlus_ADAM"
    match = re.match(r"^\d+_(.+?)_QA", path.stem)
    if match is None:
        raise ValueError(f"Unexpected file name: {path.name}")
    return match.group(1)


def load_medquad(raw_dir=RAW_DIR):
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. Download them with: python download_data.py"
        )
    if len(csv_files) < EXPECTED_FILES:
        # a half-finished download would quietly build a half-finished index
        print(f"Warning: only {len(csv_files)} of {EXPECTED_FILES} CSV files are in {raw_dir}. "
              "Run python download_data.py to get the rest.")

    frames = []
    for path in csv_files:
        df = pd.read_csv(path)
        df["source"] = source_from_filename(path)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    # the CSVs were saved with their pandas index as an extra column
    data = data.drop(columns=["Unnamed: 0"], errors="ignore")

    # an answer counts as empty if it is missing or only whitespace
    data["answer"] = data["answer"].fillna("").astype(str).str.strip()
    data["has_answer"] = data["answer"] != ""

    return data


if __name__ == "__main__":
    data = load_medquad()
    print(f"Total rows: {len(data)}\n")

    summary = data.groupby("source").agg(
        rows=("question", "size"),
        with_answer=("has_answer", "sum"),
    )
    summary["empty"] = summary["rows"] - summary["with_answer"]
    print(summary.sort_values("rows", ascending=False))
    print(f"\nRows with an answer: {data['has_answer'].sum()}")
    print(f"Rows with no answer: {(~data['has_answer']).sum()}\n")

    lengths = data.loc[data["has_answer"], "answer"].str.len()
    print("Answer length in characters (non-empty only):")
    print(lengths.describe(percentiles=[0.5, 0.9, 0.99]).round(0))
