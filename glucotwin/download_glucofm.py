"""Download the GlucoFM-Bench CGM benchmark from Hugging Face into data/raw/glucofm/ (train.parquet, test.parquet).

Examples (Windows PowerShell):
    python download_glucofm.py            # download once; later runs reuse the local files
    python download_glucofm.py --force    # download again and overwrite

Licence: CC BY-NC-SA 4.0 (non-commercial) plus each source cohort's licence. See the "Data" section of README.md.
"""
import argparse

import pandas as pd

from src.glucofm import GLUCOFM_DIR, HF_DATASET, SPLITS, glucose_column, load_split, split_path


def download() -> None:
    from datasets import load_dataset  # imported lazily: only needed for the download itself

    ds = load_dataset(HF_DATASET)
    GLUCOFM_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        ds[split].to_parquet(str(split_path(split)))
        print(f"Saved {split_path(split)}")


def summarize(split: str, df: pd.DataFrame) -> None:
    gcol = glucose_column(df)
    n_readings = int(df[gcol].map(len).sum())
    print(f"\n[{split}] {len(df):,} rows (one per subject), {n_readings:,} glucose readings")
    print(f"  columns: {list(df.columns)}  (glucose column: {gcol!r})")
    subjects = df.groupby("dataset")["subject_id"].nunique().sort_index()
    print(f"  subjects per cohort ({subjects.sum():,} total):")
    for cohort, n in subjects.items():
        print(f"    {cohort:<24} {n:>5,}")


def main() -> None:
    ap = argparse.ArgumentParser(description=f"Download {HF_DATASET} to {GLUCOFM_DIR}")
    ap.add_argument("--force", action="store_true", help="download again even if the Parquet files exist")
    args = ap.parse_args()

    if args.force or not all(split_path(s).exists() for s in SPLITS):
        print(f"Downloading {HF_DATASET} ...")
        download()
    else:
        print(f"Found existing files in {GLUCOFM_DIR}; skipping download (use --force to re-download).")

    for split in SPLITS:
        summarize(split, load_split(split))


if __name__ == "__main__":
    main()
