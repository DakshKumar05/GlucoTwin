"""Import a REAL dataset into GlucoTwin (stored separately from the synthetic demo in data/real/).

Examples (Windows PowerShell):
    python import_data.py --format ohio --input "C:\\data\\OhioT1DM"            # folder of *.xml files
    python import_data.py --format csv  --input "C:\\data\\my_cgm"              # folder with timeseries.csv (+ EHR csvs)
    python import_data.py --format csv  --input export.csv --map glucose_mgdl="Glucose Value (mg/dL)"
    python import_data.py --format csv  --input new_week.csv --append           # add new data, keep the old
    python import_data.py --report                                              # describe the imported dataset

After importing:  python train.py --source real      (or just open the dashboard and pick "real")
Models retrain automatically whenever the imported data changes.
"""
import argparse
import sys

from src import config as C
from src.data_loader import load_from_sqlite, merge_tables, save_to_sqlite
from src.importers import dataset_report, import_csv, import_ohiot1dm


def parse_maps(items):
    out = {}
    for item in items or []:
        if "=" not in item:
            sys.exit(f"--map expects target=source_column, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Import a real dataset into GlucoTwin")
    ap.add_argument("--format", choices=["ohio", "csv"])
    ap.add_argument("--input", help="file or folder to import")
    ap.add_argument("--map", action="append", metavar="TARGET=COLUMN",
                    help="explicit column mapping for csv, e.g. glucose_mgdl='Glucose Value (mg/dL)' (repeatable)")
    ap.add_argument("--append", action="store_true", help="merge with the existing imported dataset")
    ap.add_argument("--report", action="store_true", help="only print a report of the imported dataset")
    args = ap.parse_args()
    db = C.db_path("real")

    if args.report and not args.input:
        if not db.exists():
            sys.exit(f"No imported dataset at {db}")
        print(dataset_report(load_from_sqlite(db)))
        return
    if not (args.format and args.input):
        ap.error("--format and --input are required (or use --report)")

    if args.format == "ohio":
        tables, notes = import_ohiot1dm(args.input)
    else:
        tables, notes = import_csv(args.input, parse_maps(args.map))
    for n in notes:
        print(" -", n)
    if args.append and db.exists():
        tables = merge_tables(load_from_sqlite(db), tables)
        print(" - merged with existing imported data")
    save_to_sqlite(tables, db)
    print(f"\nSaved to {db}\n")
    print(dataset_report(tables))
    print("\nNext: python train.py --source real   (models are retrained from scratch on this data)")


if __name__ == "__main__":
    main()
