"""Command-line entry point: load data, train all models from scratch, print evaluation.

Usage:
    python train.py                          # synthetic demo data (generated if missing)
    python train.py --regenerate             # regenerate the synthetic cohort first
    python train.py --patients 20 --days 10 --regenerate
    python train.py --source real            # data imported with import_data.py
    python train.py --source real --split time_only
    python train.py --export-csv             # also write every table to data/csv/
"""
import argparse

import pandas as pd

from src import config as C
from src.data_loader import ensure_data, export_csv
from src.pipeline import build_dataset, train_all


def main() -> None:
    ap = argparse.ArgumentParser(description="Train and evaluate GlucoTwin")
    ap.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--split", choices=["auto", "patient_time", "time_only"], default=C.SPLIT_MODE,
                    help="auto = held-out patients if >= %d patients, otherwise within-patient time split"
                         % C.MIN_PATIENTS_FOR_PATIENT_SPLIT)
    ap.add_argument("--regenerate", action="store_true", help="synthetic only: regenerate the cohort")
    ap.add_argument("--patients", type=int, default=C.N_PATIENTS)
    ap.add_argument("--days", type=int, default=C.N_DAYS)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--export-csv", action="store_true")
    args = ap.parse_args()

    tables = ensure_data(regenerate=args.regenerate and args.source == "synthetic", n_patients=args.patients,
                         n_days=args.days, seed=args.seed, source=args.source)
    print(f"Loaded {args.source} data: {len(tables['patients'])} patients, {len(tables['timeseries']):,} time-series rows")
    if args.export_csv:
        print("CSV written to", C.CSV_DIR, [p.name for p in export_csv(tables)])

    dataset = build_dataset(tables)
    for issue in dataset["issues"]:
        print("  data issue:", issue)
    art = train_all(tables, dataset=dataset, source=args.source, split_mode=args.split,
                    progress=lambda f, m: print(f"  [{f:5.0%}] {m:<70}", end="\r", flush=True))
    print()
    ev = art["evaluation"]
    pd.set_option("display.width", 200, "display.max_columns", 30)
    s = ev["split"]
    print(f"\nSplit ({s['mode']}): {s['description']}")
    print(f"{s['n_test_rows']:,} test rows from {s['n_test_patients']} patients")

    fm = ev["forecast_metrics"]
    h = C.HORIZON_MINUTES
    print(f"\nForecast metrics at {h} min (test rows only):")
    cols = ["config_label", "family_label", "selected", "MAE", "RMSE", "calib_MAE", "interval_coverage", "n"]
    print(fm[fm.horizon_min == h][[c for c in cols if c in fm]].round(2).to_string(index=False))
    print("\nAblation (MAE difference, mg/dL; negative = better; 95% block-bootstrap CI):")
    print(ev["ablation"].round(2).to_string(index=False))
    print("\nRisk classification (test):")
    rcols = ["event", "config", "status", "positives", "base_rate", "precision", "recall", "F1", "PR_AUC", "Brier", "ECE"]
    rm = ev["risk_metrics"]
    print(rm[[c for c in rcols if c in rm]].round(3).to_string(index=False))
    print(f"\nArtifacts saved to {C.artifact_path(args.source)} "
          f"({art['meta']['train_seconds']} s, backend {art['meta']['gbm_backend']})")
    if args.source == "synthetic":
        print("Reminder: metrics on synthetic data do not establish real-world clinical performance.")
    else:
        print("Reminder: retrospective metrics on a small dataset are not clinical validation.")


if __name__ == "__main__":
    main()
