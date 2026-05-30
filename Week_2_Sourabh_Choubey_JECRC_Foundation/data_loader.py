# STEP 1 — DATA LOADER
# Load the raw CSV, print what's inside, run basic checks, save it.

import pandas as pd
import os
import json

RAW  = "tesla_deliveries_dataset_2015_2025.csv"
OUT  = "outputs/01_loaded_data.csv"
RPT  = "outputs/01_report.json"

def main():
    os.makedirs("outputs", exist_ok=True)
    print("\n=== STEP 1: DATA LOADER ===")

    df = pd.read_csv(RAW)
    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Columns : {list(df.columns)}")
    print(f"Years   : {int(df.Year.min())} – {int(df.Year.max())}")
    print(f"Regions : {sorted(df.Region.unique())}")
    print(f"Models  : {sorted(df.Model.unique())}")
    print(f"Missing : {df.isnull().sum().sum()} values")
    print(f"Dupes   : {df.duplicated().sum()} rows")

    # Sanity checks — flag anything suspicious early
    checks = [
        (df.Year.between(2000, 2030).all(), "Year values out of range"),
        (df.Month.between(1, 12).all(), "Month values out of 1-12"),
        ((df.Estimated_Deliveries >= 0).all(), "Negative delivery counts"),
        ((df.Avg_Price_USD > 0).all(), "Zero or negative prices"),
    ]
    all_ok = True
    for passed, msg in checks:
        if not passed:
            print(f"  [WARNING] {msg}")
            all_ok = False
    if all_ok:
        print("  [OK] All checks passed.")

    # Save data + a JSON summary report
    df.to_csv(OUT, index=False)
    report = {
        "shape"   : list(df.shape),
        "missing" : df.isnull().sum().to_dict(),
        "dupes"   : int(df.duplicated().sum()),
        "years"   : [int(df.Year.min()), int(df.Year.max())],
        "stats"   : df.describe().round(2).to_dict(),
    }
    json.dump(report, open(RPT, "w"), indent=2, default=str)
    print(f"Saved → {OUT}  |  {RPT}\n")
    return df

if __name__ == "__main__":
    main()