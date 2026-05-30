# STEP 2 — PREPROCESSING
# Fix types → fill gaps → drop duplicates → cap outliers → encode categories.

import pandas as pd
import numpy as np
import os

IN  = "outputs/01_loaded_data.csv"
OUT = "outputs/02_preprocessed_data.csv"

NUM_COLS = ["Year","Month","Estimated_Deliveries","Production_Units",
            "Battery_Capacity_kWh","Range_km","Charging_Stations","Avg_Price_USD","CO2_Saved_tons"]
CAT_COLS = ["Region", "Model", "Source_Type"]
CLIP_COLS = ["Estimated_Deliveries","Production_Units","Avg_Price_USD",
             "CO2_Saved_tons","Range_km","Charging_Stations"]

def main():
    os.makedirs("outputs", exist_ok=True)
    print("\n=== STEP 2: PREPROCESSING ===")
    df = pd.read_csv(IN)

    # 1. Fix data types
    for c in NUM_COLS: df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CAT_COLS: df[c] = df[c].astype(str).str.strip()
    print("Types fixed.")

    # 2. Fill missing values (median for numbers, mode for text)
    for c in df.select_dtypes("number"):
        n = df[c].isnull().sum()
        if n: df[c].fillna(df[c].median(), inplace=True); print(f"  Filled {n} NaN in '{c}' → median")
    for c in CAT_COLS:
        n = df[c].isnull().sum()
        if n: df[c].fillna(df[c].mode()[0], inplace=True); print(f"  Filled {n} NaN in '{c}' → mode")
    if df.isnull().sum().sum() == 0: print("No missing values found.")

    # 3. Drop duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Duplicates removed: {before - len(df)}")

    # 4. Cap outliers with IQR fences (clip, don't delete)
    for c in CLIP_COLS:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        lo, hi = q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1)
        n = ((df[c] < lo) | (df[c] > hi)).sum()
        df[c] = df[c].clip(lo, hi)
        if n: print(f"  Capped {n} outliers in '{c}'")
    print("Outlier capping done.")

    # 5. Label-encode categories (keep original columns too)
    for c in CAT_COLS:
        mapping = {v: i for i, v in enumerate(sorted(df[c].unique()))}
        df[c + "_Code"] = df[c].map(mapping)
        print(f"  Encoded '{c}' → '{c}_Code': {mapping}")

    print(f"Final: {df.shape[0]:,} rows × {df.shape[1]} cols")
    df.to_csv(OUT, index=False)
    print(f"Saved → {OUT}\n")
    return df

if __name__ == "__main__":
    main()