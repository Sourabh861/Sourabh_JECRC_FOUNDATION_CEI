# STEP 4 — FEATURE ENGINEERING
# Build new features from existing columns.
# More meaningful inputs → better model predictions.

import pandas as pd
import numpy as np
import os

IN  = "outputs/02_preprocessed_data.csv"
OUT = "outputs/04_feature_engineered.csv"

def main():
    os.makedirs("outputs", exist_ok=True)
    print("\n=== STEP 4: FEATURE ENGINEERING ===")
    df = pd.read_csv(IN)
    print(f"Input: {df.shape[1]} columns")

    # 1. TIME FEATURES — quarter/season matter more than raw month numbers
    df["Quarter"] = ((df.Month - 1) // 3 + 1).astype(int)
    df["Season"] = df.Month.map({12:"Winter",1:"Winter",2:"Winter",
                                            3:"Spring",4:"Spring",5:"Spring",
                                            6:"Summer",7:"Summer",8:"Summer",
                                            9:"Autumn",10:"Autumn",11:"Autumn"})
    df["Is_Quarter_End"] = df.Month.isin([3,6,9,12]).astype(int)  # delivery push months
    df["Is_Year_End"]    = (df.Month == 12).astype(int)
    df["Years_Since_2015"] = df.Year - 2015
    print("Time features added.")

    # 2. RATIO / EFFICIENCY FEATURES — capture business performance in one number
    safe = lambda col: df[col].replace(0, np.nan)
    df["Delivery_Rate"]           = (df.Estimated_Deliveries / safe("Production_Units")).round(4)   # < 1 = backlog
    df["Price_Per_Km"]            = (df.Avg_Price_USD / safe("Range_km")).round(2)
    df["Price_Per_kWh"]           = (df.Avg_Price_USD / safe("Battery_Capacity_kWh")).round(2)
    df["CO2_Per_1k_Deliveries"]   = (df.CO2_Saved_tons / (df.Estimated_Deliveries/1000).replace(0,np.nan)).round(4)
    df["Production_Surplus"]      = df.Production_Units - df.Estimated_Deliveries  # +ve = over-produced
    df["Stations_Per_1k_Deliveries"] = (df.Charging_Stations / (df.Estimated_Deliveries/1000).replace(0,np.nan)).round(4)
    print("Ratio features added.")

    # 3. LAG & ROLLING FEATURES — what happened last month / last quarter?
    df["_ord"] = df.Year * 12 + df.Month
    df = df.sort_values(["Model","Region","_ord"])
    grp = df.groupby(["Model","Region"])

    for lag in [1,2,3]:
        df[f"Deliveries_Lag_{lag}"] = grp["Estimated_Deliveries"].shift(lag)
        df[f"Production_Lag_{lag}"] = grp["Production_Units"].shift(lag)

    df["Roll3_Mean"]  = grp["Estimated_Deliveries"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df["Roll3_Std"]   = grp["Estimated_Deliveries"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).std())
    df["MoM_Growth"]  = grp["Estimated_Deliveries"].pct_change().replace([np.inf,-np.inf], np.nan).round(4)
    df["YoY_Growth"]  = grp["Estimated_Deliveries"].transform(lambda x: x.pct_change(12)).replace([np.inf,-np.inf], np.nan).round(4)
    df.drop(columns=["_ord"], inplace=True)
    print("Lag & rolling features added.")

    # 4. LOG TRANSFORMS — reduce skew in large-range columns
    for col in ["Estimated_Deliveries","Production_Units","CO2_Saved_tons","Charging_Stations"]:
        df[f"Log_{col}"] = np.log1p(df[col].clip(lower=0))
    print("Log transforms added.")

    # 5. ONE-HOT ENCODING — convert text labels to 0/1 columns for ML
    for col, prefix in [("Region","Region"),("Model","Model"),("Season","Season"),("Quarter","Q")]:
        dummies = pd.get_dummies(df[col].astype(str), prefix=prefix)
        df = pd.concat([df, dummies], axis=1)
    print("One-hot encoding done.")

    # 6. Fill NaN from lag operations (first rows of each group have no history)
    lag_cols = [c for c in df.columns if any(k in c for k in ["Lag_","Roll3","MoM_","YoY_"])]
    df[lag_cols] = df[lag_cols].fillna(df[lag_cols].median())
    print(f"NaNs filled in {len(lag_cols)} lag/rolling columns.")

    print(f"Output: {df.shape[1]} columns  (+{df.shape[1]-15} new features)")
    df.to_csv(OUT, index=False)
    print(f"Saved → {OUT}\n")
    return df

if __name__ == "__main__":
    main()