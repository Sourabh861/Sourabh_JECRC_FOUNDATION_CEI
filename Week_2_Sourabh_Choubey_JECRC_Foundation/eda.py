# STEP 3 — EXPLORATORY DATA ANALYSIS
# Pure EDA — no files saved, just clean printed insights.

import pandas as pd
import numpy as np

IN = "outputs/02_preprocessed_data.csv"

NUM = ["Estimated_Deliveries","Production_Units","Avg_Price_USD",
       "CO2_Saved_tons","Range_km","Battery_Capacity_kWh","Charging_Stations"]

def divider(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")

def main():
    print("\n=== STEP 3: EDA ===")
    df = pd.read_csv(IN)
    print(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # 1. Basic stats for every numeric column
    divider("1. Descriptive Statistics")
    print(df[NUM].describe().round(2).to_string())

    # 2. How many records per category
    divider("2. Category Counts")
    for col in ["Region", "Model", "Source_Type"]:
        print(f"\n  {col}:")
        for val, cnt in df[col].value_counts().items():
            pct = cnt / len(df) * 100
            print(f"    {val:<30} {cnt:>5,}  ({pct:.1f}%)")

    # 3. Yearly delivery & production totals
    divider("3. Yearly Trend")
    yr = df.groupby("Year")[["Estimated_Deliveries","Production_Units"]].sum()
    yr["Delivery_Rate_%"] = (yr.Estimated_Deliveries / yr.Production_Units * 100).round(1)
    print(yr.to_string())

    # 4. Monthly seasonality — which months deliver the most?
    divider("4. Monthly Seasonality (avg deliveries)")
    mon = df.groupby("Month")["Estimated_Deliveries"].mean().round(0)
    for month, avg in mon.items():
        bar = "█" * int(avg / mon.max() * 30)
        print(f"  Month {month:>2}  {bar:<30}  {avg:>8,.0f}")

    # 5. Average deliveries by Region and Model
    divider("5. Avg Deliveries by Region & Model")
    print("\n  By Region:")
    print(df.groupby("Region")["Estimated_Deliveries"].mean().round(0).sort_values(ascending=False).to_string())
    print("\n  By Model:")
    print(df.groupby("Model")["Estimated_Deliveries"].mean().round(0).sort_values(ascending=False).to_string())

    # 6. Correlation with the target column
    divider("6. Correlation with Estimated_Deliveries")
    corr = df[NUM].corr()["Estimated_Deliveries"].drop("Estimated_Deliveries").sort_values(ascending=False)
    for feat, val in corr.items():
        direction = "▲" if val > 0 else "▼"
        bar = "█" * int(abs(val) * 20)
        print(f"  {direction} {feat:<30} {bar:<20}  {val:+.4f}")

    # 7. Full correlation matrix
    divider("7. Full Correlation Matrix")
    print(df[NUM].corr().round(3).to_string())

    # 8. Price vs Deliveries — breakdown by Model
    divider("8. Avg Price & Deliveries by Model")
    summary = df.groupby("Model").agg(
        Avg_Price   = ("Avg_Price_USD",          "mean"),
        Avg_Deliveries = ("Estimated_Deliveries", "mean"),
        Total_Deliveries = ("Estimated_Deliveries","sum"),
    ).round(0)
    print(summary.sort_values("Avg_Deliveries", ascending=False).to_string())

    # 9. CO2 savings summary
    divider("9. CO2 Savings by Region")
    co2 = df.groupby("Region")["CO2_Saved_tons"].agg(["mean","sum","max"]).round(1)
    co2.columns = ["Avg_CO2","Total_CO2","Max_CO2"]
    print(co2.sort_values("Total_CO2", ascending=False).to_string())

    # 10. Skewness — high skew means log-transform will help
    divider("10. Skewness of Numeric Features")
    skew = df[NUM].skew().round(3).sort_values(ascending=False)
    for col, val in skew.items():
        flag = "  ← high skew, consider log transform" if abs(val) > 1 else ""
        print(f"  {col:<35} {val:+.3f}{flag}")

    print(f"\n{'─'*55}")
    print("  EDA complete.\n")

if __name__ == "__main__":
    main()