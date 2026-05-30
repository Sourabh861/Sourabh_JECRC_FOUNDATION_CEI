# STEP 6 — HYPERPARAMETER TUNING
# Use RandomizedSearchCV to find the best settings for
# Random Forest and Gradient Boosting (+ XGBoost if installed).
# Shows before vs after improvement.

import pandas as pd
import numpy as np
import os, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import randint, uniform
try:
    from xgboost import XGBRegressor; HAS_XGB = True
except ImportError:
    HAS_XGB = False

IN      = "outputs/04_feature_engineered.csv"
OUT_CSV = "outputs/06_tuned_metrics.csv"
OUT_PRM = "outputs/06_best_params.json"
PLOTS   = "plots"
TARGET  = "Estimated_Deliveries"
N_ITER  = 8   
FEATS   = ["Year","Month","Quarter","Is_Year_End","Is_Quarter_End","Years_Since_2015",
           "Production_Units","Avg_Price_USD","Battery_Capacity_kWh","Range_km",
           "CO2_Saved_tons","Charging_Stations","Delivery_Rate","Price_Per_Km",
           "Price_Per_kWh","CO2_Per_1k_Deliveries","Production_Surplus",
           "Stations_Per_1k_Deliveries","Deliveries_Lag_1","Deliveries_Lag_2",
           "Deliveries_Lag_3","Production_Lag_1","Roll3_Mean","Roll3_Std",
           "MoM_Growth","YoY_Growth","Region_Code","Model_Code","Source_Code"]

def score_row(name, y_true, y_pred, cv_r2=None):
    return {"Model":name,
            "R2"    : round(r2_score(y_true,y_pred),4),
            "RMSE"  : round(np.sqrt(mean_squared_error(y_true,y_pred)),2),
            "MAE"   : round(mean_absolute_error(y_true,y_pred),2),
            "MAPE_%": round(np.nanmean(np.abs((y_true-y_pred)/np.where(y_true==0,np.nan,y_true)))*100,2),
            "CV_R2" : round(cv_r2,4) if cv_r2 else None}

def tune(name, model, space, Xtr, Xte, ytr, yte):
    print(f"\n  Tuning {name} (n_iter={N_ITER}, cv=3) ...")
    search = RandomizedSearchCV(model, space, n_iter=N_ITER, cv=3, scoring="r2",
                                 n_jobs=-1, random_state=42, refit=True, verbose=0)
    search.fit(Xtr, ytr)
    pred = search.best_estimator_.predict(Xte)
    r    = score_row(f"{name} (tuned)", yte, pred, search.best_score_)
    r["Best_Params"] = json.dumps(search.best_params_, default=str)
    print(f"  → R²={r['R2']}  RMSE={r['RMSE']:,.0f}  CV R²={r['CV_R2']}")
    return r, search

def main():
    os.makedirs("outputs", exist_ok=True); os.makedirs(PLOTS, exist_ok=True)
    print("\n=== STEP 6: HYPERPARAMETER TUNING ===")

    df   = pd.read_csv(IN)
    feats = [c for c in FEATS if c in df.columns]
    X = df[feats].fillna(df[feats].median()); y = df[TARGET]
    X, y = X[y.notna()], y[y.notna()]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    Xtr, Xte, ytr, yte = Xtr.values, Xte.values, ytr.values, yte.values
    print(f"Train: {len(Xtr):,}  Test: {len(Xte):,}")

    # Search spaces — ranges of values to try for each hyperparameter
    rf_space = {"n_estimators":randint(100,500),"max_depth":[None,6,8,10,12,15],
                "min_samples_split":randint(2,15),"min_samples_leaf":randint(1,8),
                "max_features":["sqrt","log2",0.5,0.7],"bootstrap":[True,False]}

    gb_space = {"n_estimators":randint(50,200),"learning_rate":uniform(0.05,0.15),
                "max_depth":randint(2,5),"subsample":uniform(0.7,0.3)}

    rows, searches = [], {}
    r, s = tune("Random Forest",     RandomForestRegressor(random_state=42,n_jobs=-1), rf_space, Xtr,Xte,ytr,yte)
    rows.append(r); searches["Random Forest"] = s

    r, s = tune("Gradient Boosting", GradientBoostingRegressor(random_state=42), gb_space, Xtr,Xte,ytr,yte)
    rows.append(r); searches["Gradient Boosting"] = s

    if HAS_XGB:
        xgb_space = {"n_estimators":randint(100,500),"learning_rate":uniform(0.01,0.2),
                     "max_depth":randint(2,8),"subsample":uniform(0.6,0.4),
                     "colsample_bytree":uniform(0.6,0.4),"reg_alpha":uniform(0,1)}
        r, s = tune("XGBoost", XGBRegressor(random_state=42,n_jobs=-1,verbosity=0), xgb_space, Xtr,Xte,ytr,yte)
        rows.append(r); searches["XGBoost"] = s

    tuned = pd.DataFrame(rows).sort_values("R2", ascending=False)
    print(f"\nResults after tuning:\n{tuned[['Model','R2','RMSE','MAE','MAPE_%','CV_R2']].to_string(index=False)}")

    # Before vs After comparison chart
    if os.path.exists("outputs/05_model_metrics.csv"):
        before = pd.read_csv("outputs/05_model_metrics.csv")
        before = before[before.Model.isin([r.replace(" (tuned)","") for r in tuned.Model])].copy()
        before["Type"]  = "Before"
        after           = tuned[["Model","R2","RMSE"]].copy()
        after["Model"]  = after.Model.str.replace(" (tuned)","",regex=False)
        after["Type"]   = "After"
        combined        = pd.concat([before[["Model","R2","RMSE","Type"]], after])
        fig, axes = plt.subplots(1,2,figsize=(12,5))
        for ax, metric in zip(axes, ["R2","RMSE"]):
            combined.pivot(index="Model",columns="Type",values=metric).plot(kind="bar",ax=ax,colormap="Set1",edgecolor="white")
            ax.set(title=f"{metric}: Before vs After Tuning", xlabel="")
            ax.tick_params(axis="x",rotation=20)
        plt.tight_layout()
        plt.savefig(f"{PLOTS}/tuning_before_vs_after.png",dpi=120,bbox_inches="tight"); plt.close()
        print(f"  Plot → {PLOTS}/tuning_before_vs_after.png")

    best_params = {s_name: s.best_params_ for s_name, s in searches.items()}
    tuned.to_csv(OUT_CSV, index=False)
    json.dump(best_params, open(OUT_PRM,"w"), indent=2, default=str)
    print(f"Saved → {OUT_CSV}  |  {OUT_PRM}\n")
    return tuned

if __name__ == "__main__":
    main()