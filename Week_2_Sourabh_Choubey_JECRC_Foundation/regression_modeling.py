# STEP 5 — REGRESSION MODELING
# Train 6 models to predict Estimated_Deliveries.
# Compare them on R², RMSE, MAE, MAPE.

import pandas as pd
import numpy as np
import os, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
try:
    from xgboost import XGBRegressor; HAS_XGB = True
except ImportError:
    HAS_XGB = False

IN      = "outputs/04_feature_engineered.csv"
OUT_CSV = "outputs/05_model_metrics.csv"
OUT_BEST= "outputs/05_best_model.json"
PLOTS   = "plots"
TARGET  = "Estimated_Deliveries"
FEATS   = ["Year","Month","Quarter","Is_Year_End","Is_Quarter_End","Years_Since_2015",
           "Production_Units","Avg_Price_USD","Battery_Capacity_kWh","Range_km",
           "CO2_Saved_tons","Charging_Stations","Delivery_Rate","Price_Per_Km",
           "Price_Per_kWh","CO2_Per_1k_Deliveries","Production_Surplus",
           "Stations_Per_1k_Deliveries","Deliveries_Lag_1","Deliveries_Lag_2",
           "Deliveries_Lag_3","Production_Lag_1","Roll3_Mean","Roll3_Std",
           "MoM_Growth","YoY_Growth","Region_Code","Model_Code","Source_Code"]

def score(name, y_true, y_pred, cv=None):
    r = {"Model":name, "R2":round(r2_score(y_true,y_pred),4),
         "RMSE":round(np.sqrt(mean_squared_error(y_true,y_pred)),2),
         "MAE":round(mean_absolute_error(y_true,y_pred),2),
         "MAPE_%":round(np.nanmean(np.abs((y_true-y_pred)/np.where(y_true==0,np.nan,y_true)))*100,2)}
    if cv is not None: r["CV_R2"]=round(cv.mean(),4); r["CV_Std"]=round(cv.std(),4)
    return r

def main():
    os.makedirs("outputs", exist_ok=True); os.makedirs(PLOTS, exist_ok=True)
    print("\n=== STEP 5: REGRESSION MODELING ===")

    df = pd.read_csv(IN)
    feats = [c for c in FEATS if c in df.columns]
    X = df[feats].fillna(df[feats].median())
    y = df[TARGET]
    X, y = X[y.notna()], y[y.notna()]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    Xs_tr, Xs_te = sc.fit_transform(X_tr), sc.transform(X_te)
    print(f"Train: {len(X_tr):,}  Test: {len(X_te):,}  Features: {len(feats)}\n")

    # (model, use_scaled, run_cv)
    models = {
        "Linear Regression" : (LinearRegression(),                                                      True,  True),
        "Ridge (α=10)"      : (Ridge(alpha=10),                                                         True,  True),
        "Lasso (α=1)"       : (Lasso(alpha=1, max_iter=5000),                                           True,  True),
        "Random Forest"     : (RandomForestRegressor(n_estimators=200,max_depth=12,random_state=42,n_jobs=-1), False, False),
        "Gradient Boosting" : (GradientBoostingRegressor(n_estimators=200,learning_rate=0.05,max_depth=4,random_state=42), False, False),
        "SVR"               : (SVR(kernel="rbf",C=100,epsilon=0.1),                                     True,  False),
    }
    if HAS_XGB: models["XGBoost"] = (XGBRegressor(n_estimators=200,learning_rate=0.05,max_depth=6,random_state=42,n_jobs=-1,verbosity=0), False, False)

    rows, fitted = [], {}
    for name, (m, scaled, do_cv) in models.items():
        Xtr, Xte = (Xs_tr, Xs_te) if scaled else (X_tr.values, X_te.values)
        m.fit(Xtr, y_tr.values)
        pred = m.predict(Xte)
        cv = cross_val_score(m, Xtr, y_tr.values, cv=5, scoring="r2", n_jobs=-1) if do_cv else None
        r = score(name, y_te.values, pred, cv)
        rows.append(r); fitted[name] = (m, pred, scaled)
        print(f"  {name:<28} R²={r['R2']:.4f}  RMSE={r['RMSE']:>8,.0f}  MAPE={r['MAPE_%']:.2f}%")

    results = pd.DataFrame(rows).sort_values("R2", ascending=False)
    best    = results.iloc[0]["Model"]
    bm, bp, bsc = fitted[best]
    print(f"\nBest: {best}  (R²={results.iloc[0]['R2']})")

    # Actual vs Predicted + Residuals plot for best model
    fig, axes = plt.subplots(1, 2, figsize=(13,5))
    lo, hi = y_te.min(), y_te.max()
    axes[0].scatter(y_te, bp, alpha=0.35, s=18, color="steelblue")
    axes[0].plot([lo,hi],[lo,hi],"r--", label="Perfect fit"); axes[0].legend()
    axes[0].set(title=f"Actual vs Predicted — {best}", xlabel="Actual", ylabel="Predicted")
    axes[1].hist(y_te.values-bp, bins=50, edgecolor="white", color="coral")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set(title="Residuals (should centre at 0)", xlabel="Error")
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/model_actual_vs_pred.png", dpi=120, bbox_inches="tight"); plt.close()

    # Feature importance for best model
    if hasattr(bm, "feature_importances_") or hasattr(bm, "coef_"):
        imp = bm.feature_importances_ if hasattr(bm, "feature_importances_") else np.abs(bm.coef_)
        top = np.argsort(imp)[-20:]
        fig, ax = plt.subplots(figsize=(9,7))
        ax.barh(np.array(feats)[top], imp[top])
        ax.set(title=f"Top 20 Features — {best}", xlabel="Importance")
        plt.tight_layout()
        plt.savefig(f"{PLOTS}/model_feature_importance.png", dpi=120, bbox_inches="tight"); plt.close()

    results.to_csv(OUT_CSV, index=False)
    json.dump({"best_model": best}, open(OUT_BEST,"w"))
    print(f"Saved → {OUT_CSV}\n")
    return results

if __name__ == "__main__":
    main()