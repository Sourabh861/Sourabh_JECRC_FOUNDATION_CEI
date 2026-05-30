# STEP 7 — TIME SERIES FORECASTING
# Aggregate deliveries to monthly global totals, then forecast 12 months ahead.
# Tries SARIMA, Prophet, LSTM — skips gracefully if libraries are missing.
# Holt-Winters (pure numpy) always runs as a guaranteed fallback.

import pandas as pd
import numpy as np
import os, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

IN      = "outputs/02_preprocessed_data.csv"
OUT_CSV = "outputs/07_forecasts.csv"
PLOTS   = "plots"
HORIZON = 12  

def build_series(df):
    """Aggregate all regions/models into one global monthly total."""
    ts = (df.groupby(["Year","Month"])["Estimated_Deliveries"].sum().reset_index())
    ts["Date"] = pd.to_datetime(ts.Year.astype(int).astype(str)+"-"+ts.Month.astype(int).astype(str).str.zfill(2))
    return ts.set_index("Date").sort_index()["Estimated_Deliveries"].asfreq("MS")

def forecast_sarima(ts, h):
    """Classical statsmodels SARIMA. Auto-detects if differencing is needed."""
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from statsmodels.tsa.stattools import adfuller
        d   = 1 if adfuller(ts.dropna())[1] > 0.05 else 0
        res = SARIMAX(ts, order=(2,d,2), seasonal_order=(1,1,1,12),
                      enforce_stationarity=False).fit(disp=False, maxiter=200)
        fc  = res.forecast(h)
        fc.index = pd.date_range(ts.index[-1]+pd.DateOffset(months=1), periods=h, freq="MS")
        print(f"  SARIMA: AIC={res.aic:.1f}"); return fc.clip(lower=0)
    except Exception as e: print(f"  SARIMA skipped: {e}"); return pd.Series(dtype=float)

def forecast_prophet(ts, h):
    """Facebook/Meta Prophet — handles seasonality and trend changes well."""
    try:
        from prophet import Prophet
        train = ts.reset_index().rename(columns={"Date":"ds","Estimated_Deliveries":"y"})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, seasonality_mode="multiplicative",
                    changepoint_prior_scale=0.3)
        m.fit(train)
        future = m.make_future_dataframe(periods=h, freq="MS")
        fc     = m.predict(future)
        result = fc[fc.ds > ts.index[-1]].set_index("ds")["yhat"].clip(lower=0)
        result.index = pd.DatetimeIndex(result.index)
        print("  Prophet: fit complete."); return result
    except Exception as e: print(f"  Prophet skipped: {e}"); return pd.Series(dtype=float)

def forecast_lstm(ts, h, lookback=12):
    """LSTM neural network — learns non-linear temporal patterns."""
    try:
        import tensorflow as tf; tf.get_logger().setLevel("ERROR")
        from tensorflow.keras.models   import Sequential
        from tensorflow.keras.layers   import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.preprocessing     import MinMaxScaler

        vals   = ts.values.reshape(-1,1).astype(float)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(vals)

        def make_seqs(data, lb):
            X,y = zip(*[(data[i-lb:i,0], data[i,0]) for i in range(lb,len(data))])
            return np.array(X).reshape(-1,lb,1), np.array(y)

        X, y = make_seqs(scaled, lookback)
        model = Sequential([LSTM(64,return_sequences=True,input_shape=(lookback,1)),
                             Dropout(0.2), LSTM(32), Dropout(0.2), Dense(1)])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=80, batch_size=16, validation_split=0.15,
                  callbacks=[EarlyStopping(patience=10,restore_best_weights=True)], verbose=0)

        seq = scaled[-lookback:].flatten().tolist()
        preds = []
        for _ in range(h):
            p = model.predict(np.array(seq[-lookback:]).reshape(1,lookback,1), verbose=0)[0][0]
            preds.append(p); seq.append(p)
        preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten().clip(0)
        idx   = pd.date_range(ts.index[-1]+pd.DateOffset(months=1), periods=h, freq="MS")
        print("  LSTM: training complete."); return pd.Series(preds, index=idx)
    except Exception as e: print(f"  LSTM skipped: {e}"); return pd.Series(dtype=float)

def forecast_holt_winters(ts, h, alpha=0.3, beta=0.1, gamma=0.3, m=12):
    """Triple Exponential Smoothing — pure numpy, always available."""
    data = ts.values.astype(float); n = len(data)
    level = np.mean(data[:m])
    trend = (np.mean(data[m:2*m]) - np.mean(data[:m])) / m
    seasonal = [data[i]/np.mean(data[:m]) for i in range(m)]
    L, T, S = [level], [trend], list(seasonal)
    for i in range(n):
        si = i % m
        sv = S[-(m-si) if si else -m]
        if i == 0: continue
        nl = alpha*(data[i]/sv) + (1-alpha)*(L[-1]+T[-1])
        nt = beta*(nl-L[-1]) + (1-beta)*T[-1]
        ns = gamma*(data[i]/nl) + (1-gamma)*sv
        L.append(nl); T.append(nt); S.append(ns)
    preds = [(L[-1]+T[-1]*i)*(S[-(m-(n+i-1)%m) if (n+i-1)%m else -m]) for i in range(1,h+1)]
    idx = pd.date_range(ts.index[-1]+pd.DateOffset(months=1), periods=h, freq="MS")
    print("  Holt-Winters: complete."); return pd.Series(np.clip(preds,0,None), index=idx)

def main():
    os.makedirs("outputs",exist_ok=True); os.makedirs(PLOTS,exist_ok=True)
    print("\n=== STEP 7: TIME SERIES FORECASTING ===")

    df = pd.read_csv(IN)
    ts = build_series(df)
    print(f"Monthly series: {len(ts)} points  ({ts.index[0].date()} → {ts.index[-1].date()})")
    print(f"Mean: {ts.mean():,.0f}  Std: {ts.std():,.0f}\n")

    # Try decomposition if statsmodels is available
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        dec  = seasonal_decompose(ts.dropna(), model="multiplicative", period=12)
        fig, axes = plt.subplots(4,1,figsize=(13,10),sharex=True)
        for ax, comp, name in zip(axes, [dec.observed,dec.trend,dec.seasonal,dec.resid],
                                        ["Observed","Trend","Seasonal","Residual"]):
            comp.plot(ax=ax); ax.set_title(name); ax.grid(True,linestyle="--",alpha=0.4)
        plt.suptitle("Time Series Decomposition",fontsize=13,y=1.01); plt.tight_layout()
        plt.savefig(f"{PLOTS}/ts_decomposition.png",dpi=120,bbox_inches="tight"); plt.close()
        print(f"  Decomposition plot → {PLOTS}/ts_decomposition.png")
    except: pass

    print("Running forecasts ...")
    forecasts = {
        "SARIMA"       : forecast_sarima(ts, HORIZON),
        "Prophet"      : forecast_prophet(ts, HORIZON),
        "LSTM"         : forecast_lstm(ts, HORIZON),
        "Holt-Winters" : forecast_holt_winters(ts, HORIZON),   # always runs
    }

    # Summary table
    rows = [{"Model":name,"Date":dt.date(),"Forecast":round(val,0)}
            for name,fc in forecasts.items() if fc is not None and len(fc)>0
            for dt,val in fc.items()]
    if rows:
        fc_df = pd.DataFrame(rows)
        print("\nForecast summary (next 12 months):")
        print(fc_df.pivot(index="Date",columns="Model",values="Forecast").to_string())
        fc_df.to_csv(OUT_CSV, index=False)

    # Combined forecast chart
    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(ts.index, ts.values/1e3, color="black", linewidth=2, label="Historical")
    colours = {"SARIMA":"steelblue","Prophet":"darkorange","LSTM":"mediumseagreen","Holt-Winters":"purple"}
    for name, fc in forecasts.items():
        if fc is not None and len(fc)>0:
            ax.plot(fc.index, fc.values/1e3, linewidth=2, linestyle="--",
                    color=colours.get(name,"gray"), label=name)
    ax.axvline(ts.index[-1], color="red", linestyle=":", alpha=0.7)
    ax.set(title="Tesla Monthly Deliveries — Historical + 12-Month Forecast",
           xlabel="Date", ylabel="Deliveries (thousands)")
    ax.legend(); ax.grid(True,linestyle="--",alpha=0.4); plt.tight_layout()
    plt.savefig(f"{PLOTS}/ts_forecast.png",dpi=120,bbox_inches="tight"); plt.close()
    print(f"\nPlot → {PLOTS}/ts_forecast.png")
    print(f"Saved → {OUT_CSV}\n")

if __name__ == "__main__":
    main()