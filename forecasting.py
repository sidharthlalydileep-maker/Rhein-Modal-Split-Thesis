"""
forecasting.py — Demand forecasting (RQ3).

Builds monthly corridor-demand forecasts with three approaches and then asks
the question that matters for the thesis: does a better forecast *change the
optimiser's decision*, or only reduce the forecast error?

Models:
  seasonal-naive  demand[t] = demand[t-12]          (baseline)
  linear          scikit-learn LinearRegression      (lagged demand + Kaub + month)
  xgboost         gradient-boosted trees             (same features)

Features (no look-ahead): demand lags 1,2,3,12; Kaub monthly-min lag 0,1;
month-of-year. Time-ordered split: train on the earliest ~75% of months, test
on the rest.

Decision test: for each test month, feed (a) the observed demand and (b) each
forecast into the optimiser under that month's actual κ_water (and rail normal),
then compare the modal split / cost / CO₂. The reported RQ3 finding is how
often, and by how much, the *decision* differs — not the MAE.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

import params as P
from optimiser import optimise
from data_loader import build_panel, observed_peaks

OUT = os.path.join(os.path.dirname(__file__), "outputs")
LAGS = [1, 2, 3, 12]


def make_features(df):
    d = df.dropna(subset=["barge_t", "rail_t", "road_t"]).copy()
    d = d.sort_values("date").reset_index(drop=True)
    d["y"] = d["demand_t"]
    for L in LAGS:
        d[f"lag{L}"] = d["y"].shift(L)
    d["kaub"] = d["kaub_w_min_cm"]
    d["kaub_lag1"] = d["kaub_w_min_cm"].shift(1)
    d["month"] = d["date"].dt.month
    d = d.dropna().reset_index(drop=True)
    return d


def _metrics(y, yhat):
    return dict(MAE=mean_absolute_error(y, yhat),
                RMSE=float(np.sqrt(mean_squared_error(y, yhat))),
                MAPE=float(np.mean(np.abs((y - yhat) / y)) * 100))


def run_forecasts(df, split=0.75):
    d = make_features(df)
    feat = [f"lag{L}" for L in LAGS] + ["kaub", "kaub_lag1", "month"]
    n = len(d)
    k = int(n * split)
    tr, te = d.iloc[:k], d.iloc[k:]
    Xtr, ytr = tr[feat], tr["y"]
    Xte, yte = te[feat], te["y"]

    preds = pd.DataFrame({"date": te["date"].values, "observed": yte.values})
    # seasonal-naive
    preds["seasonal_naive"] = te["lag12"].values
    # linear
    lin = LinearRegression().fit(Xtr, ytr)
    preds["linear"] = lin.predict(Xte)
    # random forest
    rf = RandomForestRegressor(n_estimators=400, max_depth=None,
                               min_samples_leaf=2, random_state=0, n_jobs=-1)
    rf.fit(Xtr, ytr)
    preds["random_forest"] = rf.predict(Xte)
    # xgboost
    xgr = xgb.XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.05,
                           subsample=0.9, colsample_bytree=0.9, random_state=0)
    xgr.fit(Xtr, ytr)
    preds["xgboost"] = xgr.predict(Xte)

    metrics = {m: _metrics(preds["observed"].values, preds[m].values)
               for m in ["seasonal_naive", "linear", "random_forest", "xgboost"]}
    return preds, metrics


def decision_test(df, preds, peaks, cargo="Dry bulk", lam=P.LAMBDA_BASE,
                  headroom=P.HEADROOM_BASE):
    """For each test month, compare decisions under forecast vs observed demand.

    Uses the month's ACTUAL κ_water (rail normal) so the disruption state is real
    and only the demand input differs between runs.
    """
    kmap = df.set_index("date")["kappa_water"].to_dict()
    rows = []
    for _, r in preds.iterrows():
        kw = kmap.get(pd.Timestamp(r["date"]), 1.0)
        base = optimise(cargo, kw, 1.0, r["observed"], lam, peaks, headroom)
        for model in ["seasonal_naive", "linear", "random_forest", "xgboost"]:
            fc = optimise(cargo, kw, 1.0, max(r[model], 0.0), lam, peaks, headroom)
            rows.append(dict(
                date=r["date"], model=model, kappa_water=kw,
                demand_err_pct=(r[model] - r["observed"]) / r["observed"] * 100,
                road_share_obs=base.shares["road"],
                road_share_fc=fc.shares["road"],
                road_share_diff_pp=(fc.shares["road"] - base.shares["road"]) * 100,
                cost_diff_pct=(fc.cost_eur - base.cost_eur) / base.cost_eur * 100
                if base.cost_eur else 0.0))
    dt = pd.DataFrame(rows)
    # summary per model
    summ = dt.groupby("model").agg(
        mean_abs_demand_err_pct=("demand_err_pct", lambda s: s.abs().mean()),
        mean_abs_road_share_diff_pp=("road_share_diff_pp", lambda s: s.abs().mean()),
        max_abs_road_share_diff_pp=("road_share_diff_pp", lambda s: s.abs().max()),
        mean_abs_cost_diff_pct=("cost_diff_pct", lambda s: s.abs().mean()),
    ).reset_index()
    return dt, summ


if __name__ == "__main__":
    df = build_panel(save=False)
    peaks = {m: v["tonnes"] for m, v in observed_peaks(df).items()}
    preds, metrics = run_forecasts(df)
    preds.to_csv(os.path.join(OUT, "forecasts.csv"), index=False)

    print(f"Forecast test window: {preds['date'].min():%Y-%m} → "
          f"{preds['date'].max():%Y-%m}  ({len(preds)} months)\n")
    print("Forecast accuracy (demand level):")
    md = pd.DataFrame(metrics).T
    md["MAE_Mt"] = md["MAE"] / 1e6
    print(md[["MAE_Mt", "RMSE", "MAPE"]].round({"MAE_Mt": 2, "RMSE": 0, "MAPE": 2}).to_string())

    dt, summ = decision_test(df, preds, peaks)
    dt.to_csv(os.path.join(OUT, "decision_test.csv"), index=False)
    summ.to_csv(os.path.join(OUT, "decision_test_summary.csv"), index=False)
    print("\nRQ3 — does the better forecast change the DECISION?")
    print(summ.round(2).to_string(index=False))
    print("\nReading: if road-share differences are small (≈ a few pp) even where "
          "demand error differs, the optimisation decision is robust to forecast "
          "choice — the RQ3 finding.")
