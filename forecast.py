"""
Project FORESIGHT — D3 Demand forecast model
=============================================
Weekly SKU-level forecast, 8-week horizon, evaluated with ROLLING-ORIGIN
backtesting against a seasonal-naive baseline. Metric: WAPE (bias secondary).

Leakage guard: every feature for a target week (origin + h) is computed ONLY
from weeks <= origin, except calendar/promotion attributes of the target week,
which are known in advance (planned promotions, holidays, week-of-year).

Run:  python src/forecast.py
Out:  outputs/backtest_folds.csv        per fold × horizon × model metrics
      outputs/backtest_metrics.json     headline WAPE/bias, per-horizon, per-category
      outputs/forecast.csv              8-week forecast per SKU with 80% interval + baseline
      outputs/feature_importance.csv
      outputs/model_lgbm.pkl
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

SEED = 42
HORIZON = 8                       # weeks
FORECAST_ORIGIN = pd.Timestamp("2025-11-30")
N_FOLDS = 6                        # rolling origins, 4 weeks apart
SEASON_BY_MONTH = {1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Summer", 6: "Summer",
                   7: "Monsoon", 8: "Monsoon", 9: "Monsoon", 10: "Autumn", 11: "Autumn", 12: "Winter"}
CAT_COLS = ["sku_id", "category", "season"]
FEATURES = ["h", "lag1", "lag2", "lag3", "lag4", "lag8", "lag13", "mean4", "mean8", "mean13", "std4",
            "trend4_13", "lag52_target", "lag52_mean3", "level_ratio_52", "promo_days_target",
            "holiday_days_target", "week_of_year", "month", "weeks_since_launch", "unit_price",
            "sku_id", "category", "season"]


# ----------------------------------------------------------------------------- metrics
def wape(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(np.abs(y - f).sum() / max(y.sum(), 1e-9))


def bias(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float((f - y).sum() / max(y.sum(), 1e-9))


def mape(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float)
    m = y > 0
    return float(np.mean(np.abs(y[m] - f[m]) / y[m])) if m.any() else np.nan


# ----------------------------------------------------------------------------- data
def load_weekly() -> pd.DataFrame:
    wk = pd.read_csv(PROC / "sales_weekly.csv", parse_dates=["week_end", "launch_date"])
    return wk.sort_values(["sku_id", "week_end"]).reset_index(drop=True)


def week_calendar(week_ends: pd.DatetimeIndex, wk: pd.DataFrame) -> pd.DataFrame:
    """Calendar attributes for any week (past or future). Promotions/holidays beyond the
    calendar file are assumed 0 — documented assumption."""
    known = wk.groupby("week_end")[["promo_days", "holiday_days"]].max()
    cal = pd.DataFrame(index=week_ends)
    cal["week_of_year"] = cal.index.isocalendar().week.astype(int).values
    cal["month"] = cal.index.month
    cal["season"] = cal["month"].map(SEASON_BY_MONTH)
    cal = cal.join(known.rename(columns={"promo_days": "promo_days_target",
                                          "holiday_days": "holiday_days_target"}))
    cal[["promo_days_target", "holiday_days_target"]] = cal[["promo_days_target", "holiday_days_target"]].fillna(0)
    return cal


def make_rows(wk: pd.DataFrame, origin: pd.Timestamp, horizons=range(1, HORIZON + 1), need_target=True) -> pd.DataFrame:
    """Build one row per (sku, h) for a single forecast origin using ONLY history <= origin."""
    hist = wk[wk["week_end"] <= origin]
    piv = hist.pivot(index="week_end", columns="sku_id", values="units").sort_index()
    if len(piv) < 13:
        return pd.DataFrame()
    last = piv.iloc[-1]
    feats = pd.DataFrame({"sku_id": piv.columns})
    for k in [1, 2, 3, 4, 8, 13]:
        feats[f"lag{k}"] = piv.iloc[-k].values
    feats["mean4"] = piv.iloc[-4:].mean().values
    feats["mean8"] = piv.iloc[-8:].mean().values
    feats["mean13"] = piv.iloc[-13:].mean().values
    feats["std4"] = piv.iloc[-4:].std().values
    feats["trend4_13"] = feats["mean4"] - feats["mean13"]
    # level ratio: how this year's recent level compares to the same weeks last year
    if len(piv) >= 56:
        ly = piv.iloc[-56:-52].mean()
        feats["level_ratio_52"] = (feats["mean4"].values / ly.replace(0, np.nan).values)
    else:
        feats["level_ratio_52"] = np.nan
    static = wk.drop_duplicates("sku_id").set_index("sku_id")[["category", "launch_date", "unit_price"]]
    feats = feats.merge(static, left_on="sku_id", right_index=True, how="left")
    full = wk.pivot(index="week_end", columns="sku_id", values="units").sort_index()
    all_weeks = pd.DatetimeIndex(sorted(set(full.index) | set(origin + pd.to_timedelta(7 * np.arange(1, HORIZON + 1), "D"))))
    cal = week_calendar(all_weeks, wk)
    rows = []
    for h in horizons:
        tw = origin + pd.Timedelta(days=7 * h)
        r = feats.copy()
        r["h"] = h
        r["week_end"] = tw
        lag52_week = tw - pd.Timedelta(weeks=52)
        r["lag52_target"] = piv.reindex([lag52_week]).iloc[0].values if lag52_week in piv.index else np.nan
        neigh = [lag52_week - pd.Timedelta(weeks=1), lag52_week, lag52_week + pd.Timedelta(weeks=1)]
        r["lag52_mean3"] = piv.reindex(neigh).mean().values if lag52_week in piv.index else np.nan
        for c in ["week_of_year", "month", "season", "promo_days_target", "holiday_days_target"]:
            r[c] = cal.loc[tw, c]
        r["weeks_since_launch"] = ((tw - r["launch_date"]).dt.days // 7).clip(lower=0)
        if need_target:
            r["y"] = full.reindex([tw]).iloc[0].reindex(r["sku_id"]).values if tw in full.index else np.nan
        # seasonal-naive baseline for the same target
        r["naive"] = r["lag52_target"].fillna(r["mean4"])
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def training_set(wk: pd.DataFrame, upto: pd.Timestamp) -> pd.DataFrame:
    """All (origin, h) rows whose TARGET week <= upto. Origins step weekly."""
    weeks = np.sort(wk["week_end"].unique())
    parts = []
    for o in weeks[12:]:
        o = pd.Timestamp(o)
        if o + pd.Timedelta(weeks=1) > upto:
            break
        r = make_rows(wk, o)
        parts.append(r[r["week_end"] <= upto])
    df = pd.concat(parts, ignore_index=True)
    return df.dropna(subset=["y"])


def fit(train: pd.DataFrame) -> lgb.LGBMRegressor:
    X = train[FEATURES].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    model = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.2, n_estimators=600,
                              learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
                              verbose=-1)
    model.fit(X, train["y"], categorical_feature=CAT_COLS)
    return model


def predict(model, rows: pd.DataFrame, cats: dict) -> np.ndarray:
    X = rows[FEATURES].copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cats[c])
    return np.clip(model.predict(X), 0, None)


# ----------------------------------------------------------------------------- backtest
def backtest(wk: pd.DataFrame):
    origins = [FORECAST_ORIGIN - pd.Timedelta(weeks=HORIZON + 4 * i) for i in range(N_FOLDS)][::-1]
    fold_rows, results = [], []
    for k, o in enumerate(origins, 1):
        train = training_set(wk, o)
        model = fit(train)
        cats = {c: train[c].astype("category").cat.categories for c in CAT_COLS}
        test = make_rows(wk, o).dropna(subset=["y"])
        test["model"] = predict(model, test, cats)
        test["fold"] = k
        test["origin"] = o
        fold_rows.append(test)
        for h in sorted(test["h"].unique()):
            t = test[test["h"] == h]
            results.append(dict(fold=k, origin=str(o.date()), h=int(h), n=len(t),
                                wape_model=wape(t.y, t.model), wape_naive=wape(t.y, t.naive),
                                bias_model=bias(t.y, t.model), bias_naive=bias(t.y, t.naive)))
        print(f"fold {k} origin {o.date()}  WAPE model={wape(test.y, test.model):.3f}  naive={wape(test.y, test.naive):.3f}")
    folds = pd.concat(fold_rows, ignore_index=True)
    return folds, pd.DataFrame(results)


def summarise(folds: pd.DataFrame, res: pd.DataFrame) -> dict:
    overall = dict(wape_model=wape(folds.y, folds.model), wape_naive=wape(folds.y, folds.naive),
                   mape_model=mape(folds.y, folds.model), mape_naive=mape(folds.y, folds.naive),
                   bias_model=bias(folds.y, folds.model), bias_naive=bias(folds.y, folds.naive))
    overall["improvement_pct"] = 100 * (1 - overall["wape_model"] / overall["wape_naive"])
    overall["model_wins"] = overall["wape_model"] < overall["wape_naive"]
    by_h = folds.groupby("h").apply(lambda t: pd.Series(dict(wape_model=wape(t.y, t.model), wape_naive=wape(t.y, t.naive)))).reset_index()
    by_fold = folds.groupby(["fold", "origin"]).apply(lambda t: pd.Series(dict(wape_model=wape(t.y, t.model), wape_naive=wape(t.y, t.naive)))).reset_index()
    by_fold["origin"] = by_fold["origin"].astype(str)
    by_cat = folds.groupby("category").apply(lambda t: pd.Series(dict(wape_model=wape(t.y, t.model), wape_naive=wape(t.y, t.naive)))).reset_index()
    by_sku = folds.groupby("sku_id").apply(lambda t: pd.Series(dict(wape_model=wape(t.y, t.model), wape_naive=wape(t.y, t.naive)))).reset_index()
    # interval calibration from relative residuals per horizon
    folds["rel_err"] = (folds.y - folds.model) / folds.model.clip(lower=1)
    q = folds.groupby("h")["rel_err"].quantile([0.1, 0.9]).unstack()
    sig = folds.groupby("h")["rel_err"].std()
    intervals = {int(h): dict(q10=float(q.loc[h, 0.1]), q90=float(q.loc[h, 0.9]), rel_sigma=float(sig.loc[h])) for h in q.index}
    return dict(horizon_weeks=HORIZON, n_folds=N_FOLDS, seed=SEED, overall=overall,
                by_horizon=by_h.to_dict("records"), by_fold=by_fold.to_dict("records"),
                by_category=by_cat.to_dict("records"), by_sku=by_sku.to_dict("records"), intervals=intervals,
                skus_where_model_beats_naive=int((by_sku.wape_model < by_sku.wape_naive).sum()), skus_total=int(len(by_sku)))


# ----------------------------------------------------------------------------- final forecast
def final_forecast(wk: pd.DataFrame, metrics: dict):
    train = training_set(wk, FORECAST_ORIGIN)
    model = fit(train)
    cats = {c: train[c].astype("category").cat.categories for c in CAT_COLS}
    fc = make_rows(wk, FORECAST_ORIGIN, need_target=True)
    fc["forecast"] = predict(model, fc, cats)
    iv = metrics["intervals"]
    fc["lo80"] = [max(0.0, f * (1 + iv[int(h)]["q10"])) for f, h in zip(fc.forecast, fc.h)]
    fc["hi80"] = [f * (1 + iv[int(h)]["q90"]) for f, h in zip(fc.forecast, fc.h)]
    fc["sigma"] = [f * iv[int(h)]["rel_sigma"] for f, h in zip(fc.forecast, fc.h)]
    fc = fc.rename(columns={"y": "actual"})
    cols = ["sku_id", "category", "week_end", "h", "forecast", "lo80", "hi80", "sigma", "naive", "actual",
            "promo_days_target", "unit_price"]
    fc = fc[cols].sort_values(["sku_id", "h"])
    fc["week_end"] = fc["week_end"].dt.date
    fc.to_csv(OUT / "forecast.csv", index=False)
    # true out-of-sample check on the December holdout weeks
    ho = fc.dropna(subset=["actual"])
    holdout = dict(weeks=int(ho.week_end.nunique()), wape_model=wape(ho.actual, ho.forecast),
                   wape_naive=wape(ho.actual, ho.naive), bias_model=bias(ho.actual, ho.forecast),
                   coverage80=float(((ho.actual >= ho.lo80) & (ho.actual <= ho.hi80)).mean())) if len(ho) else {}
    imp = pd.DataFrame({"feature": FEATURES, "gain": model.booster_.feature_importance("gain")})
    imp["share"] = imp["gain"] / imp["gain"].sum()
    imp.sort_values("gain", ascending=False).to_csv(OUT / "feature_importance.csv", index=False)
    joblib.dump(dict(model=model, cats=cats, features=FEATURES, origin=str(FORECAST_ORIGIN.date())), OUT / "model_lgbm.pkl")
    return holdout


def main():
    wk = load_weekly()
    folds, res = backtest(wk)
    res.to_csv(OUT / "backtest_folds.csv", index=False)
    folds.to_csv(OUT / "backtest_predictions.csv", index=False)
    metrics = summarise(folds, res)
    metrics["holdout_dec2025"] = final_forecast(wk, metrics)
    metrics["forecast_origin"] = str(FORECAST_ORIGIN.date())
    (OUT / "backtest_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    o = metrics["overall"]
    print(f"\nROLLING-ORIGIN BACKTEST ({N_FOLDS} folds × {HORIZON} weeks, {len(folds)} sku-weeks)")
    print(f"  WAPE  model {o['wape_model']:.3f}  vs  seasonal-naive {o['wape_naive']:.3f}  → {o['improvement_pct']:.1f}% better")
    print(f"  Bias  model {o['bias_model']:+.3f}  vs  naive {o['bias_naive']:+.3f}")
    print(f"  Model beats naive on {metrics['skus_where_model_beats_naive']}/{metrics['skus_total']} SKUs")
    print("  Dec-2025 holdout:", metrics["holdout_dec2025"])


if __name__ == "__main__":
    main()
