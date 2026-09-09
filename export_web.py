"""
Export one compact JSON bundle that powers the static dashboard (web/) and the
/api/score Netlify function.  Run: python src/export_web.py
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC, OUT, WEB = ROOT / "data/processed", ROOT / "outputs", ROOT / "web/data"
WEB.mkdir(parents=True, exist_ok=True)


def main():
    wk = pd.read_csv(PROC / "sales_weekly.csv", parse_dates=["week_end"])
    fc = pd.read_csv(OUT / "forecast.csv")
    risk = pd.read_csv(OUT / "risk_scores.csv")
    met = json.loads((OUT / "backtest_metrics.json").read_text())
    rs = json.loads((OUT / "risk_summary.json").read_text())
    dq = json.loads((PROC / "data_quality_log.json").read_text())
    eda = json.loads((OUT / "eda_summary.json").read_text())
    imp = pd.read_csv(OUT / "feature_importance.csv")
    hist_weeks = wk[~wk.is_holdout].week_end.drop_duplicates().sort_values().iloc[-26:]
    skus = []
    for _, r in risk.iterrows():
        f = fc[fc.sku_id == r.sku_id].sort_values("h")
        h = wk[(wk.sku_id == r.sku_id) & (wk.week_end.isin(hist_weeks))].sort_values("week_end")
        d = {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in r.items()}
        d["forecast"] = dict(weeks=f.week_end.astype(str).tolist(), mean=f.forecast.round(1).tolist(),
                             lo80=f.lo80.round(1).tolist(), hi80=f.hi80.round(1).tolist(), sigma=f.sigma.round(2).tolist(),
                             naive=f.naive.round(1).tolist(), actual=[None if pd.isna(a) else float(a) for a in f.actual])
        d["history"] = dict(weeks=h.week_end.dt.strftime("%Y-%m-%d").tolist(), units=h.units.astype(int).tolist())
        skus.append(d)
    totals = wk[~wk.is_holdout].groupby("week_end").units.sum()
    bundle = dict(
        meta=dict(client="NorthBay Living", origin=met["forecast_origin"], horizon_weeks=met["horizon_weeks"],
                  generated=str(pd.Timestamp.today().date()), skus=len(skus)),
        metrics=dict(overall=met["overall"], holdout=met["holdout_dec2025"], by_horizon=met["by_horizon"],
                     by_fold=met["by_fold"], by_category=met["by_category"], n_folds=met["n_folds"],
                     skus_model_beats_naive=met["skus_where_model_beats_naive"]),
        risk_summary=rs, eda=eda, data_quality=dq,
        feature_importance=imp.head(10).round(4).to_dict("records"),
        total_history=dict(weeks=totals.index.strftime("%Y-%m-%d").tolist(), units=totals.astype(int).tolist()),
        categories=sorted(risk.category.unique().tolist()),
        skus=skus,
    )
    (WEB / "dashboard.json").write_text(json.dumps(bundle, separators=(",", ":"), default=float))
    print(f"web/data/dashboard.json written ({(WEB / 'dashboard.json').stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
