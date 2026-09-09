"""
Project FORESIGHT — D2 EDA figures + model/risk figures for the reports
Run: python src/eda.py   → reports/figures/*.png + outputs/eda_summary.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

ROOT = Path(__file__).resolve().parents[1]
PROC, OUT, FIG = ROOT / "data/processed", ROOT / "outputs", ROOT / "reports/figures"
FIG.mkdir(parents=True, exist_ok=True)

# Validated categorical palette (fixed order) + reserved status colours
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
Q = {"Reorder now": "#e34948", "Markdown / clear": "#4a3aa7", "Watch / volatile": "#eda100", "Healthy": "#008300"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "axes.edgecolor": "#c3c2b7",
                     "axes.titleweight": "bold", "axes.titlesize": 11, "figure.dpi": 150, "savefig.bbox": "tight",
                     "savefig.facecolor": "white"})


def lakh(x, _=None):
    return f"₹{x/1e5:.0f}L" if abs(x) < 1e7 else f"₹{x/1e7:.1f}Cr"


def save(fig, name):
    fig.savefig(FIG / name); plt.close(fig)


def main():
    wk = pd.read_csv(PROC / "sales_weekly.csv", parse_dates=["week_end"])
    daily = pd.read_csv(PROC / "sales_daily_clean.csv", parse_dates=["date"])
    cal = pd.read_csv(ROOT / "data/raw/calendar.csv", parse_dates=["date"])
    sku = pd.read_csv(PROC / "sku_dim.csv")
    fc = pd.read_csv(OUT / "forecast.csv", parse_dates=["week_end"])
    risk = pd.read_csv(OUT / "risk_scores.csv")
    met = json.loads((OUT / "backtest_metrics.json").read_text())
    imp = pd.read_csv(OUT / "feature_importance.csv")
    summary = {}

    # 1. Total weekly demand — seasonality
    tot = wk.groupby("week_end")["units"].sum()
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(tot.index, tot.values, color=C[0], lw=2)
    for yr in [2024, 2025]:
        pk = tot[tot.index.year == yr]
        ax.annotate(f"peak {pk.idxmax():%b-%y}: {pk.max():,}", (pk.idxmax(), pk.max()), xytext=(30, -4), textcoords="offset points", ha="left", fontsize=8, color="#52514e")
        ax.annotate(f"trough {pk.idxmin():%b-%y}: {pk.min():,}", (pk.idxmin(), pk.min()), xytext=(8, 4), textcoords="offset points", ha="left", fontsize=8, color="#52514e")
    ax.set_title("Total weekly demand, all 50 SKUs — a clear annual cycle repeats in both years"); ax.set_ylim(tot.min()*0.9, tot.max()*1.06)
    ax.set_ylabel("Units / week"); ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    save(fig, "01_weekly_demand.png")
    m = tot.groupby(tot.index.month).mean()
    summary["peak_month"], summary["trough_month"] = int(m.idxmax()), int(m.idxmin())
    summary["peak_vs_trough_pct"] = float(100 * (m.max() / m.min() - 1))

    # 2. Monthly seasonality index
    idx = m / m.mean()
    fig, ax = plt.subplots(figsize=(6.5, 3))
    cols = [C[0] if v >= 1 else "#9a9992" for v in idx]
    ax.bar(idx.index, idx.values, color=cols, width=0.7)
    ax.axhline(1, color="#52514e", lw=1, ls="--")
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_title("Seasonality index by month (1.0 = average week)"); ax.set_ylim(0.6, 1.4)
    for i, v in idx.items():
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, color="#52514e")
    save(fig, "02_seasonality_index.png")

    # 3. Drivers: weekday, promo, holiday
    d = daily.merge(cal[["date", "day_of_week", "is_holiday"]], on="date")
    dow = d.groupby("day_of_week")["units_sold"].mean().reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    promo = d.groupby("promo_flag")["units_sold"].mean()
    hol = d.groupby("is_holiday")["units_sold"].mean()
    fig, axs = plt.subplots(1, 3, figsize=(10, 3), gridspec_kw=dict(width_ratios=[3, 1.2, 1.2]))
    axs[0].bar(range(7), dow.values, color=[C[0] if i >= 5 else "#9a9992" for i in range(7)], width=0.7)
    axs[0].set_xticks(range(7)); axs[0].set_xticklabels([x[:3] for x in dow.index]); axs[0].set_title("Avg units per SKU-day by weekday")
    axs[1].bar(["No promo", "Promo"], promo.values, color=["#9a9992", C[1]], width=0.6); axs[1].set_title(f"Promotion lift: +{100*(promo[1]/promo[0]-1):.0f}%")
    axs[2].bar(["Normal", "Holiday"], hol.values, color=["#9a9992", C[3]], width=0.6); axs[2].set_title(f"Holiday effect: {100*(hol[1]/hol[0]-1):+.0f}%")
    for a in axs:
        for p in a.patches:
            a.text(p.get_x() + p.get_width() / 2, p.get_height() + 0.2, f"{p.get_height():.1f}", ha="center", fontsize=8, color="#52514e")
    save(fig, "03_demand_drivers.png")
    summary.update(weekend_lift_pct=float(100 * (dow[["Saturday", "Sunday"]].mean() / dow[:5].mean() - 1)),
                   promo_lift_pct=float(100 * (promo[1] / promo[0] - 1)), holiday_effect_pct=float(100 * (hol[1] / hol[0] - 1)))

    # 4. Top / bottom movers by revenue, coloured by category
    rev = wk.groupby(["sku_id", "category"])["revenue"].sum().reset_index().sort_values("revenue", ascending=False)
    cats = sorted(rev.category.unique()); cmap = {c: C[i] for i, c in enumerate(cats)}
    top, bot = rev.head(10), rev.tail(10)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, t, ttl in [(axs[0], top, "Top 10 SKUs by revenue (2024–25)"), (axs[1], bot, "Bottom 10 SKUs by revenue — dead-stock candidates")]:
        a.barh(t.sku_id[::-1], t.revenue[::-1] / 1e7, color=[cmap[c] for c in t.category[::-1]])
        a.set_title(ttl); a.set_xlabel("Revenue (₹ crore)"); a.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[c]) for c in cats]
    fig.legend(handles, cats, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=8)
    save(fig, "04_top_bottom_movers.png")
    summary["top10_revenue_share_pct"] = float(100 * top.revenue.sum() / rev.revenue.sum())
    summary["bottom10_revenue_share_pct"] = float(100 * bot.revenue.sum() / rev.revenue.sum())

    # 5. Category mix (units + revenue)
    cm = wk.groupby("category").agg(units=("units", "sum"), revenue=("revenue", "sum"))
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    x = np.arange(len(cm)); w = 0.38
    ax.bar(x - w / 2, 100 * cm.units / cm.units.sum(), w, color=C[0], label="Share of units")
    ax.bar(x + w / 2, 100 * cm.revenue / cm.revenue.sum(), w, color=C[1], label="Share of revenue")
    ax.set_xticks(x); ax.set_xticklabels(cm.index); ax.yaxis.set_major_formatter(mtick.PercentFormatter()); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Category mix — units vs revenue share")
    save(fig, "05_category_mix.png")

    # 6. Backtest: WAPE by fold and by horizon
    bf, bh = pd.DataFrame(met["by_fold"]), pd.DataFrame(met["by_horizon"]); bf["origin"] = pd.to_datetime(bf.origin).dt.strftime("%d %b")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    axs[0].plot(bf.origin, 100 * bf.wape_naive, "o--", color="#9a9992", lw=2, label="Seasonal-naive baseline")
    axs[0].plot(bf.origin, 100 * bf.wape_model, "o-", color=C[0], lw=2, label="LightGBM model")
    axs[0].set_title("Rolling-origin backtest — WAPE by fold origin, 2025"); axs[0].set_ylabel("WAPE %"); axs[0].tick_params(axis="x", labelsize=8); axs[0].legend(frameon=False, fontsize=8)
    axs[0].set_ylim(0, 100 * bf.wape_naive.max() * 1.25)
    axs[1].plot(bh.h, 100 * bh.wape_naive, "o--", color="#9a9992", lw=2, label="Seasonal-naive")
    axs[1].plot(bh.h, 100 * bh.wape_model, "o-", color=C[0], lw=2, label="LightGBM")
    axs[1].set_title("WAPE by forecast horizon (weeks ahead)"); axs[1].set_xlabel("Weeks ahead"); axs[1].set_ylim(0, 100 * bh.wape_naive.max() * 1.25); axs[1].legend(frameon=False, fontsize=8)
    save(fig, "06_backtest_wape.png")

    # 7. Forecast example — top revenue SKU, actual + baseline + model + 80% interval
    s = top.iloc[0].sku_id
    hist = wk[wk.sku_id == s].set_index("week_end")["units"].iloc[-30:]
    f = fc[fc.sku_id == s].sort_values("h")
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(hist.index, hist.values, color="#0b0b0b", lw=1.8, label="Actual demand")
    ax.fill_between(f.week_end, f.lo80, f.hi80, color=C[0], alpha=0.18, label="80% interval")
    ax.plot(f.week_end, f.forecast, color=C[0], lw=2.2, label="FORESIGHT forecast")
    ax.plot(f.week_end, f.naive, color=C[1], lw=1.6, ls="--", label="Seasonal-naive baseline")
    ho = f.dropna(subset=["actual"])
    ax.plot(ho.week_end, ho.actual, "o", color="#0b0b0b", ms=5, label="Actual (Dec holdout)")
    ax.axvline(pd.Timestamp(met["forecast_origin"]), color="#9a9992", ls=":", lw=1.2)
    ax.text(pd.Timestamp(met["forecast_origin"]), ax.get_ylim()[1] * 0.98, " forecast origin →", fontsize=8, color="#52514e", va="top")
    ax.set_title(f"{s} — 8-week forecast vs seasonal-naive, with 80% interval (top revenue SKU)"); ax.set_ylabel("Units / week"); ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left")
    save(fig, "07_forecast_example.png")

    # 8. Decisioning grid — stockout vs overstock, bubble = value at stake
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.axvspan(0.5, 1.02, ymin=0, ymax=0.5, color="#4a3aa7", alpha=0.06); ax.axhspan(0.5, 1.02, xmin=0, xmax=0.49, color="#e34948", alpha=0.06)
    ax.axvspan(0.5, 1.02, ymin=0.5, ymax=1, color="#eda100", alpha=0.06); ax.axhspan(0, 0.5, xmin=0, xmax=0.49, color="#008300", alpha=0.05)
    rj = risk.copy(); rng = np.random.default_rng(0)
    rj["x"] = (rj.overstock_score + rng.uniform(-0.02, 0.02, len(rj))).clip(0, 1); rj["y"] = (rj.stockout_score + rng.uniform(-0.02, 0.02, len(rj))).clip(0, 1)
    size = 40 + 500 * rj.value_at_stake_inr / rj.value_at_stake_inr.max()
    for q, col in Q.items():
        t = rj[rj.quadrant == q]
        ax.scatter(t.x, t.y, s=size[t.index], color=col, alpha=0.75, edgecolor="white", lw=1, label=f"{q} ({len(t)})")
    for _, r in rj.nlargest(3, "value_at_stake_inr").iterrows():
        ax.annotate(r.sku_id, (r.x, r.y), xytext=(6, 4), textcoords="offset points", fontsize=7.5, color="#0b0b0b")
    ax.axhline(0.5, color="#9a9992", lw=1); ax.axvline(0.5, color="#9a9992", lw=1)
    ax.text(0.02, 0.96, "REORDER NOW", color="#e34948", fontweight="bold", fontsize=9); ax.text(0.62, 0.96, "WATCH / VOLATILE", color="#b07800", fontweight="bold", fontsize=9)
    ax.text(0.02, 0.02, "HEALTHY", color="#008300", fontweight="bold", fontsize=9); ax.text(0.62, 0.02, "MARKDOWN / CLEAR", color="#4a3aa7", fontweight="bold", fontsize=9)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_xlabel("Overstock risk →"); ax.set_ylabel("Stockout risk →")
    ax.set_title("Decisioning view — every SKU on the stockout vs overstock grid (bubble = ₹ at stake)")
    ax.legend(frameon=False, fontsize=8, loc="center right"); ax.grid(False)
    save(fig, "08_decision_grid.png")

    # 9. Rupee impact by category
    bc = risk.groupby("category")[["sales_at_risk_inr", "locked_capital_inr"]].sum()
    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    x = np.arange(len(bc)); w = 0.38
    ax.bar(x - w / 2, bc.sales_at_risk_inr / 1e5, w, color="#e34948", label="Sales at risk (stockout)")
    ax.bar(x + w / 2, bc.locked_capital_inr / 1e5, w, color="#4a3aa7", label="Capital locked (overstock)")
    ax.set_xticks(x); ax.set_xticklabels(bc.index); ax.set_ylabel("₹ lakh"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Where the money is — rupee impact by category")
    save(fig, "09_impact_by_category.png")

    # 10. Feature importance
    t = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.barh(t.feature, 100 * t.share, color=C[0]); ax.set_xlabel("Share of model gain (%)"); ax.grid(axis="y", visible=False)
    ax.set_title("What drives the forecast — top 12 features by gain")
    save(fig, "10_feature_importance.png")

    # 11. Weeks of cover distribution
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.hist(risk.weeks_of_cover.clip(upper=30), bins=30, color=C[0], edgecolor="white")
    ax.axvline(8, color="#e34948", ls="--", lw=1.2); ax.text(8.2, ax.get_ylim()[1] * 0.9, "8-wk horizon", color="#e34948", fontsize=8)
    ax.axvline(16, color="#4a3aa7", ls="--", lw=1.2); ax.text(16.2, ax.get_ylim()[1] * 0.9, "16 wks = full overstock", color="#4a3aa7", fontsize=8)
    ax.set_xlabel("Weeks of cover (on-hand + on-order ÷ weekly forecast), capped at 30"); ax.set_ylabel("SKUs")
    ax.set_title("Inventory cover is uneven: a cluster under 3 weeks and a tail beyond 16")
    save(fig, "11_weeks_of_cover.png")

    dead = rev.tail(5).sku_id.tolist()
    summary.update(dead_stock_candidates=dead, skus=int(wk.sku_id.nunique()),
                   n_negative_margin=int((sku.margin_flag == "negative").sum()),
                   negative_margin_revenue_share_pct=float(100 * rev[rev.sku_id.isin(sku[sku.margin_flag == "negative"].sku_id)].revenue.sum() / rev.revenue.sum()),
                   top_sku=s)
    (OUT / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
