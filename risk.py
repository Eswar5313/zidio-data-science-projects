"""
Project FORESIGHT — D4 Risk scoring & decisioning
==================================================
Turns the 8-week forecast + the latest inventory position into a stockout score,
an overstock score, a recommended action and the rupee value at stake — for every SKU.
Everything is a transparent formula (no black box):

  Stockout score  = max( P[demand during lead time > on-hand],                 (stock runs out before the open order lands)
                         P[demand during lead time + safety stock > on-hand + on-order] )   (inventory position already below reorder point)
                    using the forecast mean and its backtest-calibrated sigma (normal approx.)
  Overstock score = clip( (weeks_of_cover - 8) / 8 , 0, 1 )
                    weeks_of_cover = (on-hand + on-order) / avg weekly forecast
                    → 0 at ≤8 weeks cover, 0.5 at 12 weeks, 1 at ≥16 weeks
  Sales at risk   = units we cannot serve (best estimate) × selling price
  Locked capital  = units beyond 8-week demand × unit cost
  Reorder qty     = demand over (lead time + 4-week review) + safety stock − (on-hand + on-order)

Quadrants (threshold 0.5 on each axis) map to the actions in the brief, Section 08.

Run:  python src/risk.py
Out:  outputs/risk_scores.csv, outputs/risk_summary.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
HORIZON = 8
REVIEW_WEEKS = 4
THRESH = 0.5


def score_sku(oh, oo, lead_days, safety, weekly_fc: np.ndarray, weekly_sigma: np.ndarray,
              list_price: float, unit_cost: float) -> dict:
    """Pure function so the Streamlit what-if control and the API can reuse it."""
    lt_w = max(lead_days, 0) / 7.0
    full, frac = int(np.floor(lt_w)), lt_w - np.floor(lt_w)
    fc = np.asarray(weekly_fc, float); sg = np.asarray(weekly_sigma, float)
    # demand over lead time (interpolate the partial week)
    mu_lt = fc[:full].sum() + (fc[full] * frac if full < len(fc) else fc[-1] * frac)
    sd_lt = sg[:full].sum() + (sg[full] * frac if full < len(sg) else sg[-1] * frac)
    mu_h, sd_h = fc.sum(), sg.sum()
    p_lt = float(1 - norm.cdf(oh, mu_lt, max(sd_lt, 1e-6)))
    p_rop = float(1 - norm.cdf(oh + oo - safety, mu_lt, max(sd_lt, 1e-6)))   # P(position < reorder point)
    p_h = float(1 - norm.cdf(oh + oo, mu_h, max(sd_h, 1e-6)))                 # informational: horizon exposure
    stockout = max(p_lt, p_rop)
    avg_w = max(fc.mean(), 1e-6)
    cover = (oh + oo) / avg_w
    overstock = float(np.clip((cover - HORIZON) / HORIZON, 0, 1))
    short_units = max(0.0, mu_lt - oh, mu_lt + safety - oh - oo)
    weeks_to_reorder = max(0.0, (oh + oo - safety) / avg_w - lt_w)   # when inventory position hits the reorder point
    excess_units = max(0.0, oh + oo - mu_h)
    review_span = min(len(fc), int(np.ceil(lt_w + REVIEW_WEEKS)))
    reorder = max(0.0, fc[:review_span].sum() + safety - oh - oo)
    if stockout >= THRESH and overstock < THRESH:
        quad, action = "Reorder now", f"Raise a replenishment order for ~{int(np.ceil(reorder))} units before stock runs out"
    elif overstock >= THRESH and stockout < THRESH:
        quad, action = "Markdown / clear", f"Promote or discount — {cover:.0f} weeks of cover vs 8-week horizon"
    elif stockout >= THRESH and overstock >= THRESH:
        quad, action = "Watch / volatile", "Investigate — demand is erratic; review manually before ordering"
    else:
        quad, action = "Healthy", "No action needed; leave as is"
    return dict(stockout_score=round(stockout, 3), overstock_score=round(overstock, 3),
                p_stockout_lead_time=round(p_lt, 3), p_below_reorder_point=round(p_rop, 3), p_stockout_horizon=round(p_h, 3),
                weeks_to_reorder=round(weeks_to_reorder, 1),
                demand_lead_time=round(mu_lt, 1), demand_8w=round(mu_h, 1), weeks_of_cover=round(cover, 1),
                units_short=round(short_units, 1), units_excess=round(excess_units, 1),
                sales_at_risk_inr=round(short_units * list_price, 0), locked_capital_inr=round(excess_units * unit_cost, 0),
                reorder_qty=int(np.ceil(reorder)), quadrant=quad, action=action,
                days_of_stock_on_hand=round(7 * oh / avg_w, 1))


def main():
    fc = pd.read_csv(OUT / "forecast.csv")
    inv = pd.read_csv(PROC / "inventory_latest.csv").set_index("sku_id")
    sku = pd.read_csv(PROC / "sku_dim.csv").set_index("sku_id")
    rows = []
    for s, g in fc.sort_values(["sku_id", "h"]).groupby("sku_id"):
        i, m = inv.loc[s], sku.loc[s]
        r = score_sku(i.on_hand_units, i.on_order_units, i.lead_time_days, i.safety_stock,
                      g.forecast.values, g.sigma.values, m.list_price, m.unit_cost)
        r.update(sku_id=s, product_name=m.product_name, category=m.category, subcategory=m.subcategory,
                 on_hand=int(i.on_hand_units), on_order=int(i.on_order_units), lead_time_days=int(i.lead_time_days),
                 safety_stock=int(i.safety_stock), reorder_point=int(i.reorder_point),
                 list_price=float(m.list_price), unit_cost=float(m.unit_cost), margin_flag=m.margin_flag,
                 avg_weekly_forecast=round(g.forecast.mean(), 1), inventory_value_inr=round(i.on_hand_units * m.unit_cost, 0))
        r["value_at_stake_inr"] = r["sales_at_risk_inr"] + r["locked_capital_inr"]
        rows.append(r)
    df = pd.DataFrame(rows)
    df["priority"] = df["value_at_stake_inr"].rank(ascending=False, method="first").astype(int)
    df = df.sort_values("priority")
    df.to_csv(OUT / "risk_scores.csv", index=False)
    summary = dict(
        skus=len(df), quadrant_counts=df["quadrant"].value_counts().to_dict(),
        sales_at_risk_inr=float(df.sales_at_risk_inr.sum()), locked_capital_inr=float(df.locked_capital_inr.sum()),
        inventory_value_inr=float(df.inventory_value_inr.sum()),
        reorder_list=df[df.quadrant == "Reorder now"][["sku_id", "product_name", "category", "reorder_qty", "sales_at_risk_inr", "days_of_stock_on_hand"]].to_dict("records"),
        markdown_list=df[df.quadrant == "Markdown / clear"][["sku_id", "product_name", "category", "units_excess", "locked_capital_inr", "weeks_of_cover"]].to_dict("records"),
        by_category=df.groupby("category").agg(skus=("sku_id", "count"), sales_at_risk_inr=("sales_at_risk_inr", "sum"),
                                               locked_capital_inr=("locked_capital_inr", "sum")).reset_index().to_dict("records"),
        thresholds=dict(quadrant=THRESH, overstock_cover_zero=HORIZON, overstock_cover_full=2 * HORIZON, review_weeks=REVIEW_WEEKS),
    )
    (OUT / "risk_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(df["quadrant"].value_counts().to_string())
    print(f"Sales at risk  ₹{summary['sales_at_risk_inr']:,.0f}   Locked capital ₹{summary['locked_capital_inr']:,.0f}")
    print(df[["sku_id", "quadrant", "stockout_score", "overstock_score", "on_hand", "on_order", "demand_8w", "weeks_of_cover", "sales_at_risk_inr", "locked_capital_inr", "reorder_qty"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
