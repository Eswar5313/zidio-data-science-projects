"""
Project FORESIGHT — D5 Planning dashboard (Streamlit edition)
Run locally:  streamlit run app/streamlit_app.py
Deploy:       Streamlit Community Cloud → main file app/streamlit_app.py
The same content is also published as a static site (web/) at the Netlify URL in the README.
"""
import json
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from risk import score_sku  # noqa: E402

st.set_page_config(page_title="FORESIGHT · NorthBay Living", page_icon="📦", layout="wide")
QC = {"Reorder now": "#e34948", "Markdown / clear": "#4a3aa7", "Watch / volatile": "#eda100", "Healthy": "#008300"}


def inr(v):
    return f"₹{v/1e7:.2f} Cr" if v >= 1e7 else f"₹{v/1e5:.1f} L" if v >= 1e5 else f"₹{v:,.0f}"


@st.cache_data(show_spinner="Loading forecast, risk scores and inventory…")
def load():
    out = ROOT / "outputs"
    if not (out / "risk_scores.csv").exists():
        return None
    return dict(risk=pd.read_csv(out / "risk_scores.csv"), fc=pd.read_csv(out / "forecast.csv", parse_dates=["week_end"]),
                wk=pd.read_csv(ROOT / "data/processed/sales_weekly.csv", parse_dates=["week_end"]),
                met=json.loads((out / "backtest_metrics.json").read_text()),
                dq=json.loads((ROOT / "data/processed/data_quality_log.json").read_text()))


D = load()
st.title("FORESIGHT · Demand & Inventory Intelligence")
if D is None:
    st.warning("No model outputs found. Run `python src/run_all.py` first — it regenerates everything from the raw extracts in ~1 minute.")
    st.stop()
risk, fc, wk, met = D["risk"], D["fc"], D["wk"], D["met"]
o = met["overall"]
st.caption(f"NorthBay Living · forecast origin {met['forecast_origin']} · 8-week horizon · backtest WAPE {o['wape_model']:.1%} vs seasonal-naive {o['wape_naive']:.1%}")

with st.sidebar:
    st.header("Filters")
    cats = st.multiselect("Category", sorted(risk.category.unique()), default=sorted(risk.category.unique()))
    quads = st.multiselect("Quadrant", list(QC), default=list(QC))
    q = st.text_input("Search SKU / product").strip().lower()
    st.divider()
    st.markdown("**How risk is scored**\n\n*Stockout* = probability demand over the lead time exceeds on-hand (or the position is already below the reorder point).\n\n*Overstock* = weeks of cover beyond the 8-week horizon (0 at 8 wks → 1 at 16 wks).\n\nThreshold 0.5 on each axis → four actions.")

view = risk[risk.category.isin(cats) & risk.quadrant.isin(quads)]
if q:
    view = view[view.sku_id.str.lower().str.contains(q) | view.product_name.str.lower().str.contains(q)]

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Reorder & markdown", "Forecast explorer & what-if", "Model accuracy", "Data quality"])

with tab1:
    c = st.columns(5)
    c[0].metric("Sales at risk (lead time)", inr(view.sales_at_risk_inr.sum()), f"{(view.quadrant=='Reorder now').sum()} SKUs to reorder", delta_color="inverse")
    c[1].metric("Capital locked in overstock", inr(view.locked_capital_inr.sum()), f"{(view.quadrant=='Markdown / clear').sum()} markdown candidates", delta_color="inverse")
    c[2].metric("Healthy SKUs", f"{(view.quadrant=='Healthy').sum()} / {len(view)}")
    c[3].metric("Inventory on hand (at cost)", inr(view.inventory_value_inr.sum()))
    c[4].metric("Forecast WAPE (backtest)", f"{o['wape_model']:.1%}", f"{o['improvement_pct']:.0f}% better than naive")
    if view.empty:
        st.info("No SKUs match the current filters — widen the category or quadrant selection.")
    else:
        left, right = st.columns([7, 5])
        fig = go.Figure()
        for qd, col in QC.items():
            t = view[view.quadrant == qd]
            fig.add_trace(go.Scatter(x=t.overstock_score, y=t.stockout_score, mode="markers", name=f"{qd} ({len(t)})",
                                     marker=dict(color=col, size=8 + 40 * (t.value_at_stake_inr / risk.value_at_stake_inr.max()) ** .5, opacity=.75, line=dict(color="white", width=1)),
                                     text=[f"{a} · {b}<br>at stake {inr(v)}<br>{c}" for a, b, v, c in zip(t.sku_id, t.product_name, t.value_at_stake_inr, t.action)], hoverinfo="text"))
        fig.add_hline(y=.5, line_color="#9a9992"); fig.add_vline(x=.5, line_color="#9a9992")
        fig.update_layout(title="Decisioning view — stockout vs overstock (bubble = ₹ at stake)", xaxis_title="Overstock risk →", yaxis_title="Stockout risk →",
                          xaxis_range=[-.05, 1.05], yaxis_range=[-.05, 1.05], height=480, margin=dict(t=50, b=40), legend=dict(orientation="h", y=-.15))
        left.plotly_chart(fig, use_container_width=True)
        bc = view.groupby("category")[["sales_at_risk_inr", "locked_capital_inr"]].sum() / 1e5
        f2 = go.Figure([go.Bar(name="Sales at risk", x=bc.index, y=bc.sales_at_risk_inr, marker_color="#e34948"),
                        go.Bar(name="Locked capital", x=bc.index, y=bc.locked_capital_inr, marker_color="#4a3aa7")])
        f2.update_layout(title="Rupee impact by category (₹ lakh)", barmode="group", height=300, margin=dict(t=50, b=20), legend=dict(orientation="h", y=-.25))
        right.plotly_chart(f2, use_container_width=True)
        right.markdown("**Top priorities**")
        right.dataframe(view[view.quadrant != "Healthy"].nlargest(6, "value_at_stake_inr")[["sku_id", "quadrant", "action", "value_at_stake_inr"]]
                        .rename(columns={"value_at_stake_inr": "₹ at stake"}), hide_index=True, use_container_width=True)

with tab2:
    a, b = st.columns(2)
    ro = view[view.quadrant == "Reorder now"].sort_values("sales_at_risk_inr", ascending=False)
    md = view[view.quadrant == "Markdown / clear"].sort_values("locked_capital_inr", ascending=False)
    a.subheader(f"Reorder list ({len(ro)})")
    a.dataframe(ro[["sku_id", "product_name", "category", "on_hand", "days_of_stock_on_hand", "lead_time_days", "reorder_qty", "sales_at_risk_inr"]], hide_index=True, use_container_width=True) if len(ro) else a.info("Nothing to reorder in this selection.")
    b.subheader(f"Markdown / clearance list ({len(md)})")
    b.dataframe(md[["sku_id", "product_name", "category", "on_hand", "on_order", "weeks_of_cover", "units_excess", "locked_capital_inr", "margin_flag"]], hide_index=True, use_container_width=True) if len(md) else b.info("No overstock in this selection.")
    st.subheader("All SKUs")
    st.dataframe(view[["priority", "sku_id", "product_name", "category", "quadrant", "stockout_score", "overstock_score", "on_hand", "on_order", "avg_weekly_forecast", "demand_8w", "weeks_of_cover", "reorder_qty", "sales_at_risk_inr", "locked_capital_inr", "action"]], hide_index=True, use_container_width=True, height=420)
    st.download_button("Download risk table (CSV)", view.to_csv(index=False), "foresight_risk_scores.csv", "text/csv")

with tab3:
    if view.empty:
        st.info("Widen the filters to pick a SKU.")
    else:
        sku = st.selectbox("SKU", [f"{a} · {b} · {c}" for a, b, c in zip(view.sku_id, view.product_name, view.quadrant)]).split(" ")[0]
        r = risk.set_index("sku_id").loc[sku]; f = fc[fc.sku_id == sku].sort_values("h")
        h = wk[(wk.sku_id == sku) & (~wk.is_holdout)].sort_values("week_end").tail(26)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h.week_end, y=h.units, name="Actual", line=dict(color="#0b0b0b", width=2)))
        fig.add_trace(go.Scatter(x=pd.concat([f.week_end, f.week_end[::-1]]), y=pd.concat([f.hi80, f.lo80[::-1]]), fill="toself", fillcolor="rgba(42,120,214,.18)", line=dict(color="rgba(0,0,0,0)"), name="80% interval"))
        fig.add_trace(go.Scatter(x=f.week_end, y=f.forecast, name="FORESIGHT forecast", line=dict(color="#2a78d6", width=2.5)))
        fig.add_trace(go.Scatter(x=f.week_end, y=f.naive, name="Seasonal-naive", line=dict(color="#eb6834", dash="dash")))
        ho = f.dropna(subset=["actual"])
        fig.add_trace(go.Scatter(x=ho.week_end, y=ho.actual, mode="markers", name="Actual (Dec holdout)", marker=dict(color="#0b0b0b", size=8)))
        fig.update_layout(title=f"{sku} · {r.product_name} — {r.category}/{r.subcategory}", height=420, yaxis_title="units / week", legend=dict(orientation="h", y=-.2), margin=dict(t=50))
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("What-if: change the inventory position")
        c = st.columns(4)
        lt = c[0].slider("Lead time (days)", 1, 60, int(r.lead_time_days))
        oh = c[1].number_input("On hand", 0, 100000, int(r.on_hand))
        oo = c[2].number_input("On order", 0, 100000, int(r.on_order))
        ss = c[3].number_input("Safety stock", 0, 100000, int(r.safety_stock))
        w = score_sku(oh, oo, lt, ss, f.forecast.values, f.sigma.values, r.list_price, r.unit_cost)
        m = st.columns(6)
        m[0].metric("Quadrant", w["quadrant"]); m[1].metric("Stockout risk", w["stockout_score"]); m[2].metric("Overstock risk", w["overstock_score"])
        m[3].metric("Weeks of cover", w["weeks_of_cover"]); m[4].metric("Suggested order", w["reorder_qty"]); m[5].metric("₹ at stake", inr(w["sales_at_risk_inr"] + w["locked_capital_inr"]))
        st.info(f"Over a {lt}-day lead time we expect to sell **{w['demand_lead_time']}** units against **{oh}** on hand → P(stockout before the open order lands) = **{w['p_stockout_lead_time']:.0%}**; P(position below reorder point) = **{w['p_below_reorder_point']:.0%}**. → **{w['action']}**")

with tab4:
    c = st.columns(5)
    hd = met["holdout_dec2025"]
    c[0].metric("Model WAPE", f"{o['wape_model']:.1%}"); c[1].metric("Seasonal-naive WAPE", f"{o['wape_naive']:.1%}"); c[2].metric("Improvement", f"{o['improvement_pct']:.1f}%")
    c[3].metric("Bias", f"{o['bias_model']:+.1%}"); c[4].metric("Dec-2025 holdout WAPE", f"{hd['wape_model']:.1%}", f"80% interval covered {hd['coverage80']:.0%}")
    a, b = st.columns(2)
    bf, bh = pd.DataFrame(met["by_fold"]), pd.DataFrame(met["by_horizon"])
    f1 = go.Figure([go.Scatter(x=bf.origin, y=bf.wape_model * 100, name="LightGBM", line=dict(color="#2a78d6")), go.Scatter(x=bf.origin, y=bf.wape_naive * 100, name="Seasonal-naive", line=dict(color="#9a9992", dash="dash"))])
    f1.update_layout(title="Rolling-origin backtest — WAPE % by fold origin", yaxis_range=[0, bf.wape_naive.max() * 125], height=320)
    a.plotly_chart(f1, use_container_width=True)
    f2 = go.Figure([go.Scatter(x=bh.h, y=bh.wape_model * 100, name="LightGBM", line=dict(color="#2a78d6")), go.Scatter(x=bh.h, y=bh.wape_naive * 100, name="Seasonal-naive", line=dict(color="#9a9992", dash="dash"))])
    f2.update_layout(title="WAPE % by weeks ahead", yaxis_range=[0, bh.wape_naive.max() * 125], height=320, xaxis_title="weeks ahead")
    b.plotly_chart(f2, use_container_width=True)
    imp = pd.read_csv(ROOT / "outputs/feature_importance.csv").head(12)
    st.plotly_chart(go.Figure(go.Bar(x=imp.share * 100, y=imp.feature, orientation="h", marker_color="#2a78d6")).update_layout(title="Feature importance (share of gain %)", height=380, yaxis=dict(autorange="reversed")), use_container_width=True)
    st.markdown("**Honest reading.** Six rolling origins in 2025, each retrained only on earlier data and scored on the next 8 weeks. Promotions/holidays of the target week are used because they are planned. Limitations: two years of history → one prior seasonal cycle; monthly inventory snapshots; intervals assume backtest-like errors. Retrain monthly; if WAPE drifts above the seasonal-naive line, revert to the baseline and investigate.")

with tab5:
    dq = D["dq"]
    s = dq["summary"]
    st.markdown(f"**{s['skus']}** SKUs × **{s['weeks']}** complete weeks ({s['first_week']} → {s['last_week']}) · {s['total_units']:,} units · {inr(s['total_revenue'])} revenue · **{s['unmapped_inventory_skus']}** inventory SKUs with no sales history ({inr(s['unmapped_inventory_value_latest'])} of stock) raised as a finding.")
    st.dataframe(pd.DataFrame(dq["issues"]), hide_index=True, use_container_width=True, height=560)
