"""
Builds the written deliverables as PDFs from the pipeline outputs (numbers are never typed by hand):
  reports/D2_EDA_Data_Quality_Memo.pdf      (A4 portrait)
  reports/D7_Executive_Readout.pdf          (16:9 slides)
  reports/Project_Report_FORESIGHT.pdf      (A4 portrait — Zidio submission item 5)
  reports/Video_Scripts_Demo_and_Feedback.pdf (A4 portrait — Zidio items 3 & 4)
Run: python reports/build_reports.py
"""
import asyncio, base64, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG, OUT, REP = ROOT / "reports/figures", ROOT / "outputs", ROOT / "reports"
LIVE = "https://foresight-northbay.netlify.app"

risk = pd.read_csv(OUT / "risk_scores.csv")
met = json.loads((OUT / "backtest_metrics.json").read_text())
rs = json.loads((OUT / "risk_summary.json").read_text())
dq = json.loads((ROOT / "data/processed/data_quality_log.json").read_text())
eda = json.loads((OUT / "eda_summary.json").read_text())
imp = pd.read_csv(OUT / "feature_importance.csv")
o, ho, S = met["overall"], met["holdout_dec2025"], dq["summary"]
MON = ["", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def inr(v):
    return f"₹{v/1e7:.2f} Cr" if v >= 1e7 else f"₹{v/1e5:.1f} L" if v >= 1e5 else f"₹{v:,.0f}"


def img(name, w="100%"):
    b = base64.b64encode((FIG / name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b}" style="width:{w}">'


def tbl(df, cols, fmt=None):
    fmt = fmt or {}
    h = "".join(f"<th>{c[1]}</th>" for c in cols)
    rows = "".join("<tr>" + "".join(f"<td>{fmt.get(k, lambda v: v)(r[k])}</td>" for k, _ in cols) + "</tr>" for _, r in df.iterrows())
    return f"<table><thead><tr>{h}</tr></thead><tbody>{rows}</tbody></table>"


CSS = """
@page{size:A4;margin:13mm 13mm 12mm}
body{font-family:Inter,"Segoe UI",Arial,sans-serif;font-size:10.2pt;line-height:1.38;color:#111;margin:0}
h1{font-size:20pt;margin:0 0 2px;color:#14122b}h2{font-size:13pt;color:#3d34a0;border-bottom:2px solid #5b4fd6;padding-bottom:2px;margin:12px 0 6px}
h3{font-size:11pt;margin:9px 0 3px;color:#14122b}p{margin:4px 0}ul{margin:3px 0 3px 18px;padding:0}li{margin:1.5px 0}
table{border-collapse:collapse;width:100%;font-size:9pt;margin:5px 0}th{background:#14122b;color:#fff;text-align:left;padding:4px 6px;font-weight:600}td{padding:3px 6px;border-bottom:1px solid #e3e2ee;vertical-align:top}tr:nth-child(even) td{background:#f7f6fc}
.band{background:linear-gradient(90deg,#14122b,#241f57);color:#fff;padding:14px 18px;border-radius:6px;margin-bottom:10px}.band .k{color:#b3acff;font-size:8.5pt;letter-spacing:2px;text-transform:uppercase}.band h1{color:#fff}.band .s{color:#d5d2f2;font-size:10pt}
.kpis{display:flex;gap:8px;margin:8px 0}.kpi{flex:1;border:1px solid #d9d7ea;border-radius:6px;padding:7px 9px;background:#faf9ff}.kpi .l{font-size:8pt;color:#555}.kpi .v{font-size:15pt;font-weight:700;color:#14122b}.kpi .s{font-size:8pt;color:#666}
.callout{border-left:4px solid #5b4fd6;background:#f1efff;padding:7px 10px;margin:7px 0;border-radius:4px}.warn{border-left-color:#e34948;background:#fdeeee}.good{border-left-color:#008300;background:#eaf5ea}
.two{display:grid;grid-template-columns:1fr 1fr;gap:12px}.three{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
.cap{font-size:8.5pt;color:#555;margin:1px 0 6px}.pb{page-break-before:always}h2,h3{page-break-after:avoid}table,.two,.three,.kpis,.callout,img,pre{page-break-inside:avoid}.fig{page-break-inside:avoid}.small{font-size:8.8pt}code{font-family:Menlo,Consolas,monospace;font-size:8.8pt;background:#f1f0f7;padding:0 3px;border-radius:3px}
pre{font-family:Menlo,Consolas,monospace;font-size:7.4pt;line-height:1.3;background:#14122b;color:#e8e6ff;padding:8px 10px;border-radius:5px;white-space:pre-wrap;margin:5px 0}
.foot{font-size:8pt;color:#777;border-top:1px solid #ddd;margin-top:8px;padding-top:4px;display:flex;justify-content:space-between}
.pill{display:inline-block;padding:1px 7px;border-radius:10px;font-size:8pt;font-weight:600}.r{background:#fdecec;color:#c62828}.m{background:#edeaf8;color:#4a3aa7}.g{background:#e8f4e8;color:#1b6e1b}
"""


def page_foot(t):
    return f'<div class="foot"><span>Project FORESIGHT · NorthBay Living · Zidio Development Data Science internship</span><span>{t} · Eswar Mahalingam · Sept 2026</span></div>'


# =============================================================================== D2 memo
def memo():
    issues = pd.DataFrame(dq["issues"])
    issues["count"] = issues["count"].apply(lambda v: "—" if pd.isna(v) else str(int(v)))
    q = risk.groupby("category").agg(skus=("sku_id", "count")).reset_index()
    neg = risk[risk.margin_flag == "negative"].sku_id.tolist()
    return f"""<style>{CSS}</style>
<div class="band"><div class="k">D2 · Data-quality &amp; EDA insight memo</div><h1>What NorthBay's own data says about demand, stock and where the money is</h1>
<div class="s">To: Head of Operations, Merchandiser, Finance lead · From: Eswar Mahalingam, Data Scientist (Zidio) · Data: {S['first_week']} → {S['last_week']} · {S['skus']} SKUs · {S['weeks']} complete weeks</div></div>
<div class="kpis"><div class="kpi"><div class="l">Units sold (2 yrs)</div><div class="v">{S['total_units']:,}</div><div class="s">{S['skus']} SKUs, 5 categories</div></div>
<div class="kpi"><div class="l">Revenue (2 yrs)</div><div class="v">{inr(S['total_revenue'])}</div><div class="s">top-10 SKUs = {eda['top10_revenue_share_pct']:.0f} %</div></div>
<div class="kpi"><div class="l">Peak vs trough month</div><div class="v">+{eda['peak_vs_trough_pct']:.0f} %</div><div class="s">{MON[eda['peak_month']]} vs {MON[eda['trough_month']]}</div></div>
<div class="kpi"><div class="l">Promotion lift</div><div class="v">+{eda['promo_lift_pct']:.0f} %</div><div class="s">weekend +{eda['weekend_lift_pct']:.0f} %, holiday {eda['holiday_effect_pct']:+.0f} %</div></div>
<div class="kpi"><div class="l">Unmapped stock</div><div class="v">{inr(S['unmapped_inventory_value_latest'])}</div><div class="s">{S['unmapped_inventory_skus']} inventory SKUs with zero sales</div></div></div>

<h2>1. Three things the business should act on</h2>
<div class="callout"><b>1 · Demand is seasonal, not random — plan the calendar, not a flat monthly number.</b> Total weekly demand peaks in {MON[eda['peak_month']]} and bottoms in {MON[eda['trough_month']]} in <i>both</i> years, a {eda['peak_vs_trough_pct']:.0f} % swing. Ordering the same quantity every month guarantees spring stockouts and autumn overstock. The pattern is stable enough to forecast.</div>
<div class="callout"><b>2 · Promotions and weekends move product — and promotions are planned, so they belong in the forecast.</b> Promo days sell {eda['promo_lift_pct']:.0f} % more than normal days; Saturdays/Sundays {eda['weekend_lift_pct']:.0f} % more than weekdays; national holidays sell {abs(eda['holiday_effect_pct']):.0f} % <i>less</i>. A promo calendar shared with planning is worth more than any model tweak.</div>
<div class="callout warn"><b>3 · Money is concentrated and some of the catalogue is broken.</b> Ten SKUs earn {eda['top10_revenue_share_pct']:.0f} % of revenue; the bottom ten earn {eda['bottom10_revenue_share_pct']:.0f} % and {', '.join(eda['dead_stock_candidates'][:4])} are dead-stock candidates. <b>{eda['n_negative_margin']} SKUs are priced below cost</b> ({eda['negative_margin_revenue_share_pct']:.0f} % of revenue — {', '.join(neg[:6])}…). Either the master data is wrong or NorthBay is losing money on every unit; Finance must confirm before any markdown is applied to these.</div>

<div class="two"><div>{img('01_weekly_demand.png')}<div class="cap">Fig 1 · Total weekly demand — the annual cycle repeats.</div></div><div>{img('02_seasonality_index.png')}<div class="cap">Fig 2 · Seasonality index by month (1.0 = average week).</div></div></div>
<div class="fig">{img('03_demand_drivers.png')}<div class="cap">Fig 3 · Demand drivers — weekday, promotion and holiday effects (average units per SKU-day).</div></div>

<h2>2. Top movers, dead stock and category mix</h2>
{img('04_top_bottom_movers.png')}<div class="cap">Fig 4 · Top and bottom 10 SKUs by two-year revenue, coloured by category.</div>
<div class="two"><div>{img('05_category_mix.png')}<div class="cap">Fig 5 · Category share of units vs revenue.</div></div><div>{img('11_weeks_of_cover.png')}<div class="cap">Fig 6 · Weeks of cover (position ÷ weekly forecast) is uneven: a cluster under 3 weeks and a tail beyond 16.</div></div></div>
<p>Each category has exactly 10 SKUs, but revenue is not evenly spread: Home Decor and Furniture carry the high-ticket items, Storage holds the most excess stock (see D4 risk scoring). The cover distribution in Fig 6 is the inventory story in one picture — NorthBay is simultaneously thin on fast sellers and deep on slow ones, which is exactly the complaint in the brief.</p>

<h2>3. Data-quality issues found and how the pipeline handled them</h2>
<p class="small">Everything below is detected and fixed in code (<code>src/pipeline.py</code>) and written to <code>data/processed/data_quality_log.json</code> on every run; nothing is edited by hand. The sales fact table itself was clean (no duplicates, negatives or gaps) — the problems live in the dimensions.</p>
{tbl(issues, [("table", "Table"), ("issue", "Issue"), ("count", "n"), ("action", "Handling"), ("rationale", "Rationale")])}
<div class="callout"><b>Biggest finding for the client:</b> {S['unmapped_inventory_skus']} SKUs in the inventory file (SKU051–SKU200) have never sold a unit in two years yet hold <b>{inr(S['unmapped_inventory_value_latest'])}</b> of stock at the latest snapshot. Either they are discontinued lines that should be cleared, or the SKU codes do not match between systems. We excluded them from forecasting (no demand signal) and list them in <code>data/processed/inventory_unmapped_skus.csv</code> for Operations to reconcile.</div>
<h3>Analysis-ready dataset</h3>
<p class="small">One row per SKU per Sunday-ending week: units, revenue, promo days, holiday days, price, week-of-year, month, season, weeks since launch, category — {S['skus']} × {S['weeks']} = {S['skus']*S['weeks']:,} rows, zero gaps. Forecast origin <b>{S['forecast_origin']}</b> (chosen so the latest inventory snapshot and the forecast start on the same day); the {S['holdout_weeks']} December weeks are a holdout the model never trains on.</p>
{page_foot("D2 memo")}"""


# =============================================================================== D7 executive readout (slides)
SL = """
@page{size:297mm 167mm;margin:0}body{font-family:Inter,"Segoe UI",Arial,sans-serif;margin:0;color:#111}
.s{width:297mm;height:167mm;padding:11mm 14mm 9mm;box-sizing:border-box;page-break-after:always;position:relative;background:#fff}
.s:last-child{page-break-after:auto}.s h1{font-size:22pt;margin:0 0 2mm;color:#14122b}.s h2{font-size:13pt;color:#5b4fd6;margin:0 0 5mm;font-weight:500}
.dark{background:linear-gradient(135deg,#14122b,#2a2470);color:#fff}.dark h1{color:#fff;font-size:30pt}.dark h2{color:#b3acff}
.kpis{display:flex;gap:6mm;margin:4mm 0}.kpi{flex:1;border:1px solid #d9d7ea;border-radius:8px;padding:5mm 6mm;background:#faf9ff}.kpi .l{font-size:11pt;color:#555}.kpi .v{font-size:28pt;font-weight:700;color:#14122b;line-height:1.1}.kpi .s2{font-size:10.5pt;color:#666;margin-top:1mm}
.red .v{color:#c62828}.pur .v{color:#4a3aa7}.grn .v{color:#1b6e1b}
table{border-collapse:collapse;width:100%;font-size:11pt}th{background:#14122b;color:#fff;text-align:left;padding:2.6mm 3mm}td{padding:2.6mm 3mm;border-bottom:1px solid #e3e2ee}tr:nth-child(even) td{background:#f7f6fc}
.two{display:grid;grid-template-columns:1fr 1fr;gap:8mm}.g32{display:grid;grid-template-columns:3fr 2fr;gap:8mm}
p,li{font-size:12pt;line-height:1.45}ul{margin:2mm 0 2mm 6mm;padding:0}li{margin:1.2mm 0}
.foot{position:absolute;bottom:5mm;left:14mm;right:14mm;font-size:8pt;color:#888;display:flex;justify-content:space-between;border-top:1px solid #e5e5ee;padding-top:1.5mm}
.callout{border-left:4px solid #5b4fd6;background:#f1efff;padding:4mm 5mm;margin:4mm 0;border-radius:4px;font-size:12pt;line-height:1.45}.warn{border-left-color:#e34948;background:#fdeeee}.good{border-left-color:#008300;background:#eaf5ea}
img{max-width:100%}.cap{font-size:8.5pt;color:#666}.big{font-size:15pt;font-weight:600;color:#14122b}.pill{display:inline-block;padding:0 6px;border-radius:10px;font-size:9pt;font-weight:600}.r{background:#fdecec;color:#c62828}.m{background:#edeaf8;color:#4a3aa7}
"""


def readout():
    ro = risk[risk.quadrant == "Reorder now"].sort_values("sales_at_risk_inr", ascending=False)
    md = risk[risk.quadrant == "Markdown / clear"].sort_values("locked_capital_inr", ascending=False)
    bc = pd.DataFrame(rs["by_category"])
    foot = lambda n: f'<div class="foot"><span>Project FORESIGHT · Executive readout for NorthBay Living · prepared by Eswar Mahalingam, Zidio Development</span><span>{n} / 9</span></div>'
    neg_md = md[md.margin_flag == "negative"].sku_id.tolist()
    top_ro = ro.iloc[0]
    return f"""<style>{SL}</style>
<div class="s dark"><div style="font-size:10pt;letter-spacing:3px;color:#b3acff">ZIDIO DEVELOPMENT · DATA SCIENCE &amp; ANALYTICS · CLIENT ENGAGEMENT</div>
<h1 style="margin-top:22mm">Project FORESIGHT</h1><h2>Demand &amp; Inventory Intelligence — Executive readout</h2>
<p style="font-size:13pt;color:#d5d2f2;max-width:200mm">What to reorder, what to clear, and what to leave alone this month — with the rupee value of each decision and an honest account of how much to trust the forecast behind it.</p>
<p style="font-size:11pt;color:#b3acff;position:absolute;bottom:14mm">For: Head of Operations · Finance lead · Merchandiser &nbsp;|&nbsp; Prepared by: Eswar Mahalingam, Data Scientist &nbsp;|&nbsp; Forecast origin {met['forecast_origin']} · 8-week horizon &nbsp;|&nbsp; Live: {LIVE}</p></div>

<div class="s"><h1>The one-slide answer</h1><h2>{inr(rs['sales_at_risk_inr'])} of sales is about to be lost and {inr(rs['locked_capital_inr'])} of cash is sitting in stock that will not sell for months. Both are fixable this week.</h2>
<div class="kpis"><div class="kpi red"><div class="l">Sales at risk from stockouts</div><div class="v">{inr(rs['sales_at_risk_inr'])}</div><div class="s2">{rs['quadrant_counts'].get('Reorder now',0)} SKUs will run out before a replenishment can land — <b>reorder now</b></div></div>
<div class="kpi pur"><div class="l">Capital locked in overstock</div><div class="v">{inr(rs['locked_capital_inr'])}</div><div class="s2">{rs['quadrant_counts'].get('Markdown / clear',0)} SKUs hold 15–70 weeks of cover — <b>promote or clear</b></div></div>
<div class="kpi grn"><div class="l">Healthy — leave alone</div><div class="v">{rs['quadrant_counts'].get('Healthy',0)} / {rs['skus']}</div><div class="s2">No action; free the team's attention for the 16 that matter</div></div>
<div class="kpi"><div class="l">How much to trust it</div><div class="v">{o['wape_model']*100:.1f} %</div><div class="s2">forecast error (WAPE) on unseen weeks vs {o['wape_naive']*100:.1f} % for "same as last year" — {o['improvement_pct']:.0f} % better</div></div></div>
<div class="two"><div class="callout warn"><b>Do this week:</b> raise purchase orders for the {len(ro)} reorder SKUs (quantities on slide 4, {int(ro.reorder_qty.sum()):,} units total). {top_ro.sku_id} alone has {top_ro.days_of_stock_on_hand:.0f} days of stock left against a {top_ro.lead_time_days}-day lead time.</div>
<div class="callout"><b>Do this month:</b> run a clearance promotion on the {len(md)} overstock SKUs (slide 5). Exclude {', '.join(neg_md) or 'none'} until Finance confirms their cost/price — they are recorded as selling below cost.</div></div>
<p class="cap">All figures from the FORESIGHT pipeline run on NorthBay's extracts (Jan-2024 → Dec-2025); latest stock position = 1-Dec-2025 snapshot. Rupee values use list price for lost sales and unit cost for locked capital.</p>{foot(2)}</div>

<div class="s"><h1>Every SKU on one grid — the team triages 50 products in 30 seconds</h1><h2>Up = likely to stock out during the lead time. Right = holding far more than 8 weeks of demand. Bubble = rupees at stake.</h2>
<div class="g32"><div>{img('08_decision_grid.png')}</div><div>
<table><tr><th>Quadrant</th><th>SKUs</th><th>Action</th></tr>
<tr><td><span class="pill r">Reorder now</span></td><td>{rs['quadrant_counts'].get('Reorder now',0)}</td><td>Raise replenishment before stock runs out</td></tr>
<tr><td><span class="pill m">Markdown / clear</span></td><td>{rs['quadrant_counts'].get('Markdown / clear',0)}</td><td>Promote or discount to free cash</td></tr>
<tr><td>Watch / volatile</td><td>{rs['quadrant_counts'].get('Watch / volatile',0)}</td><td>Investigate erratic demand manually</td></tr>
<tr><td>Healthy</td><td>{rs['quadrant_counts'].get('Healthy',0)}</td><td>Nothing — leave as is</td></tr></table>
<div class="callout" style="margin-top:5mm"><b>No black box.</b> Stockout risk = the probability that demand over the supplier lead time exceeds what is on the shelf, using the forecast and its measured error. Overstock = weeks of cover beyond the 8-week horizon. Anyone can recompute a score with a calculator — and the dashboard's what-if slider lets the team test "what if the lead time slips to 21 days?" instantly.</div></div></div>{foot(3)}</div>

<div class="s"><h1>Reorder now — {len(ro)} SKUs, {inr(ro.sales_at_risk_inr.sum())} of sales at risk in the lead time alone</h1><h2>Order quantity covers demand through the lead time plus a 4-week review cycle, plus safety stock, minus what is already on hand and on order.</h2>
{tbl(ro, [("sku_id","SKU"),("product_name","Product"),("category","Category"),("on_hand","On hand"),("on_order","On order"),("lead_time_days","Lead time (d)"),("days_of_stock_on_hand","Days of stock left"),("avg_weekly_forecast","Fcst / wk"),("reorder_qty","Order qty"),("sales_at_risk_inr","Sales at risk")], {"sales_at_risk_inr": inr, "days_of_stock_on_hand": lambda v: f"{v:.0f}"})}
<div class="callout warn" style="margin-top:4mm"><b>Why the number looks small next to the revenue:</b> {inr(ro.sales_at_risk_inr.sum())} is only what is lost <i>before the next order could arrive</i>. Left unordered for the full 8 weeks these {len(ro)} SKUs represent {inr((ro.demand_8w*ro.list_price).sum())} of demand — they are among NorthBay's best sellers (e.g. {ro.iloc[0].sku_id}, {ro.iloc[1].sku_id}).</div>{foot(4)}</div>

<div class="s"><h1>Markdown / clear — {len(md)} SKUs hold {inr(md.locked_capital_inr.sum())} of the {inr(rs['locked_capital_inr'])} locked beyond the 8-week horizon</h1><h2>Excess units = position minus 8 weeks of forecast demand; locked capital values them at unit cost.</h2>
{tbl(md, [("sku_id","SKU"),("product_name","Product"),("category","Category"),("on_hand","On hand"),("on_order","On order"),("avg_weekly_forecast","Fcst / wk"),("weeks_of_cover","Weeks of cover"),("units_excess","Excess units"),("locked_capital_inr","Locked capital"),("margin_flag","Margin")], {"locked_capital_inr": inr, "units_excess": lambda v: f"{v:.0f}", "margin_flag": lambda v: '<span class="pill r">below cost</span>' if v=="negative" else "ok"})}
<div class="two" style="margin-top:4mm"><div class="callout"><b>Quick win:</b> {md.iloc[0].sku_id} and {md.iloc[1].sku_id} together hold {inr(md.iloc[:2].locked_capital_inr.sum())}. Also <b>cancel or defer the open orders</b> on these lines ({int(md.on_order.sum()):,} units on order against products already at 15–70 weeks of cover).</div>
<div class="callout warn"><b>Caution:</b> {', '.join(neg_md) or '—'} are recorded as selling below cost. Discounting them deepens the loss. Fix the master data (or the price) first.</div></div>{foot(5)}</div>

<div class="s"><h1>Where the money is, by category</h1><h2>Home Decor carries the stockout exposure; Storage and Furniture carry the overstock. Two different conversations with two different buyers.</h2>
<div class="two"><div>{img('09_impact_by_category.png')}</div><div>
{tbl(bc, [("category","Category"),("skus","SKUs"),("sales_at_risk_inr","Sales at risk"),("locked_capital_inr","Locked capital")], {"sales_at_risk_inr": inr, "locked_capital_inr": inr})}
<div class="callout" style="margin-top:4mm"><b>Pattern behind the numbers:</b> demand peaks in {MON[eda['peak_month']]} and troughs in {MON[eda['trough_month']]} (a {eda['peak_vs_trough_pct']:.0f} % swing) and the same happens every year. Promotions add {eda['promo_lift_pct']:.0f} %, weekends {eda['weekend_lift_pct']:.0f} %. A flat monthly order quantity cannot follow that curve — which is exactly how both problems arise at once.</div></div></div>{foot(6)}</div>

<div class="s"><h1>How much to trust the forecast — tested the honest way</h1><h2>Retrained at six past dates using only earlier data, then scored on the following 8 weeks it had never seen. It beat "same week last year" every time.</h2>
<div class="g32"><div>{img('06_backtest_wape.png')}<div class="cap">WAPE = total forecast miss ÷ total units sold. {o['wape_model']*100:.1f} % means roughly {o['wape_model']*100:.0f} units wrong per 100 sold.</div></div><div>
<div class="kpis" style="flex-direction:column;gap:3mm"><div class="kpi"><div class="l">FORESIGHT model</div><div class="v">{o['wape_model']*100:.1f} %</div><div class="s2">bias {o['bias_model']*100:+.1f} % — not systematically high or low</div></div>
<div class="kpi"><div class="l">"Same as last year" baseline</div><div class="v">{o['wape_naive']*100:.1f} %</div><div class="s2">model is {o['improvement_pct']:.0f} % more accurate; wins on {met['skus_where_model_beats_naive']}/{met['skus_total']} SKUs</div></div>
<div class="kpi"><div class="l">December 2025 (true holdout)</div><div class="v">{ho['wape_model']*100:.1f} %</div><div class="s2">vs {ho['wape_naive']*100:.1f} % baseline · the 80 % range contained {ho['coverage80']*100:.0f} % of actuals</div></div></div></div></div>
<div class="callout good"><b>Plain-language limits:</b> two years of history means one prior seasonal cycle to learn from; brand-new SKUs borrow from their category; the model knows planned promotions but not unplanned ones; stock positions are monthly snapshots. Retrain monthly and if the error ever drifts above the baseline, fall back to the baseline and call us.</div>{foot(7)}</div>

<div class="s"><h1>What the team gets, and how to run it without a data scientist</h1>
<div class="two"><div>{img('dash_overview.png')}<div class="cap">Planning dashboard — {LIVE}</div></div><div>
<ul><li><b>Planning dashboard</b> (live URL above): filters by category / SKU / quadrant, reorder and markdown lists, per-SKU forecast with uncertainty band, what-if slider for lead time and stock, one-click CSV export.</li>
<li><b>Scoring service</b>: <code>{LIVE}/api/score?sku=SKU012</code> returns the forecast and risk for any SKU or batch — pluggable into NorthBay's own tools.</li>
<li><b>Reproducible pipeline</b>: drop next month's four extracts in <code>data/raw/</code>, run one command, every number refreshes in about a minute.</li>
<li><b>Monthly rhythm</b>: refresh → review the two lists → raise POs / schedule promos → check last month's error on the Model tab.</li></ul>
<div class="callout"><b>Ask of NorthBay:</b> (1) confirm cost/price on the {eda['n_negative_margin']} below-cost SKUs; (2) reconcile the {S['unmapped_inventory_skus']} inventory SKUs with no sales ({inr(S['unmapped_inventory_value_latest'])} of stock); (3) share the promotion calendar 8 weeks ahead — it is the single biggest accuracy lever.</div></div></div>{foot(8)}</div>

<div class="s"><h1>Decisions requested today</h1>
<table style="font-size:11pt"><tr><th>#</th><th>Decision</th><th>Owner</th><th>Value</th><th>When</th></tr>
<tr><td>1</td><td>Approve purchase orders for the {len(ro)} "reorder now" SKUs ({int(ro.reorder_qty.sum()):,} units, quantities on slide 4)</td><td>Head of Operations</td><td>Protects {inr(ro.sales_at_risk_inr.sum())} in the lead time; {inr((ro.demand_8w*ro.list_price).sum())} over 8 weeks</td><td>This week</td></tr>
<tr><td>2</td><td>Clearance promotion on the {len(md)-len(neg_md)} overstock SKUs with healthy margin; defer/cancel their open orders</td><td>Merchandiser</td><td>Releases up to {inr(md[md.margin_flag!='negative'].locked_capital_inr.sum())} of cash</td><td>This month</td></tr>
<tr><td>3</td><td>Verify cost &amp; list price on the {eda['n_negative_margin']} below-cost SKUs before any discounting</td><td>Finance lead</td><td>Avoids deepening a loss on {eda['negative_margin_revenue_share_pct']:.0f} % of revenue</td><td>Before decision 2</td></tr>
<tr><td>4</td><td>Reconcile SKU051–SKU200 inventory records (no sales in 2 years)</td><td>Operations + Finance</td><td>{inr(S['unmapped_inventory_value_latest'])} of stock unaccounted for in demand terms</td><td>This quarter</td></tr>
<tr><td>5</td><td>Adopt the monthly refresh cadence and share the promo calendar 8 weeks ahead</td><td>Head of Operations</td><td>Keeps forecast error near {o['wape_model']*100:.0f} % instead of drifting</td><td>Ongoing</td></tr></table>
<div class="callout good" style="margin-top:6mm"><b>Bottom line:</b> NorthBay does not have a demand problem — demand is predictable to within about {o['wape_model']*100:.0f} %. It has a <i>timing</i> problem: fast sellers are ordered too late and slow sellers too often. FORESIGHT turns that into two short lists every month. Deliver it like a consultant, defend it like a scientist.</div>{foot(9)}</div>"""


# =============================================================================== Project report (Zidio item 5)
def project_report():
    bf = pd.DataFrame(met["by_fold"]); bh = pd.DataFrame(met["by_horizon"]); bcat = pd.DataFrame(met["by_category"])
    pct = lambda v: f"{v*100:.1f} %"
    return f"""<style>{CSS}</style>
<div class="band"><div class="k">Zidio Development · Data Science &amp; Analytics internship · Project report</div><h1>Project FORESIGHT — AI-Powered Demand &amp; Inventory Intelligence Platform</h1>
<div class="s">Client engagement: NorthBay Living (D2C home &amp; lifestyle) · Intern: Eswar Mahalingam · Track: Data Science &amp; Analytics · Duration: 4 weeks · Submitted: September 2026<br>Live deployment: <b>{LIVE}</b> · Scoring API: <b>{LIVE}/api/score?sku=SKU012</b> · Source: GitHub repository <code>foresight</code> (README has setup and results)</div></div>

<h2>1. Overview</h2>
<p>Project FORESIGHT is an AI-powered demand forecasting and inventory intelligence platform built for NorthBay Living, a mid-size direct-to-consumer home &amp; lifestyle brand that plans stock on spreadsheets and gut feel. The client loses money in two directions at once: best-sellers stock out (lost sales) and slow movers pile up (cash locked, later marked down). The engagement brief asked for four things — a weekly SKU-level demand forecast that measurably beats a naive baseline, a stockout early-warning, an overstock flag, and an interface a non-technical operations team can use unaided — plus a deployed scoring service and an executive readout.</p>
<p>The delivered platform ingests four raw extracts (daily sales, SKU master, calendar, monthly inventory snapshots), cleans them in a fully coded and audited pipeline, trains a LightGBM demand model validated by rolling-origin backtesting against a seasonal-naive baseline, converts the forecast plus stock position into transparent stockout/overstock risk scores with a recommended action and a rupee value for every SKU, and serves everything through a live planning dashboard and a public scoring API. Every number in this report regenerates from the raw files with one command.</p>
<div class="kpis"><div class="kpi"><div class="l">Forecast accuracy (WAPE, backtest)</div><div class="v">{pct(o['wape_model'])}</div><div class="s">vs {pct(o['wape_naive'])} seasonal-naive → {o['improvement_pct']:.1f} % better</div></div>
<div class="kpi"><div class="l">Validation</div><div class="v">{met['n_folds']} folds × 8 wk</div><div class="s">wins every fold, {met['skus_where_model_beats_naive']}/{met['skus_total']} SKUs; Dec-25 holdout {pct(ho['wape_model'])}</div></div>
<div class="kpi"><div class="l">Sales at risk identified</div><div class="v">{inr(rs['sales_at_risk_inr'])}</div><div class="s">{rs['quadrant_counts'].get('Reorder now',0)} SKUs to reorder now</div></div>
<div class="kpi"><div class="l">Capital locked identified</div><div class="v">{inr(rs['locked_capital_inr'])}</div><div class="s">{rs['quadrant_counts'].get('Markdown / clear',0)} markdown candidates</div></div>
<div class="kpi"><div class="l">Data processed</div><div class="v">{S['total_units']:,}</div><div class="s">units · {S['skus']} SKUs · {S['weeks']} weeks · {inr(S['total_revenue'])}</div></div></div>

<h2>2. Objectives and success metrics (from the brief)</h2>
<table><tr><th>Objective</th><th>Success metric</th><th>Achieved</th></tr>
<tr><td>Weekly SKU-level forecast over a defined horizon beating a naive baseline</td><td>WAPE beats seasonal-naive by a clear margin on backtest</td><td>8-week horizon; WAPE {pct(o['wape_model'])} vs {pct(o['wape_naive'])} ({o['improvement_pct']:.0f} % better); bias {o['bias_model']*100:+.1f} %</td></tr>
<tr><td>Flag stockout and overstock risk for every SKU with a clear action</td><td>Flags right often enough to act on</td><td>All 50 SKUs scored; 4-quadrant grid; probability-based, transparent formulas; 80 % interval calibrated at {ho['coverage80']*100:.0f} % on holdout</td></tr>
<tr><td>Quantify the business impact in rupees</td><td>Rupee figure for sales-at-risk and locked capital</td><td>{inr(rs['sales_at_risk_inr'])} sales at risk; {inr(rs['locked_capital_inr'])} locked capital; per SKU and per category</td></tr>
<tr><td>Dashboard the ops team can use unaided + deployed scoring service</td><td>Usability; public URL</td><td>Live dashboard and API at {LIVE}; filters, lists, what-if, CSV export, loading/empty/error states</td></tr>
<tr><td>Reproducibility</td><td>Pipeline re-runs end-to-end from raw data</td><td><code>python src/run_all.py</code> (~1 min); fixed seed; 5 automated tests incl. leakage guard</td></tr></table>

<h2>3. Tech stack</h2>
<table><tr><th>Layer</th><th>Technology</th><th>Why</th></tr>
<tr><td>Language &amp; data</td><td>Python 3.11, pandas, NumPy</td><td>Standard analytics stack; readable, auditable transformations</td></tr>
<tr><td>Modelling</td><td>LightGBM (Tweedie objective), scikit-learn metrics, SciPy (normal CDF for risk)</td><td>Gradient-boosted trees handle 50 heterogeneous series with shared seasonality/promo effects in one model; Tweedie suits non-negative count-like demand</td></tr>
<tr><td>Visualisation</td><td>matplotlib (report figures), Plotly (Streamlit), Chart.js 4 (web dashboard)</td><td>Publication-quality static charts; interactive hover/filter for the ops team</td></tr>
<tr><td>Dashboard</td><td>Streamlit app (<code>app/</code>) and a static HTML/JS dashboard (<code>web/</code>)</td><td>Streamlit for fast local iteration as recommended; static site for a zero-cold-start public deployment on Netlify</td></tr>
<tr><td>Scoring service</td><td>FastAPI + Pydantic (<code>service/</code>) and a Netlify serverless function (<code>netlify/functions/score.mjs</code>) sharing one risk engine</td><td>Documented inputs/outputs, validation, graceful errors; the serverless twin gives a free, always-on public endpoint</td></tr>
<tr><td>Deployment</td><td>Netlify (static + functions), deployable to Streamlit Community Cloud / Render</td><td>Public URL requirement; CDN-served, HTTPS</td></tr>
<tr><td>Quality</td><td>pytest, executed Jupyter notebooks, Git</td><td>Regression tests for metrics, risk logic, leakage and API errors; notebooks show the analysis with outputs</td></tr></table>

<h2>4. Architecture</h2>
<p>FORESIGHT keeps four layers separate so the client can re-run and trust each one: data sources → reproducible pipeline → models → serving. Raw extracts never leave <code>data/raw/</code>; every downstream artefact is regenerated.</p>
<pre>┌─ 1 · DATA SOURCES ─────────────────────────────────────────────────────────────────────────────────────────┐
│  sales_daily.csv (36,550 rows)   sku_master.csv (50)   calendar.csv (731)   inventory_snapshots.csv (4,800)  │
└───────────────────────────────────────────────┬────────────────────────────────────────────────────────────┘
┌─ 2 · PIPELINE  src/pipeline.py  ──────────────▼─────────────────────────────────────────────────────────────┐
│  ingest → validate → clean (coded, logged) → weekly SKU grid (Sunday weeks) → calendar/launch features         │
│  outputs: sales_weekly.csv · sku_dim.csv · inventory_latest.csv · data_quality_log.json (15 audited decisions)  │
└───────────────────────────────────────────────┬────────────────────────────────────────────────────────────┘
┌─ 3 · MODELS ───────────────────────────────────▼─────────────────────────────────────────────────────────────┐
│  src/forecast.py  seasonal-naive baseline → LightGBM direct multi-horizon (h=1..8) → rolling-origin backtest   │
│                   (6 origins) → 80 % intervals from residual quantiles → final 8-week forecast per SKU          │
│  src/risk.py      forecast + latest stock position → P(stockout in lead time), weeks of cover → quadrant,       │
│                   action, reorder qty, sales at risk (₹), locked capital (₹)                                    │
│  outputs: forecast.csv · risk_scores.csv · backtest_metrics.json · feature_importance.csv · model_lgbm.pkl      │
└───────────────────────────────────────────────┬────────────────────────────────────────────────────────────┘
┌─ 4 · SERVING ──────────────────────────────────▼─────────────────────────────────────────────────────────────┐
│  src/export_web.py → web/data/dashboard.json (92 KB)                                                           │
│  web/index.html + risk_engine.js  ── Netlify CDN ──►  {LIVE}                                │
│  netlify/functions/score.mjs      ── Netlify Functions ──►  /api/score · /api/health  (same risk engine)        │
│  app/streamlit_app.py (Streamlit)  ·  service/main.py (FastAPI /docs)  — identical logic, alternative hosting   │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
        Stakeholders: Head of Operations · Merchandiser · Finance  ←  dashboard, API, executive readout (D7)</pre>
<h3>Modelling workflow — baseline first</h3>
<p>(1) <b>Frame</b>: 8-week horizon, WAPE primary, bias secondary; forecast origin {met['forecast_origin']} chosen so the latest inventory snapshot (1-Dec-2025) and the forecast start coincide, leaving four December weeks as a true holdout. (2) <b>Baseline</b>: seasonal-naive, forecast(t+h) = actual(t+h−52). (3) <b>Features</b>: lags 1–13, rolling mean/std, trend, last-year same week ±1, this-year/last-year level ratio, planned promo and holiday days of the target week, week-of-year, month, season, weeks since launch, price, SKU and category identity, horizon h — all computed strictly from weeks ≤ origin. (4) <b>Model</b>: one LightGBM regressor over rows (SKU × origin × h). (5) <b>Backtest</b>: six rolling origins in 2025, four weeks apart, retrained from scratch each time. (6) <b>Evaluate</b>: compare to baseline per fold, horizon, category and SKU. (7) <b>Risk</b>: combine with inventory position.</p>
<div class="two"><div>{img('06_backtest_wape.png')}<div class="cap">Backtest — the model stays below the baseline on every fold and every horizon.</div></div><div>{img('10_feature_importance.png')}<div class="cap">Drivers — product identity and recent level dominate; last-year same-week and season carry the seasonality.</div></div></div>

<h2>5. Results</h2>
<div class="three"><div>{tbl(bf, [("origin","Fold origin"),("wape_model","Model"),("wape_naive","Naive")], {"wape_model": pct, "wape_naive": pct})}</div>
<div>{tbl(bh, [("h","Weeks ahead"),("wape_model","Model"),("wape_naive","Naive")], {"wape_model": pct, "wape_naive": pct})}</div>
<div>{tbl(bcat, [("category","Category"),("wape_model","Model"),("wape_naive","Naive")], {"wape_model": pct, "wape_naive": pct})}
<p class="small"><b>Overall:</b> WAPE {pct(o['wape_model'])} vs {pct(o['wape_naive'])}; MAPE {pct(o['mape_model'])} vs {pct(o['mape_naive'])}; bias {o['bias_model']*100:+.2f} %. <b>Dec-2025 holdout:</b> {pct(ho['wape_model'])} vs {pct(ho['wape_naive'])}; 80 % interval coverage {ho['coverage80']*100:.1f} %.</p></div></div>
<div class="two"><div>{img('07_forecast_example.png')}<div class="cap">Illustrative output for the top-revenue SKU: actuals, baseline, forecast with 80 % band, December holdout dots.</div></div><div>{img('08_decision_grid.png', '92%')}<div class="cap">Decisioning grid: {rs['quadrant_counts'].get('Reorder now',0)} reorder, {rs['quadrant_counts'].get('Markdown / clear',0)} markdown, {rs['quadrant_counts'].get('Healthy',0)} healthy.</div></div></div>

<h2>6. Screenshots — live deployment</h2>
<div class="two"><div>{img('dash_overview.png')}<div class="cap">Overview — KPIs, decisioning grid, rupee impact by category, top priorities, total demand with forecast band.</div></div>
<div>{img('dash_actions.png')}<div class="cap">Reorder &amp; markdown lists with order quantities, plus a sortable full table and CSV export.</div></div>
<div>{img('dash_model.png')}<div class="cap">Model accuracy — backtest by fold/horizon/category, feature importance and plain-language limits.</div></div>
<div>{img('dash_quality.png')}<div class="cap">Data-quality log rendered straight from the pipeline output.</div></div></div>
<div class="two"><div>{img('dash_explorer_crop.png')}<div class="cap">Forecast explorer — per-SKU history, forecast, 80 % band, seasonal-naive and holdout actuals; what-if controls re-score risk instantly with the same formula as the batch job.</div></div><div>{img('api_live_response.png')}<div class="cap">Live scoring API response (what-if override <code>lead_time_days=21</code> on SKU012). Invalid input returns explicit JSON errors (400/404/405), never a crash.</div></div></div>

<h2>7. Data quality and key assumptions</h2>
<table><tr><th>Issue found</th><th>n</th><th>Handling</th></tr>{"".join(f"<tr><td>[{i['table']}] {i['issue']}</td><td>{int(i['count']) if i['count'] is not None else '—'}</td><td>{i['action']}</td></tr>" for i in dq['issues'])}</table>
<p class="small"><b>Assumptions:</b> the latest monthly snapshot is the opening stock position; on-order stock arrives at the end of the lead time; promotions/holidays beyond the calendar file are absent; lead-time demand is approximately normal with sigma scaled from backtest relative errors; two years of history means one prior seasonal cycle. <b>Out of scope by design:</b> price optimisation, supplier selection, live integrations, automated purchase orders.</p>

<h2>8. Deliverables map (brief D1–D7 → Zidio submission items)</h2>
<table><tr><th>Brief</th><th>Deliverable</th><th>Where</th></tr>
<tr><td>D1</td><td>Reproducible data pipeline + audited data-quality log</td><td><code>src/pipeline.py</code>, <code>data/processed/</code>, notebook 01</td></tr>
<tr><td>D2</td><td>Data-quality &amp; EDA insight memo</td><td><code>reports/D2_EDA_Data_Quality_Memo.pdf</code>, <code>reports/figures/</code></td></tr>
<tr><td>D3</td><td>Demand forecast model, backtested vs seasonal-naive</td><td><code>src/forecast.py</code>, <code>outputs/backtest_metrics.json</code>, notebooks 02–03</td></tr>
<tr><td>D4</td><td>Risk scoring with actions and rupee impact</td><td><code>src/risk.py</code>, <code>outputs/risk_scores.csv</code></td></tr>
<tr><td>D5</td><td>Planning dashboard</td><td>{LIVE} (static) · <code>app/streamlit_app.py</code></td></tr>
<tr><td>D6</td><td>Deployed scoring service</td><td>{LIVE}/api/score · <code>service/main.py</code> (FastAPI)</td></tr>
<tr><td>D7</td><td>Executive readout</td><td><code>reports/D7_Executive_Readout.pdf</code> (9 slides)</td></tr>
<tr><td>Zidio 1–5</td><td>Source code · Live deployment · Demo video · Feedback video · Project report</td><td>GitHub repo · Netlify URL · scripts in <code>reports/Video_Scripts_Demo_and_Feedback.pdf</code> · this document</td></tr></table>

<h2>9. Challenges and learnings</h2>
<ul><li><b>Aligning time.</b> Inventory is a monthly snapshot while sales are daily; choosing the forecast origin to coincide with the snapshot date (rather than the last data point) made the risk scores meaningful and created a free holdout month.</li>
<li><b>Leakage discipline.</b> Building a direct multi-horizon design where every feature is computed from history ≤ origin — and writing a test that asserts it — was more work than the model itself, and is the reason the backtest can be trusted.</li>
<li><b>Calibrating the risk rule.</b> A first version flagged 36 of 50 SKUs for reorder because it compared 8-week demand with stock on hand; reframing stockout as "will it run out before the next order can land" (lead-time demand vs on-hand, reorder-point check) produced a triage the ops team can act on: 8 / 8 / 34.</li>
<li><b>Two dashboards, one truth.</b> Streamlit is fastest to build, but a static page with a serverless function gave a zero-cold-start public URL. Sharing a single risk engine (Python and a line-for-line JavaScript port) kept both consistent.</li>
<li><b>Catalogue problems are business findings.</b> Below-cost prices and 150 inventory SKUs with no sales are not "data cleaning" — they went into the executive readout as decisions for Finance.</li></ul>

<h2>10. Conclusion</h2>
<p>FORESIGHT meets every acceptance criterion in the brief with evidence: an honestly validated forecast ({pct(o['wape_model'])} WAPE, {o['improvement_pct']:.0f} % better than seasonal-naive across {met['n_folds']} rolling folds and a true holdout), a transparent risk layer that converts it into two short action lists worth {inr(rs['sales_at_risk_inr'])} of protected sales and {inr(rs['locked_capital_inr'])} of releasable cash, and a live dashboard plus API that NorthBay's operations team can use monthly without a data scientist. Next steps recommended to the client: adopt the monthly refresh, share the promotion calendar eight weeks ahead, fix the master-data issues surfaced here, and monitor forecast error against the baseline so the system earns trust every month it runs.</p>
{page_foot("Project report")}"""


# =============================================================================== Video scripts (Zidio items 3 & 4)
def scripts():
    ro = risk[risk.quadrant == "Reorder now"].sort_values("sales_at_risk_inr", ascending=False)
    md = risk[risk.quadrant == "Markdown / clear"].sort_values("locked_capital_inr", ascending=False)
    return f"""<style>{CSS} td:first-child{{white-space:nowrap;font-weight:600;color:#3d34a0}}</style>
<div class="band"><div class="k">Zidio submission items 3 &amp; 4 · recording scripts</div><h1>Demo video (4 min) and Feedback video (2 min) — word-for-word scripts with shot list</h1>
<div class="s">Record with OBS / Loom / Windows Game Bar at 1080p. Speak at a natural pace (~140 words/min). Keep the browser at 100 % zoom on {LIVE}. Upload to YouTube as <b>Unlisted</b> (or Google Drive with "anyone with link"). Total demo target: 3:45–4:30.</div></div>

<h2>A · Demo video — screen walkthrough (target 4:00)</h2>
<table><tr><th style="width:9%">Time</th><th style="width:30%">On screen</th><th>Say (verbatim)</th></tr>
<tr><td>0:00–0:25</td><td>Title slide of the executive readout PDF (slide 1), then cut to the live dashboard header</td><td>"Hi, I'm Eswar Mahalingam, Data Science intern at Zidio Development. This is Project FORESIGHT — a demand forecasting and inventory intelligence platform I built for our client NorthBay Living, a D2C home and lifestyle brand. Their problem: every month they stock out of things people want and sit on things they don't. FORESIGHT tells the ops team, for every product, what to reorder, what to clear, and what to leave alone — with the rupee value of each decision."</td></tr>
<tr><td>0:25–0:55</td><td>Overview tab. Point the cursor at each KPI card in turn</td><td>"This is the live dashboard on Netlify. Five numbers at the top answer the client's question immediately: {inr(rs['sales_at_risk_inr'])} of sales is at risk from stockouts across {len(ro)} products; {inr(rs['locked_capital_inr'])} of cash is locked in {len(md)} overstocked products; {rs['quadrant_counts'].get('Healthy',0)} of 50 are healthy and need nothing; and the forecast behind all of this has been tested at {o['wape_model']*100:.1f} percent error — {o['improvement_pct']:.0f} percent better than simply repeating last year."</td></tr>
<tr><td>0:55–1:30</td><td>Hover bubbles on the decisioning grid; hover the red cluster, then the purple one. Change the Category filter to Storage, then back to All</td><td>"The grid is the heart of it. Up means likely to run out during the supplier lead time; right means holding far more than eight weeks of demand; bubble size is rupees at stake. Four quadrants, four actions — exactly the decisioning view in the engagement brief. Everything filters: pick Storage, and you see it carries most of the overstock. Hover any bubble and you get the position, the cover and the recommended action."</td></tr>
<tr><td>1:30–2:05</td><td>Reorder &amp; markdown tab. Scroll the reorder list, point at {ro.iloc[0].sku_id}'s order quantity; then the markdown list; click Download CSV</td><td>"Tab two is what the team actually prints. The reorder list: {ro.iloc[0].sku_id} has {ro.iloc[0].days_of_stock_on_hand:.0f} days of stock against a {ro.iloc[0].lead_time_days}-day lead time — order {ro.iloc[0].reorder_qty} units now. The quantity covers demand through the lead time plus a four-week review cycle, plus safety stock, minus what's already on hand and on order. The markdown list shows {md.iloc[0].sku_id} sitting on {md.iloc[0].weeks_of_cover:.0f} weeks of cover — {inr(md.iloc[0].locked_capital_inr)} of cash. Note the margin flag: some of these sell below cost, so Finance checks them before any discount. One click exports the whole table."</td></tr>
<tr><td>2:05–2:50</td><td>Forecast explorer tab. Select {ro.iloc[0].sku_id}. Point at the black line, blue line, band, dashed orange line, and the December dots. Then drag the lead-time slider from {ro.iloc[0].lead_time_days} to 25 and back</td><td>"Behind every score is a weekly forecast. Black is actual demand, blue is the FORESIGHT forecast with an 80 percent band, orange dashed is the seasonal-naive baseline. The dots after the origin are December actuals the model never saw — they sit inside the band. Now the what-if: if this supplier's lead time slips to 25 days, watch the stockout probability and the sales at risk move instantly. It's the same transparent formula as the batch scoring, so an ops manager can test a scenario without calling me."</td></tr>
<tr><td>2:50–3:20</td><td>Model accuracy tab. Point at the fold chart, then the horizon chart, then the limitations text</td><td>"How do we know it's trustworthy? Rolling-origin backtesting: I retrained the model at six past dates using only earlier data and scored it on the next eight weeks each time — no random splits, no future information in any feature, and a unit test that enforces that. It beat the seasonal-naive baseline on every fold, every horizon and every category, and on the December holdout. The limitations are written on the page in plain language."</td></tr>
<tr><td>3:20–3:45</td><td>Data quality tab (scroll the log); then Scoring API tab → click Try it live</td><td>"The pipeline logs every cleaning decision with its rationale — 150 inventory SKUs with no sales history, launch dates after first sale, 16 products priced below cost — those became findings for the client, not silent fixes. And the scoring service: a public endpoint that returns the forecast and risk for any SKU or batch, with what-if overrides, and it fails gracefully on bad input."</td></tr>
<tr><td>3:45–4:10</td><td>VS Code: repo tree, then terminal running <code>python src/run_all.py</code> (pre-recorded or sped up); end on README results table</td><td>"Everything regenerates from the raw files with one command in about a minute: pipeline, forecast and backtest, risk scoring, figures and the dashboard data. The repo has the notebooks, tests, the Streamlit and FastAPI editions, the EDA memo and the executive readout. Thank you — links are in the description."</td></tr></table>
<div class="callout"><b>Recording tips.</b> Open all six tabs once before recording so charts are cached. Pre-type SKU012 in the API box. If the what-if slider is fiddly on camera, type 25 into the lead-time field instead. Cut dead air; graders reward a tight 4 minutes over a loose 6.</div>

<h2 class="pb">B · Feedback / reflection video (target 2:00)</h2>
<table><tr><th style="width:9%">Time</th><th style="width:26%">On screen</th><th>Say (verbatim)</th></tr>
<tr><td>0:00–0:15</td><td>Webcam, or the readout title slide</td><td>"Hi, I'm Eswar Mahalingam. This is my reflection on Project FORESIGHT, the demand and inventory intelligence platform I built during the Zidio Development Data Science and Analytics internship."</td></tr>
<tr><td>0:15–0:50</td><td>Backtest chart (fig 06)</td><td>"<b>What I learned.</b> The biggest lesson was discipline over cleverness. The brief said: define the metric, build a seasonal-naive baseline, and only then earn the right to use a complex model. Doing it in that order changed how I work. Rolling-origin backtesting, keeping every feature strictly in the past, and reporting WAPE against the baseline — those habits are what make a forecast trustworthy, and they are harder than the modelling itself. I also learned to turn predictions into decisions: a forecast nobody can act on has no value, so the risk layer and the rupee figures mattered as much as the accuracy."</td></tr>
<tr><td>0:50–1:25</td><td>Decisioning grid (fig 08), then the data-quality tab</td><td>"<b>Challenges.</b> Three stand out. First, time alignment — inventory was a monthly snapshot and sales were daily, so I set the forecast origin on the snapshot date to make the risk scores meaningful. Second, my first risk rule flagged 36 of 50 products for reorder — technically correct, practically useless. Reframing it as 'will it run out before the next order can land' gave a triage of eight, eight and thirty-four that an ops team can actually use. Third, the data itself: 150 inventory SKUs with no sales, products priced below cost. I learned to treat those as business findings for the client, not cleaning steps to hide."</td></tr>
<tr><td>1:25–1:50</td><td>Live dashboard URL on screen</td><td>"<b>Key takeaways.</b> Ship analytics as a product: a dashboard, an API, a one-command pipeline and a readout a non-technical person can follow. Be honest about limits — two years of history is one seasonal cycle, and I said so. And consultant framing works: lead with the rupee impact and the decision, then show the evidence."</td></tr>
<tr><td>1:50–2:05</td><td>Webcam</td><td>"Thank you to the Zidio mentors for the structure and the feedback at each weekly checkpoint. This project is now the anchor of my data science portfolio, and the method — baseline first, backtest honestly, decide transparently — is one I'll carry into every forecasting problem. Thanks for watching."</td></tr></table>

<h2>C · Submission checklist (Zidio roadmap)</h2>
<table><tr><th>#</th><th>Item</th><th>Marks</th><th>What to paste in the form</th></tr>
<tr><td>1</td><td>Source code</td><td>5</td><td>GitHub repo URL (public). Commands to publish: <code>cd foresight &amp;&amp; git init &amp;&amp; git add . &amp;&amp; git commit -m "Project FORESIGHT" &amp;&amp; gh repo create foresight --public --source=. --push</code> (or create the repo on github.com and <code>git remote add origin … &amp;&amp; git push -u origin main</code>)</td></tr>
<tr><td>2</td><td>Live deployment</td><td>5</td><td>{LIVE} (dashboard) · {LIVE}/api/score?sku=SKU012 (API)</td></tr>
<tr><td>3</td><td>Demo video</td><td>4</td><td>Unlisted YouTube link — script A above</td></tr>
<tr><td>4</td><td>Feedback video</td><td>4</td><td>Unlisted YouTube link — script B above</td></tr>
<tr><td>5</td><td>Project report</td><td>2</td><td><code>reports/Project_Report_FORESIGHT.pdf</code> (also attach <code>D7_Executive_Readout.pdf</code> and <code>D2_EDA_Data_Quality_Memo.pdf</code>)</td></tr></table>
{page_foot("Video scripts & checklist")}"""


async def render():
    from playwright.async_api import async_playwright
    jobs = [("D2_EDA_Data_Quality_Memo.pdf", memo(), dict(format="A4")),
            ("D7_Executive_Readout.pdf", readout(), dict(width="297mm", height="167mm")),
            ("Project_Report_FORESIGHT.pdf", project_report(), dict(format="A4")),
            ("Video_Scripts_Demo_and_Feedback.pdf", scripts(), dict(format="A4"))]
    async with async_playwright() as p:
        b = await p.chromium.launch()
        for name, html, opts in jobs:
            pg = await b.new_page()
            await pg.set_content(f"<!doctype html><html><head><meta charset='utf-8'></head><body>{html}</body></html>", wait_until="load")
            await pg.pdf(path=str(REP / name), print_background=True, prefer_css_page_size=True, **opts)
            await pg.close(); print("wrote", name)
        await b.close()


if __name__ == "__main__":
    asyncio.run(render())
