<!-- ═══════════════ CAREER CONTROL TOWER · REPOSITORY · FORESIGHT ═══════════════ -->
<div align="center">

<img src="https://raw.githubusercontent.com/Eswar5313/Eswar5313/main/assets/headers/REPO_FORESIGHT.svg" width="100%" alt="Project FORESIGHT — Eswar Mahalingam" />

<a href="https://github.com/Eswar5313"><img src="https://img.shields.io/badge/⬅-CAREER_CONTROL_TOWER-000000?style=for-the-badge&labelColor=FFFFFF" alt="CAREER CONTROL TOWER"/></a> <a href="https://eswar5313.github.io/Eswar-Master-Project-Portfolio-2026/"><img src="https://img.shields.io/badge/✦-MASTER_PORTFOLIO-000000?style=for-the-badge&labelColor=C9CDD6" alt="MASTER PORTFOLIO"/></a> <a href="https://eswar5313.github.io/Eswar-Portfolio-Lens-Index-2026/"><img src="https://img.shields.io/badge/✦-LENS_INDEX-000000?style=for-the-badge&labelColor=FFFFFF" alt="LENS INDEX"/></a> <a href="https://foresight-northbay.netlify.app"><img src="https://img.shields.io/badge/✦-LIVE_DASHBOARD-000000?style=for-the-badge&labelColor=FFFFFF" alt="LIVE DASHBOARD"/></a> <a href="https://foresight-northbay.netlify.app/api/health"><img src="https://img.shields.io/badge/✦-SCORING_API-000000?style=for-the-badge&labelColor=C9CDD6" alt="SCORING API"/></a>

<img src="https://img.shields.io/badge/WAPE-8.8%25_vs_11.2%25_NAIVE-FFFFFF?style=for-the-badge&labelColor=000000" alt="WAPE: 8.8% vs 11.2% NAIVE"/> <img src="https://img.shields.io/badge/HOLDOUT_WAPE-9.3%25-C9CDD6?style=for-the-badge&labelColor=000000" alt="HOLDOUT WAPE: 9.3%"/> <img src="https://img.shields.io/badge/SKU_WINS-49%2F50-FFFFFF?style=for-the-badge&labelColor=000000" alt="SKU WINS: 49/50"/> <img src="https://img.shields.io/badge/REORDER_NOW-8_SKUs-C9CDD6?style=for-the-badge&labelColor=000000" alt="REORDER NOW: 8 SKUs"/>

**Zidio Development · Data Scientist (Data Science & Analytics) · 2026** — weekly SKU demand forecast · stockout early-warning · overstock flag · ops-ready dashboard + API

</div>

<img src="https://raw.githubusercontent.com/Eswar5313/Eswar5313/main/assets/divider.svg" width="100%" alt="" />

## Project FORESIGHT — AI-Powered Demand & Inventory Intelligence

**Client:** NorthBay Living (D2C home & lifestyle) · **Program:** Zidio Development — Data Science & Analytics internship · **Author:** Eswar Mahalingam

| | |
|---|---|
| **Live dashboard** | https://foresight-northbay.netlify.app |
| **Scoring API** | https://foresight-northbay.netlify.app/api/score?sku=SKU012 · health: `/api/health` |
| **Backtest (rolling-origin, 6 folds × 8 weeks)** | **WAPE 8.8 % (LightGBM) vs 11.2 % (seasonal-naive) → 21.5 % better**, bias +0.3 %, wins 6/6 folds and 49/50 SKUs |
| **Dec-2025 holdout (never seen)** | WAPE 9.3 % vs 12.4 % naive · 80 % interval covered 79.5 % of actuals |
| **Decisions this month** | 8 SKUs **reorder now** (₹47.9 L sales at risk) · 8 SKUs **markdown / clear** (₹1.48 Cr locked) · 34 healthy |

## 1. The problem
NorthBay stocks out of best-sellers and sits on slow movers because inventory is planned on gut feel. The brief asks for four things: a weekly SKU-level demand forecast that beats a naive baseline, a stockout early-warning, an overstock flag, and an interface a non-technical ops team can use without a data scientist in the room.

## 2. The data
Four simulated extracts (repository root (`*.csv`)), Jan-2024 → Dec-2025:

| Table | Grain | Rows |
|---|---|---|
| `sales_daily.csv` | SKU × day: units, revenue, price, promo flag | 36,550 (50 SKUs × 731 days) |
| `sku_master.csv` | SKU: category, subcategory, launch date, cost, list price | 50 |
| `calendar.csv` | day: week, month, season, holiday, promotion event | 731 |
| `inventory_snapshots.csv` | SKU × month: on-hand, on-order, lead time, safety stock, reorder point | 4,800 (200 SKUs × 24 months) |

`Stock_data.csv` (an equity price file) was supplied alongside; it has no join key and is excluded.

**Data-quality findings** (full log with rationale: `data_quality_log.json`, also in the dashboard's *Data quality* tab): 150 inventory SKUs (SKU051–200, ₹17.5 Cr of stock) have no sales history at all; 21 SKUs have a launch date *after* their first sale; 16 SKUs are priced below cost; 40 category/subcategory pairs look inconsistent (e.g. Kitchen → Cushion); partial calendar weeks at the series edges. Sales itself had no duplicates, negatives or missing values. Every fix is coded, none is manual.

## 3. Setup & run (reproducible in one command)
```bash
git clone <this repo> && cd foresight
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/run_all.py        # pipeline → forecast + backtest → risk → figures → dashboard data  (~1 min)
python -m pytest tests -q    # 5 tests incl. a leakage guard and API error handling
streamlit run app/streamlit_app.py            # dashboard (Streamlit edition)
uvicorn service.main:app --reload --port 8000 # scoring API (FastAPI edition) → http://localhost:8000/docs
```
Seeds are fixed (`SEED = 42`); re-running reproduces the headline numbers above.

## 4. Method (baseline first, then earn the right to be complex)
1. **Frame** — horizon 8 weeks, metric WAPE (robust for low-volume SKUs), bias as secondary. Weekly grain, Sunday-ending weeks, forecast origin **2025-11-30** so the latest inventory snapshot (2025-12-01) and the forecast start on the same day; the 4 December weeks become a true holdout.
2. **Baseline** — seasonal-naive: forecast(t+h) = actual(t+h−52).
3. **Features** — lags 1–13, rolling mean/std, trend, last-year same week (±1 wk), this-year/last-year level ratio, *planned* promo & holiday days of the target week, week-of-year, month, season, weeks since launch, price, SKU & category identity, horizon *h*.
4. **Model** — LightGBM (Tweedie objective, direct multi-horizon: one row per SKU × origin × h). Training rows only ever see history ≤ origin.
5. **Backtest** — rolling-origin CV: 6 origins in 2025, 4 weeks apart, each retrained from scratch and scored on the next 8 weeks. No random splits.
6. **Intervals** — 80 % band from per-horizon backtest residual quantiles (stretch goal; calibrated at 79.5 % on holdout).
7. **Risk** — transparent formulas (`risk.py`):
   - Stockout score = max( P[demand over lead time > on-hand], P[position − safety stock < demand over lead time] ) using forecast mean + calibrated sigma.
   - Overstock score = clip((weeks of cover − 8) / 8, 0, 1) → 0 at ≤ 8 weeks, 1 at ≥ 16.
   - Sales at risk = units short × price; locked capital = units beyond 8-week demand × cost; reorder qty = demand over (lead time + 4-week review) + safety stock − position.
   - Threshold 0.5 on each axis → **Reorder now / Markdown-clear / Watch-volatile / Healthy**, exactly the grid in the brief (§08).

## 5. Results
| Backtest slice | Model WAPE | Seasonal-naive WAPE |
|---|---|---|
| Overall (2,400 SKU-weeks) | **8.75 %** | 11.16 % |
| Worst fold (origin 13-Jul-25) | 9.4 % | 11.7 % |
| Horizon 1 wk / 8 wk | 8.0 % / 9.6 % | 10.6 % / 12.1 % |
| Dec-2025 holdout | **9.3 %** | 12.4 % |

Top drivers by gain: SKU identity (52 % — each product's own level), `mean4` (27 %), `mean8` (12 %), `mean13`, `week_of_year`, `lag52_mean3` and `season` (the seasonal signal), then promo days. See `feature_importance.csv`.

## 6. Repository layout
```
foresight/
  data/raw/                   client extracts        data/processed/  cleaned weekly dataset + data-quality log
  src/pipeline.py             D1 ingest · clean · features        src/forecast.py   D3 baseline · LightGBM · rolling backtest
  src/risk.py                 D4 stockout/overstock scoring       src/eda.py        D2 figures     src/export_web.py  dashboard bundle
  src/run_all.py              one-command reproduction            tests/            pytest suite
  notebooks/                  01_eda · 02_baseline · 03_model (executed, with outputs)
  app/streamlit_app.py        D5 dashboard (Streamlit)            web/              D5 dashboard (static, deployed on Netlify)
  service/main.py             D6 scoring API (FastAPI)            netlify/functions/score.mjs  D6 scoring API (serverless, live)
  outputs/                    forecast.csv · risk_scores.csv · backtest_metrics.json · feature_importance.csv
  reports/                    D2 EDA memo · D7 executive readout · project report · video scripts · figures/
```

## 7. Scoring API
```
GET  /api/health
GET  /api/score?sku=SKU012                          one SKU: 8-week forecast (mean, lo80, hi80, naive) + risk + action
GET  /api/score?sku=SKU012&lead_time_days=21        what-if override (also on_hand, on_order, safety_stock)
GET  /api/score?sku=SKU012,SKU020   |  ?all=true    batch
POST /api/score  {"skus":["SKU012","SKU020"],"overrides":{"lead_time_days":14}}
```
Bad input never crashes: unknown SKU → 404, non-numeric/negative override → 400, empty request → 400 with an example, bad JSON → 400.

## 8. Key assumptions & limitations
- The latest monthly snapshot is the opening stock position at the forecast origin; on-order stock arrives at the end of the lead time.
- Promotions/holidays after 31-Dec-2025 are unknown and assumed absent.
- Two years of history = one prior seasonal cycle; new SKUs lean on category peers (flag low confidence).
- Normal approximation for lead-time demand; sigma is scaled from backtest relative errors per horizon.
- Out of scope by design: price optimisation, supplier selection, live integrations, automated PO placement.

## 9. Monitoring plan (stretch)
Retrain monthly on the new snapshot. Track WAPE of the last 4 weeks vs seasonal-naive; alert if the model loses to the baseline for 2 consecutive months or bias drifts beyond ±5 %. Track realised stockouts among "Healthy" SKUs as risk-precision feedback.

## 🗂️ Files in this repository

| Group | Files |
|---|---|
| 📄 Reports | [Project_Report_FORESIGHT.pdf](Project_Report_FORESIGHT.pdf) · [D2_EDA_Data_Quality_Memo.pdf](D2_EDA_Data_Quality_Memo.pdf) · [D7_Executive_Readout.pdf](D7_Executive_Readout.pdf) · [Video_Scripts_Demo_and_Feedback.pdf](Video_Scripts_Demo_and_Feedback.pdf) · [SUBMISSION.md](SUBMISSION.md) |
| 📓 Notebooks | [01_eda.ipynb](01_eda.ipynb) · [02_baseline.ipynb](02_baseline.ipynb) · [03_model.ipynb](03_model.ipynb) |
| 🐍 Pipeline | [run_all.py](run_all.py) · [pipeline.py](pipeline.py) · [eda.py](eda.py) · [forecast.py](forecast.py) · [risk.py](risk.py) · [export_web.py](export_web.py) · [build_reports.py](build_reports.py) · [main.py](main.py) · [test_foresight.py](test_foresight.py) · [Makefile](Makefile) |
| 🌐 App & API | [index.html](index.html) (dashboard) · [risk_engine.js](risk_engine.js) · [score.mjs](score.mjs) · [streamlit_app.py](streamlit_app.py) · [netlify.toml](netlify.toml) |
| 📊 Data & outputs | sales / SKU / calendar / inventory CSVs · [forecast.csv](forecast.csv) · [risk_scores.csv](risk_scores.csv) · [backtest_metrics.json](backtest_metrics.json) · [feature_importance.csv](feature_importance.csv) |
| 🖼️ Charts | `01_weekly_demand.png` … `11_weeks_of_cover.png` · dashboard screenshots `dash_*.png` |

## 10. Deploying yourself
- **Netlify (as deployed):** `netlify deploy --prod` from the repo root (publish dir `web/`, functions in `netlify/functions/`, config in `netlify.toml`).
- **Streamlit Community Cloud:** new app → this repo → main file `streamlit_app.py`.
- **Render (FastAPI):** build `pip install -r requirements.txt`, start `uvicorn service.main:app --host 0.0.0.0 --port $PORT`.

<img src="https://raw.githubusercontent.com/Eswar5313/Eswar5313/main/assets/divider.svg" width="100%" alt="" />

<div align="center">

**Eswar Mahalingam** · B.Com · MBA · PGDLSCM · CSCMP SCPro · Six Sigma Black Belt
Data Scientist @ Zidio Development · Ghaziabad NCR, India · Open to India · EU (Blue Card) · Gulf · Immediate joiner

[![LinkedIn](https://img.shields.io/badge/✦-LINKEDIN-000000?style=for-the-badge&labelColor=C9CDD6)](https://linkedin.com/in/eswar-mahalingam)
[![Email](https://img.shields.io/badge/✦-EMAIL-000000?style=for-the-badge&labelColor=FFFFFF)](mailto:eswarmba05313@gmail.com)
[![Phone](https://img.shields.io/badge/✦-+91_9360548243-000000?style=for-the-badge&labelColor=C9CDD6)](tel:+919360548243)
[![Portfolio](https://img.shields.io/badge/✦-PORTFOLIO_SITE-000000?style=for-the-badge&labelColor=FFFFFF)](https://eswar-3d-portfolio.netlify.app)
[![Profile](https://img.shields.io/badge/⬅-CAREER_CONTROL_TOWER-000000?style=for-the-badge&labelColor=FFFFFF)](https://github.com/Eswar5313)

<img src="https://raw.githubusercontent.com/Eswar5313/Eswar5313/main/assets/kailash-footer.svg" width="100%" alt="" />

</div>
