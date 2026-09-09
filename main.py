"""
Project FORESIGHT — D6 Scoring service (FastAPI edition)
Run:    uvicorn service.main:app --reload --port 8000     → docs at /docs
Deploy: Render / Railway (start command above, port from $PORT)
A serverless JavaScript twin of this service is deployed at <netlify-url>/api/score (see README).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from risk import score_sku  # noqa: E402

app = FastAPI(title="FORESIGHT scoring API", version="1.0",
              description="Weekly SKU demand forecast (8 weeks, 80% interval, seasonal-naive baseline) + stockout/overstock risk with recommended action. NorthBay Living · Zidio Development.")

_fc = pd.read_csv(ROOT / "outputs/forecast.csv")
_risk = pd.read_csv(ROOT / "outputs/risk_scores.csv").set_index("sku_id")
_met = json.loads((ROOT / "outputs/backtest_metrics.json").read_text())
_SKUS = sorted(_risk.index)


class Overrides(BaseModel):
    lead_time_days: Optional[float] = Field(None, ge=0, le=365)
    on_hand: Optional[float] = Field(None, ge=0, le=1e7)
    on_order: Optional[float] = Field(None, ge=0, le=1e7)
    safety_stock: Optional[float] = Field(None, ge=0, le=1e7)


class BatchRequest(BaseModel):
    skus: list[str] = Field(..., min_length=1, max_length=200, examples=[["SKU012", "SKU020"]])
    overrides: Overrides = Overrides()

    @field_validator("skus")
    @classmethod
    def norm(cls, v):
        return [s.strip().upper().replace("SKU-", "SKU") for s in v]


def _score(sku: str, ov: Overrides) -> dict:
    sku = sku.strip().upper().replace("SKU-", "SKU")
    if sku not in _risk.index:
        raise HTTPException(404, f"Unknown SKU '{sku}'. Valid ids: {_SKUS[0]}–{_SKUS[-1]}")
    r = _risk.loc[sku]; f = _fc[_fc.sku_id == sku].sort_values("h")
    oh = float(ov.on_hand if ov.on_hand is not None else r.on_hand)
    oo = float(ov.on_order if ov.on_order is not None else r.on_order)
    lt = float(ov.lead_time_days if ov.lead_time_days is not None else r.lead_time_days)
    ss = float(ov.safety_stock if ov.safety_stock is not None else r.safety_stock)
    out = {k: (v.item() if hasattr(v, 'item') else v) for k, v in score_sku(oh, oo, lt, ss, f.forecast.values, f.sigma.values, float(r.list_price), float(r.unit_cost)).items()}
    return dict(sku_id=sku, product_name=str(r.product_name), category=str(r.category), subcategory=str(r.subcategory),
                inventory_position=dict(on_hand=oh, on_order=oo, lead_time_days=lt, safety_stock=ss,
                                        overridden=any(v is not None for v in ov.model_dump().values())),
                forecast=dict(origin=_met["forecast_origin"], horizon_weeks=_met["horizon_weeks"], weeks=f.week_end.astype(str).tolist(),
                              mean=f.forecast.round(1).tolist(), lo80=f.lo80.round(1).tolist(), hi80=f.hi80.round(1).tolist(),
                              seasonal_naive=f.naive.round(1).tolist(), total_8w=round(float(f.forecast.sum()), 1)),
                risk=out)


@app.get("/", include_in_schema=False)
def root():
    return {"service": "FORESIGHT scoring API", "docs": "/docs", "health": "/health", "example": "/score/SKU012"}


@app.get("/health", summary="Service status and model metadata")
def health():
    o = _met["overall"]
    return dict(status="ok", forecast_origin=_met["forecast_origin"], skus_available=len(_SKUS), model="LightGBM (tweedie) direct multi-horizon",
                backtest_wape_model=o["wape_model"], backtest_wape_seasonal_naive=o["wape_naive"])


@app.get("/score/{sku_id}", summary="Forecast + risk for one SKU (optional what-if overrides)")
def score_one(sku_id: str, lead_time_days: Optional[float] = Query(None, ge=0, le=365), on_hand: Optional[float] = Query(None, ge=0),
              on_order: Optional[float] = Query(None, ge=0), safety_stock: Optional[float] = Query(None, ge=0)):
    return _score(sku_id, Overrides(lead_time_days=lead_time_days, on_hand=on_hand, on_order=on_order, safety_stock=safety_stock))


@app.post("/score", summary="Batch: forecast + risk for many SKUs")
def score_batch(req: BatchRequest):
    results, warnings = [], []
    for s in req.skus:
        try:
            results.append(_score(s, req.overrides))
        except HTTPException as e:
            warnings.append(e.detail)
    if not results:
        raise HTTPException(404, {"error": "No valid SKUs", "details": warnings})
    body = dict(forecast_origin=_met["forecast_origin"], count=len(results), results=results,
                summary=dict(sales_at_risk_inr=sum(r["risk"]["sales_at_risk_inr"] for r in results),
                             locked_capital_inr=sum(r["risk"]["locked_capital_inr"] for r in results),
                             quadrants=pd.Series([r["risk"]["quadrant"] for r in results]).value_counts().to_dict()))
    if warnings:
        body["warnings"] = warnings
    return body


@app.get("/skus", summary="List all scoreable SKUs with their current quadrant")
def skus():
    return _risk.reset_index()[["sku_id", "product_name", "category", "quadrant", "priority"]].to_dict("records")
