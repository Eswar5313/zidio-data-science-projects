"""Smoke + logic tests:  python -m pytest tests -q"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from risk import score_sku
from forecast import wape, bias

def test_wape_bias():
    assert abs(wape([10, 20], [12, 18]) - 4 / 30) < 1e-9
    assert abs(bias([10, 20], [12, 18])) < 1e-9

def test_risk_quadrants_move_correctly():
    fc, sg = np.full(8, 100.0), np.full(8, 10.0)
    empty = score_sku(10, 0, 14, 20, fc, sg, 1000, 500)
    assert empty["quadrant"] == "Reorder now" and empty["stockout_score"] > 0.9 and empty["reorder_qty"] > 0
    piled = score_sku(3000, 0, 7, 20, fc, sg, 1000, 500)
    assert piled["quadrant"] == "Markdown / clear" and piled["overstock_score"] == 1.0 and piled["locked_capital_inr"] > 0
    ok = score_sku(400, 300, 7, 20, fc, sg, 1000, 500)
    assert ok["quadrant"] == "Healthy"

def test_outputs_consistent():
    r = pd.read_csv(ROOT / "outputs/risk_scores.csv"); f = pd.read_csv(ROOT / "outputs/forecast.csv")
    assert r.sku_id.nunique() == 50 and len(f) == 400 and (f.forecast >= 0).all() and (f.lo80 <= f.hi80).all()
    m = json.loads((ROOT / "outputs/backtest_metrics.json").read_text())
    assert m["overall"]["wape_model"] < m["overall"]["wape_naive"], "model must beat the seasonal-naive baseline on backtest"

def test_no_future_leakage_in_training_rows():
    from forecast import load_weekly, make_rows
    wk = load_weekly(); origin = pd.Timestamp("2025-06-15")
    rows = make_rows(wk, origin)
    hist = wk[wk.week_end <= origin].pivot(index="week_end", columns="sku_id", values="units")
    assert np.allclose(rows[rows.h == 1].set_index("sku_id").lag1.reindex(hist.columns), hist.iloc[-1])
    assert (rows.week_end > origin).all()

def test_api():
    from fastapi.testclient import TestClient
    from service.main import app
    c = TestClient(app)
    assert c.get("/health").status_code == 200
    j = c.get("/score/SKU012").json(); assert j["risk"]["quadrant"] and len(j["forecast"]["mean"]) == 8
    assert c.get("/score/NOPE").status_code == 404
    assert c.get("/score/SKU012?lead_time_days=-3").status_code == 422
    b = c.post("/score", json={"skus": ["SKU012", "SKU020", "BAD"]}).json(); assert b["count"] == 2 and b["warnings"]
