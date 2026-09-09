// FORESIGHT scoring service — D6
// GET  /api/score?sku=SKU012[&lead_time_days=10&on_hand=120&on_order=0]
// POST /api/score  { "skus": ["SKU012","SKU020"], "overrides": { "lead_time_days": 10 } }
// GET  /api/score?all=true        → every SKU (batch)
// GET  /api/health                → service status + model metadata
import data from "../../web/data/dashboard.json" with { type: "json" };
import { scoreSku } from "../../web/risk_engine.js";

const bySku = new Map(data.skus.map((s) => [s.sku_id, s]));
const json = (body, status = 200) =>
  new Response(JSON.stringify(body, null, 2), { status, headers: { "content-type": "application/json; charset=utf-8" } });

function num(v, name, errors, { min = 0, max = 1e7 } = {}) {
  if (v === undefined || v === null || v === "") return undefined;
  const n = Number(v);
  if (!Number.isFinite(n) || n < min || n > max) errors.push(`${name} must be a number between ${min} and ${max} (got '${v}')`);
  return n;
}

function scoreOne(id, overrides, errors) {
  const key = String(id || "").trim().toUpperCase().replace(/^SKU-?/, "SKU");
  const s = bySku.get(key);
  if (!s) { errors.push(`Unknown SKU '${id}'. Valid ids: SKU001–SKU050`); return null; }
  const inp = {
    onHand: overrides.on_hand ?? s.on_hand, onOrder: overrides.on_order ?? s.on_order,
    leadTimeDays: overrides.lead_time_days ?? s.lead_time_days, safetyStock: overrides.safety_stock ?? s.safety_stock,
    forecast: s.forecast.mean, sigma: s.forecast.sigma, listPrice: s.list_price, unitCost: s.unit_cost,
  };
  const r = scoreSku(inp);
  return {
    sku_id: s.sku_id, product_name: s.product_name, category: s.category, subcategory: s.subcategory,
    inventory_position: { on_hand: inp.onHand, on_order: inp.onOrder, lead_time_days: inp.leadTimeDays, safety_stock: inp.safetyStock,
                          overridden: Object.keys(overrides).length > 0 },
    forecast: { origin: data.meta.origin, horizon_weeks: data.meta.horizon_weeks, weeks: s.forecast.weeks, mean: s.forecast.mean,
                lo80: s.forecast.lo80, hi80: s.forecast.hi80, seasonal_naive: s.forecast.naive, total_8w: r.demand_8w },
    risk: r,
  };
}

export default async (req) => {
  const url = new URL(req.url);
  if (url.pathname.endsWith("/health")) {
    return json({ status: "ok", service: "FORESIGHT scoring API", client: data.meta.client, forecast_origin: data.meta.origin,
                  skus_available: data.skus.length, model: "LightGBM (tweedie) direct multi-horizon",
                  backtest_wape_model: data.metrics.overall.wape_model, backtest_wape_seasonal_naive: data.metrics.overall.wape_naive,
                  generated: data.meta.generated,
                  usage: { single: "GET /api/score?sku=SKU012", what_if: "GET /api/score?sku=SKU012&lead_time_days=14&on_hand=50",
                           batch: "POST /api/score {\"skus\":[\"SKU012\",\"SKU020\"]}", all: "GET /api/score?all=true" } });
  }
  const errors = [];
  let ids = [], overrides = {};
  try {
    if (req.method === "POST") {
      let body = {};
      try { body = await req.json(); } catch { return json({ error: "Body must be valid JSON, e.g. {\"skus\":[\"SKU012\"]}" }, 400); }
      ids = Array.isArray(body.skus) ? body.skus : body.sku ? [body.sku] : [];
      const o = body.overrides || {};
      for (const k of ["lead_time_days", "on_hand", "on_order", "safety_stock"]) { const v = num(o[k], k, errors, { max: k === "lead_time_days" ? 365 : 1e7 }); if (v !== undefined) overrides[k] = v; }
    } else if (req.method === "GET") {
      if (url.searchParams.get("all") === "true") ids = data.skus.map((s) => s.sku_id);
      else ids = (url.searchParams.get("sku") || url.searchParams.get("skus") || "").split(",").map((x) => x.trim()).filter(Boolean);
      for (const k of ["lead_time_days", "on_hand", "on_order", "safety_stock"]) { const v = num(url.searchParams.get(k), k, errors, { max: k === "lead_time_days" ? 365 : 1e7 }); if (v !== undefined) overrides[k] = v; }
    } else return json({ error: `Method ${req.method} not allowed. Use GET or POST.` }, 405);
    if (!ids.length) return json({ error: "Provide a SKU: GET /api/score?sku=SKU012  or  POST {\"skus\":[...]}  or  ?all=true", example: "/api/score?sku=SKU012" }, 400);
    if (ids.length > 200) return json({ error: "Batch limited to 200 SKUs per request" }, 400);
    if (errors.length) return json({ error: "Invalid input", details: errors }, 400);
    const results = ids.map((id) => scoreOne(id, overrides, errors)).filter(Boolean);
    if (!results.length) return json({ error: "No valid SKUs", details: errors }, 404);
    const body = { forecast_origin: data.meta.origin, count: results.length, results };
    if (errors.length) body.warnings = errors;
    if (results.length > 1) body.summary = {
      sales_at_risk_inr: results.reduce((a, r) => a + r.risk.sales_at_risk_inr, 0),
      locked_capital_inr: results.reduce((a, r) => a + r.risk.locked_capital_inr, 0),
      quadrants: results.reduce((a, r) => ((a[r.risk.quadrant] = (a[r.risk.quadrant] || 0) + 1), a), {}),
    };
    return json(body);
  } catch (e) {
    return json({ error: "Internal error while scoring", detail: String(e && e.message) }, 500);
  }
};

export const config = { path: ["/api/score", "/api/health"] };
