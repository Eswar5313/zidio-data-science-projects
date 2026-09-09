// Shared, dependency-free risk engine — same formulas as src/risk.py.
// Used by the dashboard (what-if control) and by the /api/score Netlify function.
export const HORIZON = 8, REVIEW_WEEKS = 4, THRESH = 0.5;

// standard normal CDF (Abramowitz–Stegun 7.1.26, |err| < 1.5e-7)
export function normCdf(x) {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989422804014327 * Math.exp(-x * x / 2);
  const p = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  return x >= 0 ? 1 - p : p;
}

export function scoreSku({ onHand, onOrder, leadTimeDays, safetyStock, forecast, sigma, listPrice, unitCost }) {
  const oh = Math.max(0, +onHand || 0), oo = Math.max(0, +onOrder || 0), ss = Math.max(0, +safetyStock || 0);
  const ltW = Math.max(0, +leadTimeDays || 0) / 7, full = Math.floor(ltW), frac = ltW - full;
  const fc = forecast.map(Number), sg = sigma.map(Number);
  const sum = (a, n) => a.slice(0, n).reduce((s, v) => s + v, 0);
  const at = (a, i) => (i < a.length ? a[i] : a[a.length - 1]);
  const muLt = sum(fc, full) + at(fc, full) * frac, sdLt = Math.max(sum(sg, full) + at(sg, full) * frac, 1e-6);
  const muH = sum(fc, fc.length), sdH = Math.max(sum(sg, sg.length), 1e-6);
  const pLt = 1 - normCdf((oh - muLt) / sdLt);
  const pRop = 1 - normCdf((oh + oo - ss - muLt) / sdLt);
  const pH = 1 - normCdf((oh + oo - muH) / sdH);
  const stockout = Math.max(pLt, pRop);
  const avgW = Math.max(muH / fc.length, 1e-6);
  const cover = (oh + oo) / avgW;
  const overstock = Math.min(1, Math.max(0, (cover - HORIZON) / HORIZON));
  const shortUnits = Math.max(0, muLt - oh, muLt + ss - oh - oo);
  const excessUnits = Math.max(0, oh + oo - muH);
  const span = Math.min(fc.length, Math.ceil(ltW + REVIEW_WEEKS));
  const reorder = Math.max(0, sum(fc, span) + ss - oh - oo);
  let quadrant, action;
  if (stockout >= THRESH && overstock < THRESH) { quadrant = "Reorder now"; action = `Raise a replenishment order for ~${Math.ceil(reorder)} units before stock runs out`; }
  else if (overstock >= THRESH && stockout < THRESH) { quadrant = "Markdown / clear"; action = `Promote or discount — ${cover.toFixed(0)} weeks of cover vs 8-week horizon`; }
  else if (stockout >= THRESH && overstock >= THRESH) { quadrant = "Watch / volatile"; action = "Investigate — demand is erratic; review manually before ordering"; }
  else { quadrant = "Healthy"; action = "No action needed; leave as is"; }
  return {
    stockout_score: +stockout.toFixed(3), overstock_score: +overstock.toFixed(3),
    p_stockout_lead_time: +pLt.toFixed(3), p_below_reorder_point: +pRop.toFixed(3), p_stockout_horizon: +pH.toFixed(3),
    demand_lead_time: +muLt.toFixed(1), demand_8w: +muH.toFixed(1), weeks_of_cover: +cover.toFixed(1),
    weeks_to_reorder: +Math.max(0, (oh + oo - ss) / avgW - ltW).toFixed(1),
    units_short: +shortUnits.toFixed(1), units_excess: +excessUnits.toFixed(1),
    sales_at_risk_inr: Math.round(shortUnits * listPrice), locked_capital_inr: Math.round(excessUnits * unitCost),
    reorder_qty: Math.ceil(reorder), days_of_stock_on_hand: +(7 * oh / avgW).toFixed(1), quadrant, action,
  };
}
