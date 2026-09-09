"""
Project FORESIGHT — D1 Data pipeline
=====================================
Ingest the four client extracts, validate, clean, and produce ONE analysis-ready
weekly SKU dataset plus a data-quality log. Every cleaning decision is coded here
(nothing manual) and written to data/processed/data_quality_log.json so the
client can audit it.

Run:  python src/pipeline.py
Out:  data/processed/sales_weekly.csv       (weekly SKU fact table + features)
      data/processed/sku_dim.csv            (cleaned SKU dimension)
      data/processed/inventory_latest.csv   (latest stock position per SKU)
      data/processed/data_quality_log.json  (issues found + how handled)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# Forecast origin: the last Sunday BEFORE the latest inventory snapshot (2025-12-01),
# so the stock position and the forecast start on the same day. Weeks after the
# origin (Dec-2025) are kept as a holdout to show forecast-vs-actual on the dashboard.
FORECAST_ORIGIN = pd.Timestamp("2025-11-30")

log: list[dict] = []


def note(table: str, issue: str, count, action: str, rationale: str):
    log.append(dict(table=table, issue=issue, count=int(count) if count is not None else None,
                    action=action, rationale=rationale))


# ----------------------------------------------------------------------------- ingest
def ingest():
    sales = pd.read_csv(RAW / "sales_daily.csv")
    inv = pd.read_csv(RAW / "inventory_snapshots.csv")
    cal = pd.read_csv(RAW / "calendar.csv")
    sku = pd.read_csv(RAW / "sku_master.csv")
    # Normalise column names to the brief's data dictionary (snake_case)
    sales = sales.rename(columns={"Date": "date", "SKU": "sku_id", "Units_Sold": "units_sold",
                                  "Revenue": "revenue", "Price": "unit_price", "Promotion": "promo_flag"})
    inv = inv.rename(columns={"Snapshot_Date": "date", "SKU": "sku_id", "Current_Stock": "on_hand_units",
                              "On_Order": "on_order_units", "Lead_Time_Days": "lead_time_days",
                              "Safety_Stock": "safety_stock", "Reorder_Point": "reorder_point",
                              "Inventory_Value": "inventory_value"})
    sku = sku.rename(columns={"SKU": "sku_id", "Product_Name": "product_name", "Category": "category",
                              "Subcategory": "subcategory", "Launch_Date": "launch_date",
                              "Cost_Price": "unit_cost", "Selling_Price": "list_price",
                              "Gross_Margin_Per_Unit": "gross_margin_per_unit"})
    if (RAW / "Stock_data.csv").exists():
        note("Stock_data.csv", "Unrelated file: equity price history (AABA ticker 2006–2017)", None,
             "Excluded from the pipeline", "Not part of the NorthBay star schema; no join key to any table.")
    return sales, inv, cal, sku


# ----------------------------------------------------------------------------- clean
def clean_sales(sales: pd.DataFrame) -> pd.DataFrame:
    n0 = len(sales)
    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    bad_dates = sales["date"].isna().sum()
    if bad_dates:
        note("sales_daily", "Unparseable dates", bad_dates, "Rows dropped", "Cannot place the sale in time.")
        sales = sales.dropna(subset=["date"])
    sales["sku_id"] = sales["sku_id"].astype(str).str.strip().str.upper()
    dups = sales.duplicated(subset=["date", "sku_id"]).sum()
    note("sales_daily", "Duplicate (date, sku) rows", dups,
         "Aggregated by sum" if dups else "None found — no action",
         "One row per SKU-day is the declared grain.")
    if dups:
        sales = sales.groupby(["date", "sku_id"], as_index=False).agg(
            units_sold=("units_sold", "sum"), revenue=("revenue", "sum"),
            unit_price=("unit_price", "mean"), promo_flag=("promo_flag", "max"))
    neg = (sales["units_sold"] < 0).sum()
    note("sales_daily", "Negative units_sold (returns?)", neg,
         "Clipped to 0" if neg else "None found — no action", "Returns are not demand.")
    sales["units_sold"] = sales["units_sold"].clip(lower=0)
    miss = sales[["units_sold", "revenue", "unit_price"]].isna().sum().sum()
    note("sales_daily", "Missing numeric values", miss,
         "Filled: units=0, revenue=units*price" if miss else "None found — no action",
         "A missing sales row means nothing sold that day.")
    sales["units_sold"] = sales["units_sold"].fillna(0)
    sales["unit_price"] = sales.groupby("sku_id")["unit_price"].transform(lambda s: s.fillna(s.median()))
    sales["revenue"] = sales["revenue"].fillna(sales["units_sold"] * sales["unit_price"])
    rev_mismatch = ((sales["revenue"] - sales["units_sold"] * sales["unit_price"]).abs() > 1).sum()
    note("sales_daily", "revenue != units_sold * unit_price (>₹1 gap)", rev_mismatch,
         "Kept as-is (revenue is authoritative)" if rev_mismatch else "None found — internally consistent",
         "Revenue is the booked figure; price is a reference.")
    note("sales_daily", "Rows retained after cleaning", len(sales), f"{n0} → {len(sales)}", "")
    return sales


def clean_sku(sku: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    sku["sku_id"] = sku["sku_id"].astype(str).str.strip().str.upper()
    sku["launch_date"] = pd.to_datetime(sku["launch_date"], errors="coerce")
    first_sale = sales.groupby("sku_id")["date"].min()
    sku = sku.merge(first_sale.rename("first_sale_date"), on="sku_id", how="left")
    late = (sku["launch_date"] > sku["first_sale_date"]).sum()
    note("sku_master", "launch_date AFTER first recorded sale", late,
         "launch_date overridden with first_sale_date (original kept as launch_date_master)",
         "Sales history is the stronger evidence; a product cannot sell before launch.")
    sku["launch_date_master"] = sku["launch_date"]
    sku["launch_date"] = sku[["launch_date", "first_sale_date"]].min(axis=1)
    negm = (sku["list_price"] < sku["unit_cost"]).sum()
    note("sku_master", "list_price below unit_cost (negative gross margin)", negm,
         "Kept; flagged margin_flag='negative' and surfaced to Finance",
         "Cannot be corrected from data — likely a master-data error or loss-leader; must be confirmed by client.")
    sku["margin_flag"] = np.where(sku["list_price"] < sku["unit_cost"], "negative", "positive")
    # Category / subcategory coherence (e.g. Kitchen→Cushion, Lighting→Cookware)
    expected = {"Chair": "Furniture", "Shelf": "Furniture", "Sofa": "Furniture", "Table": "Furniture",
                "Cabinet": "Furniture", "Cushion": "Home Decor", "Rug": "Home Decor", "Lamp": "Lighting",
                "Cookware": "Kitchen", "Organizer": "Storage"}
    mism = (sku["subcategory"].map(expected) != sku["category"]).sum()
    note("sku_master", "category/subcategory pairs that look inconsistent (e.g. Kitchen→Cushion)", mism,
         "Kept the master 'category' for reporting; added 'subcategory_group' as a sanity view",
         "Category is the client's planning hierarchy; we do not silently re-classify their catalogue.")
    sku["subcategory_group"] = sku["subcategory"].map(expected).fillna(sku["category"])
    sku["gross_margin_per_unit"] = sku["list_price"] - sku["unit_cost"]
    return sku


def clean_inventory(inv: pd.DataFrame, sales: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    inv["date"] = pd.to_datetime(inv["date"], errors="coerce")
    inv["sku_id"] = inv["sku_id"].astype(str).str.strip().str.upper()
    orphan = ~inv["sku_id"].isin(sales["sku_id"].unique())
    note("inventory_snapshots", "SKUs present in inventory but with NO sales history (SKU051–SKU200)",
         inv.loc[orphan, "sku_id"].nunique(),
         "Excluded from forecasting/risk; listed separately as 'unsellable-or-unmapped stock' for the client",
         "No demand signal → cannot forecast; likely discontinued or master-data gap. Raised as a finding.")
    orphans = inv[orphan]
    inv = inv[~orphan].copy()
    dups = inv.duplicated(subset=["date", "sku_id"]).sum()
    note("inventory_snapshots", "Duplicate (date, sku) snapshots", dups,
         "Kept last" if dups else "None found — no action", "")
    inv = inv.drop_duplicates(subset=["date", "sku_id"], keep="last")
    for c in ["on_hand_units", "on_order_units", "lead_time_days", "safety_stock", "reorder_point"]:
        neg = (inv[c] < 0).sum()
        if neg:
            note("inventory_snapshots", f"Negative {c}", neg, "Clipped to 0", "Physical quantities cannot be negative.")
        inv[c] = inv[c].clip(lower=0)
    note("inventory_snapshots", "Snapshot cadence", inv["date"].nunique(),
         "Monthly snapshots (1st of month); latest = " + str(inv["date"].max().date()),
         "Risk scoring uses the latest snapshot as the opening stock position at the forecast origin.")
    latest = inv[inv["date"] == inv["date"].max()].copy()
    return inv, latest, orphans


def clean_calendar(cal: pd.DataFrame) -> pd.DataFrame:
    cal["date"] = pd.to_datetime(cal["date"], errors="coerce")
    cal["holiday"] = cal["holiday"].fillna("None")
    cal["promotion_event"] = cal["promotion_event"].fillna("None")
    note("calendar", "Blank holiday / promotion_event cells", None,
         "Treated as 'None' (not missing)", "Blank means no event that day.")
    return cal


# ----------------------------------------------------------------------------- weekly build
def build_weekly(sales, cal, sku) -> pd.DataFrame:
    df = sales.merge(cal[["date", "is_weekend", "is_holiday", "season", "promotion_event"]], on="date", how="left")
    df["week_end"] = df["date"] + pd.to_timedelta(6 - df["date"].dt.weekday, unit="D")  # Sunday-ending weeks
    wk = df.groupby(["sku_id", "week_end"]).agg(
        units=("units_sold", "sum"), revenue=("revenue", "sum"), days=("date", "nunique"),
        promo_days=("promo_flag", "sum"), holiday_days=("is_holiday", "sum"),
        unit_price=("unit_price", "mean")).reset_index()
    partial = (wk["days"] < 7).sum()
    note("sales_weekly", "Partial weeks at series edges (fewer than 7 days)", partial,
         "Dropped", "A 3-day 'week' would look like a demand crash and poison the model.")
    wk = wk[wk["days"] == 7].drop(columns="days")
    # complete grid (every sku × every week) so silent zero-weeks are explicit
    grid = pd.MultiIndex.from_product([wk["sku_id"].unique(), np.sort(wk["week_end"].unique())],
                                      names=["sku_id", "week_end"]).to_frame(index=False)
    wk = grid.merge(wk, on=["sku_id", "week_end"], how="left")
    filled = wk["units"].isna().sum()
    note("sales_weekly", "SKU-weeks with no sales rows at all", filled, "Filled with 0 units", "No row = no sale.")
    wk[["units", "revenue", "promo_days", "holiday_days"]] = wk[["units", "revenue", "promo_days", "holiday_days"]].fillna(0)
    wk["unit_price"] = wk.groupby("sku_id")["unit_price"].transform(lambda s: s.ffill().bfill())
    # calendar features of the week itself (known in advance → safe to use for future weeks)
    wk["week_of_year"] = wk["week_end"].dt.isocalendar().week.astype(int)
    wk["month"] = wk["week_end"].dt.month
    wk["year"] = wk["week_end"].dt.year
    season_map = cal.set_index("date")["season"]
    wk["season"] = wk["week_end"].map(season_map)
    wk = wk.merge(sku[["sku_id", "category", "subcategory", "launch_date", "unit_cost", "list_price"]], on="sku_id", how="left")
    wk["weeks_since_launch"] = ((wk["week_end"] - wk["launch_date"]).dt.days // 7).clip(lower=0)
    wk = wk.sort_values(["sku_id", "week_end"]).reset_index(drop=True)
    wk["is_holdout"] = wk["week_end"] > FORECAST_ORIGIN
    return wk


def main():
    sales, inv, cal, sku = ingest()
    sales = clean_sales(sales)
    sku = clean_sku(sku, sales)
    inv, latest, orphans = clean_inventory(inv, sales)
    cal = clean_calendar(cal)
    weekly = build_weekly(sales, cal, sku)

    
    weekly.to_csv(OUT / "sales_weekly.csv", index=False)
    sku.to_csv(OUT / "sku_dim.csv", index=False)
    inv.to_csv(OUT / "inventory_clean.csv", index=False)
    latest.to_csv(OUT / "inventory_latest.csv", index=False)
    orphans.to_csv(OUT / "inventory_unmapped_skus.csv", index=False)
    sales.to_csv(OUT / "sales_daily_clean.csv", index=False)
    summary = dict(
        forecast_origin=str(FORECAST_ORIGIN.date()),
        skus=int(weekly["sku_id"].nunique()), weeks=int(weekly["week_end"].nunique()),
        first_week=str(weekly["week_end"].min().date()), last_week=str(weekly["week_end"].max().date()),
        holdout_weeks=int(weekly.loc[weekly["is_holdout"], "week_end"].nunique()),
        total_units=int(weekly["units"].sum()), total_revenue=float(weekly["revenue"].sum()),
        unmapped_inventory_skus=int(orphans["sku_id"].nunique()),
        unmapped_inventory_value_latest=float(orphans[orphans["date"] == orphans["date"].max()]["inventory_value"].sum()),
    )
    (OUT / "data_quality_log.json").write_text(json.dumps(dict(summary=summary, issues=log), indent=2, default=str))
    print("Pipeline complete.")
    print(json.dumps(summary, indent=2))
    for i in log:
        print(f"- [{i['table']}] {i['issue']} (n={i['count']}) → {i['action']}")


if __name__ == "__main__":
    main()
