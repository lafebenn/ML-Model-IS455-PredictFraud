"""
FastAPI service to score orders with fraud_model.joblib via Supabase.

Rebuilds the same feature matrix as the CRISP-DM notebook, then writes
probabilities to order_fraud_predictions.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from supabase import Client, create_client

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("FRAUD_MODEL_PATH", str(BASE_DIR / "fraud_model.joblib")))

# PostgREST page size for reads (keep below provider limits)
READ_PAGE_SIZE = 1000
# Upsert batch size (requirement)
UPSERT_CHUNK = 500
# IN(...) filter chunk size for fetches
IN_CHUNK = 200

pipeline: object | None = None
supabase: Client | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def fetch_all_rows(
    sb: Client,
    table: str,
    select: str,
    *,
    page_size: int = READ_PAGE_SIZE,
) -> list[dict]:
    """Paginate .range() until a short page."""
    out: list[dict] = []
    offset = 0
    while True:
        res = (
            sb.table(table)
            .select(select)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = res.data or []
        out.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return out


def fetch_by_in_chunks(
    sb: Client,
    table: str,
    select: str,
    column: str,
    values: list,
    chunk_size: int = IN_CHUNK,
) -> list[dict]:
    """WHERE column IN (...) in chunks to avoid URL limits."""
    out: list[dict] = []
    for i in range(0, len(values), chunk_size):
        chunk = values[i : i + chunk_size]
        if not chunk:
            continue
        res = sb.table(table).select(select).in_(column, chunk).execute()
        out.extend(res.data or [])
    return out


def load_scored_order_ids(sb: Client) -> set[int]:
    rows = fetch_all_rows(sb, "order_fraud_predictions", "order_id")
    return {int(r["order_id"]) for r in rows if r.get("order_id") is not None}


def load_all_order_ids(sb: Client) -> list[int]:
    rows = fetch_all_rows(sb, "orders", "order_id")
    return [int(r["order_id"]) for r in rows if r.get("order_id") is not None]


def order_items_aggregates(sb: Client, order_ids: list[int]) -> pd.DataFrame:
    """Mirror SQL_ITEMS: per-order aggregates from order_items + products."""
    if not order_ids:
        return pd.DataFrame(
            columns=[
                "order_id",
                "n_lines",
                "total_qty",
                "n_distinct_products",
                "sum_line_total",
                "avg_unit_price",
                "min_unit_price",
                "max_unit_price",
                "n_distinct_categories",
            ]
        )
    raw_items = fetch_by_in_chunks(
        sb,
        "order_items",
        "order_id, product_id, quantity, unit_price, line_total",
        "order_id",
        order_ids,
    )
    if not raw_items:
        return pd.DataFrame(
            columns=[
                "order_id",
                "n_lines",
                "total_qty",
                "n_distinct_products",
                "sum_line_total",
                "avg_unit_price",
                "min_unit_price",
                "max_unit_price",
                "n_distinct_categories",
            ]
        )
    oi = pd.DataFrame(raw_items)
    for col in ["quantity", "unit_price", "line_total"]:
        if col in oi.columns:
            oi[col] = pd.to_numeric(oi[col], errors="coerce")
    oi["order_id"] = oi["order_id"].astype(int)
    pids = oi["product_id"].dropna().unique().tolist()
    if pids:
        prod_rows = fetch_by_in_chunks(
            sb, "products", "product_id, category", "product_id", [int(x) for x in pids]
        )
        prods = pd.DataFrame(prod_rows) if prod_rows else pd.DataFrame(columns=["product_id", "category"])
        if not prods.empty:
            prods["product_id"] = prods["product_id"].astype(int)
            oi = oi.merge(prods, on="product_id", how="left")
        else:
            oi["category"] = pd.NA
    else:
        oi["category"] = pd.NA
    g = oi.groupby("order_id", as_index=False)
    agg = g.agg(
        n_lines=("order_id", "count"),
        total_qty=("quantity", "sum"),
        n_distinct_products=("product_id", "nunique"),
        sum_line_total=("line_total", "sum"),
        avg_unit_price=("unit_price", "mean"),
        min_unit_price=("unit_price", "min"),
        max_unit_price=("unit_price", "max"),
        n_distinct_categories=("category", "nunique"),
    )
    return agg


def build_unscored_orders_frame(sb: Client, unscored_ids: list[int]) -> pd.DataFrame:
    """Same logical joins as legacy SQLite: orders + customers + shipments + item aggregates."""
    if not unscored_ids:
        return pd.DataFrame()

    order_select = (
        "order_id, customer_id, order_datetime, billing_zip, shipping_zip, shipping_state, "
        "payment_method, device_type, ip_country, promo_used, promo_code, order_subtotal, "
        "shipping_fee, tax_amount, order_total, risk_score, is_fraud"
    )
    ord_rows = fetch_by_in_chunks(sb, "orders", order_select, "order_id", unscored_ids)
    df_o = pd.DataFrame(ord_rows)
    if df_o.empty:
        return df_o

    cust_ids = df_o["customer_id"].dropna().unique().astype(int).tolist()
    cust_select = (
        "customer_id, gender, birthdate, created_at, city, state, zip_code, "
        "customer_segment, loyalty_tier, is_active"
    )
    cust_rows = fetch_by_in_chunks(sb, "customers", cust_select, "customer_id", cust_ids)
    df_c = pd.DataFrame(cust_rows)
    rename_c = {
        "city": "cust_city",
        "state": "cust_state",
        "zip_code": "cust_zip",
        "is_active": "customer_is_active",
        "created_at": "customer_created_at",
    }
    df_c = df_c.rename(columns={k: v for k, v in rename_c.items() if k in df_c.columns})

    df = df_o.merge(df_c, on="customer_id", how="inner")

    ship_rows = fetch_by_in_chunks(
        sb,
        "shipments",
        "order_id, carrier, shipping_method, distance_band, promised_days, actual_days",
        "order_id",
        unscored_ids,
    )
    df_s = pd.DataFrame(ship_rows)
    if not df_s.empty:
        df = df.merge(df_s, on="order_id", how="left")
    else:
        for c in ["carrier", "shipping_method", "distance_band", "promised_days", "actual_days"]:
            df[c] = pd.NA

    agg = order_items_aggregates(sb, unscored_ids)
    df = df.merge(agg, on="order_id", how="left")
    return df


def wrangle_for_scoring(dfin: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Match training notebook: engineer features, return X and order_id index."""
    out = dfin.copy()
    out["order_datetime"] = pd.to_datetime(out["order_datetime"], format='ISO8601', utc=True)
    out["birthdate"] = pd.to_datetime(out["birthdate"], errors="coerce", utc=True)
    out["customer_created_at"] = pd.to_datetime(out["customer_created_at"], utc=True)
    out["account_tenure_days"] = (out["order_datetime"] - out["customer_created_at"]).dt.days
    out["age_at_order"] = (out["order_datetime"] - out["birthdate"]).dt.days / 365.25
    out["delivery_delay_days"] = out["actual_days"] - out["promised_days"]
    out["unit_price_spread"] = out["max_unit_price"] - out["min_unit_price"]
    order_ids = out["order_id"]
    drop_cols = [
        "order_id",
        "customer_id",
        "is_fraud",
        "order_datetime",
        "birthdate",
        "customer_created_at",
    ]
    X = out.drop(columns=[c for c in drop_cols if c in out.columns], errors="ignore")
    return X, order_ids


def expected_feature_columns(pipe) -> list[str]:
    ct = pipe.named_steps["preprocessor"]
    num = list(ct.transformers_[0][2])
    cat = list(ct.transformers_[1][2])
    return num + cat


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, supabase
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not url or not key:
        raise RuntimeError(
            "Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY environment variables."
        )
    pipeline = joblib.load(MODEL_PATH)
    supabase = create_client(url, key)
    yield
    pipeline = None
    supabase = None


app = FastAPI(title="Fraud scoring API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://intext-test-app.vercel.app/select-customer"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run-scoring")
def run_scoring():
    if pipeline is None or supabase is None:
        return {"error": "model or supabase not initialized", "scored": 0}

    sb = supabase
    unscored = load_all_order_ids(sb)

    if not unscored:
        return {"scored": 0}

    df = build_unscored_orders_frame(sb, unscored)
    if df.empty:
        return {"scored": 0}

    X, order_ids = wrangle_for_scoring(df)
    required = expected_feature_columns(pipeline)
    missing = [c for c in required if c not in X.columns]
    if missing:
        return {"error": f"missing columns for model: {missing}", "scored": 0}

    X = X[required]
    proba = pipeline.predict_proba(X)[:, 1]
    scored_at = _utc_now_iso()

    rows: list[dict] = []
    for oid, p in zip(order_ids.tolist(), proba.tolist()):
        pf = float(p)
        rows.append(
            {
                "order_id": int(oid),
                "fraud_probability": pf,
                "predicted_fraud": 1 if pf >= 0.5 else 0,
                "scored_at": scored_at,
            }
        )

    total = 0
    for i in range(0, len(rows), UPSERT_CHUNK):
        chunk = rows[i : i + UPSERT_CHUNK]
        sb.table("order_fraud_predictions").upsert(chunk, on_conflict="order_id").execute()
        total += len(chunk)

    return {"scored": total}
