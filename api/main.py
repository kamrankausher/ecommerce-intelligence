"""
FastAPI application — churn prediction + A/B results API.
"""
import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
import joblib
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ml.feature_engineering import build_features, FEATURE_COLS, get_db, SNAPSHOT_DATE
from experiments.ab_simulator import run_ab_test, power_analysis, run_all_experiments

ARTIFACTS = os.path.join(os.path.dirname(__file__), "..", "artifacts")
DB_PATH   = os.path.join(os.path.dirname(__file__), "..", "ecommerce.db")

_model     = None
_explainer = None
_features  = None


def load_model():
    global _model, _explainer, _features
    model_path     = os.path.join(ARTIFACTS, "model.pkl")
    explainer_path = os.path.join(ARTIFACTS, "explainer.pkl")
    features_path  = os.path.join(ARTIFACTS, "features.json")
    if not os.path.exists(model_path):
        raise RuntimeError("Model not found. Run python setup.py first.")
    _model     = joblib.load(model_path)
    _explainer = joblib.load(explainer_path)
    with open(features_path) as f:
        _features = json.load(f)
    print("Model loaded.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="E-commerce Intelligence API",
    description="Churn prediction, A/B testing, and analytics endpoints",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ──────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    customer_id: str


class ABTestRequest(BaseModel):
    experiment_id: str
    alpha: Optional[float] = 0.05


class PowerRequest(BaseModel):
    base_rate: float
    mde: float
    alpha: Optional[float] = 0.05
    power: Optional[float] = 0.80


# ── Helpers ──────────────────────────────────────────────────────────────────

def classify_risk(p: float) -> str:
    if p >= 0.65:  return "HIGH"
    if p >= 0.35:  return "MEDIUM"
    return "LOW"


def get_customer_features(customer_id: str) -> np.ndarray:
    conn = sqlite3.connect(os.path.abspath(DB_PATH))
    df = build_features(conn)
    row = df[df["customer_id"] == customer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    return row[_features].values


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict")
def predict_churn(req: PredictRequest):
    """Return churn probability, risk tier, and top SHAP drivers."""
    features = get_customer_features(req.customer_id)
    proba    = float(_model.predict_proba(features)[0][1])
    shap_vals = _explainer.shap_values(features)[0]

    # Top 3 SHAP drivers
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:3]
    top_drivers = [
        {
            "feature":    _features[i],
            "shap_value": round(float(shap_vals[i]), 4),
            "direction":  "increases" if shap_vals[i] > 0 else "decreases",
        }
        for i in top_idx
    ]

    return {
        "customer_id":       req.customer_id,
        "churn_probability": round(proba, 4),
        "risk_tier":         classify_risk(proba),
        "top_drivers":       top_drivers,
    }


@app.post("/ab-test")
def ab_test(req: ABTestRequest):
    """Run A/B test analysis for a given experiment."""
    result = run_ab_test(req.experiment_id, alpha=req.alpha)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/ab-tests")
def all_ab_tests():
    """Run all A/B tests and return results."""
    return run_all_experiments()


@app.post("/power-analysis")
def power(req: PowerRequest):
    """Calculate minimum sample size for a desired MDE."""
    return power_analysis(req.base_rate, req.mde, req.alpha, req.power)


@app.get("/kpis")
def kpis():
    """Top-line KPIs from the database."""
    conn = sqlite3.connect(os.path.abspath(DB_PATH))
    c = conn.cursor()

    total_revenue   = c.execute("SELECT ROUND(SUM(price+freight_value),2) FROM order_items").fetchone()[0]
    total_orders    = c.execute("SELECT COUNT(DISTINCT order_id) FROM orders WHERE status='delivered'").fetchone()[0]
    total_customers = c.execute("SELECT COUNT(DISTINCT customer_id) FROM customers").fetchone()[0]
    avg_order_val   = round(total_revenue / total_orders, 2) if total_orders else 0
    review_avg      = c.execute("SELECT ROUND(AVG(score),2) FROM reviews").fetchone()[0]
    late_pct        = c.execute("""
        SELECT ROUND(AVG(CASE WHEN delivered_date > estimated_delivery THEN 1.0 ELSE 0 END)*100, 1)
        FROM orders WHERE status='delivered' AND delivered_date IS NOT NULL
    """).fetchone()[0]
    conn.close()

    return {
        "total_revenue":    total_revenue,
        "total_orders":     total_orders,
        "total_customers":  total_customers,
        "avg_order_value":  avg_order_val,
        "avg_review_score": review_avg,
        "late_delivery_pct": late_pct,
    }


@app.get("/revenue-by-state")
def revenue_by_state():
    conn = sqlite3.connect(os.path.abspath(DB_PATH))
    df = pd.read_sql_query("""
        SELECT c.customer_state AS state,
               ROUND(SUM(oi.price + oi.freight_value), 2) AS revenue,
               COUNT(DISTINCT o.order_id) AS orders
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN order_items oi ON o.order_id = oi.order_id
        WHERE o.status = 'delivered'
        GROUP BY c.customer_state
        ORDER BY revenue DESC
        LIMIT 15
    """, conn)
    conn.close()
    return df.to_dict(orient="records")


@app.get("/monthly-revenue")
def monthly_revenue():
    conn = sqlite3.connect(os.path.abspath(DB_PATH))
    df = pd.read_sql_query("""
        SELECT STRFTIME('%Y-%m', o.order_date) AS month,
               ROUND(SUM(oi.price + oi.freight_value), 2) AS revenue,
               COUNT(DISTINCT o.order_id) AS orders
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        WHERE o.status = 'delivered'
        GROUP BY month
        ORDER BY month
    """, conn)
    conn.close()
    return df.to_dict(orient="records")


@app.get("/cohort-retention")
def cohort_retention():
    """Monthly cohort retention rates."""
    conn = sqlite3.connect(os.path.abspath(DB_PATH))
    df = pd.read_sql_query("""
        WITH first_orders AS (
            SELECT customer_id,
                   STRFTIME('%Y-%m', MIN(order_date)) AS cohort_month
            FROM orders WHERE status = 'delivered'
            GROUP BY customer_id
        ),
        all_orders AS (
            SELECT o.customer_id,
                   STRFTIME('%Y-%m', o.order_date) AS order_month
            FROM orders o
            WHERE status = 'delivered'
        ),
        cohort_data AS (
            SELECT fo.cohort_month,
                   ao.order_month,
                   COUNT(DISTINCT ao.customer_id) AS users
            FROM first_orders fo
            JOIN all_orders ao ON fo.customer_id = ao.customer_id
            GROUP BY fo.cohort_month, ao.order_month
        ),
        cohort_sizes AS (
            SELECT cohort_month, SUM(users) AS cohort_size
            FROM cohort_data
            WHERE cohort_month = order_month
            GROUP BY cohort_month
        )
        SELECT cd.cohort_month,
               cd.order_month,
               CAST(
                   (julianday(cd.order_month||'-01')
                   - julianday(cd.cohort_month||'-01')) / 30.0
               AS INTEGER) AS period,
               ROUND(CAST(cd.users AS FLOAT) / cs.cohort_size * 100, 1) AS retention_pct
        FROM cohort_data cd
        JOIN cohort_sizes cs ON cd.cohort_month = cs.cohort_month
        WHERE cd.cohort_month >= '2017-02'
          AND cd.cohort_month <= '2018-03'
        ORDER BY cd.cohort_month, period
    """, conn)
    conn.close()
    return df.to_dict(orient="records")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
