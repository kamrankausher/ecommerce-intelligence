"""
Feature engineering for churn prediction.
Builds RFM + behavioural features directly from the database.
"""
import sqlite3
import pandas as pd
import numpy as np
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "ecommerce.db")
SNAPSHOT_DATE = "2018-07-01"


def get_db():
    return sqlite3.connect(os.path.abspath(DB_PATH))


def build_features(conn=None) -> pd.DataFrame:
    own_conn = conn is None
    if own_conn:
        conn = get_db()

    rfm_query = """
    WITH customer_orders AS (
        SELECT
            o.customer_id,
            COUNT(DISTINCT o.order_id)                          AS frequency,
            SUM(oi.price + oi.freight_value)                    AS monetary,
            MAX(o.order_date)                                   AS last_order_date,
            MIN(o.order_date)                                   AS first_order_date,
            JULIANDAY(:snapshot) - JULIANDAY(MAX(o.order_date)) AS recency_days
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        WHERE o.status = 'delivered'
          AND o.order_date < :snapshot
        GROUP BY o.customer_id
    ),
    review_stats AS (
        SELECT
            o.customer_id,
            AVG(r.score)                        AS avg_review_score,
            COUNT(r.review_id)                  AS review_count
        FROM orders o
        LEFT JOIN reviews r ON o.order_id = r.order_id
        WHERE o.order_date < :snapshot
        GROUP BY o.customer_id
    ),
    delivery_stats AS (
        SELECT
            o.customer_id,
            AVG(CASE
                WHEN o.delivered_date > o.estimated_delivery THEN 1 ELSE 0
            END)                                AS late_delivery_rate,
            AVG(JULIANDAY(o.delivered_date)
              - JULIANDAY(o.order_date))        AS avg_delivery_days
        FROM orders o
        WHERE o.status = 'delivered'
          AND o.order_date < :snapshot
          AND o.delivered_date IS NOT NULL
        GROUP BY o.customer_id
    )
    SELECT
        co.customer_id,
        co.frequency,
        ROUND(co.monetary, 2)                           AS monetary,
        ROUND(co.recency_days, 1)                       AS recency_days,
        ROUND(co.monetary / NULLIF(co.frequency, 0), 2) AS avg_order_value,
        ROUND(JULIANDAY(:snapshot)
            - JULIANDAY(co.first_order_date), 1)        AS customer_age_days,
        COALESCE(rs.avg_review_score, 3.0)              AS avg_review_score,
        COALESCE(rs.review_count, 0)                    AS review_count,
        COALESCE(ds.late_delivery_rate, 0.0)            AS late_delivery_rate,
        COALESCE(ds.avg_delivery_days, 10.0)            AS avg_delivery_days,
        c.customer_state
    FROM customer_orders co
    JOIN customers c ON co.customer_id = c.customer_id
    LEFT JOIN review_stats rs ON co.customer_id = rs.customer_id
    LEFT JOIN delivery_stats ds ON co.customer_id = ds.customer_id
    """

    df = pd.read_sql_query(rfm_query, conn, params={"snapshot": SNAPSHOT_DATE})

    # Churn label: no order in last 90 days before snapshot
    df["churned"] = (df["recency_days"] > 90).astype(int)

    # RFM scores (quantile-based, 1-5)
    def safe_qcut(series, q, labels):
        try:
            return pd.qcut(series, q=q, labels=labels, duplicates="drop").astype(int)
        except ValueError:
            # Fall back to rank-based scoring if too many duplicates
            return pd.cut(series, bins=len(labels), labels=labels, duplicates="drop").astype(int)

    df["r_score"] = safe_qcut(df["recency_days"], q=5, labels=[5, 4, 3, 2, 1])
    df["f_score"] = safe_qcut(df["frequency"].clip(upper=df["frequency"].quantile(0.99)), q=5, labels=[1, 2, 3, 4, 5])
    df["m_score"] = safe_qcut(df["monetary"].clip(upper=df["monetary"].quantile(0.99)), q=5, labels=[1, 2, 3, 4, 5])
    df["rfm_score"] = df["r_score"] + df["f_score"] + df["m_score"]

    # State encoding (top 5 states + other)
    top_states = ["SP", "RJ", "MG", "RS", "PR"]
    df["state_encoded"] = df["customer_state"].apply(
        lambda s: top_states.index(s) + 1 if s in top_states else 0
    )

    # Log transforms for skewed features
    df["log_monetary"] = np.log1p(df["monetary"])
    df["log_recency"] = np.log1p(df["recency_days"])

    if own_conn:
        conn.close()
    return df


FEATURE_COLS = [
    "frequency", "log_monetary", "log_recency", "avg_order_value",
    "customer_age_days", "avg_review_score", "review_count",
    "late_delivery_rate", "avg_delivery_days", "rfm_score",
    "r_score", "f_score", "m_score", "state_encoded",
]

DISPLAY_FEATURE_NAMES = {
    "frequency": "Order Frequency",
    "log_monetary": "Total Spend (log)",
    "log_recency": "Days Since Last Order (log)",
    "avg_order_value": "Avg Order Value",
    "customer_age_days": "Customer Age (days)",
    "avg_review_score": "Avg Review Score",
    "review_count": "Review Count",
    "late_delivery_rate": "Late Delivery Rate",
    "avg_delivery_days": "Avg Delivery Days",
    "rfm_score": "RFM Score",
    "r_score": "Recency Score",
    "f_score": "Frequency Score",
    "m_score": "Monetary Score",
    "state_encoded": "Customer State",
}


if __name__ == "__main__":
    df = build_features()
    print(f"Features shape: {df.shape}")
    print(f"Churn rate: {df['churned'].mean():.1%}")
    print(df[FEATURE_COLS].describe().round(2))
