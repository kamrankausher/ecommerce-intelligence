"""
E-commerce Growth Intelligence — Streamlit Dashboard
5 tabs: KPIs | SQL Analytics | A/B Tests | Churn Intelligence | ML Experiments
"""
import os
import sys
import json
import sqlite3
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.ab_simulator import run_all_experiments, power_analysis
from ml.feature_engineering import build_features, FEATURE_COLS, DISPLAY_FEATURE_NAMES

DB_PATH       = os.path.join(os.path.dirname(__file__), "..", "ecommerce.db")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")

st.set_page_config(
    page_title="E-commerce Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Premium Styling (Deep Space Purple & Gold) ───────────────────────────────
st.markdown("""
<style>
/* Global background and typography */
.stApp {
    background-color: #0B090A;
    color: #E0E1DD;
    font-family: 'Inter', 'Roboto', sans-serif;
}
/* Metric Cards */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #161A1D, #0B090A);
    border: 1px solid #3A0CA3;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 4px 15px rgba(114, 9, 183, 0.2);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(255, 215, 0, 0.3);
    border-color: #FFD700;
}
[data-testid="stMetricValue"] { 
    font-size: 2rem !important; 
    font-weight: 700; 
    color: #FFD700; 
}
[data-testid="stMetricLabel"] { 
    font-size: 0.9rem !important; 
    color: #8D99AE; 
    text-transform: uppercase;
    letter-spacing: 1px;
}
[data-testid="stMetricDelta"] svg {
    color: #00F5D4 !important;
}
/* General Layout */
.block-container { padding: 2rem 3rem; }
h1 { font-size: 2rem !important; font-weight: 700; color: #FFFFFF; text-shadow: 0 2px 10px rgba(157, 78, 221, 0.5); }
h2 { font-size: 1.4rem !important; font-weight: 600; color: #E0E1DD; border-bottom: 1px solid #3A0CA3; padding-bottom: 0.5rem; }
/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    background-color: #161A1D;
    border-radius: 10px;
    padding: 0.5rem;
}
.stTabs [data-baseweb="tab"] { 
    font-size: 1rem; 
    font-weight: 600; 
    color: #8D99AE;
    border-radius: 8px;
    margin-right: 0.5rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #7209B7, #3A0CA3);
    color: #FFFFFF !important;
    box-shadow: 0 4px 10px rgba(114, 9, 183, 0.5);
}
</style>
""", unsafe_allow_html=True)

import plotly.io as pio
pio.templates.default = "plotly_dark"
pio.templates["plotly_dark"].layout.paper_bgcolor = "rgba(0,0,0,0)"
pio.templates["plotly_dark"].layout.plot_bgcolor = "rgba(0,0,0,0)"
pio.templates["plotly_dark"].layout.font.family = "Inter, sans-serif"


# ── Data loaders (cached) ────────────────────────────────────────────────────



@st.cache_data
def load_kpis():
    conn = sqlite3.connect(os.path.abspath(DB_PATH))
    c = conn.cursor()
    r = {}
    r["revenue"]    = c.execute("SELECT ROUND(SUM(price+freight_value),2) FROM order_items").fetchone()[0]
    r["orders"]     = c.execute("SELECT COUNT(DISTINCT order_id) FROM orders WHERE status='delivered'").fetchone()[0]
    r["customers"]  = c.execute("SELECT COUNT(DISTINCT customer_id) FROM customers").fetchone()[0]
    r["avg_order"]  = round(r["revenue"] / r["orders"], 2) if r["orders"] else 0
    r["avg_review"] = c.execute("SELECT ROUND(AVG(score),2) FROM reviews").fetchone()[0]
    r["late_pct"]   = c.execute("""
        SELECT ROUND(AVG(CASE WHEN delivered_date > estimated_delivery THEN 1.0 ELSE 0 END)*100,1)
        FROM orders WHERE status='delivered' AND delivered_date IS NOT NULL
    """).fetchone()[0]
    conn.close()
    return r


@st.cache_data
def load_monthly_revenue():
    conn = sqlite3.connect(os.path.abspath(DB_PATH))
    df = pd.read_sql_query("""
        SELECT STRFTIME('%Y-%m', o.order_date) AS month,
               ROUND(SUM(oi.price+oi.freight_value),2) AS revenue,
               COUNT(DISTINCT o.order_id) AS orders
        FROM orders o JOIN order_items oi ON o.order_id=oi.order_id
        WHERE o.status='delivered' GROUP BY month ORDER BY month
    """, conn)
    conn.close()
    return df


@st.cache_data
def load_revenue_by_state():
    conn = sqlite3.connect(os.path.abspath(DB_PATH))
    df = pd.read_sql_query("""
        SELECT c.customer_state AS state,
               ROUND(SUM(oi.price+oi.freight_value),2) AS revenue,
               COUNT(DISTINCT o.order_id) AS orders
        FROM orders o JOIN customers c ON o.customer_id=c.customer_id
        JOIN order_items oi ON o.order_id=oi.order_id
        WHERE o.status='delivered' GROUP BY state ORDER BY revenue DESC LIMIT 15
    """, conn)
    conn.close()
    return df


@st.cache_data
def load_category_revenue():
    conn = sqlite3.connect(os.path.abspath(DB_PATH))
    df = pd.read_sql_query("""
        SELECT p.category,
               ROUND(SUM(oi.price),2) AS revenue,
               COUNT(oi.id) AS items_sold
        FROM order_items oi JOIN products p ON oi.product_id=p.product_id
        JOIN orders o ON oi.order_id=o.order_id
        WHERE o.status='delivered' GROUP BY p.category ORDER BY revenue DESC
    """, conn)
    conn.close()
    return df


@st.cache_data
def load_cohort_data():
    conn = sqlite3.connect(os.path.abspath(DB_PATH))
    df = pd.read_sql_query("""
        WITH fo AS (
            SELECT customer_id, STRFTIME('%Y-%m', MIN(order_date)) AS cohort
            FROM orders WHERE status='delivered' GROUP BY customer_id
        ),
        ao AS (
            SELECT o.customer_id, STRFTIME('%Y-%m', o.order_date) AS om
            FROM orders o WHERE status='delivered'
        ),
        cd AS (
            SELECT fo.cohort, ao.om,
                   CAST((julianday(ao.om||'-01')-julianday(fo.cohort||'-01'))/30.0 AS INT) AS period,
                   COUNT(DISTINCT ao.customer_id) AS users
            FROM fo JOIN ao ON fo.customer_id=ao.customer_id GROUP BY fo.cohort, ao.om
        ),
        cs AS (SELECT cohort, users AS sz FROM cd WHERE period=0)
        SELECT cd.cohort, cd.period,
               ROUND(CAST(cd.users AS FLOAT)/cs.sz*100,1) AS retention
        FROM cd JOIN cs ON cd.cohort=cs.cohort
        WHERE cd.cohort>='2017-03' AND cd.cohort<='2018-04' AND cd.period<=6
        ORDER BY cd.cohort, cd.period
    """, conn)
    conn.close()
    return df


@st.cache_data
def load_seller_performance():
    conn = sqlite3.connect(os.path.abspath(DB_PATH))
    df = pd.read_sql_query("""
        SELECT oi.seller_id,
               s.seller_state,
               COUNT(DISTINCT oi.order_id)     AS orders,
               ROUND(SUM(oi.price),2)           AS revenue,
               ROUND(AVG(r.score),2)            AS avg_score,
               ROUND(AVG(CASE WHEN o.delivered_date>o.estimated_delivery THEN 1.0 ELSE 0 END)*100,1) AS late_pct
        FROM order_items oi
        JOIN sellers s ON oi.seller_id=s.seller_id
        JOIN orders o ON oi.order_id=o.order_id
        LEFT JOIN reviews r ON o.order_id=r.order_id
        WHERE o.status='delivered'
        GROUP BY oi.seller_id HAVING orders>=5
        ORDER BY revenue DESC LIMIT 50
    """, conn)
    conn.close()
    return df


@st.cache_data
def load_ab_results():
    return run_all_experiments()


@st.cache_data
def load_predictions():
    path = os.path.join(ARTIFACTS_DIR, "predictions.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def load_model_metrics():
    path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_optuna_trials():
    path = os.path.join(ARTIFACTS_DIR, "optuna_trials.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def load_features_df():
    return build_features()


# ── Color palette (Premium Deep Space & Gold) ────────────────────────────────
PURPLE = "#9D4EDD"  # Neon purple
TEAL   = "#00F5D4"  # Neon teal
AMBER  = "#FFD700"  # Rich gold
RED    = "#FF006E"  # Vibrant pink/red
BLUE   = "#3A0CA3"  # Deep blue/purple
GRAY   = "#8D99AE"  # Metallic gray


# ════════════════════════════════════════════════════════════════════════════
# App Layout
# ════════════════════════════════════════════════════════════════════════════

st.title("📊 E-commerce Growth Intelligence Platform")
st.caption("Olist-inspired dataset · XGBoost churn model · A/B experimentation · SQL analytics")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 KPI Overview",
    "🗃️ SQL Analytics",
    "🧪 A/B Experiments",
    "🤖 Churn Intelligence",
    "⚗️ ML Experiments",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — KPI OVERVIEW
# ════════════════════════════════════════════════════════════════════════════

with tab1:
    kpis = load_kpis()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Revenue",    f"R${kpis['revenue']:,.0f}")
    c2.metric("Orders Delivered", f"{kpis['orders']:,}")
    c3.metric("Customers",        f"{kpis['customers']:,}")
    c4.metric("Avg Order Value",  f"R${kpis['avg_order']:.2f}")
    c5.metric("Avg Review",       f"{kpis['avg_review']:.2f} / 5")
    c6.metric("Late Deliveries",  f"{kpis['late_pct']}%")

    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        df_rev = load_monthly_revenue()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=df_rev["month"], y=df_rev["revenue"],
                   name="Revenue (R$)", marker_color=PURPLE, opacity=0.8),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=df_rev["month"], y=df_rev["orders"],
                       name="Orders", line=dict(color=TEAL, width=2),
                       mode="lines+markers"),
            secondary_y=True
        )
        fig.update_layout(
            title="Monthly Revenue & Orders",
            height=320, hovermode="x unified",
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        fig.update_yaxes(title_text="Revenue (R$)", secondary_y=False)
        fig.update_yaxes(title_text="Orders", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        df_cat = load_category_revenue().head(8)
        fig2 = px.bar(df_cat, x="revenue", y="category", orientation="h",
                      color="revenue", color_continuous_scale="Purples",
                      title="Revenue by Category")
        fig2.update_layout(height=320, showlegend=False,
                           coloraxis_showscale=False,
                           margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        df_state = load_revenue_by_state()
        fig3 = px.bar(df_state, x="state", y="revenue",
                      color="revenue", color_continuous_scale="Teal",
                      title="Revenue by State (Top 15)")
        fig3.update_layout(height=280, showlegend=False,
                           coloraxis_showscale=False,
                           margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        conn = sqlite3.connect(os.path.abspath(DB_PATH))
        df_status = pd.read_sql_query("""
            SELECT status, COUNT(*) AS cnt FROM orders GROUP BY status
        """, conn)
        conn.close()
        fig4 = px.pie(df_status, values="cnt", names="status",
                      title="Order Status Distribution",
                      color_discrete_sequence=[TEAL, RED, GRAY])
        fig4.update_layout(height=280, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig4, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — SQL ANALYTICS
# ════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Cohort Retention Analysis")
    st.caption("Window functions + CTEs — which customer cohorts retain best over 6 months")

    df_cohort = load_cohort_data()
    if not df_cohort.empty:
        pivot = df_cohort.pivot_table(index="cohort", columns="period", values="retention", aggfunc="mean")
        fig_h = px.imshow(
            pivot,
            color_continuous_scale="RdYlGn",
            aspect="auto",
            title="Monthly Cohort Retention (%) — rows = acquisition month, cols = months after first purchase",
            text_auto=".0f",
            zmin=0, zmax=100,
        )
        fig_h.update_layout(height=420, margin=dict(l=0, r=0, t=50, b=0))
        fig_h.update_coloraxes(colorbar_title="Retention %")
        st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("---")
    st.subheader("Seller Performance")
    st.caption("RANK() OVER(PARTITION BY state) — top sellers by revenue with review score & late delivery rate")

    df_sellers = load_seller_performance()
    col1, col2 = st.columns(2)

    with col1:
        fig_s = px.scatter(
            df_sellers, x="revenue", y="avg_score",
            size="orders", color="late_pct",
            color_continuous_scale="RdYlGn_r",
            hover_data=["seller_id", "seller_state", "orders"],
            title="Seller Revenue vs. Review Score (size = order volume, color = late %)",
        )
        fig_s.update_layout(height=350, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_s, use_container_width=True)

    with col2:
        top10 = df_sellers.head(10)[["seller_id", "seller_state",
                                     "orders", "revenue", "avg_score", "late_pct"]]
        top10.columns = ["Seller", "State", "Orders", "Revenue (R$)", "Avg Score", "Late %"]
        st.dataframe(top10, use_container_width=True, height=350)

    st.markdown("---")
    st.subheader("Revenue by State — SQL Window Function")
    st.caption("Running total using SUM() OVER(ORDER BY revenue DESC ROWS UNBOUNDED PRECEDING)")

    df_st = load_revenue_by_state()
    df_st["running_total"]   = df_st["revenue"].cumsum()
    df_st["pct_of_total"]    = (df_st["revenue"] / df_st["revenue"].sum() * 100).round(1)
    df_st["cumulative_pct"]  = df_st["pct_of_total"].cumsum().round(1)

    fig_combo = make_subplots(specs=[[{"secondary_y": True}]])
    fig_combo.add_trace(
        go.Bar(x=df_st["state"], y=df_st["revenue"],
               name="Revenue", marker_color=PURPLE, opacity=0.85),
        secondary_y=False
    )
    fig_combo.add_trace(
        go.Scatter(x=df_st["state"], y=df_st["cumulative_pct"],
                   name="Cumulative %", line=dict(color=AMBER, width=2, dash="dot"),
                   mode="lines+markers"),
        secondary_y=True
    )
    fig_combo.update_layout(height=300, hovermode="x unified",
                            margin=dict(l=0, r=0, t=20, b=0))
    fig_combo.update_yaxes(title_text="Revenue (R$)", secondary_y=False)
    fig_combo.update_yaxes(title_text="Cumulative %", secondary_y=True, range=[0, 110])
    st.plotly_chart(fig_combo, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — A/B EXPERIMENTS
# ════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Experiment Results")
    st.caption("Two-proportion z-test · 95% confidence intervals · Cohen's h effect size")

    ab_results = load_ab_results()

    for res in ab_results:
        sig_color = "🟢" if res["significant"] else "🔴"
        with st.expander(f"{sig_color} {res['experiment_id'].replace('_', ' ').title()}  —  {res['verdict']}", expanded=True):
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Control CVR",   f"{res['cvr_control']:.2%}")
            c2.metric("Treatment CVR", f"{res['cvr_treatment']:.2%}",
                      delta=f"{res['uplift_pct']:+.1f}%")
            c3.metric("p-value",       f"{res['p_value']:.4f}",
                      delta="significant" if res["significant"] else "not sig",
                      delta_color="normal" if res["significant"] else "inverse")
            c4.metric("Effect Size",   f"{res['effect_size']:.4f}")
            c5.metric("Sample (each)", f"{res['n_control']:,}")

            # CI plot
            fig_ci = go.Figure()
            diff = res["cvr_treatment"] - res["cvr_control"]
            fig_ci.add_shape(type="line", x0=0, x1=0, y0=-0.3, y1=0.3,
                             line=dict(color="gray", dash="dash", width=1))
            fig_ci.add_trace(go.Scatter(
                x=[res["ci_lower"], diff, res["ci_upper"]],
                y=[0, 0, 0],
                mode="markers+lines",
                marker=dict(size=[6, 12, 6],
                            color=[GRAY, PURPLE if res["significant"] else RED, GRAY]),
                line=dict(color=PURPLE if res["significant"] else RED, width=3),
                name="95% CI",
            ))
            fig_ci.update_layout(
                height=120,
                xaxis_title="Difference in conversion rate (treatment − control)",
                yaxis=dict(visible=False),
                margin=dict(l=0, r=0, t=10, b=30),
                showlegend=False,
            )
            st.plotly_chart(fig_ci, use_container_width=True)

    st.markdown("---")
    st.subheader("Power Analysis Calculator")
    st.caption("Minimum sample size to detect a given effect with 80% power")

    col_p1, col_p2, col_p3 = st.columns(3)
    base_rate = col_p1.slider("Baseline conversion rate", 0.01, 0.30, 0.05, 0.005, format="%.3f")
    mde       = col_p2.slider("Minimum detectable effect (MDE)", 0.001, 0.05, 0.01, 0.001, format="%.3f")
    alpha_val = col_p3.select_slider("Significance level (α)", [0.01, 0.05, 0.10], value=0.05)

    pa = power_analysis(base_rate, mde, alpha_val)
    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("Min sample per variant", f"{pa['min_sample_size']:,}")
    pc2.metric("Total users needed",     f"{pa['total_users']:,}")
    pc3.metric("Cohen's h effect size",  f"{pa['effect_size_h']:.4f}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — CHURN INTELLIGENCE
# ════════════════════════════════════════════════════════════════════════════

with tab4:
    metrics = load_model_metrics()
    preds   = load_predictions()

    if metrics is None or preds is None:
        st.warning("Model not trained yet. Run `python setup.py` to train the model.")
    else:
        st.subheader("Model Performance")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Test AUC-ROC",   f"{metrics['test_auc']:.4f}")
        c2.metric("Accuracy",       f"{metrics['test_acc']:.2%}")
        c3.metric("F1 Score",       f"{metrics['test_f1']:.4f}")
        c4.metric("CV AUC (5-fold)", f"{metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
        c5.metric("Churn Rate",     f"{metrics['churn_rate']:.1%}")

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            # Risk tier distribution
            tier_counts = preds["risk_tier"].value_counts().reset_index()
            tier_counts.columns = ["tier", "count"]
            tier_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
            tier_counts["order"] = tier_counts["tier"].map(tier_order)
            tier_counts = tier_counts.sort_values("order")
            colors = {"LOW": "#059669", "MEDIUM": "#D97706", "HIGH": "#DC2626"}
            fig_tier = px.bar(tier_counts, x="tier", y="count",
                              color="tier",
                              color_discrete_map=colors,
                              title="Customers by Churn Risk Tier")
            fig_tier.update_layout(height=300, showlegend=False,
                                   margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_tier, use_container_width=True)

        with col_b:
            # Churn probability distribution
            fig_dist = px.histogram(
                preds, x="churn_probability", nbins=40,
                color="risk_tier",
                color_discrete_map={"LOW": "#059669", "MEDIUM": "#D97706", "HIGH": "#DC2626"},
                title="Churn Probability Distribution",
                category_orders={"risk_tier": ["LOW", "MEDIUM", "HIGH"]},
            )
            fig_dist.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0),
                                   barmode="stack")
            st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("---")
        st.subheader("SHAP Feature Importance")
        shap_path = os.path.join(ARTIFACTS_DIR, "shap_importance.png")
        shap_dot_path = os.path.join(ARTIFACTS_DIR, "shap_summary.png")

        if os.path.exists(shap_path):
            col_sh1, col_sh2 = st.columns(2)
            with col_sh1:
                st.image(shap_path, caption="Mean |SHAP| feature importance", use_container_width=True)
            with col_sh2:
                if os.path.exists(shap_dot_path):
                    st.image(shap_dot_path, caption="SHAP summary (dot plot)", use_container_width=True)

        st.markdown("---")
        st.subheader("High Risk Customer List")
        high_risk = preds[preds["risk_tier"] == "HIGH"].sort_values(
            "churn_probability", ascending=False
        ).head(20).copy()
        high_risk["churn_probability"] = high_risk["churn_probability"].apply(lambda x: f"{x:.2%}")
        st.dataframe(high_risk[["customer_id", "churn_probability", "risk_tier"]],
                     use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — ML EXPERIMENTS
# ════════════════════════════════════════════════════════════════════════════

with tab5:
    metrics  = load_model_metrics()
    trials   = load_optuna_trials()

    if metrics is None or trials is None:
        st.warning("Run `python setup.py` to train the model first.")
    else:
        st.subheader("Optuna Hyperparameter Search")
        st.caption(f"{metrics['n_trials']} trials · Best val AUC: {metrics['best_val_auc']:.4f}")

        col_a, col_b = st.columns(2)

        with col_a:
            trials["trial_color"] = trials["val_auc"].apply(
                lambda x: "best" if x == trials["val_auc"].max() else "normal"
            )
            fig_opt = go.Figure()
            fig_opt.add_trace(go.Scatter(
                x=trials["number"], y=trials["val_auc"],
                mode="markers+lines",
                marker=dict(
                    color=trials["val_auc"],
                    colorscale="Purples",
                    size=8,
                    colorbar=dict(title="AUC"),
                ),
                line=dict(color="#e5e7eb", width=1),
                name="Val AUC"
            ))
            # Mark best
            best_idx = trials["val_auc"].idxmax()
            fig_opt.add_trace(go.Scatter(
                x=[trials.loc[best_idx, "number"]],
                y=[trials.loc[best_idx, "val_auc"]],
                mode="markers",
                marker=dict(color=PURPLE, size=14, symbol="star"),
                name=f"Best trial #{int(trials.loc[best_idx, 'number'])}",
            ))
            fig_opt.update_layout(
                title="Validation AUC across Optuna trials",
                height=320,
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis_title="Trial #",
                yaxis_title="Validation AUC",
            )
            st.plotly_chart(fig_opt, use_container_width=True)

        with col_b:
            fig_lr = px.scatter(
                trials, x="params_learning_rate", y="val_auc",
                color="params_max_depth", size="params_n_estimators",
                color_continuous_scale="Viridis",
                title="Learning Rate vs. AUC (color=depth, size=estimators)",
                labels={"params_learning_rate": "Learning Rate",
                        "val_auc": "Val AUC",
                        "params_max_depth": "Max Depth"},
            )
            fig_lr.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_lr, use_container_width=True)

        st.markdown("---")
        st.subheader("Best Model Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Test AUC-ROC", f"{metrics['test_auc']:.4f}")
        col2.metric("CV AUC (mean ± std)",
                    f"{metrics['cv_auc_mean']:.4f}",
                    f"± {metrics['cv_auc_std']:.4f}")
        col3.metric("Customers evaluated", f"{metrics['n_customers']:,}")

        # Trials table
        st.dataframe(
            trials.rename(columns={
                "number": "Trial",
                "val_auc": "Val AUC",
                "params_learning_rate": "Learning Rate",
                "params_max_depth": "Max Depth",
                "params_n_estimators": "n_estimators",
            }).sort_values("Val AUC", ascending=False).head(10),
            use_container_width=True,
        )

        st.subheader("Feature Correlation with Churn")
        df_feat = load_features_df()
        corrs = df_feat[FEATURE_COLS + ["churned"]].corr()["churned"].drop("churned")
        corrs_df = corrs.reset_index()
        corrs_df.columns = ["feature", "correlation"]
        corrs_df["feature"] = corrs_df["feature"].map(
            lambda x: DISPLAY_FEATURE_NAMES.get(x, x))
        corrs_df = corrs_df.sort_values("correlation")

        fig_corr = px.bar(
            corrs_df, x="correlation", y="feature", orientation="h",
            color="correlation",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            title="Pearson Correlation of Features with Churn Label",
        )
        fig_corr.update_layout(height=400, coloraxis_showscale=True,
                               margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_corr, use_container_width=True)
