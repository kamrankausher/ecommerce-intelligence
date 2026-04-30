# E-commerce Growth Intelligence Platform

> End-to-end data science system covering churn prediction, A/B experimentation, SQL analytics, and MLOps — built on a synthetic Brazilian e-commerce dataset.

---

## 🎯 What this demonstrates

| Skill | Where |
|---|---|
| Advanced SQL (window functions, CTEs, cohort analysis) | `sql/`, `api/main.py` |
| A/B testing (z-test, power analysis, confidence intervals) | `experiments/ab_simulator.py` |
| ML pipeline (XGBoost + Optuna + SHAP + cross-validation) | `ml/train.py` |
| MLflow experiment tracking + model registry | `ml/train.py`, `mlruns/` |
| FastAPI production API with Pydantic validation | `api/main.py` |
| Docker + CI/CD (GitHub Actions → Docker Hub → Render) | `infra/`, `.github/workflows/` |
| Pytest test suite (17 tests, runs in CI) | `tests/test_all.py` |
| Streamlit analytics dashboard (5 tabs) | `dashboard/app.py` |

---

## 🚀 Quickstart (3 commands)

```bash
git clone https://github.com/kamrankausher/ecommerce-intelligence.git
cd ecommerce-intelligence
pip install -r requirements.txt && python setup.py
```

Then:

```bash
# Dashboard
streamlit run dashboard/app.py

# API server
uvicorn api.main:app --reload

# Tests
pytest tests/ -v

# MLflow UI
mlflow ui --backend-store-uri mlruns/
```

---

## 📁 Project structure

```
ecommerce-intelligence/
├── data/
│   └── generate_data.py        # Synthetic e-commerce dataset (8k customers, 18k orders)
├── sql/
│   ├── 01_revenue_analysis.sql # Monthly revenue with window functions
│   ├── 02_cohort_retention.sql # Cohort retention using CTEs + DATE_TRUNC
│   ├── 03_seller_performance.sql # RANK() OVER(PARTITION BY state)
│   ├── 04_rfm_features.sql     # RFM scoring in pure SQL
│   └── 05_ab_results_query.sql # A/B experiment data pull
├── experiments/
│   ├── ab_simulator.py         # Two-proportion z-test + power analysis
│   └── experiments.json        # Experiment definitions
├── ml/
│   ├── feature_engineering.py  # RFM + behavioural features from SQLite
│   ├── train.py                # XGBoost + Optuna (25 trials) + MLflow + SHAP
│   └── evaluate.py             # Model evaluation utilities
├── api/
│   └── main.py                 # FastAPI: /predict, /ab-test, /kpis, /cohort-retention
├── dashboard/
│   └── app.py                  # Streamlit: 5-tab analytics dashboard
├── tests/
│   └── test_all.py             # 17 pytest tests (runs in CI)
├── infra/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .github/workflows/
│   └── ci.yml                  # Test → Build → Push → Deploy
├── setup.py                    # One-command setup
└── requirements.txt
```

---

## 📊 Model results

| Metric | Score |
|---|---|
| Test AUC-ROC | 1.000 |
| Accuracy | 99.7% |
| F1 Score | 0.998 |
| CV AUC (5-fold) | 1.000 ± 0.000 |

---

## 🧪 A/B experiment results

| Experiment | p-value | Result |
|---|---|---|
| Checkout button color | 0.0363 | ✅ Significant |
| Email subject line | 0.0000 | ✅ Significant |
| Discount 10% vs 15% | 0.5418 | ❌ No difference |

---

## 🔌 API endpoints

```
GET  /health               → model status
GET  /kpis                 → top-line business metrics
POST /predict              → churn probability + SHAP drivers
POST /ab-test              → run A/B test for experiment_id
GET  /ab-tests             → all experiment results
POST /power-analysis       → minimum sample size calculator
GET  /monthly-revenue      → revenue time series
GET  /revenue-by-state     → state breakdown
GET  /cohort-retention     → cohort matrix
```

---

## Tech stack

Python · XGBoost · SHAP · MLflow · Optuna · FastAPI · Streamlit · Plotly · SQLite · SciPy · statsmodels · Docker · GitHub Actions · pytest
