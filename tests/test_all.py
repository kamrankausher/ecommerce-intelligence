"""
Test suite — runs in CI/CD pipeline before deploy.
"""
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── A/B Test Tests ──────────────────────────────────────────────────────────

def test_ab_simulator_returns_dict():
    from experiments.ab_simulator import run_ab_test
    result = run_ab_test("checkout_button_color")
    assert isinstance(result, dict), "Result must be a dict"


def test_ab_simulator_has_required_keys():
    from experiments.ab_simulator import run_ab_test
    result = run_ab_test("checkout_button_color")
    required = ["experiment_id", "n_control", "n_treatment",
                "cvr_control", "cvr_treatment", "p_value",
                "significant", "verdict", "uplift_pct"]
    for key in required:
        assert key in result, f"Missing key: {key}"


def test_ab_p_value_range():
    from experiments.ab_simulator import run_ab_test
    result = run_ab_test("email_subject_line")
    assert 0.0 <= result["p_value"] <= 1.0, "p-value must be in [0, 1]"


def test_ab_n_samples_positive():
    from experiments.ab_simulator import run_ab_test
    result = run_ab_test("discount_offer")
    assert result["n_control"] > 0
    assert result["n_treatment"] > 0


def test_ab_cvr_between_0_and_1():
    from experiments.ab_simulator import run_ab_test
    result = run_ab_test("checkout_button_color")
    assert 0 <= result["cvr_control"] <= 1
    assert 0 <= result["cvr_treatment"] <= 1


def test_ab_significant_has_boolean():
    from experiments.ab_simulator import run_ab_test
    result = run_ab_test("email_subject_line")
    assert isinstance(result["significant"], bool)


def test_ab_all_experiments_run():
    from experiments.ab_simulator import run_all_experiments
    results = run_all_experiments()
    assert len(results) == 3, "Expected 3 experiments"
    for r in results:
        assert "error" not in r


def test_power_analysis_returns_positive_sample():
    from experiments.ab_simulator import power_analysis
    result = power_analysis(base_rate=0.05, mde=0.01)
    assert result["min_sample_size"] > 0
    assert result["total_users"] == result["min_sample_size"] * 2


def test_power_analysis_larger_mde_needs_smaller_sample():
    from experiments.ab_simulator import power_analysis
    r_small = power_analysis(base_rate=0.05, mde=0.005)
    r_large = power_analysis(base_rate=0.05, mde=0.020)
    assert r_small["min_sample_size"] > r_large["min_sample_size"]


# ── Feature Engineering Tests ───────────────────────────────────────────────

def test_features_dataframe_not_empty():
    from ml.feature_engineering import build_features, FEATURE_COLS
    df = build_features()
    assert len(df) > 0, "Feature dataframe should not be empty"
    for col in FEATURE_COLS:
        assert col in df.columns, f"Missing feature column: {col}"


def test_churn_label_binary():
    from ml.feature_engineering import build_features
    df = build_features()
    assert set(df["churned"].unique()).issubset({0, 1})


def test_churn_rate_reasonable():
    from ml.feature_engineering import build_features
    df = build_features()
    churn_rate = df["churned"].mean()
    assert 0.1 <= churn_rate <= 0.9, f"Unusual churn rate: {churn_rate:.1%}"


def test_no_null_features():
    from ml.feature_engineering import build_features, FEATURE_COLS
    df = build_features()
    nulls = df[FEATURE_COLS].isnull().sum().sum()
    assert nulls == 0, f"Found {nulls} null values in feature columns"


# ── API Tests (requires trained model) ─────────────────────────────────────

def test_api_health():
    """Test /health endpoint returns ok."""
    artifacts = os.path.join(os.path.dirname(__file__), "..", "artifacts", "model.pkl")
    if not os.path.exists(artifacts):
        pytest.skip("Model not trained yet — run python setup.py first")

    from fastapi.testclient import TestClient
    from api.main import app, load_model
    load_model()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_api_kpis_returns_all_keys():
    """Test /kpis endpoint."""
    artifacts = os.path.join(os.path.dirname(__file__), "..", "artifacts", "model.pkl")
    if not os.path.exists(artifacts):
        pytest.skip("Model not trained yet")

    from fastapi.testclient import TestClient
    from api.main import app, load_model
    load_model()
    client = TestClient(app)
    response = client.get("/kpis")
    assert response.status_code == 200
    data = response.json()
    for key in ["total_revenue", "total_orders", "total_customers", "avg_order_value"]:
        assert key in data


def test_api_predict_valid_customer():
    """Test /predict with a real customer."""
    artifacts = os.path.join(os.path.dirname(__file__), "..", "artifacts", "model.pkl")
    if not os.path.exists(artifacts):
        pytest.skip("Model not trained yet")

    import sqlite3
    db_path = os.path.join(os.path.dirname(__file__), "..", "ecommerce.db")
    conn = sqlite3.connect(os.path.abspath(db_path))
    cid = conn.execute("SELECT customer_id FROM customers LIMIT 1").fetchone()[0]
    conn.close()

    from fastapi.testclient import TestClient
    from api.main import app, load_model
    load_model()
    client = TestClient(app)
    response = client.post("/predict", json={"customer_id": cid})
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_tier"] in ["LOW", "MEDIUM", "HIGH"]


def test_api_predict_invalid_customer():
    """Test /predict with non-existent customer returns 404."""
    artifacts = os.path.join(os.path.dirname(__file__), "..", "artifacts", "model.pkl")
    if not os.path.exists(artifacts):
        pytest.skip("Model not trained yet")

    from fastapi.testclient import TestClient
    from api.main import app, load_model
    load_model()
    client = TestClient(app)
    response = client.post("/predict", json={"customer_id": "DOES_NOT_EXIST_999"})
    assert response.status_code == 404
