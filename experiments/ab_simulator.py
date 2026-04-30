"""
A/B test analysis engine.
Runs statistical tests on experiment data from the database.
"""
import sqlite3
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower, NormalIndPower
from statsmodels.stats.proportion import proportions_ztest
import warnings
warnings.filterwarnings("ignore")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "ecommerce.db")


def get_experiment_data(experiment_id: str, conn=None) -> pd.DataFrame:
    own_conn = conn is None
    if own_conn:
        conn = sqlite3.connect(os.path.abspath(DB_PATH))
    query = """
    SELECT variant, customer_id, converted, event_date
    FROM ab_events
    WHERE experiment_id = ?
    """
    df = pd.read_sql_query(query, conn, params=(experiment_id,))
    if own_conn:
        conn.close()
    return df


def run_ab_test(experiment_id: str, alpha: float = 0.05) -> dict:
    """Run a full A/B test analysis and return results dict."""
    df = get_experiment_data(experiment_id)
    if df.empty:
        return {"error": f"No data for experiment: {experiment_id}"}

    control   = df[df["variant"] == "control"]["converted"].values
    treatment = df[df["variant"] == "treatment"]["converted"].values

    n_control   = len(control)
    n_treatment = len(treatment)
    cvr_control   = control.mean()
    cvr_treatment = treatment.mean()
    uplift = (cvr_treatment - cvr_control) / cvr_control if cvr_control > 0 else 0

    # Two-proportion z-test
    count = np.array([treatment.sum(), control.sum()])
    nobs  = np.array([n_treatment, n_control])
    z_stat, p_value = proportions_ztest(count, nobs)

    # Confidence interval for the difference
    se = np.sqrt(
        cvr_control * (1 - cvr_control) / n_control +
        cvr_treatment * (1 - cvr_treatment) / n_treatment
    )
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = cvr_treatment - cvr_control
    ci_lower = diff - z_crit * se
    ci_upper = diff + z_crit * se

    # Cohen's h effect size
    effect_size = 2 * (np.arcsin(np.sqrt(cvr_treatment)) -
                       np.arcsin(np.sqrt(cvr_control)))

    significant = p_value < alpha

    return {
        "experiment_id":   experiment_id,
        "n_control":       int(n_control),
        "n_treatment":     int(n_treatment),
        "cvr_control":     round(float(cvr_control), 4),
        "cvr_treatment":   round(float(cvr_treatment), 4),
        "uplift_pct":      round(float(uplift) * 100, 2),
        "z_statistic":     round(float(z_stat), 4),
        "p_value":         round(float(p_value), 6),
        "significant":     bool(significant),
        "alpha":           alpha,
        "ci_lower":        round(float(ci_lower), 4),
        "ci_upper":        round(float(ci_upper), 4),
        "effect_size":     round(float(effect_size), 4),
        "verdict":         "WINNER ✓" if significant and cvr_treatment > cvr_control
                           else ("LOSER ✗" if significant else "NO DIFFERENCE"),
    }


def power_analysis(base_rate: float, mde: float, alpha: float = 0.05,
                   power: float = 0.8) -> dict:
    """Calculate minimum sample size for a given effect."""
    effect_size = 2 * (np.arcsin(np.sqrt(base_rate + mde)) -
                       np.arcsin(np.sqrt(base_rate)))

    analysis = NormalIndPower()
    n = analysis.solve_power(
        effect_size=abs(effect_size),
        alpha=alpha,
        power=power,
        alternative="two-sided"
    )
    return {
        "base_rate":       base_rate,
        "mde":             mde,
        "min_sample_size": int(np.ceil(n)),
        "total_users":     int(np.ceil(n)) * 2,
        "alpha":           alpha,
        "power":           power,
        "effect_size_h":   round(effect_size, 4),
    }


def run_all_experiments() -> list:
    conn = sqlite3.connect(os.path.abspath(DB_PATH))
    experiments = pd.read_sql_query(
        "SELECT DISTINCT experiment_id FROM ab_events", conn
    )["experiment_id"].tolist()
    conn.close()
    return [run_ab_test(exp_id) for exp_id in experiments]


if __name__ == "__main__":
    results = run_all_experiments()
    for r in results:
        print(f"\n{'='*55}")
        print(f"Experiment: {r['experiment_id']}")
        print(f"  Control CVR:   {r['cvr_control']:.2%}  (n={r['n_control']:,})")
        print(f"  Treatment CVR: {r['cvr_treatment']:.2%}  (n={r['n_treatment']:,})")
        print(f"  Uplift:        {r['uplift_pct']:+.1f}%")
        print(f"  p-value:       {r['p_value']:.4f}  {'(significant)' if r['significant'] else '(not significant)'}")
        print(f"  95% CI:        [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]")
        print(f"  Verdict:       {r['verdict']}")
