"""
setup.py — Run this once to:
  1. Generate synthetic e-commerce data
  2. Load into SQLite
  3. Train XGBoost churn model (Optuna + MLflow + SHAP)
  4. Run A/B test analysis
  5. Print final summary

After this, run: streamlit run dashboard/app.py
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def banner(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def step(n, msg):
    print(f"\n[{n}] {msg}")
    print("-" * 40)


def main():
    banner("E-commerce Growth Intelligence Platform — Setup")
    total_start = time.time()

    # ── Step 1: Generate Data ─────────────────────────────────────────────
    step(1, "Generating synthetic e-commerce data")
    from data.generate_data import main as gen_data
    gen_data()

    # ── Step 2: Train Model ───────────────────────────────────────────────
    step(2, "Training XGBoost churn model (Optuna + MLflow + SHAP)")
    print("  This takes ~2–3 minutes for 25 trials...\n")
    from ml.train import train
    metrics = train()

    # ── Step 3: Run A/B Tests ─────────────────────────────────────────────
    step(3, "Running A/B experiment analysis")
    from experiments.ab_simulator import run_all_experiments
    ab_results = run_all_experiments()

    # ── Summary ───────────────────────────────────────────────────────────
    banner("Setup Complete!")
    elapsed = time.time() - total_start
    print(f"\n  Time elapsed: {elapsed:.0f}s")

    print("\n  Model metrics:")
    print(f"    Test AUC-ROC:  {metrics['test_auc']:.4f}")
    print(f"    Accuracy:      {metrics['test_acc']:.2%}")
    print(f"    F1 Score:      {metrics['test_f1']:.4f}")
    print(f"    CV AUC (5-fold): {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")

    print("\n  A/B test results:")
    for r in ab_results:
        sig = "[YES]" if r["significant"] else "[NO]"
        print(f"    {r['experiment_id']:<30} p={r['p_value']:.4f}  {sig}")

    print("\n" + "="*60)
    print("  NEXT STEPS:")
    print("="*60)
    print("\n  1. Launch dashboard:")
    print("       streamlit run dashboard/app.py")
    print("\n  2. Start API server:")
    print("       python -m uvicorn api.main:app --reload")
    print("\n  3. Run tests:")
    print("       pytest tests/ -v")
    print("\n  4. View MLflow experiments:")
    print("       mlflow ui --backend-store-uri mlruns/")
    print()


if __name__ == "__main__":
    main()
