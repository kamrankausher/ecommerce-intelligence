"""
Model training: XGBoost churn classifier
- Optuna hyperparameter tuning (25 trials)
- MLflow experiment tracking (every trial logged)
- SHAP explainability
- Model saved to disk + MLflow registry
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import mlflow
import mlflow.xgboost
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, classification_report, accuracy_score,
    confusion_matrix, f1_score
)
from sklearn.preprocessing import label_binarize
import joblib

from ml.feature_engineering import build_features, FEATURE_COLS

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
MLFLOW_DIR    = os.path.join(os.path.dirname(__file__), "..", "mlruns")
EXPERIMENT    = "churn-prediction"
MODEL_NAME    = "churn-xgboost"
N_TRIALS      = 25


def train():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    from pathlib import Path
    mlflow_uri = Path(os.path.abspath(MLFLOW_DIR)).as_uri()
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(EXPERIMENT)

    print("Building features...")
    df = build_features()
    X = df[FEATURE_COLS].values
    y = df["churned"].values
    customer_ids = df["customer_id"].values
    print(f"  Dataset: {len(df):,} customers | churn rate: {y.mean():.1%}")

    indices = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    test_customer_ids = customer_ids[idx_test]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

    print(f"Running Optuna ({N_TRIALS} trials)...")

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 400),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            "scale_pos_weight": scale_pos,
            "random_state": 42,
            "eval_metric": "logloss",
            "verbosity": 0,
        }
        model = XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            mlflow.log_params(params)
            mlflow.log_metric("val_auc", auc)

        return auc

    with mlflow.start_run(run_name="optuna_search"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

        best_params = study.best_params
        best_params["scale_pos_weight"] = scale_pos
        best_params["random_state"] = 42
        best_params["eval_metric"] = "logloss"
        best_params["verbosity"] = 0

        print(f"  Best val AUC: {study.best_value:.4f}")
        print("Training final model on full train set...")

        final_model = XGBClassifier(**best_params)
        final_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Evaluate
        proba_test = final_model.predict_proba(X_test)[:, 1]
        pred_test  = final_model.predict(X_test)

        test_auc  = roc_auc_score(y_test, proba_test)
        test_acc  = accuracy_score(y_test, pred_test)
        test_f1   = f1_score(y_test, pred_test)
        cv_scores = cross_val_score(
            XGBClassifier(**best_params), X_train, y_train,
            cv=StratifiedKFold(n_splits=5), scoring="roc_auc", n_jobs=-1
        )

        print(f"\n  Test AUC:      {test_auc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test F1:       {test_f1:.4f}")
        print(f"  CV AUC:        {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "test_auc":    test_auc,
            "test_acc":    test_acc,
            "test_f1":     test_f1,
            "cv_auc_mean": cv_scores.mean(),
            "cv_auc_std":  cv_scores.std(),
            "n_trials":    N_TRIALS,
        })

        # SHAP
        print("Computing SHAP values...")
        explainer  = shap.TreeExplainer(final_model)
        shap_vals  = explainer.shap_values(X_test[:500])

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_vals, X_test[:500],
            feature_names=FEATURE_COLS,
            show=False, plot_type="bar"
        )
        plt.title("Feature Importance (SHAP values)")
        plt.tight_layout()
        shap_bar_path = os.path.join(ARTIFACTS_DIR, "shap_importance.png")
        plt.savefig(shap_bar_path, dpi=120, bbox_inches="tight")
        plt.close()

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_vals, X_test[:500],
            feature_names=FEATURE_COLS,
            show=False
        )
        plt.title("SHAP Summary Plot")
        plt.tight_layout()
        shap_dot_path = os.path.join(ARTIFACTS_DIR, "shap_summary.png")
        plt.savefig(shap_dot_path, dpi=120, bbox_inches="tight")
        plt.close()

        mlflow.log_artifact(shap_bar_path)
        mlflow.log_artifact(shap_dot_path)

        # Save model + explainer
        model_path    = os.path.join(ARTIFACTS_DIR, "model.pkl")
        explainer_path = os.path.join(ARTIFACTS_DIR, "explainer.pkl")
        feature_path  = os.path.join(ARTIFACTS_DIR, "features.json")

        joblib.dump(final_model, model_path)
        joblib.dump(explainer, explainer_path)
        with open(feature_path, "w") as f:
            json.dump(FEATURE_COLS, f)

        mlflow.xgboost.log_model(final_model, artifact_path="model",
                                  registered_model_name=MODEL_NAME)
        mlflow.log_artifact(feature_path)

        # Save metrics for dashboard
        metrics = {
            "test_auc":      round(test_auc, 4),
            "test_acc":      round(test_acc, 4),
            "test_f1":       round(test_f1, 4),
            "cv_auc_mean":   round(float(cv_scores.mean()), 4),
            "cv_auc_std":    round(float(cv_scores.std()), 4),
            "churn_rate":    round(float(y.mean()), 4),
            "n_customers":   int(len(df)),
            "n_trials":      N_TRIALS,
            "best_val_auc":  round(study.best_value, 4),
        }
        metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Save test predictions for dashboard
        preds_df = pd.DataFrame({
            "customer_id":       test_customer_ids,
            "churn_probability": proba_test,
            "predicted":         pred_test,
            "actual":            y_test,
        })
        preds_df["risk_tier"] = preds_df["churn_probability"].apply(
            lambda p: "HIGH" if p >= 0.65 else ("MEDIUM" if p >= 0.35 else "LOW")
        )
        preds_df.to_csv(os.path.join(ARTIFACTS_DIR, "predictions.csv"), index=False)

        # Save optuna history for dashboard
        trials_df = study.trials_dataframe()[
            ["number", "value", "params_learning_rate",
             "params_max_depth", "params_n_estimators"]
        ].rename(columns={"value": "val_auc"})
        trials_df.to_csv(os.path.join(ARTIFACTS_DIR, "optuna_trials.csv"), index=False)

        print(f"\nAll artifacts saved to: {ARTIFACTS_DIR}")

    return metrics


if __name__ == "__main__":
    train()
