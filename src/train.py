"""
src/train.py
────────────
Train three candidate models, compare them, and persist the best one.

Usage:
    python -m src.train
"""

import json
import os

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.config import (
    CATEGORICAL_FEATURES,
    COMPARISON_PATH,
    FEATURE_IMPORTANCE_PATH,
    MODEL_PATH,
    NUMERICAL_FEATURES,
)
from src.preprocess import build_pipeline, load_data


# ── Helpers ────────────────────────────────────────────────────────────────────

def train_and_evaluate(
    name: str,
    estimator,
    X_train,
    X_test,
    y_train,
    y_test,
) -> tuple:
    """
    Build a full sklearn Pipeline (preprocessor + estimator), fit it on the
    training split, and evaluate on the test split.

    Returns
    -------
    fitted_pipeline : sklearn.pipeline.Pipeline
    metrics         : dict with keys "name", "rmse", "r2"
    """
    pipe = Pipeline([
        ("preprocessor", build_pipeline()),
        ("model", estimator),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = float(r2_score(y_test, y_pred))

    return pipe, {"name": name, "rmse": round(rmse, 2), "r2": round(r2, 4)}


def get_feature_importance(pipeline: Pipeline) -> dict:
    """
    Extract feature importances from the best tree-based model.

    Returns a dict {feature_name: importance} sorted descending, top 15.
    Returns an empty dict for models without feature_importances_.
    """
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return {}

    preprocessor = pipeline.named_steps["preprocessor"]

    # OHE feature names
    ohe = preprocessor.named_transformers_["cat"]
    ohe_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()

    # All feature names in transform order
    all_names = ohe_names + NUMERICAL_FEATURES

    importances = model.feature_importances_
    importance_map = dict(zip(all_names, importances))

    # Sort descending, return top 15
    sorted_map = dict(
        sorted(importance_map.items(), key=lambda x: x[1], reverse=True)[:15]
    )
    return {k: round(float(v), 6) for k, v in sorted_map.items()}


# ── Main training script ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Car Price Prediction — Model Training")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading and preprocessing data …")
    X, y = load_data()
    print(f"      Dataset: {X.shape[0]} rows × {X.shape[1]} features")

    # 2. Split
    print("\n[2/5] Splitting into train (80%) / test (20%) …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Define models
    models = [
        ("Linear Regression",  LinearRegression()),
        ("Random Forest",      RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ("XGBoost",            XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                            random_state=42, n_jobs=-1, verbosity=0)),
    ]

    # 4. Train & evaluate
    print("\n[3/5] Training and evaluating models …\n")
    results = []
    pipelines = {}

    header = f"{'Model':<22} {'RMSE':>14} {'R²':>8}"
    print(header)
    print("-" * len(header))

    for name, estimator in models:
        pipe, metrics = train_and_evaluate(name, estimator, X_train, X_test, y_train, y_test)
        results.append(metrics)
        pipelines[name] = pipe
        print(f"{name:<22} ₹{metrics['rmse']:>12,.0f}   {metrics['r2']:>6.4f}")

    # 5. Select best model (lowest RMSE)
    print("\n[4/5] Selecting best model …")
    best_metrics = min(results, key=lambda m: m["rmse"])
    best_name    = best_metrics["name"]
    best_pipe    = pipelines[best_name]
    print(f"      🏆  Winner: {best_name} (RMSE ₹{best_metrics['rmse']:,.0f}, R² {best_metrics['r2']})")

    # 6. Persist
    print("\n[5/5] Saving artefacts …")
    os.makedirs("models", exist_ok=True)

    joblib.dump(best_pipe, MODEL_PATH)
    print(f"      ✓ Best model saved  → {MODEL_PATH}")

    with open(COMPARISON_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"      ✓ Comparison data   → {COMPARISON_PATH}")

    importance = get_feature_importance(best_pipe)
    with open(FEATURE_IMPORTANCE_PATH, "w") as f:
        json.dump(importance, f, indent=2)
    print(f"      ✓ Feature importance → {FEATURE_IMPORTANCE_PATH}")

    print("\n" + "=" * 60)
    print("  Training complete.")
    print("=" * 60)
