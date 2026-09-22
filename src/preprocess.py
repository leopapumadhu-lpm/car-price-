"""
src/preprocess.py
─────────────────
Data loading and feature engineering for the CarDekho dataset.

Real CSV columns:
    name, year, selling_price, km_driven, fuel, seller_type,
    transmission, owner, mileage(km/ltr/kg), engine, max_power, seats

No 'torque' column exists in this version of the dataset.
mileage and engine are already numeric floats.
max_power is a string like "74" or "103.52" — needs float conversion.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    DATA_PATH, TARGET_COL, MILEAGE_COL,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, CURRENT_YEAR,
)


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Read and clean the CardDekho CSV.

    Returns
    -------
    X : pd.DataFrame  — features (ALL_FEATURES columns)
    y : pd.Series     — selling_price
    """
    df = pd.read_csv(DATA_PATH)

    # ── Normalise string columns ───────────────────────────────────────────────
    for col in ["fuel", "seller_type", "transmission", "owner"]:
        df[col] = df[col].str.strip().str.lower()

    # ── Derived feature: car age ───────────────────────────────────────────────
    df["car_age"] = CURRENT_YEAR - df["year"]

    # ── Rename mileage column → 'mileage' ─────────────────────────────────────
    # Already float in this dataset (no unit strings)
    df = df.rename(columns={MILEAGE_COL: "mileage"})

    # ── Fix max_power: object → float ─────────────────────────────────────────
    # Values like "74", "103.52", "" (blank string for some rows)
    df["max_power"] = (
        df["max_power"]
        .replace("", np.nan)
        .astype(str)
        .str.strip()
        .replace("nan", np.nan)
    )
    df["max_power"] = pd.to_numeric(df["max_power"], errors="coerce")

    # ── Fill numeric NaNs with column medians ─────────────────────────────────
    for col in ["mileage", "engine", "max_power", "seats"]:
        df[col] = df[col].fillna(df[col].median())

    # ── Drop unused columns ───────────────────────────────────────────────────
    df = df.drop(columns=["name", "year"], errors="ignore")

    # ── Drop rows still containing NaN ────────────────────────────────────────
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    df = df.dropna(subset=feature_cols + [TARGET_COL])

    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()

    return X, y


def build_pipeline() -> ColumnTransformer:
    """
    Return a fitted-ready ColumnTransformer (preprocessor only — no estimator).

    Usage in train.py:
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([
            ("preprocessor", build_pipeline()),
            ("model", some_estimator),
        ])
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return preprocessor


if __name__ == "__main__":
    X, y = load_data()
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print()
    print(X.head())
    print()
    print(X.dtypes)
