import os
import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
_KAGGLE_PATH = "/kaggle/input/car-price-prediction-dataset/cardekho.csv"
_LOCAL_PATH = "data/cardekho.csv"
DATA_PATH = _KAGGLE_PATH if os.path.exists(_KAGGLE_PATH) else _LOCAL_PATH

MODEL_PATH = "models/best_model.pkl"
COMPARISON_PATH = "models/model_comparison.json"
FEATURE_IMPORTANCE_PATH = "models/feature_importance.json"

# ── Target ─────────────────────────────────────────────────────────────────────
TARGET_COL = "selling_price"

# ── Actual column name in the CSV for mileage ─────────────────────────────────
MILEAGE_COL = "mileage(km/ltr/kg)"   # raw CSV column name

# ── Feature lists (names AFTER preprocessing, i.e. in the DataFrame fed to ML) ─
CATEGORICAL_FEATURES = ["fuel", "seller_type", "transmission", "owner"]
NUMERICAL_FEATURES = ["car_age", "km_driven", "mileage", "engine", "max_power", "seats"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# ── Derived ────────────────────────────────────────────────────────────────────
CURRENT_YEAR = datetime.datetime.now().year
