# Car Price Prediction

A full end-to-end car price prediction application built entirely in Python. It trains and compares three ML models on the CarDekho dataset, exposes the best model through a FastAPI REST API, and provides an interactive Streamlit dashboard.

---

## Project Structure

```
car-price/
├── data/
│   └── cardekho.csv          # Raw dataset (download from Kaggle — see below)
├── notebooks/
│   └── eda.ipynb             # Exploratory data analysis (optional)
├── src/
│   ├── config.py             # Shared constants (paths, feature names)
│   ├── preprocess.py         # Data loading and feature engineering pipeline
│   ├── train.py              # Model training, comparison, and persistence
│   └── predict.py            # Prediction helper used by the API
├── api/
│   └── main.py               # FastAPI app (/predict, /models, /features, /health)
├── app/
│   └── streamlit_app.py      # Streamlit frontend
├── models/
│   ├── best_model.pkl        # Saved best model + preprocessor pipeline
│   ├── model_comparison.json # RMSE / R² for all three models
│   └── feature_importance.json
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the dataset

**Option A — Local path (default for local development)**

1. Go to [https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
2. Download `cardekho.csv` (or the equivalent CSV from that dataset page)
3. Place it at `data/cardekho.csv` inside this project directory

**Option B — Kaggle environment**

When running inside a Kaggle notebook, the dataset is automatically available at:

```
/kaggle/input/car-price-prediction-dataset/cardekho.csv
```

`src/config.py` checks this path first; if it exists, it is used automatically. No configuration change is needed.

---

## Running the Full Stack

Run each step in order. Steps 2 and 3 must run simultaneously in separate terminals.

### Step 1 — Train the model

```bash
python -m src.train
```

This reads `data/cardekho.csv`, trains Linear Regression, Random Forest, and XGBoost models, selects the best one by RMSE, and writes:
- `models/best_model.pkl`
- `models/model_comparison.json`
- `models/feature_importance.json`

### Step 2 — Start the FastAPI backend

```bash
uvicorn api.main:app --reload --port 8000
```

The API starts on [http://localhost:8000](http://localhost:8000). Interactive docs are available at [http://localhost:8000/docs](http://localhost:8000/docs).

### Step 3 — Start the Streamlit frontend

In a **second terminal**:

```bash
streamlit run app/streamlit_app.py
```

The app opens at [http://localhost:8501](http://localhost:8501).

---

## Streamlit Pages

| Page | Description |
|---|---|
| **Predict** | Enter car details (fuel type, transmission, km driven, engine size, etc.) using sidebar widgets and click **Predict Price** to get an estimated selling price from the API. The result is shown as a large metric callout with a Plotly gauge. |
| **Model Comparison** | Bar chart showing RMSE and R² for all three trained models (Linear Regression, Random Forest, XGBoost), sourced from `GET /models`. |
| **Feature Importance** | Horizontal bar chart of the top feature importances from the best tree-based model, sourced from `GET /features`. |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | Accepts raw car features as JSON, returns `{"predicted_price": <float>}` |
| `GET` | `/models` | Returns model comparison metrics (RMSE, R²) |
| `GET` | `/features` | Returns feature importance values |
| `GET` | `/health` | Returns `{"status": "ok"}` |

---

## Key Dependencies

| Library | Purpose |
|---|---|
| pandas, numpy | Data loading and manipulation |
| scikit-learn | Preprocessing pipeline and Linear Regression / Random Forest |
| xgboost | Gradient boosted tree model |
| fastapi + uvicorn | REST API server |
| streamlit | Interactive Python frontend |
| plotly | Interactive charts in Streamlit |
| joblib | Model serialization |
| pydantic | Request validation in FastAPI |
