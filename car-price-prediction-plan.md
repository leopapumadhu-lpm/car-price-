# Car Price Prediction — Project Plan

## Top-Level Overview

Build a full end-to-end car price prediction application using Python exclusively.

- **Data**: CarDekho dataset (`cardekho.csv`) from Kaggle
- **Backend**: FastAPI REST API that loads the trained model and serves predictions
- **Frontend**: Streamlit application with interactive form (sliders, dropdowns) and charts (model comparison, feature importance, price distribution)
- **ML Pipeline**: Train and compare Linear Regression, Random Forest, and XGBoost; persist the best model; expose via the API
- **Deployment**: Single-machine local run — `uvicorn` for the API, `streamlit run` for the UI

---

## Project Structure

```
car-price/
├── data/
│   └── cardekho.csv                  # Raw dataset
├── notebooks/
│   └── eda.ipynb                     # Exploratory data analysis (optional)
├── src/
│   ├── preprocess.py                 # Data loading and feature engineering
│   ├── train.py                      # Model training, comparison, and persistence
│   ├── predict.py                    # Prediction helper used by the API
│   └── config.py                     # Shared constants (paths, feature names)
├── api/
│   └── main.py                       # FastAPI app with /predict and /models endpoints
├── app/
│   └── streamlit_app.py              # Streamlit frontend
├── models/
│   └── best_model.pkl                # Saved best model + preprocessor pipeline
├── requirements.txt
└── README.md
```

---

## Sub-Tasks

---

### Sub-Task 1 — Project Scaffolding & Dataset Setup

**Status**: `[x] done`

**Intent**
Create the full directory structure, install dependencies, and make the raw dataset available at the expected path so every downstream task has a stable foundation.

**Expected Outcomes**
- All directories exist (`data/`, `src/`, `api/`, `app/`, `models/`, `notebooks/`)
- `requirements.txt` lists all dependencies (pandas, numpy, scikit-learn, xgboost, fastapi, uvicorn, streamlit, plotly, joblib, python-multipart)
- `src/config.py` defines shared constants: `DATA_PATH`, `MODEL_PATH`, feature column names, and target column name
- `README.md` contains run instructions

**Todo List**
1. Create the directory layout listed above
2. Write `requirements.txt` with pinned-or-minimum versions
3. Write `src/config.py` with all shared constants
4. Copy/symlink `cardekho.csv` into `data/` (or document the path assumption)
5. Write `README.md` with setup and run steps

**Relevant Context**
- Dataset path assumption: `data/cardekho.csv` (local copy of the Kaggle file)
- No virtual-env tooling is prescribed — user manages their Python env

---

### Sub-Task 2 — Data Preprocessing & Feature Engineering

**Status**: `[x] done`

**Intent**
Clean the raw CSV and produce a reusable preprocessing pipeline (scikit-learn `Pipeline` + `ColumnTransformer`) that can be saved alongside the model so the API receives raw user inputs and the pipeline handles all transforms.

**Expected Outcomes**
- `src/preprocess.py` exposes a `build_pipeline()` function returning a scikit-learn `Pipeline`
- Raw data issues handled: missing values, inconsistent string formatting (e.g., `"Petrol"` vs `" petrol"`), derived features (car age from year), dropping unused columns
- Categorical features (fuel type, seller type, transmission, owner) are one-hot or ordinal encoded inside the pipeline
- Numerical features (km_driven, engine, max_power, mileage, seats) are scaled where appropriate
- A `load_data()` function returns a clean `(X, y)` tuple ready for model fitting

**Todo List**
1. Read and inspect the cardekho.csv columns and dtypes
2. Define feature lists (numerical, categorical, target) in `src/config.py`
3. Implement `load_data()` in `src/preprocess.py` — clean strings, derive `car_age`, drop `name`/`year` columns, handle nulls
4. Implement `build_pipeline()` — `ColumnTransformer` for encoding + scaling wrapped in a `Pipeline`
5. Add a quick `__main__` block that prints shape and sample after transform (smoke test)

**Relevant Context**
- Key columns in cardekho.csv: `name`, `year`, `selling_price` (target), `km_driven`, `fuel`, `seller_type`, `transmission`, `owner`, `mileage`, `engine`, `max_power`, `torque`, `seats`
- `car_age = current_year - year` replaces the raw `year` column
- `torque` is often dropped due to mixed units / poor signal

---

### Sub-Task 3 — Model Training, Comparison & Persistence

**Status**: `[x] done`

**Intent**
Train three candidate models, evaluate them on a held-out test set using RMSE and R², select the best, and persist the full pipeline (preprocessor + model) to disk so the API can load it once at startup.

**Expected Outcomes**
- `src/train.py` is a runnable script (`python -m src.train`) that produces `models/best_model.pkl`
- Three models trained: `LinearRegression`, `RandomForestRegressor`, `XGBRegressor`
- Cross-validated RMSE and R² printed/logged for each model
- Best model selected automatically (lowest RMSE on test split)
- Comparison results saved to `models/model_comparison.json` for the Streamlit dashboard
- Feature importances saved to `models/feature_importance.json` (from best tree model)

**Todo List**
1. Implement `train_and_evaluate(pipeline, X, y)` helper returning metrics dict
2. Build the three pipelines (same preprocessor, different estimator)
3. Run 5-fold cross-validation on each; record mean RMSE and R²
4. Train each on full train split, evaluate on test split
5. Serialize best pipeline with `joblib.dump` to `models/best_model.pkl`
6. Write `models/model_comparison.json` and `models/feature_importance.json`

**Relevant Context**
- `src/preprocess.py` — `build_pipeline()` and `load_data()` are consumed here
- `src/config.py` — `MODEL_PATH`, `COMPARISON_PATH`, `FEATURE_IMPORTANCE_PATH`
- Use `train_test_split` with `random_state=42` and `test_size=0.2`
- XGBoost: `n_estimators=300`, `learning_rate=0.05`, `max_depth=6`; Random Forest: `n_estimators=200`

---

### Sub-Task 4 — FastAPI Backend

**Status**: `[x] done`

**Intent**
Expose the trained model as a lightweight REST API so the Streamlit frontend can call it for predictions and retrieve model metadata without coupling to the training code.

**Expected Outcomes**
- `api/main.py` starts cleanly with `uvicorn api.main:app --reload`
- `POST /predict` accepts a JSON body matching raw feature inputs and returns `{"predicted_price": <float>}`
- `GET /models` returns the model comparison metrics from `models/model_comparison.json`
- `GET /features` returns feature importance data from `models/feature_importance.json`
- `GET /health` returns `{"status": "ok"}`
- Pydantic `CarFeatures` request model validates all inputs with sensible ranges

**Todo List**
1. Define `CarFeatures` Pydantic model with all input fields and validation constraints
2. Load `best_model.pkl` once at app startup using a FastAPI `lifespan` context
3. Implement `POST /predict` endpoint — call `model.predict(input_df)`, return price
4. Implement `GET /models`, `GET /features`, `GET /health` endpoints
5. Add CORS middleware so Streamlit (running on a different port) can call the API
6. Test endpoints manually with `curl` or FastAPI's built-in `/docs` Swagger UI

**Relevant Context**
- `src/predict.py` wraps `model.predict()` and converts raw dict → single-row DataFrame
- The model pipeline expects the same raw column names as in training (before preprocessing)
- Default API port: `8000`; Streamlit port: `8501`

---

### Sub-Task 5 — Streamlit Frontend

**Status**: `[x] done`

**Intent**
Build an interactive Streamlit UI that lets users input car features and see a predicted price, alongside visual dashboards for model performance and feature importance — all in pure Python.

**Expected Outcomes**
- `app/streamlit_app.py` runs with `streamlit run app/streamlit_app.py`
- **Sidebar**: dropdowns and sliders for all car features (fuel type, transmission, owner, km driven, car age, engine, power, seats)
- **Page 1 — Predict**: sends form data to `POST /predict`, displays predicted price prominently, shows a Plotly gauge or number callout
- **Page 2 — Model Comparison**: bar chart (Plotly) of RMSE and R² for all three models (data from `GET /models`)
- **Page 3 — Feature Importance**: horizontal bar chart of feature importances (from `GET /features`)
- Clean layout with `st.columns`, `st.metric`, `st.plotly_chart`

**Todo List**
1. Set up page config (`st.set_page_config`) and navigation (`st.sidebar.radio`)
2. Build the sidebar input form (all features as widgets)
3. Implement Predict page — call API on button click, display result with `st.metric` + Plotly gauge
4. Implement Model Comparison page — fetch `GET /models`, render grouped bar chart
5. Implement Feature Importance page — fetch `GET /features`, render horizontal bar chart
6. Handle API errors gracefully (connection refused, bad response) with `st.error()`

**Relevant Context**
- API base URL set as a constant at the top of `streamlit_app.py` (default: `http://localhost:8000`)
- Use `requests` library for HTTP calls from Streamlit to FastAPI
- `plotly.express` for all charts; no matplotlib required
- Feature value ranges for sliders should match training data distributions

---

### Sub-Task 6 — Integration Testing & README

**Status**: `[x] done`

**Intent**
Verify the full pipeline works end-to-end: train → API up → Streamlit prediction returns a reasonable price. Document how to run the whole stack.

**Expected Outcomes**
- Running `python -m src.train` produces `models/best_model.pkl` without errors
- API starts and `/health`, `/predict`, `/models`, `/features` all return 200
- Streamlit connects to the API and shows a predicted price for a test input
- `README.md` has clear step-by-step run instructions

**Todo List**
1. Run `python -m src.train` end-to-end; fix any import or path errors
2. Start API, hit all endpoints with sample payloads; verify responses
3. Start Streamlit, test prediction form, model comparison page, feature importance page
4. Update `README.md` with full setup + run instructions
5. Add a `.gitignore` excluding `models/*.pkl`, `data/`, `__pycache__`

**Relevant Context**
- Startup order: train first, then API, then Streamlit
- Both API and Streamlit must be running simultaneously for the app to work

---

## Dependency Graph

```
Sub-Task 1 (Scaffold)
    └── Sub-Task 2 (Preprocess)
            └── Sub-Task 3 (Train)
                    ├── Sub-Task 4 (API)
                    └── Sub-Task 5 (Streamlit)
                            └── Sub-Task 6 (Integration)
```

---

## Key Technology Choices

| Layer | Library | Reason |
|---|---|---|
| Data | pandas, numpy | Standard tabular data handling |
| ML | scikit-learn, xgboost | Pipeline API + best-in-class tree boosting |
| Backend | FastAPI + uvicorn | Async, Pydantic validation, auto Swagger docs |
| Frontend | Streamlit | Pure Python interactive UI, zero JS |
| Charts | Plotly Express | Interactive charts, native Streamlit support |
| Model I/O | joblib | Efficient sklearn/xgboost serialization |
