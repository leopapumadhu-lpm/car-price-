"""
app/streamlit_app.py
────────────────────
Streamlit frontend for the Car Price Prediction application.

Pages
─────
1. 🔮 Predict Price       — form input + predicted price gauge
2. 📊 Model Comparison    — RMSE and R² bar charts for all three models
3. 📈 Feature Importance  — horizontal bar chart of top features
4. 🔍 EDA                 — exploratory data analysis charts from the dataset

Run:
    streamlit run app/streamlit_app.py
"""

import datetime
import os

import numpy as np
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"


# ── API helper ─────────────────────────────────────────────────────────────────
def call_api(method: str, endpoint: str, payload: dict | None = None):
    """
    Call the FastAPI backend.

    Returns parsed JSON on success, or None (after showing st.error) on failure.
    """
    url = f"{API_URL}{endpoint}"
    try:
        if method.upper() == "POST":
            resp = requests.post(url, json=payload, timeout=10)
        else:
            resp = requests.get(url, timeout=10)

        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"API error {resp.status_code}: {resp.json().get('detail', resp.text)}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Cannot connect to the API. "
            "Make sure the FastAPI server is running:\n\n"
            "```\nuvicorn api.main:app --reload --port 8000\n```"
        )
        return None
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        return None


# ── Brand defaults (computed once from dataset) ────────────────────────────────
@st.cache_data(show_spinner=False)
def load_brand_defaults():
    """
    Return a dict of brand → median feature values derived from the dataset.
    Used to auto-populate sidebar sliders when a brand is selected.
    """
    _dp = os.path.join(os.path.dirname(__file__), "..", "data", "cardekho.csv")
    df = pd.read_csv(_dp)
    df["brand"] = df["name"].str.split().str[0]
    df["car_age"] = datetime.datetime.now().year - df["year"]
    df["max_power_num"] = pd.to_numeric(
        df["max_power"].replace("", np.nan).astype(str).str.strip().replace("nan", np.nan),
        errors="coerce",
    )
    df = df.rename(columns={"mileage(km/ltr/kg)": "mileage"})

    # Global fallback medians (used when a brand subgroup has NaN medians)
    g_engine    = float(df["engine"].median())
    g_power     = float(df["max_power_num"].median())
    g_mileage   = float(df["mileage"].median())
    g_seats     = float(df["seats"].median())

    def _int_snap(val, fallback, snap, lo, hi):
        v = val if not (isinstance(val, float) and np.isnan(val)) else fallback
        return max(lo, min(hi, int(round(float(v) / snap) * snap)))

    def _flt_clamp(val, fallback, lo, hi):
        v = val if not (isinstance(val, float) and np.isnan(val)) else fallback
        return max(lo, min(hi, round(float(v), 1)))

    defaults = {}
    for brand, grp in df.groupby("brand"):
        defaults[brand] = {
            "car_age":      _int_snap(grp["car_age"].median(),         10,        1,    0,  25),
            "km_driven":    _int_snap(grp["km_driven"].median(),    50_000,    1_000,    0, 500_000),
            "mileage":      _flt_clamp(grp["mileage"].median(),    g_mileage,  5.0, 40.0),
            "engine":       _int_snap(grp["engine"].median(),       g_engine,     50,  500, 5_000),
            "max_power":    _flt_clamp(grp["max_power_num"].median(), g_power, 30.0, 500.0),
            "seats":        _int_snap(grp["seats"].median(),            5,         1,    2,  14),
            "fuel":         grp["fuel"].str.lower().mode()[0],
            "transmission": grp["transmission"].str.lower().mode()[0],
        }
    return defaults

BRAND_DEFAULTS = load_brand_defaults()
ALL_BRANDS = ["(select brand)"] + sorted(BRAND_DEFAULTS.keys())

# ── Sidebar — input form ───────────────────────────────────────────────────────
st.sidebar.title("🚗 Car Features")
st.sidebar.markdown("---")

# ── Brand selector ─────────────────────────────────────────────────────────────
brand = st.sidebar.selectbox(
    "🏷️ Car Brand",
    ALL_BRANDS,
    index=0,
    help="Selecting a brand auto-fills typical values for that brand's cars.",
)

# Resolve defaults: use brand medians if a brand is selected, else fall back to global defaults
if brand != "(select brand)":
    bd = BRAND_DEFAULTS[brand]
    _def_age       = bd["car_age"]
    _def_km        = bd["km_driven"]
    _def_mileage   = bd["mileage"]
    _def_engine    = bd["engine"]
    _def_power     = bd["max_power"]
    _def_seats     = bd["seats"]
    _def_fuel      = bd["fuel"]
    _def_trans     = bd["transmission"]
else:
    _def_age, _def_km, _def_mileage = 5, 50_000, 18.0
    _def_engine, _def_power, _def_seats = 1_200, 82.0, 5
    _def_fuel, _def_trans = "petrol", "manual"

if brand != "(select brand)":
    st.sidebar.caption(f"ℹ️ Defaults auto-filled for **{brand}** (edit below as needed)")

st.sidebar.markdown("---")

fuel = st.sidebar.selectbox(
    "Fuel Type",
    ["petrol", "diesel", "cng", "lpg"],
    index=["petrol", "diesel", "cng", "lpg"].index(_def_fuel)
          if _def_fuel in ["petrol", "diesel", "cng", "lpg"] else 0,
)
seller_type = st.sidebar.selectbox(
    "Seller Type",
    ["individual", "dealer", "trustmark dealer"],
    index=0,
)
transmission = st.sidebar.selectbox(
    "Transmission",
    ["manual", "automatic"],
    index=["manual", "automatic"].index(_def_trans)
          if _def_trans in ["manual", "automatic"] else 0,
)
owner = st.sidebar.selectbox(
    "Owner",
    ["first owner", "second owner", "third owner", "fourth & above owner", "test drive car"],
    index=0,
)
car_age = st.sidebar.slider("Car Age (years)", min_value=0, max_value=25, value=_def_age)
km_driven = st.sidebar.number_input(
    "Km Driven",
    min_value=0,
    max_value=500_000,
    value=_def_km,
    step=1_000,
)
mileage = st.sidebar.slider(
    "Mileage (km/ltr/kg)",
    min_value=5.0,
    max_value=40.0,
    value=_def_mileage,
    step=0.5,
)
engine = st.sidebar.slider(
    "Engine Capacity (CC)",
    min_value=500,
    max_value=5_000,
    value=_def_engine,
    step=50,
)
max_power = st.sidebar.slider(
    "Max Power (bhp)",
    min_value=30.0,
    max_value=500.0,
    value=_def_power,
    step=1.0,
)
seats = st.sidebar.slider("Seats", min_value=2, max_value=14, value=_def_seats)

# Build payload (must match CarFeatures Pydantic model)
payload = {
    "fuel": fuel,
    "seller_type": seller_type,
    "transmission": transmission,
    "owner": owner,
    "car_age": car_age,
    "km_driven": float(km_driven),
    "mileage": mileage,
    "engine": float(engine),
    "max_power": max_power,
    "seats": float(seats),
}

# ── Navigation ─────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🔮 Predict Price", "📊 Model Comparison", "📈 Feature Importance", "🔍 EDA"],
    index=0,
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Predict Price
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Predict Price":
    st.title("🔮 Car Price Predictor")
    st.markdown(
        "Adjust the car features in the **sidebar** and click **Predict Price** "
        "to get an estimated selling price."
    )
    st.markdown("---")

    col_btn, col_space = st.columns([1, 3])
    with col_btn:
        predict_clicked = st.button("🚀 Predict Price", use_container_width=True, type="primary")

    if predict_clicked:
        with st.spinner("Calling model…"):
            result = call_api("POST", "/predict", payload)

        if result:
            price = result["predicted_price"]
            st.success("✅ Prediction complete!")
            st.markdown("### 💰 Estimated Selling Price")

            # Large metric
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.metric(
                    label="Predicted Price (INR)",
                    value=f"₹ {price:,.0f}",
                )
            with col2:
                st.metric(label="Fuel Type", value=fuel.capitalize())
            with col3:
                st.metric(label="Transmission", value=transmission.capitalize())

            st.markdown("---")

            # Plotly gauge
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=price,
                    number={"prefix": "₹", "valueformat": ",.0f"},
                    title={"text": "Predicted Selling Price (INR)", "font": {"size": 18}},
                    gauge={
                        "axis": {"range": [0, 5_000_000], "tickformat": ",.0f"},
                        "bar": {"color": "#3b82d4"},
                        "steps": [
                            {"range": [0, 500_000],       "color": "#e5f0ff"},
                            {"range": [500_000, 1_500_000], "color": "#bfdbfe"},
                            {"range": [1_500_000, 3_000_000], "color": "#93c5fd"},
                            {"range": [3_000_000, 5_000_000], "color": "#60a5fa"},
                        ],
                        "threshold": {
                            "line": {"color": "#1d4ed8", "width": 4},
                            "thickness": 0.75,
                            "value": price,
                        },
                    },
                )
            )
            fig.update_layout(height=350, margin=dict(t=60, b=20, l=40, r=40))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Selected Car Features")
    display_row = {"Brand": brand if brand != "(select brand)" else "—"} | payload
    st.dataframe(
        pd.DataFrame([display_row]).rename(columns={
            "fuel": "Fuel",
            "seller_type": "Seller Type",
            "transmission": "Transmission",
            "owner": "Owner",
            "car_age": "Car Age (yrs)",
            "km_driven": "Km Driven",
            "mileage": "Mileage",
            "engine": "Engine (CC)",
            "max_power": "Max Power (bhp)",
            "seats": "Seats",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")
    st.markdown(
        "Performance of all three trained models evaluated on the **20% held-out test set**."
    )
    st.markdown("---")

    data = call_api("GET", "/models")

    if data:
        df = pd.DataFrame(data)
        df.columns = [c.capitalize() for c in df.columns]

        # Highlight best model
        best_row = df.loc[df["Rmse"].idxmin()]
        st.info(
            f"🏆 **Best Model: {best_row['Name']}** — "
            f"RMSE ₹{best_row['Rmse']:,.0f} | R² {best_row['R2']:.4f}"
        )

        st.subheader("Raw Metrics")
        styled = df.style.highlight_min(subset=["Rmse"], color="#d1fae5") \
                         .highlight_max(subset=["R2"],   color="#d1fae5") \
                         .format({"Rmse": "₹{:,.0f}", "R2": "{:.4f}"})
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Visual Comparison")

        col1, col2 = st.columns(2)

        with col1:
            fig_rmse = px.bar(
                df,
                x="Name",
                y="Rmse",
                title="RMSE by Model (lower is better ↓)",
                color="Name",
                color_discrete_sequence=px.colors.qualitative.Set2,
                text_auto=".3s",
            )
            fig_rmse.update_layout(
                showlegend=False,
                yaxis_title="RMSE (₹)",
                xaxis_title="",
            )
            fig_rmse.update_traces(textposition="outside")
            st.plotly_chart(fig_rmse, use_container_width=True)

        with col2:
            fig_r2 = px.bar(
                df,
                x="Name",
                y="R2",
                title="R² Score by Model (higher is better ↑)",
                color="Name",
                color_discrete_sequence=px.colors.qualitative.Set2,
                text_auto=".4f",
            )
            fig_r2.update_layout(
                showlegend=False,
                yaxis_title="R² Score",
                xaxis_title="",
                yaxis_range=[0, 1],
            )
            fig_r2.update_traces(textposition="outside")
            st.plotly_chart(fig_r2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Feature Importance":
    st.title("📈 Feature Importance")
    st.markdown(
        "Top features driving the price prediction from the best tree-based model."
    )
    st.markdown("---")

    data = call_api("GET", "/features")

    if data is not None:
        if not data:
            st.info(
                "ℹ️ Feature importance is not available for linear models. "
                "Retrain with a tree-based model (Random Forest or XGBoost) selected as best."
            )
        else:
            df = pd.DataFrame(
                {"Feature": list(data.keys()), "Importance": list(data.values())}
            ).sort_values("Importance", ascending=False).reset_index(drop=True)

            st.subheader(f"Top {len(df)} Features")

            fig = px.bar(
                df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Feature Importance (best model)",
                color="Importance",
                color_continuous_scale="Blues",
                text_auto=".4f",
            )
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                xaxis_title="Importance Score",
                yaxis_title="",
                coloraxis_showscale=False,
                height=500,
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("Raw Importance Values")
            st.dataframe(
                df.style.background_gradient(subset=["Importance"], cmap="Blues"),
                use_container_width=True,
                hide_index=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — EDA  (mirrors notebooks/eda.ipynb, rendered with Plotly)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 EDA":
    st.title("🔍 Exploratory Data Analysis")
    st.markdown(
        "Interactive version of [`notebooks/eda.ipynb`](../notebooks/eda.ipynb) — "
        "all charts are built live from `data/cardekho.csv`."
    )
    st.markdown("---")

    # ── Load & clean (same logic as src/preprocess.py) ────────────────────────
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cardekho.csv")

    @st.cache_data(show_spinner="Loading dataset…")
    def load_eda_data():
        df = pd.read_csv(DATA_PATH)
        df["car_age"] = datetime.datetime.now().year - df["year"]
        df["max_power"] = pd.to_numeric(
            df["max_power"].replace("", np.nan).astype(str).str.strip().replace("nan", np.nan),
            errors="coerce",
        )
        df = df.rename(columns={"mileage(km/ltr/kg)": "mileage"})
        for col in ["fuel", "seller_type", "transmission", "owner"]:
            df[col] = df[col].str.strip()
        return df

    df = load_eda_data()
    CAP_99 = df["selling_price"].quantile(0.99)  # used to cap charts

    # ── Section selector ──────────────────────────────────────────────────────
    section = st.selectbox(
        "Jump to section",
        [
            "1. Dataset Overview",
            "2. Missing Values",
            "3. Target Distribution — Selling Price",
            "4. Categorical Feature Distributions",
            "5. Numerical Feature Distributions",
            "6. Correlation Heatmap",
            "7. Price vs Key Features",
            "8. Outlier Analysis",
            "9. Key Findings Summary",
        ],
    )

    st.markdown("---")

    # ── 1. Dataset Overview ───────────────────────────────────────────────────
    if section.startswith("1"):
        st.subheader("1. Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", f"{len(df):,}")
        col2.metric("Columns", len(df.columns))
        col3.metric("Year Range", f"{int(df['year'].min())} – {int(df['year'].max())}")

        st.markdown("**First 10 rows**")
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

        st.markdown("**Data types & non-null counts**")
        info_df = pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "non_null": df.notnull().sum(),
            "null": df.isnull().sum(),
        })
        st.dataframe(info_df, use_container_width=True)

        st.markdown("**Descriptive statistics**")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    # ── 2. Missing Values ─────────────────────────────────────────────────────
    elif section.startswith("2"):
        st.subheader("2. Missing Values")
        missing = df.isnull().sum().sort_values(ascending=False)
        missing_pct = (missing / len(df) * 100).round(2)
        miss_df = pd.DataFrame({"Column": missing.index, "Count": missing.values, "Pct (%)": missing_pct.values})
        miss_df = miss_df[miss_df["Count"] > 0].reset_index(drop=True)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(miss_df, use_container_width=True, hide_index=True)
        with col2:
            fig = px.bar(
                miss_df, x="Pct (%)", y="Column", orientation="h",
                title="Missing Values (%)",
                color="Pct (%)", color_continuous_scale="Reds",
                text_auto=".2f",
            )
            fig.update_layout(coloraxis_showscale=False, yaxis_title="")
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        st.info("mileage, engine, max_power and seats each have ~2.7% missing — filled with column median during training.")

    # ── 3. Target Distribution ────────────────────────────────────────────────
    elif section.startswith("3"):
        st.subheader("3. Target Distribution — Selling Price")
        price = df["selling_price"].dropna()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Mean",   f"₹{price.mean():,.0f}")
        c2.metric("Median", f"₹{price.median():,.0f}")
        c3.metric("Std",    f"₹{price.std():,.0f}")
        c4.metric("Min",    f"₹{price.min():,.0f}")
        c5.metric("Max",    f"₹{price.max():,.0f}")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                df, x="selling_price", nbins=60,
                title="Raw Price Distribution",
                labels={"selling_price": "Selling Price (₹)"},
                color_discrete_sequence=["#3b82d4"],
            )
            fig.update_layout(bargap=0.05)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            df_log = df.copy()
            df_log["log_price"] = np.log1p(df_log["selling_price"])
            fig = px.histogram(
                df_log, x="log_price", nbins=60,
                title="Log-Transformed Price Distribution",
                labels={"log_price": "log(1 + Price)"},
                color_discrete_sequence=["#7c5cd8"],
            )
            fig.update_layout(bargap=0.05)
            st.plotly_chart(fig, use_container_width=True)

        st.caption("The raw distribution is heavily right-skewed. Log transformation reveals a near-normal shape, which benefits linear models.")

    # ── 4. Categorical Feature Distributions ──────────────────────────────────
    elif section.startswith("4"):
        st.subheader("4. Categorical Feature Distributions")
        cat_cols = ["fuel", "seller_type", "transmission", "owner"]

        tab1, tab2 = st.tabs(["Count", "Median Price"])

        with tab1:
            col1, col2 = st.columns(2)
            for i, col in enumerate(cat_cols):
                vc = df[col].value_counts().reset_index()
                vc.columns = [col, "count"]
                fig = px.bar(vc, x=col, y="count", title=f"{col.replace('_',' ').title()} — Count",
                             color=col, color_discrete_sequence=px.colors.qualitative.Set2,
                             text_auto=True)
                fig.update_layout(showlegend=False, xaxis_title="")
                fig.update_traces(textposition="outside")
                (col1 if i % 2 == 0 else col2).plotly_chart(fig, use_container_width=True)

        with tab2:
            col1, col2 = st.columns(2)
            for i, col in enumerate(cat_cols):
                grp = df.groupby(col)["selling_price"].median().reset_index().sort_values("selling_price", ascending=False)
                grp.columns = [col, "median_price"]
                fig = px.bar(grp, x=col, y="median_price",
                             title=f"Median Price by {col.replace('_',' ').title()}",
                             color=col, color_discrete_sequence=px.colors.qualitative.Set2,
                             text_auto=".3s")
                fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Median Price (₹)")
                fig.update_traces(textposition="outside")
                (col1 if i % 2 == 0 else col2).plotly_chart(fig, use_container_width=True)

    # ── 5. Numerical Feature Distributions ────────────────────────────────────
    elif section.startswith("5"):
        st.subheader("5. Numerical Feature Distributions")
        num_cols = [
            ("car_age",   "Car Age (yrs)"),
            ("km_driven", "Km Driven"),
            ("mileage",   "Mileage (km/l)"),
            ("engine",    "Engine (CC)"),
            ("max_power", "Max Power (bhp)"),
            ("seats",     "Seats"),
        ]
        col1, col2 = st.columns(2)
        for i, (col, lbl) in enumerate(num_cols):
            fig = px.histogram(
                df, x=col, nbins=50,
                title=lbl,
                labels={col: lbl},
                color_discrete_sequence=["#3b82d4"],
            )
            med = df[col].median()
            fig.add_vline(x=med, line_dash="dash", line_color="#dc2626",
                          annotation_text=f"Median: {med:.1f}", annotation_position="top right")
            fig.update_layout(bargap=0.05)
            (col1 if i % 2 == 0 else col2).plotly_chart(fig, use_container_width=True)

    # ── 6. Correlation Heatmap ────────────────────────────────────────────────
    elif section.startswith("6"):
        st.subheader("6. Correlation Heatmap")
        corr_cols = ["selling_price", "car_age", "km_driven", "mileage", "engine", "max_power", "seats"]
        corr = df[corr_cols].dropna().corr().round(3)

        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Pearson Correlation Matrix",
            aspect="auto",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Correlation with `selling_price` (sorted)**")
        corr_with_price = corr["selling_price"].drop("selling_price").sort_values(ascending=False)
        corr_df = corr_with_price.reset_index()
        corr_df.columns = ["Feature", "Correlation"]
        fig2 = px.bar(corr_df, x="Feature", y="Correlation",
                      color="Correlation", color_continuous_scale="RdBu_r",
                      title="Correlation with Selling Price", text_auto=".3f")
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        fig2.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

        st.info("**max_power** has the strongest positive correlation (≈0.74) with selling price, followed by **engine** (≈0.63). **car_age** and **km_driven** are negatively correlated.")

    # ── 7. Price vs Key Features ──────────────────────────────────────────────
    elif section.startswith("7"):
        st.subheader("7. Price vs Key Features")
        sub = df[df["selling_price"] <= CAP_99].copy()

        tab1, tab2, tab3 = st.tabs(["Scatter Plots", "Box Plots", "Price vs Car Age"])

        with tab1:
            scatter_cols = [
                ("max_power", "Max Power (bhp)"),
                ("car_age",   "Car Age (yrs)"),
                ("km_driven", "Km Driven"),
                ("engine",    "Engine (CC)"),
            ]
            col1, col2 = st.columns(2)
            for i, (col, lbl) in enumerate(scatter_cols):
                fig = px.scatter(
                    sub, x=col, y="selling_price", color="fuel",
                    title=f"Price vs {lbl}",
                    labels={col: lbl, "selling_price": "Selling Price (₹)", "fuel": "Fuel"},
                    opacity=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                (col1 if i % 2 == 0 else col2).plotly_chart(fig, use_container_width=True)

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.box(sub, x="fuel", y="selling_price",
                             title="Price Distribution by Fuel Type",
                             color="fuel", color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Selling Price (₹)")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.box(sub, x="transmission", y="selling_price",
                             title="Price Distribution by Transmission",
                             color="transmission", color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Selling Price (₹)")
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            age_price = df[df["car_age"] <= 25].groupby("car_age")["selling_price"].median().reset_index()
            fig = px.line(
                age_price, x="car_age", y="selling_price",
                title="Median Selling Price vs Car Age",
                labels={"car_age": "Car Age (yrs)", "selling_price": "Median Price (₹)"},
                markers=True,
                color_discrete_sequence=["#3b82d4"],
            )
            fig.update_traces(fill="tozeroy", fillcolor="rgba(59,130,212,0.1)")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Median price drops steeply as car age increases — older cars sell for significantly less.")

    # ── 8. Outlier Analysis ───────────────────────────────────────────────────
    elif section.startswith("8"):
        st.subheader("8. Outlier Analysis")

        outlier_cols = [
            ("selling_price", "Selling Price"),
            ("km_driven",     "Km Driven"),
            ("max_power",     "Max Power (bhp)"),
        ]

        col1, col2, col3 = st.columns(3)
        cols_ui = [col1, col2, col3]
        for ui_col, (col, lbl) in zip(cols_ui, outlier_cols):
            data = df[col].dropna()
            q1, q3 = data.quantile(0.25), data.quantile(0.75)
            iqr = q3 - q1
            n_out = int(((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)).sum())
            fig = px.box(df, y=col, title=f"{lbl}<br><sup>{n_out} outliers ({n_out/len(data)*100:.1f}%)</sup>",
                         color_discrete_sequence=["#3b82d4"])
            fig.update_layout(yaxis_title=lbl)
            ui_col.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Top 10 Most Expensive Cars")
        top10 = df.nlargest(10, "selling_price")[
            ["name", "year", "selling_price", "fuel", "transmission", "owner"]
        ].reset_index(drop=True)
        top10.index += 1
        st.dataframe(
            top10.style.format({"selling_price": "₹{:,.0f}"}),
            use_container_width=True,
        )

    # ── 9. Key Findings Summary ───────────────────────────────────────────────
    elif section.startswith("9"):
        st.subheader("9. Key Findings Summary")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Dataset**")
            st.markdown(f"- **{len(df):,} records** across {int(df['year'].min())}–{int(df['year'].max())}")
            st.markdown(f"- Price range: ₹{df['selling_price'].min():,.0f} – ₹{df['selling_price'].max():,.0f}")
            st.markdown(f"- Median price: **₹{df['selling_price'].median():,.0f}**")
            st.markdown(f"- Most common fuel: **{df['fuel'].mode()[0]}**")
            st.markdown(f"- Most common transmission: **{df['transmission'].mode()[0]}**")
            st.markdown(f"- Most common owner type: **{df['owner'].mode()[0]}**")

            st.markdown("**Missing values**")
            miss = df.isnull().sum()
            for col in miss[miss > 0].index:
                st.markdown(f"- `{col}`: {miss[col]} rows ({miss[col]/len(df)*100:.1f}%) — filled with median")

        with col2:
            st.markdown("**Top predictors (Random Forest feature importance)**")
            importance_data = {
                "max_power":  0.724,
                "car_age":    0.191,
                "km_driven":  0.027,
                "mileage":    0.019,
                "engine":     0.019,
            }
            fig = px.bar(
                x=list(importance_data.values()),
                y=list(importance_data.keys()),
                orientation="h",
                title="Feature Importance (top 5)",
                labels={"x": "Importance", "y": "Feature"},
                color=list(importance_data.values()),
                color_continuous_scale="Blues",
                text_auto=".3f",
            )
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                coloraxis_showscale=False,
                height=280,
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Model performance**")
            model_data = {
                "Model": ["Linear Regression", "Random Forest", "XGBoost"],
                "RMSE (₹)": [445_015, 141_976, 144_081],
                "R²": [0.6979, 0.9692, 0.9683],
            }
            mdf = pd.DataFrame(model_data)
            st.dataframe(
                mdf.style
                   .highlight_min(subset=["RMSE (₹)"], color="#d1fae5")
                   .highlight_max(subset=["R²"],       color="#d1fae5")
                   .format({"RMSE (₹)": "₹{:,.0f}", "R²": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )
