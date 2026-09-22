import pandas as pd
from src.config import ALL_FEATURES


def make_prediction(model, input_dict: dict) -> float:
    """Convert raw input dict to single-row DataFrame and predict."""
    df = pd.DataFrame([input_dict])[ALL_FEATURES]
    prediction = model.predict(df)[0]
    return float(prediction)
