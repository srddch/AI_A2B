import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
import math
from pathlib import Path
from tensorflow.keras.models import load_model


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "saved_models"
DATA_DIR = BASE_DIR / "data" / "processed"

lstm_model = load_model(MODEL_DIR / "lstm_model.keras")
gru_model = load_model(MODEL_DIR / "gru_model.keras")

scaler = joblib.load(MODEL_DIR / "scaler.pkl")
encoder = joblib.load(MODEL_DIR / "onehot_encoder.pkl")

RF_MODEL_PATH = MODEL_DIR / "rf_model.pkl"

if RF_MODEL_PATH.exists():
    rf_model = joblib.load(RF_MODEL_PATH)
else:
    rf_model = None


_historical_data = None


def _load_historical_data():
    global _historical_data

    if _historical_data is None:
        train_df = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["timestamp"])
        val_df = pd.read_csv(DATA_DIR / "val.csv", parse_dates=["timestamp"])
        test_df = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["timestamp"])

        _historical_data = pd.concat(
            [train_df, val_df, test_df],
            ignore_index=True
        )

    return _historical_data


def get_past_4_flows(site_id, departure_time):
    df = _load_historical_data()

    site_data = df[df["site_number"] == site_id].copy()

    if site_data.empty:
        raise ValueError(f"Site {site_id} has no historical data")

    site_data = site_data.sort_values("timestamp").reset_index(drop=True)

    time_diffs = abs(site_data["timestamp"] - pd.to_datetime(departure_time))
    closest_idx = time_diffs.idxmin()

    row = site_data.iloc[closest_idx]

    return [
        float(row["flow_t_minus_4"]),
        float(row["flow_t_minus_3"]),
        float(row["flow_t_minus_2"]),
        float(row["flow_t_minus_1"])
    ]


def predict_travel_time_by_departure(edge_features, model_name="lstm"):
    past_4_flows = get_past_4_flows(
        edge_features["from_site"],
        edge_features["departure_time"]
    )

    full_edge_features = {
        "from_site": edge_features["from_site"],
        "to_site": edge_features["to_site"],
        "distance_km": edge_features["distance_km"],
        "past_4_flows": past_4_flows
    }

    return predict_travel_time(full_edge_features, model_name)


def predict_flow(site_id, past_4_flows, model_name="lstm"):
    model_name = model_name.lower()

    if model_name == "rf":
        if rf_model is None:
            raise FileNotFoundError(
                "rf_model.pkl not found. Please run: python models/train_rf_for_ui.py"
            )

        X_rf = np.array(past_4_flows).reshape(1, -1)
        pred = rf_model.predict(X_rf)
        return float(pred[0])

    site_encoded = encoder.transform([[site_id]]).flatten()

    past_4_flows_np = np.array(past_4_flows).reshape(-1, 1)
    past_4_flows_scaled = scaler.transform(past_4_flows_np).flatten()

    X_seq = []

    for i in range(4):
        step_features = np.concatenate([
            site_encoded,
            [past_4_flows_scaled[i]]
        ])
        X_seq.append(step_features)

    X_seq = np.array([X_seq])

    if model_name == "lstm":
        model = lstm_model
    elif model_name == "gru":
        model = gru_model
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    pred = model.predict(X_seq, verbose=0)
    return float(pred[0, 0])


def flow_to_travel_time(distance_km, flow_veh_per_15min):
    flow_per_hour = flow_veh_per_15min * 4.0

    if flow_per_hour <= 351.0:
        speed = 60.0
    else:
        a = -1.4648375
        b = 93.75
        c = -flow_per_hour

        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            speed = 32.0
        else:
            v1 = (-b + math.sqrt(discriminant)) / (2 * a)
            v2 = (-b - math.sqrt(discriminant)) / (2 * a)

            if flow_per_hour <= 1500.0:
                speed = max(v1, v2)
            else:
                speed = min(v1, v2)

        speed = min(max(speed, 1.0), 60.0)

    return (distance_km / speed) * 60.0 + 0.5


def predict_travel_time(edge_features, model_name="lstm"):
    future_flow = predict_flow(
        site_id=edge_features["from_site"],
        past_4_flows=edge_features["past_4_flows"],
        model_name=model_name
    )

    return flow_to_travel_time(
        edge_features["distance_km"],
        future_flow
    )