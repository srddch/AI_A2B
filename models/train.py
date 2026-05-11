from preprocessing.loader import load_raw_data
from preprocessing.cleaner import clean_data
from preprocessing.sequence_builder import (
    build_sequences,
    train_test_split_sequence
)

from models.baseline_model import BaselineModel


def train_baseline_model():

    path = "../data/raw/Scats+Data+October+2006.xls"

    raw_df = load_raw_data(path)

    cleaned_df = clean_data(raw_df)

    traffic_series = cleaned_df["traffic_flow"].values

    X, y = build_sequences(
        traffic_series,
        window_size=12
    )

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split_sequence(
        X,
        y,
        train_ratio=0.8
    )

    model = BaselineModel()

    model.train_model(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    model.save_model(
        "../models/saved_models/rf_model.pkl"
    )

    print("Baseline RF trained successfully.")
    print("Prediction samples:", predictions[:10])

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions
    }


if __name__ == "__main__":
    train_baseline_model()