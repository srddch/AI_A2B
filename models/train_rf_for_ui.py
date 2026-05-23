import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from models.rf_feature_utils import build_rf_feature


TRAIN_FILE = os.path.join(BASE_DIR, "data", "processed", "train.csv")
VAL_FILE = os.path.join(BASE_DIR, "data", "processed", "val.csv")

ENCODER_FILE = os.path.join(BASE_DIR, "models", "saved_models", "onehot_encoder.pkl")
SAVE_MODEL_PATH = os.path.join(BASE_DIR, "models", "saved_models", "rf_model.pkl")

SEQ_LEN = 4


def build_dataset(df, encoder):
    X = []
    y = []

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="mixed"
    )

    for site, group in df.groupby("site_number"):
        group = group.sort_values("timestamp").reset_index(drop=True)

        flows = group["flow_target"].values
        timestamps = group["timestamp"].values

        if len(flows) <= SEQ_LEN:
            continue

        for i in range(len(flows) - SEQ_LEN):
            past_4_flows = flows[i:i + SEQ_LEN]
            target = flows[i + SEQ_LEN]
            target_time = timestamps[i + SEQ_LEN]

            feature = build_rf_feature(
                site_id=site,
                past_4_flows=past_4_flows,
                timestamp=target_time,
                encoder=encoder
            )

            X.append(feature)
            y.append(target)

    return np.array(X), np.array(y)


def main():
    print("Loading data...")

    train_df = pd.read_csv(TRAIN_FILE)

    if os.path.exists(VAL_FILE):
        val_df = pd.read_csv(VAL_FILE)
        df = pd.concat([train_df, val_df], ignore_index=True)
        print("Using train + validation data.")
    else:
        df = train_df
        print("Using train data only.")

    encoder = joblib.load(ENCODER_FILE)

    print("Building improved Random Forest features...")
    X, y = build_dataset(df, encoder)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("RF feature number:", X.shape[1])

    model = RandomForestRegressor(
        n_estimators=1000,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.7,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

    print("Training improved Random Forest...")
    model.fit(X, y)

    os.makedirs(os.path.dirname(SAVE_MODEL_PATH), exist_ok=True)
    joblib.dump(model, SAVE_MODEL_PATH, compress=3)

    print("Improved Random Forest trained successfully.")
    print(f"Saved to: {SAVE_MODEL_PATH}")


if __name__ == "__main__":
    main()