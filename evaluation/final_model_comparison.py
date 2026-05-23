import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestRegressor


TRAIN_FILE = "data/processed/train.csv"
TEST_FILE = "data/processed/test.csv"

LSTM_MODEL = "models/saved_models/lstm_model.keras"
GRU_MODEL = "models/saved_models/gru_model.keras"

SCALER_FILE = "models/saved_models/scaler.pkl"
ENCODER_FILE = "models/saved_models/onehot_encoder.pkl"

OUTPUT_DIR = "docs/figures"
OUTPUT_FIG = "final_one_day_model_comparison.png"



SEQ_LEN = 4
SITE_ID = 3002


POINTS_PER_DAY = 96



def make_time_labels():
    labels = []

    for i in range(POINTS_PER_DAY):
        hour = (i * 15) // 60
        minute = (i * 15) % 60
        labels.append(f"{hour:02d}:{minute:02d}")

    return labels


def smooth(data, window=3):
    return pd.Series(data).rolling(
        window=window,
        center=True,
        min_periods=1
    ).mean().values


def build_rf_dataset(df):
    X = []
    y = []

    flows = df["flow_target"].values

    for i in range(len(flows) - SEQ_LEN):
        X.append(flows[i:i + SEQ_LEN])
        y.append(flows[i + SEQ_LEN])

    return np.array(X), np.array(y)


def build_deep_input(seq, site_id, scaler, encoder):
    site_encoded = encoder.transform([[site_id]]).flatten()

    seq_scaled = scaler.transform(
        np.array(seq).reshape(-1, 1)
    ).flatten()

    X_seq = []

    for t in range(SEQ_LEN):
        step = np.concatenate([
            site_encoded,
            [seq_scaled[t]]
        ])
        X_seq.append(step)

    return np.array([X_seq])



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    train_df = train_df[train_df["site_number"] == SITE_ID].reset_index(drop=True)
    test_df = test_df[test_df["site_number"] == SITE_ID].reset_index(drop=True)

    if len(test_df) < POINTS_PER_DAY + SEQ_LEN:
        raise ValueError("Test data is not enough for one full day.")

    scaler = joblib.load(SCALER_FILE)
    encoder = joblib.load(ENCODER_FILE)

    lstm_model = load_model(LSTM_MODEL)
    gru_model = load_model(GRU_MODEL)

    X_rf, y_rf = build_rf_dataset(train_df)

    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_rf, y_rf)


    day_df = test_df.iloc[:POINTS_PER_DAY + SEQ_LEN].reset_index(drop=True)
    day_flows = day_df["flow_target"].values

    true_data = []
    lstm_predictions = []
    gru_predictions = []
    rf_predictions = []

    for i in range(POINTS_PER_DAY):
        seq = day_flows[i:i + SEQ_LEN]
        true_value = day_flows[i + SEQ_LEN]

        X_deep = build_deep_input(
            seq=seq,
            site_id=SITE_ID,
            scaler=scaler,
            encoder=encoder
        )

        lstm_pred = lstm_model.predict(X_deep, verbose=0)[0][0]
        gru_pred = gru_model.predict(X_deep, verbose=0)[0][0]
        rf_pred = rf_model.predict([seq])[0]

        true_data.append(true_value)
        lstm_predictions.append(lstm_pred)
        gru_predictions.append(gru_pred)
        rf_predictions.append(rf_pred)


    lstm_predictions = smooth(lstm_predictions, window=3)
    gru_predictions = smooth(gru_predictions, window=3)
    rf_predictions = smooth(rf_predictions, window=3)


    time_labels = make_time_labels()
    x = np.arange(POINTS_PER_DAY)

    plt.figure(figsize=(14, 6))

    plt.plot(x, true_data, label="True Data", linewidth=1.8)
    plt.plot(x, lstm_predictions, label="LSTM", linewidth=1.8)
    plt.plot(x, gru_predictions, label="GRU", linewidth=1.8)
    plt.plot(x, rf_predictions, label="Random Forest", linewidth=1.8)

    tick_positions = list(range(0, POINTS_PER_DAY, 12))
    tick_labels = [time_labels[i] for i in tick_positions]

    plt.xticks(tick_positions, tick_labels, rotation=45)

    plt.title("One-Day Traffic Flow Prediction Comparison")
    plt.xlabel("Time of Day")
    plt.ylabel("Traffic Flow")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FIG)
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Saved figure to: {save_path}")


if __name__ == "__main__":
    main()