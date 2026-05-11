import os
import pandas as pd

from models.train import train_baseline_model
from evaluation.metrics import calculate_metrics
from evaluation.plots import plot_predictions


def run_evaluation():
    """
    Run evaluation for Random Forest baseline model.
    This file generates:
    1. metrics.csv
    2. predictions.csv
    3. prediction plot
    """

    # =========================
    # 1. Make output folders
    # =========================

    os.makedirs("../data/model_output", exist_ok=True)
    os.makedirs("../docs/figures", exist_ok=True)

    # =========================
    # 2. Train model and get results
    # =========================

    result = train_baseline_model()

    y_test = result["y_test"]
    predictions = result["predictions"]

    # =========================
    # 3. Calculate metrics
    # =========================

    metrics = calculate_metrics(
        y_test,
        predictions
    )

    print("\n===== RANDOM FOREST EVALUATION =====")
    print(metrics)

    # =========================
    # 4. Save metrics.csv
    # =========================

    metrics_df = pd.DataFrame([
        {
            "Model": "Random Forest",
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "MAPE": metrics["MAPE"]
        }
    ])

    metrics_df.to_csv(
        "../data/model_output/metrics.csv",
        index=False
    )

    # =========================
    # 5. Save predictions.csv
    # =========================

    predictions_df = pd.DataFrame({
        "actual": y_test,
        "predicted": predictions
    })

    predictions_df.to_csv(
        "../data/model_output/predictions.csv",
        index=False
    )

    # =========================
    # 6. Plot prediction figure
    # =========================

    plot_predictions(
        y_true=y_test[:300],
        y_pred=predictions[:300],
        model_name="Random Forest",
        save_path="../docs/figures/random_forest_prediction.png"
    )

    print("\nEvaluation finished successfully.")
    print("Saved: data/model_output/metrics.csv")
    print("Saved: data/model_output/predictions.csv")
    print("Saved: docs/figures/random_forest_prediction.png")


if __name__ == "__main__":
    run_evaluation()