import pandas as pd

from evaluation.metrics import calculate_metrics


def compare_models(results_dict):
    """
    Compare multiple models

    results_dict format:

    {
        "LSTM": (y_true, y_pred),
        "GRU": (y_true, y_pred),
        "RF": (y_true, y_pred)
    }
    """

    comparison_results = []

    for model_name, values in results_dict.items():

        y_true, y_pred = values

        metrics = calculate_metrics(
            y_true,
            y_pred
        )

        comparison_results.append({
            "Model": model_name,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "MAPE": metrics["MAPE"]
        })

    df = pd.DataFrame(comparison_results)

    df = df.sort_values(
        by="RMSE",
        ascending=True
    )

    return df