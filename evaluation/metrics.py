import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(y_true, y_pred):

    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    non_zero_mask = y_true != 0

    if np.sum(non_zero_mask) == 0:
        mape = 0.0
    else:
        mape = np.mean(
            np.abs(
                (y_true[non_zero_mask] - y_pred[non_zero_mask])
                / y_true[non_zero_mask]
            )
        ) * 100

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape)
    }