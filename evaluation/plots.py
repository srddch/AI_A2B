import matplotlib.pyplot as plt


def plot_predictions(
    y_true,
    y_pred,
    model_name,
    save_path
):
    """
    Plot Actual vs Predicted
    """

    plt.figure(figsize=(12, 6))

    plt.plot(
        y_true,
        label="Actual"
    )

    plt.plot(
        y_pred,
        label="Predicted"
    )

    plt.title(
        f"{model_name} Prediction vs Actual"
    )

    plt.xlabel("Time")

    plt.ylabel("Traffic Flow")

    plt.legend()

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()