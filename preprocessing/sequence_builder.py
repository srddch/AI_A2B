import numpy as np


def build_sequences(series, window_size=12):
    """
    Build sliding window sequences from a 1D traffic flow series.

    Example:
    [1,2,3,4] with window_size=3
    X = [1,2,3]
    y = 4
    """

    series = np.array(series, dtype=float)

    X = []
    y = []

    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])

    return np.array(X), np.array(y)


def train_test_split_sequence(X, y, train_ratio=0.8):
    """
    Time series split.
    Do not shuffle.
    """

    split_index = int(len(X) * train_ratio)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test