import numpy as np

from preprocessing.sequence_builder import (
    build_sequences,
    train_test_split_sequence
)


def test_sequence_builder():

    data = np.array([
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10
    ])

    X, y = build_sequences(
        data,
        window_size=3
    )

    assert X.shape[0] == 7
    assert X.shape[1] == 3
    assert len(y) == 7


def test_train_test_split():

    X = np.random.rand(100, 12)
    y = np.random.rand(100)

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split_sequence(X, y)

    assert len(X_train) == 80
    assert len(X_test) == 20