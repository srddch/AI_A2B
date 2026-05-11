import numpy as np

from models.baseline_model import BaselineModel


def test_baseline_model():

    X_train = np.random.rand(100, 12)
    y_train = np.random.rand(100)

    X_test = np.random.rand(20, 12)

    model = BaselineModel()

    model.train_model(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    assert len(predictions) == 20