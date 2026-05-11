import joblib
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


class BaselineModel:
    """
    Random Forest baseline model
    """

    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state
        )

    def train_model(self, X_train, y_train):
        """
        Train Random Forest model
        """
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, X):
        """
        Predict traffic flow
        """
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """
        Evaluate MAE
        """
        y_pred = self.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)

        return {
            "mae": float(mae)
        }

    def save_model(self, path):
        """
        Save model
        """
        joblib.dump(self.model, path)

    def load_model(self, path):
        """
        Load model
        """
        self.model = joblib.load(path)
        return self.model