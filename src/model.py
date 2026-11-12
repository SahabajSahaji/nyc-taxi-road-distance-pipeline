
import joblib
import pandas as pd
from pathlib import Path


class TaxiFarePredictor:
    """Load and use the trained NYC taxi fare model."""

    def __init__(self, model_path=None):

        if model_path is None:
            model_path = (
                Path(__file__).resolve().parent.parent
                / "models"
                / "xgboost_taxi_fare_model.joblib"
            )

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

        self.bundle = joblib.load(self.model_path)

        self.model = self.bundle["model"]
        self.imputer = self.bundle["imputer"]
        self.feature_columns = self.bundle["feature_columns"]

    def predict(self, features):
        """
        Predict taxi fare from engineered features.

        features must contain the model's expected feature columns.
        """

        if isinstance(features, dict):
            features = pd.DataFrame([features])

        missing_columns = [
            col for col in self.feature_columns
            if col not in features.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing model features: {missing_columns}"
            )

        features = features[self.feature_columns]

        features_imputed = self.imputer.transform(features)

        predictions = self.model.predict(features_imputed)

        return predictions
