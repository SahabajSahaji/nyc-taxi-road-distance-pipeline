
import numpy as np
import pandas as pd
import pytest

from src.model import TaxiFarePredictor


@pytest.fixture
def predictor():
    return TaxiFarePredictor()


def test_model_loads(predictor):
    assert predictor.model is not None
    assert predictor.imputer is not None


def test_expected_feature_count(predictor):
    assert len(predictor.feature_columns) == 27
    assert predictor.model.n_features_in_ == 27


def test_prediction_returns_numeric_value(predictor):
    features = pd.DataFrame(
        np.zeros((1, 27)),
        columns=predictor.feature_columns
    )

    prediction = predictor.predict(features)

    assert len(prediction) == 1
    assert np.isfinite(prediction[0])


def test_prediction_with_missing_values(predictor):
    features = pd.DataFrame(
        np.zeros((1, 27)),
        columns=predictor.feature_columns
    )

    features.loc[0, "road_distance_km"] = np.nan
    features.loc[0, "road_duration_min"] = np.nan

    prediction = predictor.predict(features)

    assert len(prediction) == 1
    assert np.isfinite(prediction[0])


def test_missing_feature_raises_error(predictor):
    features = pd.DataFrame(
        np.zeros((1, 26)),
        columns=predictor.feature_columns[:26]
    )

    with pytest.raises(ValueError, match="Missing model features"):
        predictor.predict(features)
