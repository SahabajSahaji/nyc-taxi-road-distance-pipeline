
from unittest.mock import patch
from src.pipeline import TaxiFarePipeline


@patch("src.pipeline.TaxiFarePredictor")
@patch("src.pipeline.OSRMRouter")
def test_successful_fare_prediction(mock_router, mock_predictor):
    # Mock OSRM response
    mock_router.return_value.get_road_distance.return_value = (
        4.79, 8.34, "success"
    )

    # Mock model prediction
    mock_predictor.return_value.predict.return_value = [13.30]

    pipeline = TaxiFarePipeline()

    result = pipeline.predict_fare(
        pickup_latitude=40.7580,
        pickup_longitude=-73.9855,
        dropoff_latitude=40.7829,
        dropoff_longitude=-73.9654,
        pickup_datetime="2015-01-15 12:00:00 UTC",
        passenger_count=1
    )

    assert result["predicted_fare"] == 13.30
    assert result["road_distance_km"] == 4.79
    assert result["road_duration_min"] == 8.34
    assert result["routing_status"] == "success"


@patch("src.pipeline.TaxiFarePredictor")
@patch("src.pipeline.OSRMRouter")
def test_unroutable_trip(mock_router, mock_predictor):
    mock_router.return_value.get_road_distance.return_value = (
        None, None, "unroutable"
    )

    mock_predictor.return_value.predict.return_value = [13.76]

    pipeline = TaxiFarePipeline()

    result = pipeline.predict_fare(
        pickup_latitude=40.73832,
        pickup_longitude=-74.02597,
        dropoff_latitude=40.72692,
        dropoff_longitude=-74.01399,
        pickup_datetime="2015-01-15 12:00:00 UTC",
        passenger_count=1
    )

    assert result["predicted_fare"] == 13.76
    assert result["road_distance_km"] is None
    assert result["road_duration_min"] is None
    assert result["routing_status"] == "unroutable"

import pytest


def test_invalid_pickup_latitude():
    pipeline = TaxiFarePipeline()

    with pytest.raises(ValueError, match="Invalid pickup latitude"):
        pipeline.predict_fare(
            pickup_latitude=100,
            pickup_longitude=-73.9855,
            dropoff_latitude=40.7829,
            dropoff_longitude=-73.9654,
            pickup_datetime="2015-01-15 12:00:00 UTC",
            passenger_count=1
        )

def test_invalid_passenger_count():
    pipeline = TaxiFarePipeline()

    with pytest.raises(
        ValueError,
        match="Passenger count must be between 1 and 6"
    ):
        pipeline.predict_fare(
            pickup_latitude=40.7580,
            pickup_longitude=-73.9855,
            dropoff_latitude=40.7829,
            dropoff_longitude=-73.9654,
            pickup_datetime="2015-01-15 12:00:00 UTC",
            passenger_count=0
        )