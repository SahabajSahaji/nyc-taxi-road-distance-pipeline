
import numpy as np
import pandas as pd

from src.feature_engineering import create_features


def sample_trip():
    return pd.DataFrame({
        "passenger_count": [1],
        "pickup_latitude": [40.7580],
        "pickup_longitude": [-73.9855],
        "dropoff_latitude": [40.7829],
        "dropoff_longitude": [-73.9654],
        "pickup_datetime": ["2015-01-15 12:00:00 UTC"],
        "road_distance_km": [4.7886],
        "road_duration_min": [8.3367]
    })


def test_haversine_feature():
    result = create_features(sample_trip())

    assert np.isclose(
        result["haversine_distance_km"].iloc[0],
        3.245,
        atol=0.001
    )


def test_temporal_features():
    result = create_features(sample_trip())

    assert result["pickup_hour"].iloc[0] == 7
    assert result["pickup_day"].iloc[0] == 15
    assert result["pickup_day_of_week"].iloc[0] == 3
    assert result["pickup_month"].iloc[0] == 1
    assert result["is_weekend"].iloc[0] == 0


def test_location_clusters():
    result = create_features(sample_trip())

    assert result["pickup_location_lat"].iloc[0] == 40.758
    assert result["pickup_location_lon"].iloc[0] == -73.986
    assert result["dropoff_location_lat"].iloc[0] == 40.783
    assert result["dropoff_location_lon"].iloc[0] == -73.965


def test_road_features():
    result = create_features(sample_trip())

    assert np.isclose(
        result["road_distance_ratio"].iloc[0],
        1.4756,
        atol=0.001
    )

    assert result["osrm_route_missing"].iloc[0] == 0
    assert result["osrm_zero_distance_anomaly"].iloc[0] == 0


def test_zero_distance_anomaly():
    df = sample_trip()

    # Different coordinates but an unreliable zero road distance
    df["road_distance_km"] = 0
    df["road_duration_min"] = 0

    result = create_features(df)

    assert result["osrm_zero_distance_anomaly"].iloc[0] == 1
    assert result["osrm_route_missing"].iloc[0] == 1
    assert pd.isna(result["road_distance_km"].iloc[0])
    assert pd.isna(result["road_duration_min"].iloc[0])


def test_missing_route():
    df = sample_trip()

    df["road_distance_km"] = np.nan
    df["road_duration_min"] = np.nan

    result = create_features(df)

    assert result["osrm_route_missing"].iloc[0] == 1
    assert result["road_distance_ratio_missing"].iloc[0] == 1
    assert result["estimated_speed_missing"].iloc[0] == 1
