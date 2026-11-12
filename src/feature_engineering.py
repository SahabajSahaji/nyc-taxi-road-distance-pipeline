
import numpy as np
import pandas as pd

from src.distance import haversine_distance


def create_features(df):
    """
    Generate the 27 features used by the trained XGBoost model.

    Required input columns:
        passenger_count
        pickup_longitude
        pickup_latitude
        dropoff_longitude
        dropoff_latitude
        pickup_datetime
        road_distance_km
        road_duration_min
    """

    df = df.copy()

    # 1. Haversine distance
    df["haversine_distance_km"] = haversine_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"]
    )

    # 2. Zero-distance indicator
    df["is_zero_distance"] = (
        df["haversine_distance_km"] == 0
    ).astype(int)

    # 3. Detect unreliable OSRM zero-distance results
    anomaly_mask = (
        (df["road_distance_km"] == 0)
        & (df["haversine_distance_km"] > 0)
    )

    df["osrm_zero_distance_anomaly"] = (
        anomaly_mask.astype(int)
    )

    # 4. Remove unreliable routing values
    routing_features = [
        "road_distance_km",
        "road_duration_min"
    ]

    df.loc[anomaly_mask, routing_features] = np.nan

    # 5. Road-distance features
    df["road_distance_ratio"] = (
        df["road_distance_km"]
        / df["haversine_distance_km"].replace(0, np.nan)
    )

    df["distance_difference_km"] = (
        df["road_distance_km"]
        - df["haversine_distance_km"]
    )

    df["estimated_speed_kmh"] = (
        df["road_distance_km"]
        / (df["road_duration_min"] / 60)
    )

    df["estimated_speed_kmh"] = (
        df["estimated_speed_kmh"].replace(
            [np.inf, -np.inf], np.nan
        )
    )

    # 6. Missing-value indicators
    df["osrm_route_missing"] = (
        df["road_distance_km"].isna().astype(int)
    )

    df["road_distance_ratio_missing"] = (
        df["road_distance_ratio"].isna().astype(int)
    )

    df["estimated_speed_missing"] = (
        df["estimated_speed_kmh"].isna().astype(int)
    )

    # 7. Convert pickup datetime to New York local time
    pickup_datetime_ny = (
        pd.to_datetime(
            df["pickup_datetime"],
            utc=True
        ).dt.tz_convert("America/New_York")
    )

    df["pickup_hour"] = pickup_datetime_ny.dt.hour
    df["pickup_day"] = pickup_datetime_ny.dt.day
    df["pickup_day_of_week"] = (
        pickup_datetime_ny.dt.dayofweek
    )
    df["pickup_month"] = pickup_datetime_ny.dt.month
    df["pickup_year"] = pickup_datetime_ny.dt.year

    df["is_weekend"] = (
        pickup_datetime_ny.dt.dayofweek >= 5
    ).astype(int)

    df["is_rush_hour"] = (
        (pickup_datetime_ny.dt.dayofweek < 5)
        & (
            pickup_datetime_ny.dt.hour.between(7, 9)
            | pickup_datetime_ny.dt.hour.between(16, 19)
        )
    ).astype(int)

    # 8. Geographical location clusters
    df["pickup_location_lat"] = (
        df["pickup_latitude"].round(3)
    )

    df["pickup_location_lon"] = (
        df["pickup_longitude"].round(3)
    )

    df["dropoff_location_lat"] = (
        df["dropoff_latitude"].round(3)
    )

    df["dropoff_location_lon"] = (
        df["dropoff_longitude"].round(3)
    )

    return df
