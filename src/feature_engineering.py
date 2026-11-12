
import numpy as np
import pandas as pd

from src.distance import haversine_distance


def create_features(df):
    """
    Generate geographical and temporal features
    from NYC taxi trip data.
    """

    df = df.copy()

    # Haversine distance
    df["haversine_distance_km"] = haversine_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"]
    )

    # Convert pickup datetime to New York local time
    pickup_datetime = pd.to_datetime(
        df["pickup_datetime"],
        utc=True
    )

    pickup_datetime_ny = pickup_datetime.dt.tz_convert(
        "America/New_York"
    )

    # Temporal features
    df["pickup_hour"] = pickup_datetime_ny.dt.hour
    df["pickup_day"] = pickup_datetime_ny.dt.day
    df["pickup_day_of_week"] = pickup_datetime_ny.dt.dayofweek
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

    # Road distance features, if routing columns exist
    if "road_distance_km" in df.columns:

        df["road_distance_ratio"] = (
            df["road_distance_km"]
            / df["haversine_distance_km"].replace(0, np.nan)
        )

        df["distance_difference_km"] = (
            df["road_distance_km"]
            - df["haversine_distance_km"]
        )

    if {"road_distance_km", "road_duration_min"}.issubset(df.columns):

        df["estimated_speed_kmh"] = (
            df["road_distance_km"]
            / (df["road_duration_min"] / 60)
        )

        df["estimated_speed_kmh"] = (
            df["estimated_speed_kmh"].replace(
                [np.inf, -np.inf], np.nan
            )
        )

    return df
