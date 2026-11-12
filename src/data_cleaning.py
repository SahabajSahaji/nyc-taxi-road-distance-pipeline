
import pandas as pd


def clean_taxi_data(df):
    """
    Clean NYC taxi dataset by removing records with
    missing or invalid coordinates.

    Returns:
        cleaned_df: DataFrame containing valid records.
        quality_report: Summary of removed records.
    """

    required_columns = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "fare_amount"
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    cleaned_df = df.copy()
    original_rows = len(cleaned_df)

    # Remove missing coordinates
    cleaned_df = cleaned_df.dropna(
        subset=required_columns[:4]
    )

    # Validate geographical coordinates
    valid_coordinates = (
        cleaned_df["pickup_longitude"].between(-180, 180)
        & cleaned_df["dropoff_longitude"].between(-180, 180)
        & cleaned_df["pickup_latitude"].between(-90, 90)
        & cleaned_df["dropoff_latitude"].between(-90, 90)
    )

    cleaned_df = cleaned_df.loc[valid_coordinates].copy()

    quality_report = {
        "original_rows": original_rows,
        "cleaned_rows": len(cleaned_df),
        "removed_rows": original_rows - len(cleaned_df)
    }

    return cleaned_df, quality_report
