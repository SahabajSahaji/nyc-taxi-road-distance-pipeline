
import pandas as pd
import pytest

from src.data_cleaning import clean_taxi_data


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "pickup_longitude": [-73.9855, None, 200],
        "pickup_latitude": [40.7580, 40.7000, 40.7000],
        "dropoff_longitude": [-73.9654, -73.9600, -73.9500],
        "dropoff_latitude": [40.7829, 40.7700, 40.7800],
        "fare_amount": [12.0, 15.0, 20.0]
    })


def test_clean_valid_records(sample_data):
    cleaned, report = clean_taxi_data(sample_data)

    assert len(cleaned) == 1
    assert report["original_rows"] == 3
    assert report["cleaned_rows"] == 1
    assert report["removed_rows"] == 2


def test_original_dataframe_unchanged(sample_data):
    original = sample_data.copy()

    clean_taxi_data(sample_data)

    pd.testing.assert_frame_equal(sample_data, original)


def test_missing_required_column():
    df = pd.DataFrame({
        "pickup_longitude": [-73.9855]
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        clean_taxi_data(df)


def test_empty_dataframe():
    df = pd.DataFrame(columns=[
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "fare_amount"
    ])

    cleaned, report = clean_taxi_data(df)

    assert len(cleaned) == 0
    assert report["removed_rows"] == 0
