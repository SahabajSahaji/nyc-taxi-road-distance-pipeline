
import pandas as pd

from src.distance import haversine_distance
from src.data_cleaning import clean_taxi_data
from src.routing import OSRMRouter
from src.feature_engineering import create_features
from src.model import TaxiFarePredictor


# 1. Test data cleaning
df = pd.DataFrame({
    "pickup_longitude": [-73.9855],
    "pickup_latitude": [40.7580],
    "dropoff_longitude": [-73.9654],
    "dropoff_latitude": [40.7829],
    "fare_amount": [12.0],
    "pickup_datetime": ["2015-01-15 12:00:00 UTC"]
})

cleaned_df, report = clean_taxi_data(df)
assert len(cleaned_df) == 1
print("1. Data cleaning: PASS")


# 2. Test Haversine distance
distance = haversine_distance(
    40.7580, -73.9855,
    40.7829, -73.9654
)
assert round(distance, 3) == 3.245
print("2. Haversine distance: PASS")


# 3. Test OSRM routing
router = OSRMRouter()
road_distance, duration, status = router.get_road_distance(
    40.7580, -73.9855,
    40.7829, -73.9654
)

assert status == "success"
assert road_distance > 0
print("3. OSRM routing: PASS")


# 4. Test feature engineering
df["road_distance_km"] = road_distance
df["road_duration_min"] = duration

featured_df = create_features(df)

assert "pickup_hour" in featured_df.columns
assert "road_distance_ratio" in featured_df.columns
print("4. Feature engineering: PASS")


# 5. Test model loading
predictor = TaxiFarePredictor()
assert len(predictor.feature_columns) == 27
print("5. Model loading: PASS")

print("\nAll integration checks passed!")
