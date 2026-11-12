
import pandas as pd

from src.routing import OSRMRouter
from src.feature_engineering import create_features
from src.model import TaxiFarePredictor


class TaxiFarePipeline:
    """End-to-end NYC taxi fare prediction pipeline."""

    def __init__(self):
        self.router = OSRMRouter()
        self.predictor = TaxiFarePredictor()

    def predict_fare(
        self,
        pickup_latitude,
        pickup_longitude,
        dropoff_latitude,
        dropoff_longitude,
        pickup_datetime,
        passenger_count=1
    ):
        
        if not (-90 <= pickup_latitude <= 90):
            raise ValueError("Invalid pickup latitude")

        if not (-180 <= pickup_longitude <= 180):
            raise ValueError("Invalid pickup longitude")

        if not (-90 <= dropoff_latitude <= 90):
            raise ValueError("Invalid dropoff latitude")

        if not (-180 <= dropoff_longitude <= 180):
            raise ValueError("Invalid dropoff longitude")

        if not (1 <= passenger_count <= 6):
            raise ValueError("Passenger count must be between 1 and 6")
        
        # Get road distance and duration
        road_distance, road_duration, status = (
            self.router.get_road_distance(
                pickup_latitude,
                pickup_longitude,
                dropoff_latitude,
                dropoff_longitude
            )
        )

        # Prepare input data
        trip = pd.DataFrame([{
            "passenger_count": passenger_count,
            "pickup_latitude": pickup_latitude,
            "pickup_longitude": pickup_longitude,
            "dropoff_latitude": dropoff_latitude,
            "dropoff_longitude": dropoff_longitude,
            "pickup_datetime": pickup_datetime,
            "road_distance_km": road_distance,
            "road_duration_min": road_duration
        }])

        # Generate model features
        features = create_features(trip)

        # Select exact training feature order
        features = features[self.predictor.feature_columns]

        # Predict fare
        predicted_fare = self.predictor.predict(features)[0]

        return {
            "predicted_fare": round(float(predicted_fare), 2),
            "road_distance_km": road_distance,
            "road_duration_min": road_duration,
            "routing_status": status
        }
