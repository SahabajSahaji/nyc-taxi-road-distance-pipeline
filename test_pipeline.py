from src.pipeline import TaxiFarePipeline

pipeline = TaxiFarePipeline()

result = pipeline.predict_fare(
    pickup_latitude=40.7580,
    pickup_longitude=-73.9855,
    dropoff_latitude=40.7829,
    dropoff_longitude=-73.9654,
    pickup_datetime="2015-01-15 12:00:00 UTC",
    passenger_count=1
)

print("Taxi Fare Prediction Results")
print("-" * 35)

for key, value in result.items():
    print(f"{key}: {value}")