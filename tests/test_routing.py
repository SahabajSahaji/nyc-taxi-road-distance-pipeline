
from unittest.mock import Mock, patch

from src.routing import OSRMRouter


def test_successful_route():
    router = OSRMRouter()

    mock_response = Mock()
    mock_response.json.return_value = {
        "code": "Ok",
        "routes": [{
            "distance": 4788.6,
            "duration": 500.2
        }]
    }

    with patch("src.routing.requests.get", return_value=mock_response):
        distance, duration, status = router.get_road_distance(
            40.7580, -73.9855,
            40.7829, -73.9654
        )

    assert abs(distance - 4.7886) < 1e-6
    assert round(duration, 4) == 8.3367
    assert status == "success"


def test_unroutable_route():
    router = OSRMRouter()

    mock_response = Mock()
    mock_response.json.return_value = {
        "code": "NoRoute",
        "message": "No route found between points"
    }

    with patch("src.routing.requests.get", return_value=mock_response):
        distance, duration, status = router.get_road_distance(
            40.738315, -74.025967,
            40.726922, -74.013992
        )

    assert distance is None
    assert duration is None
    assert status == "unroutable"


def test_connection_error():
    import requests

    router = OSRMRouter()

    with patch(
        "src.routing.requests.get",
        side_effect=requests.exceptions.ConnectionError
    ):
        distance, duration, status = router.get_road_distance(
            40.7580, -73.9855,
            40.7829, -73.9654
        )

    assert distance is None
    assert duration is None
    assert status == "retryable_error"


def test_timeout_error():
    import requests

    router = OSRMRouter()

    with patch(
        "src.routing.requests.get",
        side_effect=requests.exceptions.Timeout
    ):
        distance, duration, status = router.get_road_distance(
            40.7580, -73.9855,
            40.7829, -73.9654
        )

    assert distance is None
    assert duration is None
    assert status == "retryable_error"
