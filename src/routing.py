
import requests


class OSRMRouter:
    """Client for the local OSRM routing server."""

    def __init__(self, base_url="http://localhost:5000", timeout=10):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get_road_distance(
        self,
        pickup_lat,
        pickup_lon,
        dropoff_lat,
        dropoff_lon
    ):
        """
        Calculate road distance and duration using OSRM.

        Returns:
            (distance_km, duration_min, status)
        """

        coordinates = (
            f"{pickup_lon},{pickup_lat};"
            f"{dropoff_lon},{dropoff_lat}"
        )

        url = f"{self.base_url}/route/v1/driving/{coordinates}"

        try:
            response = requests.get(
                url,
                params={"overview": "false"},
                timeout=self.timeout
            )

            
            data = response.json()
            if data.get("code") == "NoRoute":
                return None ,None ,"unroutable"

            if data.get("code") != "Ok" or not data.get("routes"):
                return None, None, "retryable_error"

            response.raise_for_status()

            route = data["routes"][0]

            distance_km = route["distance"] / 1000
            duration_min = route["duration"] / 60

            return distance_km, duration_min, "success"

        except requests.exceptions.Timeout:
            return None, None, "retryable_error"

        except requests.exceptions.ConnectionError:
            return None, None, "retryable_error"

        except requests.exceptions.RequestException:
            return None, None, "retryable_error"

        except (KeyError, ValueError, TypeError):
            return None, None, "retryable_error"
