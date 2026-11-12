
import numpy as np


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate straight-line distance between two geographical points.

    Parameters:
        lat1: Pickup latitude
        lon1: Pickup longitude
        lat2: Dropoff latitude
        lon2: Dropoff longitude

    Returns:
        Distance in kilometers.
    """

    earth_radius_km = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return earth_radius_km * c
