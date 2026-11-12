
import numpy as np

from src.distance import haversine_distance


def test_known_distance():
    """Test distance between Times Square and Central Park."""

    distance = haversine_distance(
        40.7580, -73.9855,
        40.7829, -73.9654
    )

    assert np.isclose(distance, 3.245, atol=0.001)


def test_same_coordinates():
    """Distance between identical coordinates should be zero."""

    distance = haversine_distance(
        40.7580, -73.9855,
        40.7580, -73.9855
    )

    assert distance == 0


def test_distance_symmetry():
    """Distance should be identical in both directions."""

    distance_1 = haversine_distance(
        40.7580, -73.9855,
        40.7829, -73.9654
    )

    distance_2 = haversine_distance(
        40.7829, -73.9654,
        40.7580, -73.9855
    )

    assert np.isclose(distance_1, distance_2)


def test_vectorized_distance():
    """Test calculation with multiple coordinate pairs."""

    distances = haversine_distance(
        np.array([40.7580, 40.7580]),
        np.array([-73.9855, -73.9855]),
        np.array([40.7829, 40.7580]),
        np.array([-73.9654, -73.9855])
    )

    assert len(distances) == 2
    assert np.isclose(distances[0], 3.245, atol=0.001)
    assert distances[1] == 0
