import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from hypergraphpercol.geometry import minimum_enclosing_ball


def test_minimum_enclosing_ball_triangle():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    center, radius_sq = minimum_enclosing_ball(points)
    np.testing.assert_allclose(center, np.array([0.5, 0.5]))
    assert radius_sq == pytest.approx(0.5)


def test_minimum_enclosing_ball_square():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    center, radius_sq = minimum_enclosing_ball(points)
    np.testing.assert_allclose(center, np.array([0.5, 0.5]))
    assert radius_sq == pytest.approx(0.5)


def test_minimum_enclosing_ball_all_points_inside():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(10, 3))
    center, radius_sq = minimum_enclosing_ball(points)
    squared_distances = np.sum((points - center) ** 2, axis=1)
    assert np.all(squared_distances <= radius_sq + 1e-9)
