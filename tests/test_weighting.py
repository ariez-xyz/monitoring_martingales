import math
import torch

from monitor.weighting import UniformWeights, OptimalTemporalWeights
from tests.fixtures import NormalIncrementAdapter

def test_uniform_weights_match_formula():
    """Uniform weights should be 1/m in the sliding window and 0 everywhere else."""
    drift_history = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    weighting = UniformWeights(radius=1)

    weights = weighting(drift_history, target=2)

    assert weights is not None
    expected = torch.tensor([0.0, 1 / 3, 1 / 3, 1 / 3, 0.0])
    assert torch.allclose(weights, expected)
    assert abs(float(weights.sum()) - 1.0) < 1e-6

    assert weighting(drift_history, target=0) is None
    assert weighting(drift_history, target=4) is None

def test_optimal_temporal_weights_radius_correct():
    adapter = NormalIncrementAdapter() # rho = 1

    weighting = OptimalTemporalWeights(adapter, 0.5)
    # optimal_window_length = 1.404845239428869 
    assert weighting.get_radius() == 0

    weighting = OptimalTemporalWeights(adapter, 0.01)
    # optimal_window_length = 2.1964567435801996
    assert weighting.get_radius() == 1

    weighting = OptimalTemporalWeights(adapter, 0.000001)
    # optimal_window_length = 3.072928182547332
    assert weighting.get_radius() == 1


def test_uniform_weights_se_and_de_match_formula_for_fixture():
    """UniformWeights should implement the centered-window formulas exactly."""
    adapter = NormalIncrementAdapter(mean=-1.0, sigma=0.0, initial_value=10.0)
    weighting = UniformWeights(radius=1)

    # For this fixture, B = 1 and rho = 1.
    # radius = 1 => m = 3
    expected_se = math.sqrt(2 * math.log(2 / 0.01) / 3)
    expected_de = 2 * (3**2 - 1) / (4 * 3)

    assert math.isclose(weighting.SE(adapter, 0.01), expected_se)
    assert math.isclose(weighting.DE(adapter), expected_de)


def test_optimal_temporal_weights_se_and_de_delegate_to_uniform_weights():
    """OptimalTemporalWeights should match the uniform window it chooses."""
    adapter = NormalIncrementAdapter(mean=-1.0, sigma=0.0, initial_value=10.0)
    delta = 0.01

    optimal = OptimalTemporalWeights(adapter, delta)
    uniform = UniformWeights(radius=optimal.get_radius())

    assert optimal.get_radius() == 1
    assert math.isclose(optimal.SE(adapter, delta), uniform.SE(adapter, delta))
    assert math.isclose(optimal.DE(adapter), uniform.DE(adapter))
