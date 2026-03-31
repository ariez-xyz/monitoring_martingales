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
