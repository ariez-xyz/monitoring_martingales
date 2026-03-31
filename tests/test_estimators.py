from monitor.estimators import HistoryEstimator
from monitor.weighting import UniformWeights
from tests.fixtures import NormalIncrementAdapter

def test_history_estimator_delay():
    """HistoryEstimator should trail the stream by the window radius."""
    adapter = NormalIncrementAdapter(mean=-1.0, sigma=0.0, initial_value=10.0, max_steps=100)
    estimator = HistoryEstimator(weighting=UniformWeights(radius=2), delta=0.01)

    for _ in range(4):
        adapter.step()

    verdict, lower, upper, info = estimator(adapter)
    assert verdict == "?"
    assert lower == float("-inf")
    assert upper == float("inf")
    assert info["delay"] == 2
    assert info["reason"] == "insufficient history"

    adapter.step()
    verdict, lower, upper, info = estimator(adapter)

    assert info["delay"] == 2
    assert info["current_drift_index"] == 4
    assert info["target"] == 2
    assert lower <= upper

def test_history_estimator_averages_sliding_window():
    """HistoryEstimator should return the average of the sliding window."""
    adapter = NormalIncrementAdapter(mean=-1.0, sigma=0.0, initial_value=10.0, max_steps=100)
    adapter.drift_history = [10,9,8,7,6,5,4,3,2,0]

    estimator = HistoryEstimator(weighting=UniformWeights(radius=4), delta=0.01)
    verdict, lower, upper, info = estimator(adapter)

    assert info["current_drift_index"] == 9
    assert info["target"] == 5
    assert info["delay"] == 4
    assert abs(info["weighted_mean"] - 44/9) < 1e-6
    assert lower <= 44/9 <= upper
    assert verdict in {"T", "F", "?"}
