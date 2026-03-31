import math

from monitor import HypothesisTestingMonitor
from tests.fixtures import NormalIncrementAdapter


def test_e_process_seed_is_1():
    """
	Test E-process 
    E_0 = 1
    """
    adapter = NormalIncrementAdapter(mean=-1.0, sigma=0.0, initial_value=10.0)
    monitor = HypothesisTestingMonitor(adapter=adapter, delta=0.1)

    assert monitor.E(0) == 1
    assert monitor.E_cache == [1]


def test_plugin_bet_matches_formula():
    """
    Bet satisfies: 
      0 <= beta_k <= 1/B
      beta_k = min(1/B, max(0, S_k/(V_k + B^2)))
    """
    for mean in -1.0, 1.0:
        adapter = NormalIncrementAdapter(mean=mean, sigma=0.0, initial_value=10.0, max_steps=11)
        monitor = HypothesisTestingMonitor(adapter=adapter, delta=0.1)

        results = list(monitor.run()) # Run monitor
        beta = [monitor.beta(i) for i in range(11)]

        assert len(monitor.Delta) == len(beta) == len(results)

        for k in range(len(beta)):
            assert math.isclose(monitor.S(k), k * mean)
            assert math.isclose(monitor.V(k), k * mean**2)
            assert 0.0 <= beta[k] <= 1.0 # 1/B=1
            expected_beta = min(1, max(0, k * mean / ((k * mean**2) + 1)))
            assert math.isclose(beta[k], expected_beta)


def test_e_process_matches_product_formula():
    """
    E_n=prod_{k=0}^{n-1} (1+ beta_k * Delta_k)
    """
    adapter = NormalIncrementAdapter(mean=1.0, sigma=0.0, initial_value=5.0, max_steps=5)
    monitor = HypothesisTestingMonitor(adapter=adapter, delta=0.1)

    results = list(monitor.run()) # Run monitor

    expected = 1.0
    for k in range(len(results)):
        expected *= 1.0 + monitor.beta(k) * monitor.Delta[k]

    assert math.isclose(monitor.E(len(results)), expected)


def test_monitor_refutes_when_e_value_crosses_threshold():
    """
    Monitor refutes as soon as E_n >= 1/delta
    """
    adapter = NormalIncrementAdapter(mean=1.0, sigma=0.0, initial_value=50.0, max_steps=50)
    delta = 0.2
    monitor = HypothesisTestingMonitor(adapter=adapter, delta=delta)

    results = list(monitor.run())

    assert results, "Expected at least one monitoring step"

    first_f_index = next(i for i, (verdict, _) in enumerate(results) if verdict == "F")

    # Once rejected, the monitor should stay rejected.
    for later_verdict, later_info in results[first_f_index:]:
        assert later_verdict == "F"
        assert later_info["e_value"] >= 1.0 / delta # e_value never decreases because drift is always 1

    # e_value should be less than 1/delta
    for earlier_verdict, earlier_info in results[:first_f_index]:
        assert earlier_verdict == "?"
        assert earlier_info["e_value"] < 1.0 / delta
