"""
Integration tests that run the full monitoring pipeline.
These tests also serve as demonstrations of the monitoring system.

Run with: pytest tests/test_integration.py -v -s
"""
import numpy as np


def test_full_monitor_sablas():
    """
    Full integration test: run the monitor on the Sablas drone system.

    This demonstrates the complete monitoring pipeline:
    1. Create a dynamical system adapter (SablasDrone)
    2. Create an estimator (SamplingEstimator)
    3. Run the monitor loop and collect results
    4. Validate that the monitor produces meaningful safety assessments
    """
    from monitor.adapters import SablasDrone
    from monitor.estimators import SamplingEstimator

    print("\n")
    print("=" * 70)
    print("  FULL MONITOR INTEGRATION TEST (Sablas Drone)")
    print("=" * 70)
    print()
    print("  Running the monitoring loop on the Sablas drone system.")
    print("  The estimator classifies safety at each step:")
    print("    T = Certified safe (upper CI < 0)")
    print("    F = Certified unsafe (lower CI > 0)")
    print("    ? = Inconclusive (CI straddles 0)")
    print()

    drone = SablasDrone(vis_every=0)
    estimator = SamplingEstimator(delta=0.01)

    # Run and collect results
    results = []
    max_steps = 200
    for _ in range(max_steps):
        if drone.done():
            drone.reset()
        result = estimator(drone)
        results.append(result)
        drone.step()

    # Analyze results
    safety_counts = {"T": 0, "F": 0, "?": 0}
    ci_widths = []

    for safety, lower, upper, info in results:
        safety_counts[safety] += 1
        ci_widths.append(upper - lower)

    total = len(results)

    print("-" * 70)
    print("  RESULTS")
    print("-" * 70)
    print()
    print(f"  Total steps:          {total}")
    print(f"  Certified safe (T):   {safety_counts['T']} ({safety_counts['T']/total*100:.1f}%)")
    print(f"  Certified unsafe (F): {safety_counts['F']} ({safety_counts['F']/total*100:.1f}%)")
    print(f"  Inconclusive (?):     {safety_counts['?']} ({safety_counts['?']/total*100:.1f}%)")
    print()
    print(f"  Mean CI width:        {np.mean(ci_widths):.4f}")
    print(f"  Min CI width:         {np.min(ci_widths):.4f}")
    print(f"  Max CI width:         {np.max(ci_widths):.4f}")
    print()
    print("=" * 70)

    # Basic sanity checks
    assert total > 0, "Should have run at least one step"
    assert all(s in ["T", "F", "?"] for s, _, _, _ in results), "Invalid safety labels"
    assert all(l <= u for _, l, u, _ in results), "Lower should be <= upper"


def test_full_monitor_pendulum():
    """
    Full integration test: run the monitor on the inverted pendulum system.

    This demonstrates monitoring with the neural_clbf pendulum adapter,
    which has:
    - Learned CLF (Control Lyapunov Function)
    - Dynamic Lipschitz constant estimation
    - Configurable dt and noise
    """
    from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum
    from monitor.estimators import SamplingEstimator, AnalyticEstimator

    print("\n")
    print("=" * 70)
    print("  FULL MONITOR INTEGRATION TEST (Inverted Pendulum)")
    print("=" * 70)
    print()

    adapter = NeuralCLBFPendulum(dt=0.01, noise_level=0.0)

    print(f"  System: Inverted Pendulum with learned CLF")
    print(f"  dt = {adapter.dt}")
    print(f"  gamma (Lipschitz) = {adapter.get_lipschitz_constant():.4f}")
    print()

    # Test with SamplingEstimator
    print("  --- SamplingEstimator ---")
    estimator = SamplingEstimator(delta=0.01)

    results = []
    for _ in range(100):
        if adapter.done():
            break
        result = estimator(adapter)
        results.append(result)
        adapter.step()

    safety_counts = {"T": 0, "F": 0, "?": 0}
    for safety, _, _, _ in results:
        safety_counts[safety] += 1

    print(f"  Results: T={safety_counts['T']}, F={safety_counts['F']}, ?={safety_counts['?']}")

    # Reset and test with AnalyticEstimator
    adapter.reset()

    print("\n  --- AnalyticEstimator ---")
    estimator = AnalyticEstimator()

    results = []
    for _ in range(100):
        if adapter.done():
            break
        result = estimator(adapter)
        results.append(result)
        adapter.step()

    safety_counts = {"T": 0, "F": 0, "?": 0}
    for safety, _, _, _ in results:
        safety_counts[safety] += 1

    print(f"  Results: T={safety_counts['T']}, F={safety_counts['F']}, ?={safety_counts['?']}")

    print()
    print("=" * 70)


def test_history_estimator_with_lookahead():
    """
    Integration test: HistoryEstimator with delayed estimation for lookahead.

    This demonstrates the full pipeline with:
    1. OptimalTemporalWeights for automatic neighbor selection
    2. Delayed estimation (delay="auto") for symmetric lookahead
    3. Comparison with non-delayed estimation

    The key insight: with delay, we can use "future" observations to improve
    the estimate, at the cost of slightly delayed results.
    """
    from monitor import NeuralCertificateMonitor
    from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum
    from monitor.estimators import HistoryEstimator, AnalyticEstimator
    from monitor.weighting import OptimalTemporalWeights, RecentWeights

    print("\n")
    print("=" * 70)
    print("  HISTORY ESTIMATOR WITH LOOKAHEAD TEST")
    print("=" * 70)
    print()
    print("  Comparing HistoryEstimator configurations:")
    print("  1. RecentWeights + no delay (baseline)")
    print("  2. OptimalTemporalWeights + no delay (optimal m but no lookahead)")
    print("  3. OptimalTemporalWeights + auto delay (full lookahead)")
    print()

    # Create adapter
    adapter = NeuralCLBFPendulum(dt=0.01)

    # Run a trajectory to build history
    for _ in range(200):
        adapter.step()

    print(f"  History length: {len(adapter.state_history)} states")
    print()

    # Get optimal m for reference
    optimal = OptimalTemporalWeights()
    m = optimal.get_optimal_m(adapter)
    print(f"  Optimal m = {m}")
    print()

    # Create different estimator configurations
    configs = [
        ("RecentWeights(k=10), delay=0", HistoryEstimator(weighting=RecentWeights(k=10), delay=0)),
        ("OptimalTemporalWeights, delay=0", HistoryEstimator(weighting=OptimalTemporalWeights(), delay=0)),
        ("OptimalTemporalWeights, delay=auto", HistoryEstimator(weighting=OptimalTemporalWeights(), delay="auto")),
    ]

    # Also get ground truth from analytic estimator for comparison
    analytic = AnalyticEstimator()

    print("-" * 70)
    print(f"  {'Config':<40} {'Safety':<8} {'CI Width':<12} {'Target'}")
    print("-" * 70)

    for name, estimator in configs:
        safety, lower, upper, info = estimator(adapter)
        ci_width = upper - lower
        target = info.get('target', 'N/A')
        print(f"  {name:<40} {safety:<8} {ci_width:<12.4f} {target}")

    # Get analytic estimate for reference
    _, analytic_er, _, _ = analytic(adapter)
    print()
    print(f"  Analytic E[R] (ground truth): {analytic_er:.4f}")
    print()
    print("=" * 70)


def test_monitor_demo():
    """
    Demo: Visual comparison of monitoring with different estimators.

    This can be used as a standalone demo to understand the monitor behavior.
    Shows step-by-step progression with multiple estimator types.
    """
    from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum
    from monitor.estimators import SamplingEstimator, AnalyticEstimator, HistoryEstimator
    from monitor.weighting import RecentWeights, OptimalTemporalWeights

    print("\n")
    print("=" * 70)
    print("  MONITOR DEMO: Comparing Estimators")
    print("=" * 70)
    print()
    print("  Running pendulum for 1000 steps, comparing estimator types:")
    print("  - Sampling:  Monte Carlo estimation of E[R]")
    print("  - Analytic:  Exact E[R] using V(E[Y])")
    print("  - History:   Past trajectory (no lookahead)")
    print("  - Hist+LA:   History with lookahead (delay=auto)")
    print()

    adapter = NeuralCLBFPendulum(dt=0.01)
    k = 10

    sampling = SamplingEstimator(delta=0.01)
    analytic = AnalyticEstimator()
    history = HistoryEstimator(weighting=RecentWeights(k), delta=0.01)
    history_la = HistoryEstimator(weighting=OptimalTemporalWeights(), delta=0.01, delay="auto")

    n_steps = 1000

    # Get optimal m for reference
    optimal_m = OptimalTemporalWeights().compute_optimal_m(adapter)
    warmup_la = optimal_m  # Lookahead version needs more warmup

    print(f"  OptimalTemporalWeights m = {optimal_m}")
    print()

    print("-" * 115)
    print(f"  {'Step':<6} {'V(x)':<10} {'Sampling':<20} {'Analytic':<12} {'History':<20} {'Hist+LA':<20}")
    print(f"  {'':<6} {'':<10} {'[lower, upper]':<20} {'E[R]':<12} {'[lower, upper]':<20} {'[lower, upper]':<20}")
    print("-" * 115)

    for step in range(n_steps):
        if adapter.done():
            break

        V = float(adapter.get_certificate_value())

        # Get estimates
        s_safety, s_lo, s_hi, _ = sampling(adapter)
        a_safety, a_er, _, _ = analytic(adapter)

        if step >= k:
            h_safety, h_lo, h_hi, _ = history(adapter)
            h_str = f"[{h_lo:+.3f}, {h_hi:+.3f}]"
        else:
            h_str = "(warmup)"

        if step >= warmup_la:
            hla_safety, hla_lo, hla_hi, _ = history_la(adapter)
            hla_str = f"[{hla_lo:+.3f}, {hla_hi:+.3f}]"
        else:
            hla_str = "(warmup)"

        print(f"  {step:<6} {V:<10.4f} [{s_lo:+.3f}, {s_hi:+.3f}] {s_safety}  {a_er:<+12.4f} {h_str:<20} {hla_str}")

        adapter.step()

    print("-" * 115)
    print()
    print("  Legend:")
    print("    V(x)      = Certificate value at current state")
    print("    Sampling  = CI from Monte Carlo (512-8192 samples)")
    print("    Analytic  = Point estimate using E[next_state]")
    print("    History   = CI from trajectory history (no lookahead)")
    print("    Hist+LA   = CI with OptimalTemporalWeights + delay=auto (uses lookahead)")
    print()
    print("=" * 70)


def test_base():
    """Basic smoke test - just runs the monitor without assertions."""
    from monitor import NeuralCertificateMonitor
    from monitor.adapters import SablasDrone
    from monitor.estimators import SamplingEstimator

    drone = SablasDrone(vis_every=0)
    estimator = SamplingEstimator()
    m = NeuralCertificateMonitor(drone, estimator)
    # Consume the generator to ensure the loop runs
    for _ in m.run():
        pass
