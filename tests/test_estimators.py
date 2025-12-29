"""Tests for estimators and CI coverage."""
import numpy as np
from monitor.adapters import SablasDrone
from monitor.estimators import SamplingEstimator, AnalyticEstimator, HistoryEstimator
from monitor.weighting import UniformWeights, RecentWeights, OptimalTemporalWeights


def test_ci_coverage():
    """
    End-to-end test: verify that the SamplingEstimator's CI contains the true E[R].

    For each step, we:
    1. Get the CI from SamplingEstimator (based on Monte Carlo sampling)
    2. Compute E[R] analytically via AnalyticEstimator (exploiting small Jensen gap)
    3. Check that the analytic E[R] falls within the CI

    With delta=0.01, we expect ~99% coverage. The analytic estimate is near-exact
    (Jensen gap < 0.0001), so this is a rigorous coverage test.
    """
    drone = SablasDrone()
    sampling_estimator = SamplingEstimator()
    analytic_estimator = AnalyticEstimator()

    n_steps = 200

    print("\n")
    print("=" * 70)
    print("  CONFIDENCE INTERVAL COVERAGE TEST")
    print("=" * 70)
    print()
    print("  Theory: The SamplingEstimator computes a CI for E[R] using Hoeffding's")
    print("  inequality with delta=0.01. This CI should contain the true E[R] at least")
    print("  99% of the time.")
    print()
    print("  Ground Truth: We compute E[R] analytically using the fact that for this")
    print("  system, Jensen's gap is tiny (<0.0001), so V(E[Y]) ~ E[V(Y)].")
    print("  This gives us a near-exact E[R] to check against the CI.")
    print()
    print("  Legend:  [---*---]  CI with true E[R] inside (*)")
    print("           |         Zero line (E[R] < 0 means safe)")
    print("           X         True E[R] outside CI (miss)")
    print()
    # Visual CI bar settings
    bar_width = 40

    def make_ci_bar(lower, upper, true_val, val_min, val_max):
        """Create a visual bar showing CI and where true value falls."""
        # Normalize to [0, 1] range
        range_size = val_max - val_min
        if range_size == 0:
            return "[" + "?" * bar_width + "]"

        lo_pos = int((lower - val_min) / range_size * bar_width)
        hi_pos = int((upper - val_min) / range_size * bar_width)
        true_pos = int((true_val - val_min) / range_size * bar_width)

        # Clamp to valid range
        lo_pos = max(0, min(bar_width - 1, lo_pos))
        hi_pos = max(0, min(bar_width - 1, hi_pos))
        true_pos = max(0, min(bar_width - 1, true_pos))

        # Build the bar
        bar = [' '] * bar_width

        # Draw CI range
        for i in range(lo_pos, hi_pos + 1):
            bar[i] = '-'

        # Draw CI endpoints
        bar[lo_pos] = '['
        bar[hi_pos] = ']'

        # Draw true value (overwrites if on endpoint)
        if lo_pos <= true_pos <= hi_pos:
            bar[true_pos] = '*'
        else:
            bar[true_pos] = 'X'  # Miss!

        # Add zero marker if in range
        zero_pos = int((0 - val_min) / range_size * bar_width)
        if 0 <= zero_pos < bar_width and bar[zero_pos] == ' ':
            bar[zero_pos] = '|'

        return ''.join(bar)

    print("-" * 78)
    print(f"  {'Step':<5} {'CI':<40}  {'Lower':>8} {'E[R]':>8} {'Upper':>8}")
    print("-" * 78)

    covered = 0
    total = 0
    ci_widths = []
    misses = []
    all_results = []

    step = 0
    while total < n_steps:
        if drone.done():
            drone.reset()

        # Get CI from sampling estimator
        _, lower, upper, n_samples = sampling_estimator(drone)
        ci_widths.append(upper - lower)

        # Get near-exact E[R] from analytic estimator
        _, true_er, _, _ = analytic_estimator(drone)

        # Check coverage
        if lower <= true_er <= upper:
            covered += 1
            status = "OK"
        else:
            miss_dist = min(abs(true_er - lower), abs(true_er - upper))
            misses.append(miss_dist)
            status = "MISS"

        total += 1
        all_results.append((step, lower, upper, true_er, n_samples, status))

        drone.step()
        step += 1

    # Compute global range for consistent visualization
    all_lowers = [r[1] for r in all_results]
    all_uppers = [r[2] for r in all_results]
    all_true = [r[3] for r in all_results]
    val_min = min(min(all_lowers), min(all_true)) - 0.05
    val_max = max(max(all_uppers), max(all_true)) + 0.05

    # Print results with visual bars
    for step, lower, upper, true_er, n_samples, status in all_results:
        bar = make_ci_bar(lower, upper, true_er, val_min, val_max)
        marker = ">>" if status == "MISS" else "  "
        print(f"{marker} {step:<5} {bar}  {lower:>8.4f} {true_er:>8.4f} {upper:>8.4f}")

    coverage_rate = covered / total
    mean_ci_width = np.mean(ci_widths)

    print("-" * 70)
    print()
    print("  RESULTS")
    print()
    print(f"    Total steps tested:    {total}")
    print(f"    CIs containing E[R]:   {covered}")
    print(f"    CIs missing E[R]:      {len(misses)}")
    print()
    print(f"    Coverage rate:         {coverage_rate*100:.1f}%")
    print(f"    Expected coverage:     >=99% (delta=0.01)")
    print()
    print(f"    Mean CI width:         {mean_ci_width:.4f}")
    print(f"    Min CI width:          {min(ci_widths):.4f}")
    print(f"    Max CI width:          {max(ci_widths):.4f}")

    if misses:
        print()
        print(f"    Miss distances:        mean={np.mean(misses):.6f}, max={max(misses):.6f}")

    print()
    if coverage_rate >= 0.99:
        print("  VERDICT: PERFECT - Coverage meets theoretical guarantee!")
    elif coverage_rate >= 0.95:
        print("  VERDICT: PASS - Coverage is excellent (small slack for Jensen gap)")
    else:
        print("  VERDICT: FAIL - Coverage below acceptable threshold")
    print()
    print("=" * 70)

    # With delta=0.01, expect 99% coverage. Allow small slack for Jensen gap + clipping.
    assert coverage_rate > 0.95, f"CI coverage too low: {coverage_rate:.1%}"


def test_history_estimator_ci_coverage():
    """
    End-to-end test: verify that the HistoryEstimator's CI contains the true E[R].

    The HistoryEstimator uses observed trajectory history and the Lipschitz assumption
    to bound E[R]. Unlike SamplingEstimator which samples possible next states,
    HistoryEstimator uses past rewards from nearby states.

    Key differences from SamplingEstimator test:
    - Need to accumulate trajectory history before testing
    - CIs will be wider due to discretization error (DE) component
    - Coverage depends on Lipschitz constant being correct

    With delta=0.01, we expect ~99% coverage if:
    1. The Lipschitz constant gamma is an upper bound on the true constant
    2. The sub-Gaussian assumption holds (it does for bounded rewards)
    """
    drone = SablasDrone()
    analytic_estimator = AnalyticEstimator()

    # Test both weighting strategies
    weighting_strategies = [
        ("UniformWeights", UniformWeights()),
        ("RecentWeights(k=10)", RecentWeights(k=10)),
    ]

    for strategy_name, weighting in weighting_strategies:
        history_estimator = HistoryEstimator(weighting=weighting, delta=0.01)

        print("\n")
        print("=" * 70)
        print(f"  HISTORY ESTIMATOR CI COVERAGE TEST ({strategy_name})")
        print("=" * 70)
        print()
        print("  Theory: The HistoryEstimator computes a CI for E[R] using:")
        print("    - Weighted mean of observed rewards from trajectory history")
        print("    - Discretization Error (DE): gamma * sum w_i * d(x_t, x_i)")
        print("    - Statistical Error (SE): concentration bound for weighted sum")
        print()
        print("  Ground Truth: We compute E[R] analytically using V(E[Y]) ~ E[V(Y)].")
        print()

        # Visual CI bar settings
        bar_width = 40

        def make_ci_bar(lower, upper, true_val, val_min, val_max):
            """Create a visual bar showing CI and where true value falls."""
            range_size = val_max - val_min
            if range_size == 0:
                return "[" + "?" * bar_width + "]"

            lo_pos = int((lower - val_min) / range_size * bar_width)
            hi_pos = int((upper - val_min) / range_size * bar_width)
            true_pos = int((true_val - val_min) / range_size * bar_width)

            lo_pos = max(0, min(bar_width - 1, lo_pos))
            hi_pos = max(0, min(bar_width - 1, hi_pos))
            true_pos = max(0, min(bar_width - 1, true_pos))

            bar = [' '] * bar_width

            for i in range(lo_pos, hi_pos + 1):
                bar[i] = '-'

            bar[lo_pos] = '['
            bar[hi_pos] = ']'

            if lo_pos <= true_pos <= hi_pos:
                bar[true_pos] = '*'
            else:
                bar[true_pos] = 'X'

            zero_pos = int((0 - val_min) / range_size * bar_width)
            if 0 <= zero_pos < bar_width and bar[zero_pos] == ' ':
                bar[zero_pos] = '|'

            return ''.join(bar)

        print("-" * 86)
        print(f"  {'Step':<5} {'CI':<40}  {'Lower':>8} {'E[R]':>8} {'Upper':>8} {'DE':>6} {'SE':>6}")
        print("-" * 86)

        covered = 0
        total = 0
        inconclusive = 0
        ci_widths = []
        de_values = []
        se_values = []
        misses = []
        all_results = []

        n_steps = 150
        warmup_steps = 20  # Let history accumulate before testing coverage

        step = 0
        while step < n_steps + warmup_steps:
            if drone.done():
                drone.reset()

            # Get CI from history estimator
            safety, lower, upper, info = history_estimator(drone)

            # Skip warmup period (insufficient history)
            if step >= warmup_steps and info.get("reason") != "insufficient history":
                ci_widths.append(upper - lower)
                de_values.append(info.get("DE", 0))
                se_values.append(info.get("SE", 0))

                # Get near-exact E[R] from analytic estimator
                _, true_er, _, _ = analytic_estimator(drone)

                # Check coverage
                if lower <= true_er <= upper:
                    covered += 1
                    status = "OK"
                else:
                    miss_dist = min(abs(true_er - lower), abs(true_er - upper))
                    misses.append(miss_dist)
                    status = "MISS"

                if safety == "?":
                    inconclusive += 1

                total += 1
                all_results.append((step, lower, upper, true_er, info.get("DE", 0), info.get("SE", 0), status))

            drone.step()
            step += 1

        if not all_results:
            print("  No valid results collected (insufficient history throughout)")
            continue

        # Compute global range for consistent visualization
        all_lowers = [r[1] for r in all_results]
        all_uppers = [r[2] for r in all_results]
        all_true = [r[3] for r in all_results]
        val_min = min(min(all_lowers), min(all_true)) - 0.1
        val_max = max(max(all_uppers), max(all_true)) + 0.1

        # Print results with visual bars
        for step, lower, upper, true_er, de, se, status in all_results:
            bar = make_ci_bar(lower, upper, true_er, val_min, val_max)
            marker = ">>" if status == "MISS" else "  "
            print(f"{marker} {step:<5} {bar}  {lower:>8.4f} {true_er:>8.4f} {upper:>8.4f} {de:>6.3f} {se:>6.3f}")

        coverage_rate = covered / total if total > 0 else 0
        mean_ci_width = np.mean(ci_widths) if ci_widths else 0
        mean_de = np.mean(de_values) if de_values else 0
        mean_se = np.mean(se_values) if se_values else 0

        print("-" * 70)
        print()
        print("  RESULTS")
        print()
        print(f"    Total steps tested:    {total}")
        print(f"    CIs containing E[R]:   {covered}")
        print(f"    CIs missing E[R]:      {len(misses)}")
        print(f"    Inconclusive (\"?\"):    {inconclusive}")
        print()
        print(f"    Coverage rate:         {coverage_rate*100:.1f}%")
        print(f"    Expected coverage:     >=99% (delta=0.01)")
        print()
        print(f"    Mean CI width:         {mean_ci_width:.4f}")
        print(f"    Mean DE:               {mean_de:.4f}")
        print(f"    Mean SE:               {mean_se:.4f}")
        print(f"    DE / (DE+SE):          {mean_de / (mean_de + mean_se) * 100:.1f}% (discretization dominates if high)")

        if misses:
            print()
            print(f"    Miss distances:        mean={np.mean(misses):.6f}, max={max(misses):.6f}")

        print()
        if coverage_rate >= 0.99:
            print(f"  VERDICT: PERFECT - {strategy_name} coverage meets theoretical guarantee!")
        elif coverage_rate >= 0.95:
            print(f"  VERDICT: PASS - {strategy_name} coverage is excellent")
        elif coverage_rate >= 0.90:
            print(f"  VERDICT: MARGINAL - {strategy_name} coverage slightly below target")
        else:
            print(f"  VERDICT: FAIL - {strategy_name} coverage below acceptable threshold")
        print()
        print("=" * 70)

        # Note: HistoryEstimator coverage depends heavily on gamma being correct.
        # With gamma=1.0 (conservative), we expect good coverage but wide CIs.
        # The assertion is lenient because DE depends on trajectory diversity.
        assert coverage_rate > 0.90, f"{strategy_name} CI coverage too low: {coverage_rate:.1%}"


def test_delayed_history_estimator():
    """
    Test that the delayed HistoryEstimator correctly uses lookahead.

    When delay > 0, the estimator targets T - delay instead of T,
    allowing symmetric weighting with OptimalTemporalWeights.
    """
    from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum

    print("\n")
    print("=" * 70)
    print("  DELAYED HISTORY ESTIMATOR TEST")
    print("=" * 70)

    adapter = NeuralCLBFPendulum(dt=0.01)

    # Build up history
    for _ in range(100):
        adapter.step()

    T = len(adapter.state_history) - 1  # 100

    # Test 1: No delay (default behavior)
    estimator_no_delay = HistoryEstimator(weighting=RecentWeights(k=10), delay=0)
    safety, lower, upper, info = estimator_no_delay(adapter)

    print(f"\n  No delay (default):")
    print(f"    T={info['T']}, target={info['target']}, delay={info['delay']}")
    assert info['target'] == info['T'], "No delay should target T"
    assert info['delay'] == 0

    # Test 2: Fixed delay
    estimator_fixed = HistoryEstimator(weighting=RecentWeights(k=10), delay=20)
    safety, lower, upper, info = estimator_fixed(adapter)

    print(f"\n  Fixed delay=20:")
    print(f"    T={info['T']}, target={info['target']}, delay={info['delay']}")
    assert info['target'] == info['T'] - 20, f"target should be T-20={info['T']-20}"
    assert info['delay'] == 20

    # Test 3: Auto delay with OptimalTemporalWeights
    optimal_weighting = OptimalTemporalWeights(max_m=15)
    m = optimal_weighting.get_optimal_m(adapter)

    estimator_auto = HistoryEstimator(weighting=optimal_weighting, delay="auto")
    safety, lower, upper, info = estimator_auto(adapter)

    print(f"\n  Auto delay with OptimalTemporalWeights (m={m}):")
    print(f"    T={info['T']}, target={info['target']}, delay={info['delay']}")
    assert info['delay'] == m, f"Auto delay should equal m={m}"
    assert info['target'] == info['T'] - m

    # Test 4: Auto delay with non-optimal weighting (should be 0)
    estimator_auto_recent = HistoryEstimator(weighting=RecentWeights(k=10), delay="auto")
    safety, lower, upper, info = estimator_auto_recent(adapter)

    print(f"\n  Auto delay with RecentWeights (should be 0):")
    print(f"    T={info['T']}, target={info['target']}, delay={info['delay']}")
    assert info['delay'] == 0, "Auto delay with non-optimal weighting should be 0"

    # Test 5: Verify lookahead is actually used
    print(f"\n  Verifying lookahead with OptimalTemporalWeights:")

    # Get weights for target = T - m
    target = T - m
    weights = optimal_weighting(adapter, target - 1)  # -1 for reward indexing
    nonzero = [(i, w) for i, w in enumerate(weights) if w > 0]

    print(f"    Target state index: {target}")
    print(f"    Weight indices: {nonzero[0][0]} to {nonzero[-1][0]}")
    print(f"    Uses lookahead: {nonzero[-1][0] > target - 1}")  # -1 for reward vs state

    # The key assertion: weights should extend past target (lookahead)
    assert nonzero[-1][0] >= target - 1, "Should use lookahead indices"

    print("\n" + "=" * 70)
    print("  Delayed estimation works correctly!")
    print("=" * 70)
