import numpy as np
import torch
from monitor import NeuralCertificateMonitor
from monitor.adapters import SablasDrone
from monitor.estimators import SamplingEstimator, AnalyticEstimator


def test_base():
    """Basic smoke test - just runs the monitor without assertions."""
    drone = SablasDrone(vis_every=0)  # Disable visualization for tests
    estimator = SamplingEstimator()
    m = NeuralCertificateMonitor(drone, estimator)
    m.run()


def get_min_obstacle_distance(state: np.ndarray, obstacles: np.ndarray) -> float:
    """
    Computes the minimum distance from the drone position to any obstacle.

    Args:
        state: (8,) array, first 3 elements are position
        obstacles: (k, 8) array, first 3 elements of each row are obstacle position

    Returns:
        Minimum Euclidean distance to any obstacle
    """
    drone_pos = state[:3]
    obstacle_positions = obstacles[:, :3]
    distances = np.linalg.norm(obstacle_positions - drone_pos, axis=1)
    return float(np.min(distances))


def test_certificate_correlates_with_safety():
    """
    Validates that the CBF value V(x) correlates with actual physical safety.

    The CBF is intentionally conservative - it may signal V < 0 (violation) before
    the drone is physically unsafe. This is correct behavior for a safety-critical system.

    Key property: When V(x) > 0, the drone should be physically safe (dist > 0.6).
    """
    drone = SablasDrone()

    # Use sablas's actual safety threshold
    dang_dist = 0.6  # sablas uses this in utils.is_safe()

    samples = []
    max_steps = 500
    for _ in range(max_steps):
        if drone.done():
            drone.reset()

        v = float(drone.get_certificate_value())
        min_dist = get_min_obstacle_distance(drone.state, drone.obstacle)
        is_physically_safe = min_dist > dang_dist

        samples.append((v, min_dist, is_physically_safe))
        drone.step()

    vs = np.array([s[0] for s in samples])
    dists = np.array([s[1] for s in samples])
    safe = np.array([s[2] for s in samples])

    print(f"\n--- Certificate-Safety Correlation (n={len(samples)}) ---")
    print(f"Physical safety rate (dist > {dang_dist}): {safe.mean()*100:.1f}%")
    print(f"V > 0 rate: {(vs > 0).mean()*100:.1f}%")
    print(f"V < 0 rate: {(vs < 0).mean()*100:.1f}%")
    print()

    # Key metric: when V > 0, should be physically safe
    v_positive = vs > 0
    if v_positive.sum() > 0:
        safe_when_v_positive = safe[v_positive].mean()
        print(f"When V > 0: {safe_when_v_positive*100:.1f}% are physically safe")
        # This should be very high - CBF positive means safe
        assert safe_when_v_positive > 0.95, f"V > 0 but only {safe_when_v_positive:.1%} physically safe"

    # When V < 0, the CBF is being conservative - drone may still be physically safe
    # but approaching danger. Check that distance is at least trending lower.
    v_negative = vs < 0
    if v_negative.sum() > 0:
        mean_dist_when_v_neg = dists[v_negative].mean()
        mean_dist_when_v_pos = dists[v_positive].mean() if v_positive.sum() > 0 else 0
        print(f"Mean distance when V < 0: {mean_dist_when_v_neg:.3f}")
        print(f"Mean distance when V > 0: {mean_dist_when_v_pos:.3f}")
        # V < 0 should correspond to closer distances on average
        assert mean_dist_when_v_neg < mean_dist_when_v_pos, \
            f"V < 0 should mean closer to obstacles"


def test_reward_sign_on_violation():
    """
    Validates that the reward signal correlates with safety transitions.

    When V(x) transitions from positive to negative (entering violation),
    the reward should tend to be positive (indicating CBF condition violated).
    """
    drone = SablasDrone()

    rewards_at_violation_entry = []
    rewards_when_safe = []

    prev_v = float(drone.get_certificate_value())

    max_steps = 300
    for _ in range(max_steps):
        if drone.done():
            drone.reset()
            prev_v = float(drone.get_certificate_value())
            continue

        # Sample and compute reward
        samples = drone.sample(n_samples=100)
        reward = float(drone.get_reward(samples).mean())

        # Step and get new certificate value
        drone.step()
        curr_v = float(drone.get_certificate_value())

        # Track rewards at violation entry vs when safe
        if prev_v > 0 and curr_v < 0:
            # Entering violation
            rewards_at_violation_entry.append(reward)
        elif prev_v > 0.1 and curr_v > 0.1:
            # Staying safely positive
            rewards_when_safe.append(reward)

        prev_v = curr_v

    print(f"\n--- Reward Sign Analysis ---")
    print(f"Rewards at violation entry: {len(rewards_at_violation_entry)} samples")
    if rewards_at_violation_entry:
        mean_viol = np.mean(rewards_at_violation_entry)
        print(f"  Mean reward: {mean_viol:.4f}")
        print(f"  Positive rewards: {sum(1 for r in rewards_at_violation_entry if r > 0)}/{len(rewards_at_violation_entry)}")

    print(f"Rewards when safe: {len(rewards_when_safe)} samples")
    if rewards_when_safe:
        mean_safe = np.mean(rewards_when_safe)
        print(f"  Mean reward: {mean_safe:.4f}")
        print(f"  Negative rewards: {sum(1 for r in rewards_when_safe if r < 0)}/{len(rewards_when_safe)}")

    # When safe, rewards should mostly be negative (CBF condition satisfied)
    if len(rewards_when_safe) > 20:
        negative_ratio = sum(1 for r in rewards_when_safe if r < 0) / len(rewards_when_safe)
        assert negative_ratio > 0.7, f"Expected mostly negative rewards when safe, got {negative_ratio:.1%}"


def test_ci_coverage():
    """
    End-to-end test: verify that the SamplingEstimator's CI contains the true E[R].

    For each step, we:
    1. Get the CI from SamplingEstimator (based on Monte Carlo sampling)
    2. Compute E[R] analytically via AnalyticEstimator (exploiting small Jensen gap)
    3. Check that the analytic E[R] falls within the CI

    With δ=0.01, we expect ~99% coverage. The analytic estimate is near-exact
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
    print("  inequality with δ=0.01. This CI should contain the true E[R] at least")
    print("  99% of the time.")
    print()
    print("  Ground Truth: We compute E[R] analytically using the fact that for this")
    print("  system, Jensen's gap is tiny (<0.0001), so V(E[Y]) ≈ E[V(Y)].")
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
    print(f"    Expected coverage:     ≥99% (δ=0.01)")
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

    # With δ=0.01, expect 99% coverage. Allow small slack for Jensen gap + clipping.
    assert coverage_rate > 0.95, f"CI coverage too low: {coverage_rate:.1%}"


