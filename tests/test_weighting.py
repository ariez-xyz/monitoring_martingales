"""Tests for weighting strategies."""
from monitor.weighting import UniformWeights, RecentWeights, OptimalTemporalWeights


def test_weighting_strategies():
    """
    Smoke test for weighting strategies including OptimalTemporalWeights.

    Verifies:
    1. All strategies produce valid weights (sum to 1, non-negative)
    2. OptimalTemporalWeights computes reasonable m* based on adapter parameters
    3. Weights are applied correctly at different timesteps
    """
    from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum

    print("\n")
    print("=" * 70)
    print("  WEIGHTING STRATEGIES TEST")
    print("=" * 70)

    # Create adapter and run some steps to build history
    adapter = NeuralCLBFPendulum(dt=0.01)
    n_steps = 50
    for _ in range(n_steps):
        adapter.step()

    # Get adapter parameters for context
    gamma = adapter.get_lipschitz_constant()
    reward_lo, reward_hi = adapter.get_reward_bounds()
    sigma = (reward_hi - reward_lo) / 2

    print(f"\n  Adapter parameters:")
    print(f"    dt = {adapter.dt}")
    print(f"    gamma (Lipschitz) = {gamma:.4f}")
    print(f"    sigma (reward bound) = {sigma:.1f}")
    print(f"    History length = {n_steps + 1} steps")

    # Test strategies
    strategies = [
        ("UniformWeights", UniformWeights()),
        ("RecentWeights(k=10)", RecentWeights(k=10)),
        ("OptimalTemporalWeights", OptimalTemporalWeights()),
    ]

    print(f"\n  Strategy comparison at t={n_steps}:")
    print("-" * 70)

    for name, strategy in strategies:
        weights = strategy(adapter, n_steps)

        # Validate weights
        assert len(weights) == n_steps + 1, f"{name}: wrong length"
        assert abs(sum(weights) - 1.0) < 1e-6, f"{name}: weights don't sum to 1"
        assert all(w >= 0 for w in weights), f"{name}: negative weights"

        # Find non-zero weights
        nonzero_indices = [i for i, w in enumerate(weights) if w > 0]
        n_nonzero = len(nonzero_indices)
        first_nonzero = nonzero_indices[0] if nonzero_indices else -1
        last_nonzero = nonzero_indices[-1] if nonzero_indices else -1

        # Weight statistics
        nonzero_weights = [w for w in weights if w > 0]
        max_weight = max(nonzero_weights)
        min_weight = min(nonzero_weights)

        print(f"\n  {name}:")
        print(f"    Non-zero weights: {n_nonzero} (indices {first_nonzero}-{last_nonzero})")
        print(f"    Weight range: [{min_weight:.4f}, {max_weight:.4f}]")

        if isinstance(strategy, OptimalTemporalWeights):
            m_star = strategy.get_optimal_m(adapter)
            print(f"    Optimal m* = {m_star}")
            print(f"    (computed from m* = (c1/gamma)^(2/3) where c1 = sigma*sqrt(log(2/delta)/2))")

    # Show weight distribution for OptimalTemporalWeights at different times
    print("\n" + "-" * 70)
    print("  OptimalTemporalWeights at different timesteps:")
    print("-" * 70)

    optimal = OptimalTemporalWeights()
    m_star = optimal.get_optimal_m(adapter)

    print(f"\n  {'t':<6} {'History':<10} {'Weighted':<10} {'Indices':<20}")
    print(f"  {'':<6} {'length':<10} {'count':<10} {'(non-zero)':<20}")
    print("  " + "-" * 50)

    for t in [5, 10, 20, 30, 50]:
        if t <= n_steps:
            weights = optimal(adapter, t)
            nonzero = [(i, w) for i, w in enumerate(weights) if w > 0]
            indices = f"{nonzero[0][0]}-{nonzero[-1][0]}" if nonzero else "none"
            print(f"  {t:<6} {t+1:<10} {len(nonzero):<10} {indices:<20}")

    # Visual weight distribution for t=50
    print("\n" + "-" * 70)
    print(f"  Weight distribution at t={n_steps} (m*={m_star}):")
    print("-" * 70)

    weights = optimal(adapter, n_steps)
    nonzero = [(i, w) for i, w in enumerate(weights) if w > 0]

    if len(nonzero) <= 20:
        # Show all weights
        print(f"\n  {'Index':<8} {'Weight':<12} {'Bar':<30}")
        print("  " + "-" * 50)
        for i, w in nonzero:
            bar_len = int(w * 200)  # Scale for visibility
            bar = "#" * bar_len
            print(f"  {i:<8} {w:<12.4f} {bar}")
    else:
        # Show summary for many weights
        print(f"\n  {len(nonzero)} weights from index {nonzero[0][0]} to {nonzero[-1][0]}")
        print(f"  Each weight = {nonzero[0][1]:.4f} (uniform over window)")

    print("\n" + "=" * 70)
    print("  All weighting strategies work correctly!")
    print("=" * 70)


def test_weighting_with_lookahead():
    """
    Test that weighting strategies handle lookahead correctly.

    When t (target) is not at the end of history, strategies should be able
    to use "future" observations (indices > t) for weighting.

    Expected behavior:
    - UniformWeights: weights [0, t] uniformly (no lookahead, simple baseline)
    - RecentWeights(k): weights [t-k+1, t] (backward from target, no lookahead)
    - OptimalTemporalWeights: weights [t-m, t+m] symmetric around t (uses lookahead!)
    """
    from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum

    print("\n")
    print("=" * 70)
    print("  WEIGHTING WITH LOOKAHEAD TEST")
    print("=" * 70)

    # Create adapter with enough history
    adapter = NeuralCLBFPendulum(dt=0.01)
    for _ in range(100):
        adapter.step()

    history_len = len(adapter.state_history)  # 101 (initial + 100 steps)
    T = history_len - 1  # 100, latest index

    print(f"\n  History length: {history_len} states (indices 0-{T})")

    # Test 1: UniformWeights - should weight [0, t] regardless of history length
    print("\n  --- UniformWeights ---")
    uniform = UniformWeights()

    for target in [50, 80, T]:
        weights = uniform(adapter, target)
        nonzero = [(i, w) for i, w in enumerate(weights) if w > 0]
        print(f"  target={target}: {len(weights)} weights, non-zero at indices {nonzero[0][0]}-{nonzero[-1][0]}")

        assert len(weights) == history_len, f"Should return history_len weights"
        assert abs(sum(weights) - 1.0) < 1e-6, "Weights should sum to 1"
        assert nonzero[0][0] == 0, "UniformWeights should start at 0"
        assert nonzero[-1][0] == target, "UniformWeights should end at target"

    # Test 2: RecentWeights - should weight [t-k+1, t] (backward only)
    print("\n  --- RecentWeights(k=10) ---")
    recent = RecentWeights(k=10)

    for target in [50, 80, T]:
        weights = recent(adapter, target)
        nonzero = [(i, w) for i, w in enumerate(weights) if w > 0]
        expected_start = max(0, target - 9)  # k=10 means indices t-9 to t
        print(f"  target={target}: non-zero at indices {nonzero[0][0]}-{nonzero[-1][0]} (expected {expected_start}-{target})")

        assert len(weights) == history_len, f"Should return history_len weights"
        assert nonzero[0][0] == expected_start, f"Should start at {expected_start}"
        assert nonzero[-1][0] == target, f"Should end at {target}"

    # Test 3: OptimalTemporalWeights WITH lookahead
    # This is the key test - when target < T, it should use indices after target
    print("\n  --- OptimalTemporalWeights (with lookahead) ---")
    optimal = OptimalTemporalWeights(max_m=10)  # cap m for predictable test
    m = optimal.get_optimal_m(adapter)
    print(f"  m = {m}")

    # Case A: target at end (no lookahead possible)
    target = T
    weights = optimal(adapter, target)
    nonzero = [(i, w) for i, w in enumerate(weights) if w > 0]
    print(f"  target={target} (at end): indices {nonzero[0][0]}-{nonzero[-1][0]} (backward only)")
    assert nonzero[-1][0] == target, "Should end at target when at end of history"

    # Case B: target in middle (lookahead should be used)
    target = 50
    weights = optimal(adapter, target)
    nonzero = [(i, w) for i, w in enumerate(weights) if w > 0]
    expected_start = max(0, target - m)
    expected_end = min(T, target + m)
    print(f"  target={target} (mid): indices {nonzero[0][0]}-{nonzero[-1][0]} (expected ~{expected_start}-{expected_end})")

    # THIS IS THE KEY ASSERTION - should use lookahead
    assert nonzero[-1][0] > target, f"Should use lookahead! Got end={nonzero[-1][0]}, target={target}"
    assert nonzero[-1][0] == expected_end, f"Should end at {expected_end}"

    # Case C: target near start (limited backward, full forward)
    target = 5
    weights = optimal(adapter, target)
    nonzero = [(i, w) for i, w in enumerate(weights) if w > 0]
    expected_start = max(0, target - m)
    expected_end = min(T, target + m)
    print(f"  target={target} (near start): indices {nonzero[0][0]}-{nonzero[-1][0]} (expected {expected_start}-{expected_end})")

    assert nonzero[0][0] == expected_start
    assert nonzero[-1][0] == expected_end

    print("\n" + "=" * 70)
    print("  Lookahead weighting works correctly!")
    print("=" * 70)
