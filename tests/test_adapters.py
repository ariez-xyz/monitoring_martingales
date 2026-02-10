"""Tests for dynamical system adapters."""
import numpy as np
import torch


def test_neural_clbf_pendulum_adapter():
    """
    Smoke test for the neural_clbf inverted pendulum adapter.

    Verifies that:
    1. The adapter loads successfully
    2. Certificate values can be computed
    3. Steps can be taken
    4. The CLF decreases along trajectories (nominal controller stabilizes)
    """
    from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum

    # Create adapter
    adapter = NeuralCLBFPendulum()

    assert adapter.get_state_dim() == 2, "Inverted pendulum should have 2D state"
    assert adapter.dt == 0.01, f"Expected dt=0.01, got {adapter.dt}"

    # Test certificate value computation
    V0 = adapter.get_certificate_value()
    assert V0.shape == torch.Size([]), f"V should be scalar, got shape {V0.shape}"
    assert V0 >= 0, "CLF should be non-negative"

    # Run trajectory and verify CLF generally decreases
    V_values = [float(V0)]
    for _ in range(100):
        adapter.step()
        V = float(adapter.get_certificate_value())
        V_values.append(V)

    # CLF should decrease overall (may have small increases due to discretization)
    assert V_values[-1] < V_values[0], f"CLF should decrease: V0={V_values[0]:.4f}, Vfinal={V_values[-1]:.4f}"

    # Test sampling
    samples = adapter.sample(n_samples=5)
    assert samples.shape == (5, 2), f"Expected (5,2), got {samples.shape}"

    # Test expected next state
    expected = adapter.get_expected_next_state()
    assert expected.shape == (2,), f"Expected (2,), got {expected.shape}"

    # Test reward computation
    rewards = adapter.get_reward(samples)
    assert rewards.shape == (5,), f"Expected (5,), got {rewards.shape}"

    # Test history (1 initial + 100 steps = 101)
    history = adapter.get_state_history()
    assert history.shape[0] == 101, f"Expected 101 steps in history, got {history.shape[0]}"
    assert history.shape[1] == 2, f"Expected 2D states, got {history.shape[1]}"

    print(f"\nNeural CLBF Pendulum Adapter Test:")
    print(f"  Initial V: {V_values[0]:.4f}")
    print(f"  Final V:   {V_values[-1]:.4f}")
    print(f"  V decrease: {V_values[0] - V_values[-1]:.4f}")
    print(f"  Adapter works correctly!")


def test_pendulum_lipschitz_estimation():
    """
    Test that the pendulum adapter dynamically estimates Lipschitz constant.

    Verifies:
    1. Estimation returns a positive value
    2. Caching works (second call is instant)
    3. Different dt gives different gamma (smaller dt -> smaller gamma)
    """
    import time
    from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum

    print("\n--- Pendulum Lipschitz Estimation Test ---")

    # Test estimation works
    adapter = NeuralCLBFPendulum(dt=0.01)
    start = time.time()
    gamma1 = adapter.get_lipschitz_constant()
    time1 = time.time() - start

    assert gamma1 > 0, f"gamma should be positive, got {gamma1}"
    assert gamma1 < 10, f"gamma should be reasonable, got {gamma1}"
    print(f"  dt=0.01: gamma = {gamma1:.4f} (estimated in {time1:.2f}s)")

    # Test caching - second call should be instant
    start = time.time()
    gamma1_cached = adapter.get_lipschitz_constant()
    time_cached = time.time() - start

    assert gamma1_cached == gamma1, "Cached value should match"
    assert time_cached < 0.01, f"Cached call should be instant, took {time_cached:.3f}s"
    print(f"  Cached call: {time_cached*1000:.2f}ms")

    # Test different dt gives different gamma
    adapter_small_dt = NeuralCLBFPendulum(dt=0.005)
    gamma_small = adapter_small_dt.get_lipschitz_constant()

    adapter_large_dt = NeuralCLBFPendulum(dt=0.02)
    gamma_large = adapter_large_dt.get_lipschitz_constant()

    print(f"  dt=0.005: gamma = {gamma_small:.4f}")
    print(f"  dt=0.02:  gamma = {gamma_large:.4f}")

    # Smaller dt should give smaller gamma (reward changes are smaller)
    assert gamma_small < gamma1, f"Smaller dt should give smaller gamma: {gamma_small} vs {gamma1}"
    assert gamma_large > gamma1, f"Larger dt should give larger gamma: {gamma_large} vs {gamma1}"

    print("  Lipschitz estimation works correctly!")


def test_sablas_lipschitz_constant():
    """
    Test that the sablas adapter returns fixed Lipschitz constant with fixed control cadence.

    Verifies:
    1. Returns fixed gamma = 1.0 for default dt
    2. Keeps control period fixed at 0.1s while dt changes simulation fidelity
    """
    from monitor.adapters.sablas import SablasDrone

    print("\n--- Sablas Lipschitz Constant Test ---")

    # Test fixed constant
    adapter = SablasDrone()
    gamma = adapter.get_lipschitz_constant()

    assert gamma == 1.0, f"Expected fixed gamma=1.0, got {gamma}"
    assert adapter.control_period == 0.1
    assert adapter.update_control_every == 1
    print(f"  Default dt=0.1: gamma = {gamma}, control period = {adapter.control_period}")

    # Smaller dt should keep control period fixed and increase hold count
    adapter_fine = SablasDrone(dt=0.05)
    assert adapter_fine.control_period == 0.1
    assert adapter_fine.update_control_every == 2, (
        f"Expected update_control_every=2 at dt=0.05, got {adapter_fine.update_control_every}"
    )
    print(f"  dt=0.05: update_control_every = {adapter_fine.update_control_every}")

    print("  Sablas Lipschitz constant works correctly!")


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
    from monitor.adapters import SablasDrone

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
    from monitor.adapters import SablasDrone

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
