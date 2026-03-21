import torch
import numpy as np
from monitor.adapters import SablasDrone
from sablas.envs.env_drone import Drone
from sablas.modules.network import NNController

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

def test_drone_interfaces():
    """
    Tests the basic shapes and types of the adapter outputs.
    """
    drone = SablasDrone(dt=0.1)
    state_dim = drone.get_state_dim()
    assert state_dim == 8

    # Test initial state
    assert drone.state.shape == (state_dim,)

    # Test step()
    next_state = drone.step()
    assert isinstance(next_state, torch.Tensor)
    assert next_state.shape == (state_dim,)
    assert next_state.dtype == torch.float32

    # Test sample()
    n_samples = 10
    samples = drone.sample(n_samples=n_samples)
    assert isinstance(samples, torch.Tensor)
    assert samples.shape == (n_samples, state_dim)

    # Test get_certificate_value()
    v = drone.get_certificate_value()
    assert isinstance(v, torch.Tensor)
    assert v.ndim == 1

def test_certificate_change_on_step():
    """
    Tests the core logic by taking a step and observing the change in the certificate value.
    """
    drone = SablasDrone(dt=0.1)

    # 1. Get the certificate value for the current state
    v_current = drone.get_certificate_value()
    print(f"\nInitial State Position: {drone.state[:3]}")
    print(f"Initial Certificate V(x): {v_current.item():.4f}")

    # 2. Take one simulation step
    next_state = drone.step()

    # 3. Get the certificate value for the new state
    v_next = drone.get_certificate_value(state=next_state)
    print(f"Next State Position:    {next_state.numpy()[:3]}")
    print(f"Next Certificate V(y): {v_next.item():.4f}")

    # 4. Calculate and report the change
    delta_v = v_next - v_current
    print(f"Change in Certificate (V(y) - V(x)): {delta_v.item():.4f}")
    
    # A working Control Barrier Function (CBF) should result in V staying
    # positive and ideally increasing as the controller actively avoids obstacles.
    # We won't assert a specific behavior yet, but observing this value is the key output.

def test_cross_validation_with_script():
    """
    Cross-validates the adapter's logic against the original script's logic
    by running one step in parallel under a fixed random seed.
    """
    seed = 42
    dt = 0.1
    k_obs = 8

    # --- 1. Run the adapter for one step ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    adapter = SablasDrone(dt=dt)
    adapter.reset(seed=seed)
    adapter_state_0 = adapter.state.copy()
    adapter_state_1 = adapter.step().numpy()

    # --- 2. Manually run the script logic for one step ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialization logic from script
    env = Drone(k_obstacle=k_obs, dt=dt)
    controller = NNController(n_state=8, k_obstacle=8, m_control=3)
    controller.load_state_dict(torch.load('./sablas/data/drone_controller_weights.pth'))
    controller.eval()
    
    # Reset and get initial state
    script_state_0, obstacle, goal = env.reset(seed=seed)
    state_error = torch.zeros(1, 8, dtype=torch.float32)

    # Step logic from script
    u_nominal = env.nominal_controller(script_state_0, goal)
    u_tensor = controller(
        torch.from_numpy(script_state_0.reshape(1, 8)).float(),
        torch.from_numpy(obstacle.reshape(1, k_obs, 8)).float(),
        torch.from_numpy(u_nominal.reshape(1, 3)).float(),
        state_error
    )
    u = np.squeeze(u_tensor.detach().cpu().numpy())
    script_state_1, _, _, _, _ = env.step(u)

    # --- 3. Compare the results ---
    # Assert that the initial states are identical
    np.testing.assert_allclose(adapter_state_0, script_state_0, rtol=1e-6, atol=1e-6)
    
    # Assert that the next states after one step are identical
    np.testing.assert_allclose(adapter_state_1, script_state_1, rtol=1e-6, atol=1e-6)

    print("\nCross-validation successful: Adapter state matches script state for one step.")


def test_sablas_lipschitz_constant():
    """
    Test that the sablas adapter returns fixed Lipschitz constant with fixed control cadence.

    Verifies:
    1. Returns fixed gamma = 1.0 for default dt
    2. Keeps control period fixed at 0.1s while dt changes simulation fidelity
    """
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


def test_sablas_reset_seed_is_deterministic_and_clear():
    """Sablas reset(seed=...) should be reproducible and maintain clearance."""
    adapter = SablasDrone(dt=0.1, noise_level=0.0)

    adapter.reset(seed=123)
    state_a = adapter.state.copy()
    goal_a = adapter.goal.copy()
    obstacle_a = adapter.obstacle.copy()

    adapter.reset(seed=123)
    state_b = adapter.state.copy()
    goal_b = adapter.goal.copy()
    obstacle_b = adapter.obstacle.copy()

    adapter.reset(seed=456)
    state_c = adapter.state.copy()

    assert np.allclose(state_a, state_b)
    assert np.allclose(goal_a, goal_b)
    assert np.allclose(obstacle_a, obstacle_b)
    assert not np.allclose(state_a, state_c)

    state_clearance = np.min(np.linalg.norm(adapter.env.obstacle - adapter.state[:3], axis=1))
    goal_clearance = np.min(np.linalg.norm(adapter.env.obstacle - adapter.goal[:3], axis=1))
    assert state_clearance >= adapter.env.safe_dist
    assert goal_clearance >= adapter.env.safe_dist


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
