"""
Empirically estimate the Lipschitz constant γ for the expected reward function.

The Lipschitz assumption is: |E[ρ(x)] - E[ρ(x')]| ≤ γ · d(x, x')

Follows sablas test_drone.py approach: run for a large fixed number of steps,
collecting diverse states across many episode resets.
"""

import numpy as np
import torch
from monitor.adapters import SablasDrone

EVAL_STEPS = 50000  # Fewer than CBF bounds since we need pairwise comparisons


def estimate_expected_reward(adapter: SablasDrone, state: torch.Tensor, obstacle: np.ndarray, n_samples: int = 100) -> float:
    """Estimate E[R(x)] by sampling next states."""
    original_state = adapter.state.copy()
    original_obstacle = adapter.obstacle.copy()

    adapter.state = state.numpy()
    adapter.obstacle = obstacle

    next_states = adapter.sample(state, n_samples=n_samples)
    rewards = adapter.get_reward(next_states, cur_state=state)

    adapter.state = original_state
    adapter.obstacle = original_obstacle
    return float(rewards.mean())


def main():
    adapter = SablasDrone()

    # Collect states and obstacles from long run
    print(f"Collecting states over {EVAL_STEPS} steps...")
    states = []
    obstacles = []
    num_resets = 0

    for i in range(EVAL_STEPS):
        if adapter.done():
            adapter.reset()
            num_resets += 1

        states.append(torch.from_numpy(adapter.state.copy()).float())
        obstacles.append(adapter.obstacle.copy())
        adapter.step()

        if (i + 1) % 10000 == 0:
            print(f"  Progress: {i + 1}/{EVAL_STEPS} steps, {num_resets} resets")

    print(f"Collected {len(states)} states across {num_resets} resets")

    # Sample pairs and compute ratios
    print("Estimating expected rewards and computing ratios...")
    n_pairs = 1000
    ratios = []

    rng = np.random.default_rng(42)
    indices = rng.choice(len(states), size=(n_pairs, 2), replace=True)

    # Cache expected rewards
    reward_cache = {}

    for i, (idx1, idx2) in enumerate(indices):
        if idx1 == idx2:
            continue

        state1, state2 = states[idx1], states[idx2]
        obs1, obs2 = obstacles[idx1], obstacles[idx2]

        # Get or compute expected rewards
        if idx1 not in reward_cache:
            reward_cache[idx1] = estimate_expected_reward(adapter, state1, obs1)
        if idx2 not in reward_cache:
            reward_cache[idx2] = estimate_expected_reward(adapter, state2, obs2)

        er1, er2 = reward_cache[idx1], reward_cache[idx2]
        dist = adapter.distance(state1, state2)

        if dist > 1e-6:  # avoid division by zero
            ratio = abs(er1 - er2) / dist
            ratios.append(ratio)

        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{n_pairs} pairs")

    ratios = np.array(ratios)

    print(f"\n=== Lipschitz Constant Estimation ===")
    print(f"Pairs analyzed: {len(ratios)}")
    print(f"Ratio |E[R(x)] - E[R(x')]| / d(x,x'):")
    print(f"  Mean:   {ratios.mean():.4f}")
    print(f"  Std:    {ratios.std():.4f}")
    print(f"  Median: {np.median(ratios):.4f}")
    print(f"  95th:   {np.percentile(ratios, 95):.4f}")
    print(f"  99th:   {np.percentile(ratios, 99):.4f}")
    print(f"  Max:    {ratios.max():.4f}")

    # Use max with margin since Lipschitz must hold for ALL pairs
    margin = 1.2  # 20% margin for unobserved pairs
    suggested_gamma = ratios.max() * margin
    print(f"\nSuggested γ (max with {int((margin-1)*100)}% margin): {suggested_gamma:.4f}")


if __name__ == "__main__":
    main()
