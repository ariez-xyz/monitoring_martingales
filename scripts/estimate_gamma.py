"""
Empirically estimate the Lipschitz constant γ for the expected reward function.

The Lipschitz assumption is: |E[ρ(x)] - E[ρ(x')]| ≤ γ · d(x, x')

Follows sablas test_drone.py approach: run for a large fixed number of steps,
collecting diverse states across many episode resets.

Uses AnalyticEstimator for exact E[R] computation (exploiting small Jensen gap).
"""

import numpy as np
import torch
from monitor.adapters import SablasDrone
from monitor.estimators import AnalyticEstimator

EVAL_STEPS = 50000  # Fewer than CBF bounds since we need pairwise comparisons


def compute_expected_reward(adapter: SablasDrone, state: torch.Tensor, obstacle: np.ndarray, noise: np.ndarray) -> float:
    """Compute E[R(x)] analytically by temporarily setting adapter state."""
    # Save current state
    original_state = adapter.state.copy()
    original_obstacle = adapter.obstacle.copy()
    original_noise = adapter.env.noise.copy()

    # Set to the historical state
    adapter.state = state.numpy()
    adapter.obstacle = obstacle
    adapter.env.noise = noise

    # Use analytic estimator (exploits small Jensen gap)
    estimator = AnalyticEstimator()
    _, reward, _, _ = estimator(adapter)

    # Restore state
    adapter.state = original_state
    adapter.obstacle = original_obstacle
    adapter.env.noise = original_noise

    return reward


def main():
    adapter = SablasDrone()

    # Collect states, obstacles, and noise from long run
    print(f"Collecting states over {EVAL_STEPS} steps...")
    states = []
    obstacles = []
    noises = []
    num_resets = 0

    for i in range(EVAL_STEPS):
        if adapter.done():
            adapter.reset()
            num_resets += 1

        states.append(torch.from_numpy(adapter.state.copy()).float())
        obstacles.append(adapter.obstacle.copy())
        noises.append(adapter.env.noise.copy())
        adapter.step()

        if (i + 1) % 10000 == 0:
            print(f"  Progress: {i + 1}/{EVAL_STEPS} steps, {num_resets} resets")

    print(f"Collected {len(states)} states across {num_resets} resets")

    # Compute E[R] for all states
    print("Computing E[R] for all states...")
    expected_rewards = []
    for i in range(len(states)):
        er = compute_expected_reward(adapter, states[i], obstacles[i], noises[i])
        expected_rewards.append(er)
        if (i + 1) % 10000 == 0:
            print(f"  Progress: {i + 1}/{len(states)}")

    # Compute ratios for temporally close pairs
    print("Computing Lipschitz ratios for temporally close pairs...")
    max_temporal_gap = 50  # Only consider pairs within 50 steps of each other
    ratios = []

    for i in range(len(states)):
        for j in range(i + 1, min(i + max_temporal_gap + 1, len(states))):
            dist = adapter.distance(states[i], states[j])
            if dist > 1e-6:  # avoid division by zero
                ratio = abs(expected_rewards[i] - expected_rewards[j]) / dist
                ratios.append(ratio)

        if (i + 1) % 10000 == 0:
            print(f"  Progress: {i + 1}/{len(states)}")

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
