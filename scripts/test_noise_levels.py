#!/usr/bin/env python3
"""Test CLF behavior under different noise levels.

For a CLF, we want E[V(x) - V(Y)] > 0 (V decreasing in expectation).
This script tests at what noise level this condition starts failing.

Noise model: uniform ball with radius = noise_scale * dt
"""
import torch
import numpy as np
from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum


def test_noise_level(noise_scale: float, n_episodes: int = 20, max_steps: int = 200):
    """Test CLF performance at a given noise level.

    Returns:
        mean_reward: Average reward (V(x) - V(Y)) across all steps
        violation_rate: Fraction of steps where reward < 0
        clf_decreasing: Whether V decreases on average over trajectories
    """
    adapter = NeuralCLBFPendulum(noise_scale=noise_scale)

    all_rewards = []
    violations = 0
    total_steps = 0

    initial_V_sum = 0
    final_V_sum = 0

    for ep in range(n_episodes):
        adapter.reset()
        initial_V_sum += float(adapter.get_certificate_value())
        prev_state = adapter.state.clone()

        for step in range(max_steps):
            next_state = adapter.step()
            reward = float(adapter.get_reward(next_state, prev_state))
            all_rewards.append(reward)

            if reward < 0:
                violations += 1
            total_steps += 1

            prev_state = next_state.clone()

            if adapter.done():
                break

        final_V_sum += float(adapter.get_certificate_value())

    mean_reward = np.mean(all_rewards)
    violation_rate = violations / total_steps
    mean_V_decrease = (initial_V_sum - final_V_sum) / n_episodes

    return {
        'mean_reward': mean_reward,
        'std_reward': np.std(all_rewards),
        'violation_rate': violation_rate,
        'mean_V_decrease': mean_V_decrease,
        'total_steps': total_steps,
    }


def main():
    print("Testing CLF behavior under different noise levels\n")
    print("For CLF: want E[V(x) - V(Y)] > 0 (V decreasing)\n")
    print("Noise model: uniform ball, radius = noise_scale * dt\n")

    # Get dt for reference
    adapter = NeuralCLBFPendulum()
    dt = adapter.dt
    print(f"dt = {dt}\n")

    # Test range of noise scales
    # noise_scale=1 means radius=dt, noise_scale=10 means radius=10*dt
    noise_scales = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

    results = []
    for scale in noise_scales:
        radius = scale * dt
        print(f"Testing noise_scale={scale:.1f} (radius={radius:.4f})...", end=" ", flush=True)
        res = test_noise_level(scale)
        results.append((scale, res))
        print(f"mean_reward={res['mean_reward']:.4f}, violations={res['violation_rate']:.1%}")

    # Summary table
    print("\n" + "=" * 85)
    print(f"{'noise_scale':>12} | {'radius':>8} | {'E[reward]':>10} | {'std':>8} | {'violations':>10} | {'V decrease':>10}")
    print("-" * 85)

    for scale, res in results:
        radius = scale * dt
        status = "OK" if res['mean_reward'] > 0 else "FAIL"
        print(f"{scale:>12.1f} | {radius:>8.4f} | {res['mean_reward']:>10.4f} | {res['std_reward']:>8.4f} | "
              f"{res['violation_rate']:>9.1%} | {res['mean_V_decrease']:>10.4f}  [{status}]")

    # Find threshold
    print("\n" + "=" * 85)
    threshold = None
    for scale, res in results:
        if res['mean_reward'] <= 0:
            threshold = scale
            break

    if threshold:
        print(f"CLF condition fails at noise_scale >= {threshold:.1f} (radius >= {threshold * dt:.4f})")
    else:
        print("CLF condition holds for all tested noise levels!")

    # Plot results
    try:
        import matplotlib.pyplot as plt

        scales = [s for s, _ in results]
        radii = [s * dt for s in scales]
        mean_rewards = [r['mean_reward'] for _, r in results]
        std_rewards = [r['std_reward'] for _, r in results]
        violation_rates = [r['violation_rate'] for _, r in results]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Mean reward vs noise
        ax = axes[0]
        ax.errorbar(radii, mean_rewards, yerr=std_rewards, marker='o', capsize=5)
        ax.axhline(y=0, color='r', linestyle='--', label='CLF threshold')
        ax.fill_between(radii, 0, max(mean_rewards) * 1.1, alpha=0.1, color='green', label='CLF satisfied')
        if min(mean_rewards) < 0:
            ax.fill_between(radii, min(mean_rewards) * 1.1, 0, alpha=0.1, color='red', label='CLF violated')
        ax.set_xlabel('Noise radius')
        ax.set_ylabel('E[V(x) - V(Y)]')
        ax.set_title('Mean Reward vs Noise Level')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Violation rate vs noise
        ax = axes[1]
        ax.plot(radii, [v * 100 for v in violation_rates], marker='s', color='orange')
        ax.set_xlabel('Noise radius')
        ax.set_ylabel('Violation rate (%)')
        ax.set_title('Per-step CLF Violations vs Noise Level')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('scripts/output/pendulum_noise_test.png', dpi=150)
        print(f"\nPlot saved to scripts/output/pendulum_noise_test.png")
        plt.show()

    except Exception as e:
        print(f"\nSkipping visualization: {e}")


if __name__ == "__main__":
    main()
