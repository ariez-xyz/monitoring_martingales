#!/usr/bin/env python3
"""Compare violation rates for constant vs step-scaled noise.

Tests the intuition that constant noise causes more violations near the origin
(where steps are tiny) compared to step-scaled noise.

Note: This script adds noise manually to compare different strategies.
The adapter's built-in noise uses dt-scaled (Option B): radius = noise_level * dt.
"""
import torch
import numpy as np
from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum


def sample_uniform_ball(n_samples: int, dim: int) -> torch.Tensor:
    """Sample uniformly from unit ball."""
    direction = torch.randn(n_samples, dim)
    direction = direction / torch.norm(direction, dim=1, keepdim=True)
    radius = torch.rand(n_samples, 1) ** (1.0 / dim)
    return direction * radius


def run_experiment(adapter, noise_type: str, param: float, n_episodes: int = 30):
    """
    Run experiment collecting per-step data.

    Args:
        noise_type: 'constant' (radius = param) or 'step_scaled' (radius = param * step_size)
        param: noise radius (constant) or fraction of step size (step_scaled)
    """
    dt = adapter.dt
    data = []

    for ep in range(n_episodes):
        adapter.reset()

        for step in range(300):
            state = adapter.state.clone()
            dist_from_origin = float(torch.norm(state))

            # Compute deterministic next state
            state_batch = state.unsqueeze(0)
            with torch.no_grad():
                f, g = adapter.dynamics.control_affine_dynamics(state_batch)
                u = adapter.dynamics.u_nominal(state_batch)
                xdot = f.squeeze(-1) + (g @ u.unsqueeze(-1)).squeeze(-1)
                det_next = state + dt * xdot.squeeze(0)

            step_size = float(torch.norm(det_next - state))

            # Compute noise radius based on type
            if noise_type == 'constant':
                noise_radius = param
            else:  # step_scaled
                noise_radius = param * step_size

            # Add uniform ball noise
            noise = sample_uniform_ball(1, 2).squeeze(0) * noise_radius
            next_state = det_next + noise

            # Check violation
            V_cur = float(adapter.get_certificate_value(state))
            V_next = float(adapter.get_certificate_value(next_state))
            violated = V_next > V_cur

            data.append({
                'dist': dist_from_origin,
                'step_size': step_size,
                'noise_radius': noise_radius,
                'noise_to_step_ratio': noise_radius / step_size if step_size > 1e-6 else float('inf'),
                'violated': violated,
                'V_cur': V_cur,
                'V_next': V_next,
            })

            adapter.state = next_state
            adapter.state_history.append(next_state.clone())

            if adapter.done():
                break

    return data


def analyze_by_distance(data, bins=None):
    """Analyze violation rates binned by distance from origin."""
    if bins is None:
        bins = [0, 0.1, 0.3, 0.5, 1.0, 2.0, 4.0]

    print(f"{'Dist range':<15} {'Count':<8} {'Violations':<12} {'Avg step':<12} {'Noise/Step':<12}")
    print("-" * 65)

    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        in_bin = [d for d in data if lo <= d['dist'] < hi]
        if len(in_bin) == 0:
            continue

        count = len(in_bin)
        viol_rate = sum(d['violated'] for d in in_bin) / count
        avg_step = np.mean([d['step_size'] for d in in_bin])
        avg_ratio = np.mean([min(d['noise_to_step_ratio'], 100) for d in in_bin])

        print(f"[{lo:.1f}, {hi:.1f}){'':<6} {count:<8} {viol_rate:<12.1%} {avg_step:<12.4f} {avg_ratio:<12.2f}")

    total_viol = sum(d['violated'] for d in data) / len(data)
    print("-" * 65)
    print(f"{'TOTAL':<15} {len(data):<8} {total_viol:<12.1%}")

    return total_viol


def main():
    print("Comparing constant vs step-scaled noise\n")
    print("Hypothesis: constant noise causes more violations near origin")
    print("            (where steps are tiny but noise isn't)\n")

    adapter = NeuralCLBFPendulum()

    # Run experiments
    print("Running constant noise (radius=0.02)...")
    data_const = run_experiment(adapter, 'constant', 0.02)

    print("Running step-scaled noise (20% of step)...")
    data_scaled = run_experiment(adapter, 'step_scaled', 0.20)

    # Analyze
    print("\n" + "=" * 65)
    print("CONSTANT NOISE (radius = 0.02)")
    print("=" * 65)
    viol_const = analyze_by_distance(data_const)

    print("\n" + "=" * 65)
    print("STEP-SCALED NOISE (radius = 20% of step size)")
    print("=" * 65)
    viol_scaled = analyze_by_distance(data_scaled)

    # Summary
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)

    # Compare near-origin vs far-from-origin
    near_const = [d for d in data_const if d['dist'] < 0.3]
    far_const = [d for d in data_const if d['dist'] >= 0.5]
    near_scaled = [d for d in data_scaled if d['dist'] < 0.3]
    far_scaled = [d for d in data_scaled if d['dist'] >= 0.5]

    if near_const and far_const:
        near_viol_const = sum(d['violated'] for d in near_const) / len(near_const)
        far_viol_const = sum(d['violated'] for d in far_const) / len(far_const)
        print(f"Constant noise:    near origin {near_viol_const:.1%}, far {far_viol_const:.1%}, ratio {near_viol_const/far_viol_const:.1f}x")

    if near_scaled and far_scaled:
        near_viol_scaled = sum(d['violated'] for d in near_scaled) / len(near_scaled)
        far_viol_scaled = sum(d['violated'] for d in far_scaled) / len(far_scaled)
        print(f"Step-scaled noise: near origin {near_viol_scaled:.1%}, far {far_viol_scaled:.1%}, ratio {near_viol_scaled/far_viol_scaled:.1f}x")

    # Plot if possible
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Violation rate vs distance
        ax = axes[0]
        bins = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]

        for data, label, color in [(data_const, 'Constant', 'red'),
                                    (data_scaled, 'Step-scaled', 'blue')]:
            rates = []
            centers = []
            for i in range(len(bins) - 1):
                lo, hi = bins[i], bins[i + 1]
                in_bin = [d for d in data if lo <= d['dist'] < hi]
                if len(in_bin) > 10:
                    rates.append(sum(d['violated'] for d in in_bin) / len(in_bin))
                    centers.append((lo + hi) / 2)
            ax.plot(centers, [r * 100 for r in rates], 'o-', label=label, color=color)

        ax.set_xlabel('Distance from origin')
        ax.set_ylabel('Violation rate (%)')
        ax.set_title('Violation Rate vs Distance from Origin')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Noise/step ratio vs distance
        ax = axes[1]
        for data, label, color in [(data_const, 'Constant', 'red'),
                                    (data_scaled, 'Step-scaled', 'blue')]:
            ratios = []
            centers = []
            for i in range(len(bins) - 1):
                lo, hi = bins[i], bins[i + 1]
                in_bin = [d for d in data if lo <= d['dist'] < hi]
                if len(in_bin) > 10:
                    avg_ratio = np.mean([min(d['noise_to_step_ratio'], 10) for d in in_bin])
                    ratios.append(avg_ratio)
                    centers.append((lo + hi) / 2)
            ax.plot(centers, ratios, 'o-', label=label, color=color)

        ax.set_xlabel('Distance from origin')
        ax.set_ylabel('Noise radius / Step size')
        ax.set_title('Noise-to-Step Ratio vs Distance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='ratio=1')

        plt.tight_layout()
        plt.savefig('scripts/output/noise_scaling_comparison.png', dpi=150)
        print(f"\nPlot saved to scripts/output/noise_scaling_comparison.png")
        plt.show()

    except Exception as e:
        print(f"\nSkipping plot: {e}")


if __name__ == "__main__":
    main()
