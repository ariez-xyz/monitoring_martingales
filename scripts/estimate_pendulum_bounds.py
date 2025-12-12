#!/usr/bin/env python3
"""Estimate certificate bounds for the neural_clbf inverted pendulum.

Samples states from the training region [-2, 2]² and computes V(x) for each.
"""
import torch
import numpy as np
from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum


def estimate_bounds_grid(adapter, n_points_per_dim=100):
    """Estimate V bounds by evaluating on a grid."""
    theta = torch.linspace(-2.0, 2.0, n_points_per_dim)
    theta_dot = torch.linspace(-2.0, 2.0, n_points_per_dim)

    # Create grid
    grid_theta, grid_theta_dot = torch.meshgrid(theta, theta_dot, indexing='ij')
    states = torch.stack([grid_theta.flatten(), grid_theta_dot.flatten()], dim=1)

    # Evaluate V for all states
    V_values = []
    batch_size = 1000
    for i in range(0, len(states), batch_size):
        batch = states[i:i+batch_size]
        with torch.no_grad():
            V = adapter.controller.V(batch)
        V_values.append(V)

    V_all = torch.cat(V_values).squeeze()

    return V_all, states, (grid_theta, grid_theta_dot)


def estimate_bounds_trajectory(adapter, n_episodes=100, max_steps=500):
    """Estimate V bounds by running trajectories."""
    V_values = []

    for ep in range(n_episodes):
        adapter.reset()  # Random initial state within bounds
        V_values.append(float(adapter.get_certificate_value()))

        for _ in range(max_steps):
            adapter.step()
            V_values.append(float(adapter.get_certificate_value()))
            if adapter.done():
                break

    return torch.tensor(V_values)


def main():
    print("Loading pendulum adapter...")
    adapter = NeuralCLBFPendulum()

    # Check state limits
    upper, lower = adapter.dynamics.state_limits
    print(f"\nState limits from dynamics:")
    print(f"  theta:     [{lower[0]:.1f}, {upper[0]:.1f}] rad")
    print(f"  theta_dot: [{lower[1]:.1f}, {upper[1]:.1f}] rad/s")

    # Method 1: Grid evaluation
    print(f"\n--- Grid Evaluation ({100}x{100} = 10,000 points) ---")
    V_grid, states, (grid_theta, grid_theta_dot) = estimate_bounds_grid(adapter)

    print(f"V range: [{V_grid.min():.4f}, {V_grid.max():.4f}]")
    print(f"V mean:  {V_grid.mean():.4f}")
    print(f"V std:   {V_grid.std():.4f}")

    # Find where max/min occur
    max_idx = V_grid.argmax()
    min_idx = V_grid.argmin()
    print(f"V_max at state: {states[max_idx].tolist()}")
    print(f"V_min at state: {states[min_idx].tolist()}")

    # Method 2: Trajectory evaluation (what states are actually visited)
    print(f"\n--- Trajectory Evaluation (100 episodes, up to 500 steps each) ---")
    V_traj = estimate_bounds_trajectory(adapter)

    print(f"V range: [{V_traj.min():.4f}, {V_traj.max():.4f}]")
    print(f"V mean:  {V_traj.mean():.4f}")
    print(f"V std:   {V_traj.std():.4f}")
    print(f"Total samples: {len(V_traj)}")

    # Suggested bounds
    grid_max = float(V_grid.max())
    traj_max = float(V_traj.max())
    suggested_max = max(grid_max, traj_max) * 1.1  # 10% margin

    print(f"\n--- Suggested Bounds ---")
    print(f"Certificate bounds: (0.0, {suggested_max:.1f})")
    print(f"Reward bounds: ({-suggested_max:.1f}, {suggested_max:.1f})")

    # Optional: visualize V landscape
    try:
        import matplotlib.pyplot as plt

        V_grid_2d = V_grid.reshape(100, 100).numpy()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Heatmap
        ax = axes[0]
        im = ax.imshow(V_grid_2d, extent=[-2, 2, -2, 2], origin='lower',
                       aspect='auto', cmap='viridis')
        ax.set_xlabel('θ (rad)')
        ax.set_ylabel('θ̇ (rad/s)')
        ax.set_title('CLF V(θ, θ̇)')
        ax.plot(0, 0, 'r*', markersize=15, label='Goal (V=0)')
        ax.legend()
        plt.colorbar(im, ax=ax, label='V')

        # Histogram of trajectory V values
        ax = axes[1]
        ax.hist(V_traj.numpy(), bins=50, density=True, alpha=0.7, label='Trajectory')
        ax.axvline(V_grid.max(), color='r', linestyle='--', label=f'Grid max: {grid_max:.2f}')
        ax.set_xlabel('V')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of V values during trajectories')
        ax.legend()

        plt.tight_layout()
        plt.savefig('scripts/output/pendulum_clf_bounds.png', dpi=150)
        print(f"\nPlot saved to scripts/output/pendulum_clf_bounds.png")
        plt.show()

    except Exception as e:
        print(f"\nSkipping visualization: {e}")


if __name__ == "__main__":
    main()
