#!/usr/bin/env python3
"""Estimate Lipschitz constant γ and reward bounds for the pendulum adapter.

Two notions of γ:
1. Spatial γ = |ΔR| / d(x,x')  - reward change per unit state distance
2. Temporal γ = |ΔR| / k       - reward change per timestep

The optimal m formula uses γ in: DE = γ × Σ w_i × distance_i
- If distance is spatial, use spatial γ
- If distance is temporal (k steps), use temporal γ
"""
import torch
import numpy as np
import math
from collections import defaultdict
from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum


def collect_trajectories(adapter, n_episodes=100, max_steps=300):
    """Collect trajectory data: states and rewards per episode."""
    trajectories = []

    for ep in range(n_episodes):
        adapter.reset()
        states = []
        rewards = []

        for step in range(max_steps):
            state = adapter.state.clone().numpy()

            # Compute reward R(x) = V(x) - V(next)
            state_batch = adapter.state.unsqueeze(0)
            with torch.no_grad():
                f, g = adapter.dynamics.control_affine_dynamics(state_batch)
                u = adapter.dynamics.u_nominal(state_batch)
                xdot = f.squeeze(-1) + (g @ u.unsqueeze(-1)).squeeze(-1)
                next_state = adapter.state + adapter.dt * xdot.squeeze(0)

            V_cur = float(adapter.get_certificate_value(adapter.state))
            V_next = float(adapter.get_certificate_value(next_state))
            reward = V_cur - V_next

            states.append(state)
            rewards.append(reward)

            adapter.step()
            if adapter.done():
                break

        if len(states) > 30:
            trajectories.append((np.array(states), np.array(rewards)))

    return trajectories


def compute_gamma_stats(trajectories, max_k=20):
    """Compute both spatial and temporal γ for temporal neighbors."""
    spatial_ratios = []  # |ΔR| / dist
    temporal_ratios = [] # |ΔR| / k
    raw_diffs = []       # |ΔR|
    all_rewards = []

    by_k = defaultdict(lambda: {'spatial': [], 'temporal': [], 'raw': [], 'dist': []})

    for states, rewards in trajectories:
        all_rewards.extend(rewards)
        n = len(states)

        for k in range(1, min(max_k + 1, n)):
            for t in range(k, n):
                dist = np.linalg.norm(states[t] - states[t-k])
                if dist < 1e-6:
                    continue

                diff = abs(rewards[t] - rewards[t-k])

                spatial_ratios.append(diff / dist)
                temporal_ratios.append(diff / k)
                raw_diffs.append(diff)

                by_k[k]['spatial'].append(diff / dist)
                by_k[k]['temporal'].append(diff / k)
                by_k[k]['raw'].append(diff)
                by_k[k]['dist'].append(dist)

    return {
        'spatial': np.array(spatial_ratios),
        'temporal': np.array(temporal_ratios),
        'raw': np.array(raw_diffs),
        'rewards': np.array(all_rewards),
        'by_k': by_k
    }


def compute_optimal_m(gamma, sigma, delta=0.05):
    """Compute optimal m given γ and σ."""
    c1 = sigma * math.sqrt(math.log(2.0 / delta) / 2.0)
    if gamma <= 0:
        return float('inf'), c1
    m_star = (c1 / gamma) ** (2.0 / 3.0)
    return m_star, c1


def main():
    print("=" * 70)
    print("ESTIMATING γ AND OPTIMAL m FOR PENDULUM")
    print("=" * 70)

    adapter = NeuralCLBFPendulum()
    dt = adapter.dt
    cert_lo, cert_hi = adapter.certificate_bounds

    print(f"\ndt = {dt}")
    print(f"Certificate bounds: ({cert_lo}, {cert_hi})")

    # Collect data
    print(f"\nCollecting trajectories...")
    trajectories = collect_trajectories(adapter, n_episodes=100, max_steps=300)
    total_samples = sum(len(r) for _, r in trajectories)
    print(f"Collected {len(trajectories)} trajectories, {total_samples} samples")

    # Compute γ statistics
    stats = compute_gamma_stats(trajectories, max_k=20)

    # Reward bounds
    r_min, r_max = stats['rewards'].min(), stats['rewards'].max()
    sigma_empirical = (r_max - r_min) / 2
    sigma_cert = (cert_hi - cert_lo) / 2

    print(f"\n" + "-" * 70)
    print("REWARD BOUNDS")
    print("-" * 70)
    print(f"Empirical: [{r_min:.4f}, {r_max:.4f}]  →  σ = {sigma_empirical:.4f}")
    print(f"Certificate: [{cert_lo - cert_hi}, {cert_hi - cert_lo}]  →  σ = {sigma_cert:.1f}")

    # γ statistics
    gamma_spatial_99 = np.percentile(stats['spatial'], 99)
    gamma_spatial_max = stats['spatial'].max()
    gamma_temporal_99 = np.percentile(stats['temporal'], 99)
    gamma_temporal_max = stats['temporal'].max()

    print(f"\n" + "-" * 70)
    print("LIPSCHITZ γ (99th percentile)")
    print("-" * 70)
    print(f"Spatial  (|ΔR|/dist): {gamma_spatial_99:.4f}  (max: {gamma_spatial_max:.4f})")
    print(f"Temporal (|ΔR|/k):    {gamma_temporal_99:.4f}  (max: {gamma_temporal_max:.4f})")

    # Show by k
    print(f"\n" + "-" * 70)
    print("γ BY TEMPORAL DISTANCE k")
    print("-" * 70)
    print(f"{'k':<4} {'|ΔR|/dist':<12} {'|ΔR|/k':<12} {'|ΔR|':<12} {'dist':<10}")
    print(f"{'':4} {'(spatial)':<12} {'(temporal)':<12} {'(raw)':<12} {'(mean)':<10}")
    print("-" * 54)
    for k in [1, 2, 5, 10, 15, 20]:
        if k in stats['by_k'] and len(stats['by_k'][k]['spatial']) > 10:
            bk = stats['by_k'][k]
            print(f"{k:<4} {np.percentile(bk['spatial'], 99):<12.4f} "
                  f"{np.percentile(bk['temporal'], 99):<12.4f} "
                  f"{np.percentile(bk['raw'], 99):<12.4f} "
                  f"{np.mean(bk['dist']):<10.4f}")

    # Optimal m for all combinations
    print(f"\n" + "-" * 70)
    print("OPTIMAL m* = (c1/γ)^(2/3)")
    print("-" * 70)
    print(f"{'γ type':<12} {'σ type':<12} {'γ':<10} {'σ':<10} {'c1':<10} {'m*':<10}")
    print("-" * 64)

    for gamma_name, gamma in [('spatial', gamma_spatial_99), ('temporal', gamma_temporal_99)]:
        for sigma_name, sigma in [('empirical', sigma_empirical), ('certificate', sigma_cert)]:
            m_star, c1 = compute_optimal_m(gamma, sigma)
            print(f"{gamma_name:<12} {sigma_name:<12} {gamma:<10.4f} {sigma:<10.4f} {c1:<10.4f} {m_star:<10.1f}")

    # Example pairs
    print(f"\n" + "-" * 70)
    print("EXAMPLE ADJACENT PAIRS (k=1)")
    print("-" * 70)
    print(f"{'x_t':<22} {'x_{t-1}':<22} {'R_t':<8} {'R_{t-1}':<8} {'|ΔR|':<8} {'dist':<8}")
    print("-" * 70)

    states, rewards = trajectories[0]
    for t in range(1, min(8, len(states))):
        dist = np.linalg.norm(states[t] - states[t-1])
        diff = abs(rewards[t] - rewards[t-1])
        print(f"{str(np.round(states[t], 2)):<22} {str(np.round(states[t-1], 2)):<22} "
              f"{rewards[t]:<8.4f} {rewards[t-1]:<8.4f} {diff:<8.4f} {dist:<8.4f}")

    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"For temporal weighting with empirical bounds:")
    m_recommended, _ = compute_optimal_m(gamma_temporal_99, sigma_empirical)
    print(f"  γ = {gamma_temporal_99:.4f} (temporal, 99th)")
    print(f"  σ = {sigma_empirical:.4f} (empirical)")
    print(f"  m* = {m_recommended:.1f}  →  use m = {int(round(m_recommended))}")

    print(f"\nFor conservative bounds:")
    m_conservative, _ = compute_optimal_m(gamma_temporal_99, sigma_cert)
    print(f"  γ = {gamma_temporal_99:.4f} (temporal, 99th)")
    print(f"  σ = {sigma_cert:.1f} (certificate)")
    print(f"  m* = {m_conservative:.1f}  →  use m = {int(round(m_conservative))}")


if __name__ == "__main__":
    main()
