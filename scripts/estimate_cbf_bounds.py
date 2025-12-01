"""
Empirically estimate tight bounds on CBF values.

Follows sablas test_drone.py approach: run for a large fixed number of steps,
continuing through episode resets to cover diverse configurations.
"""

import numpy as np
import torch
from monitor.adapters import SablasDrone

EVAL_STEPS = 1000000  # Match sablas config


def main():
    # Temporarily disable bounds checking by using wide bounds
    adapter = SablasDrone()
    adapter.certificate_bounds = (-100.0, 100.0)

    print(f"Collecting CBF values over {EVAL_STEPS} steps...")
    cbf_values_trajectory = []
    cbf_values_sampled = []
    n_samples_per_step = 20
    num_resets = 0

    for i in range(EVAL_STEPS):
        if adapter.done():
            adapter.reset()
            num_resets += 1

        # Current state CBF
        v = float(adapter.get_certificate_value())
        cbf_values_trajectory.append(v)

        # Sample some next states for additional coverage
        if i % 10 == 0:  # Sample every 10th step to save time
            sampled_states = adapter.sample(n_samples=n_samples_per_step)
            sampled_cbf = adapter.get_certificate_value(sampled_states)
            cbf_values_sampled.extend(sampled_cbf.tolist())

        adapter.step()

        if (i + 1) % 20000 == 0:
            print(f"  Progress: {i + 1}/{EVAL_STEPS} steps, {num_resets} resets")

    cbf_trajectory = np.array(cbf_values_trajectory)
    cbf_sampled = np.array(cbf_values_sampled)
    cbf_all = np.concatenate([cbf_trajectory, cbf_sampled])

    print(f"\n=== CBF Bounds Estimation ===")
    print(f"Total resets: {num_resets}")

    print(f"\nTrajectory states ({len(cbf_trajectory)} samples):")
    print(f"  Min:  {cbf_trajectory.min():.4f}")
    print(f"  Max:  {cbf_trajectory.max():.4f}")
    print(f"  Mean: {cbf_trajectory.mean():.4f}")

    print(f"\nSampled next states ({len(cbf_sampled)} samples):")
    print(f"  Min:  {cbf_sampled.min():.4f}")
    print(f"  Max:  {cbf_sampled.max():.4f}")
    print(f"  Mean: {cbf_sampled.mean():.4f}")

    print(f"\nCombined ({len(cbf_all)} samples):")
    print(f"  Min:     {cbf_all.min():.4f}")
    print(f"  Max:     {cbf_all.max():.4f}")
    print(f"  0.1th percentile:  {np.percentile(cbf_all, 0.1):.4f}")
    print(f"  99.9th percentile: {np.percentile(cbf_all, 99.9):.4f}")

    # Suggest bounds with some margin
    margin = 0.2
    suggested_lo = cbf_all.min() - margin
    suggested_hi = cbf_all.max() + margin

    print(f"\nSuggested bounds (with {margin} margin): ({suggested_lo:.2f}, {suggested_hi:.2f})")


if __name__ == "__main__":
    main()
