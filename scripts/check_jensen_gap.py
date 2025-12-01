"""
Check how large the Jensen gap is: |E[V(Y)] - V(E[Y])| for the sablas CBF.

If this gap is small, we could use V(E[Y]) as a proxy for E[V(Y)] to get
a more precise estimate of E[R] for testing purposes.
"""

import numpy as np
import torch
from monitor.adapters import SablasDrone


def estimate_expected_next_state(adapter: SablasDrone, n_samples: int = 10000) -> np.ndarray:
    """
    Estimate E[Y] where Y ~ P(x) is the next state.

    The dynamics are: Y = x + (dsdt + noise) * dt
    where noise is sticky (95% keep, 5% resample from N(0, noise_std)).

    E[Y] = x + dsdt * dt + E[noise] * dt

    For sticky noise starting from current env.noise:
    - With prob 0.95: noise = env.noise (current)
    - With prob 0.05: noise ~ N(0, noise_std)

    So E[noise] = 0.95 * env.noise + 0.05 * 0 = 0.95 * env.noise
    (since E[N(0, std)] = 0)

    But actually let's just estimate via Monte Carlo to be safe.
    """
    state = adapter.state.copy()
    samples = adapter.sample(torch.from_numpy(state).float(), n_samples=n_samples)
    return samples.mean(dim=0).numpy()


def main():
    adapter = SablasDrone()

    n_mc_samples = 10000  # For E[V(Y)]
    n_steps = 500

    gaps = []
    relative_gaps = []
    ev_values = []
    ve_values = []

    for i in range(n_steps):
        if adapter.done():
            adapter.reset()
            continue

        # E[V(Y)] via Monte Carlo
        samples = adapter.sample(n_samples=n_mc_samples)
        v_samples = adapter.get_certificate_value(samples)
        e_v_y = float(v_samples.mean())

        # V(E[Y])
        e_y = estimate_expected_next_state(adapter, n_samples=n_mc_samples)
        v_e_y = float(adapter.get_certificate_value(torch.from_numpy(e_y).float()))

        gap = abs(e_v_y - v_e_y)
        gaps.append(gap)
        ev_values.append(e_v_y)
        ve_values.append(v_e_y)

        # Relative gap (compared to |E[V(Y)]|)
        if abs(e_v_y) > 0.01:
            relative_gaps.append(gap / abs(e_v_y))

        adapter.step()

        if (i + 1) % 100 == 0:
            print(f"Step {i+1}/{n_steps}")

    gaps = np.array(gaps)
    ev_values = np.array(ev_values)
    ve_values = np.array(ve_values)
    relative_gaps = np.array(relative_gaps)

    print(f"\n=== Jensen Gap Analysis (n={len(gaps)}) ===")
    print(f"\nAbsolute gap |E[V(Y)] - V(E[Y])|:")
    print(f"  Mean:   {gaps.mean():.6f}")
    print(f"  Std:    {gaps.std():.6f}")
    print(f"  Median: {np.median(gaps):.6f}")
    print(f"  95th:   {np.percentile(gaps, 95):.6f}")
    print(f"  Max:    {gaps.max():.6f}")

    print(f"\nRelative gap |E[V(Y)] - V(E[Y])| / |E[V(Y)]|:")
    print(f"  Mean:   {relative_gaps.mean():.4f}")
    print(f"  Median: {np.median(relative_gaps):.4f}")
    print(f"  95th:   {np.percentile(relative_gaps, 95):.4f}")

    print(f"\nE[V(Y)] statistics:")
    print(f"  Mean:   {ev_values.mean():.4f}")
    print(f"  Std:    {ev_values.std():.4f}")
    print(f"  Range:  [{ev_values.min():.4f}, {ev_values.max():.4f}]")

    print(f"\nV(E[Y]) statistics:")
    print(f"  Mean:   {ve_values.mean():.4f}")
    print(f"  Std:    {ve_values.std():.4f}")
    print(f"  Range:  [{ve_values.min():.4f}, {ve_values.max():.4f}]")

    # Compare to typical reward magnitude
    # Reward = V(x) - V(Y) - alpha(V(x)), so the V(Y) term matters
    # CI width is typically ~0.1-0.5, so gap should be << this to be useful
    print(f"\n=== Interpretation ===")
    print(f"For V(E[Y]) to be a useful proxy for E[V(Y)]:")
    print(f"  Gap should be << CI width (typically 0.1-0.5)")
    print(f"  Current mean gap: {gaps.mean():.4f}")
    if gaps.mean() < 0.05:
        print(f"  -> Gap is small, V(E[Y]) could be a reasonable proxy!")
    else:
        print(f"  -> Gap is significant, V(E[Y]) is not a good proxy.")


if __name__ == "__main__":
    main()
