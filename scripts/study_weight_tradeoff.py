"""
Study the DE/SE tradeoff for uniform weights over k nearest neighbors.

Setup:
- Sequence of observations X_0, ..., X_T (conceptually N(0,1) but we only need the theory)
- Temporal distance: d(x_i, x_j) = |i - j|
- Uniform weights over 2m neighbors (m on each side of target index t)
- SE = sqrt(σ² * log(2/δ) / (2m))  — concentration bound
- DE = γ * (m+1)/2                  — average distance times Lipschitz constant

Goal: Understand how m and γ interact, find optimal m* for various γ.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, log


def compute_se(m: int, sigma: float = 1.0, delta: float = 0.01) -> float:
    """Statistical error: concentration bound for 2m uniform-weighted samples."""
    if m == 0:
        return float('inf')
    return sqrt(sigma**2 * log(2/delta) / (2 * m))


def compute_de(m: int, gamma: float) -> float:
    """Discretization error: γ * average distance to 2m neighbors."""
    if m == 0:
        return 0.0
    # Neighbors at distances 1, 2, ..., m on each side
    # Average distance = (1 + 2 + ... + m) * 2 / (2m) = (m+1)/2
    return gamma * (m + 1) / 2


def compute_total_error(m: int, gamma: float, sigma: float = 1.0, delta: float = 0.01) -> float:
    """Total CI half-width: SE + DE."""
    return compute_se(m, sigma, delta) + compute_de(m, gamma)


def find_optimal_m(gamma: float, max_m: int = 1000, sigma: float = 1.0, delta: float = 0.01) -> tuple:
    """Find the m that minimizes total error."""
    best_m = 1
    best_error = float('inf')
    for m in range(1, max_m + 1):
        error = compute_total_error(m, gamma, sigma, delta)
        if error < best_error:
            best_error = error
            best_m = m
    return best_m, best_error


def theoretical_optimal_m(gamma: float, sigma: float = 1.0, delta: float = 0.01) -> float:
    """
    Analytical optimal m (continuous approximation).

    Total error ≈ c1/√m + c2*m where:
      c1 = σ * √(log(2/δ) / 2)
      c2 = γ / 2

    d/dm [c1/√m + c2*m] = -c1/(2*m^(3/2)) + c2 = 0
    => m^(3/2) = c1 / (2*c2)
    => m* = (c1 / (2*c2))^(2/3)
    """
    if gamma == 0:
        return float('inf')  # Use all data
    c1 = sigma * sqrt(log(2/delta) / 2)
    c2 = gamma / 2
    return (c1 / (2 * c2)) ** (2/3)


def main():
    # Parameters
    sigma = 1.0
    delta = 0.01
    max_m = 200

    # Gamma values to study
    gammas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]

    m_values = np.arange(1, max_m + 1)

    # ==========================================================================
    # Figure 1: SE, DE, and Total Error vs m for various γ
    # ==========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle('DE/SE Tradeoff: Uniform Weights over 2m Neighbors\n' +
                 f'(σ={sigma}, δ={delta})', fontsize=14, fontweight='bold')

    for idx, gamma in enumerate(gammas):
        ax = axes[idx // 3, idx % 3]

        se_values = [compute_se(m, sigma, delta) for m in m_values]
        de_values = [compute_de(m, gamma) for m in m_values]
        total_values = [s + d for s, d in zip(se_values, de_values)]

        ax.plot(m_values, se_values, 'b-', label='SE', linewidth=2)
        ax.plot(m_values, de_values, 'r-', label='DE', linewidth=2)
        ax.plot(m_values, total_values, 'k-', label='Total', linewidth=2.5)

        # Mark optimal m
        if gamma > 0:
            opt_m, opt_error = find_optimal_m(gamma, max_m, sigma, delta)
            ax.axvline(x=opt_m, color='green', linestyle='--', alpha=0.7)
            ax.plot(opt_m, opt_error, 'go', markersize=10, label=f'm*={opt_m}')
            ax.set_title(f'γ = {gamma}\nOptimal m* = {opt_m}, Error = {opt_error:.3f}')
        else:
            ax.set_title(f'γ = {gamma}\n(No DE, use all data)')

        ax.set_xlabel('m (neighbors on each side)')
        ax.set_ylabel('Error')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_m)

        # Set reasonable y-limit
        y_max = min(3.0, max(total_values[:50]) * 1.2)
        ax.set_ylim(0, y_max)

    plt.tight_layout()
    plt.savefig('scripts/output/weight_tradeoff_curves.png', dpi=150, bbox_inches='tight')
    print("Saved: scripts/output/weight_tradeoff_curves.png")

    # ==========================================================================
    # Figure 2: Optimal m* vs γ
    # ==========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('How Optimal m* Depends on Lipschitz Constant γ\n' +
                 f'(σ={sigma}, δ={delta})', fontsize=14, fontweight='bold')

    gamma_range = np.linspace(0.01, 1.0, 100)

    # Empirical optimal m*
    opt_m_empirical = [find_optimal_m(g, max_m=500, sigma=sigma, delta=delta)[0] for g in gamma_range]

    # Theoretical optimal m*
    opt_m_theoretical = [theoretical_optimal_m(g, sigma, delta) for g in gamma_range]

    # Left plot: m* vs γ
    ax = axes[0]
    ax.plot(gamma_range, opt_m_empirical, 'b-', linewidth=2, label='Empirical (discrete)')
    ax.plot(gamma_range, opt_m_theoretical, 'r--', linewidth=2, label='Theoretical (continuous)')
    ax.set_xlabel('γ (Lipschitz constant)')
    ax.set_ylabel('Optimal m*')
    ax.set_title('Optimal Neighborhood Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right plot: optimal error vs γ
    ax = axes[1]
    opt_errors = [find_optimal_m(g, max_m=500, sigma=sigma, delta=delta)[1] for g in gamma_range]
    ax.plot(gamma_range, opt_errors, 'k-', linewidth=2)
    ax.set_xlabel('γ (Lipschitz constant)')
    ax.set_ylabel('Optimal Total Error (SE + DE)')
    ax.set_title('Minimum Achievable CI Width')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('scripts/output/weight_tradeoff_optimal.png', dpi=150, bbox_inches='tight')
    print("Saved: scripts/output/weight_tradeoff_optimal.png")

    # ==========================================================================
    # Figure 3: Heatmap of total error over (m, γ) space
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(10, 8))

    m_range = np.arange(1, 151)
    gamma_range_heatmap = np.linspace(0.01, 0.5, 100)

    error_matrix = np.zeros((len(gamma_range_heatmap), len(m_range)))
    for i, g in enumerate(gamma_range_heatmap):
        for j, m in enumerate(m_range):
            error_matrix[i, j] = compute_total_error(m, g, sigma, delta)

    # Clip for better visualization
    error_matrix = np.clip(error_matrix, 0, 2.0)

    im = ax.imshow(error_matrix, aspect='auto', origin='lower', cmap='viridis_r',
                   extent=[m_range[0], m_range[-1], gamma_range_heatmap[0], gamma_range_heatmap[-1]])

    # Overlay optimal m* curve
    opt_m_curve = [find_optimal_m(g, max_m=150, sigma=sigma, delta=delta)[0] for g in gamma_range_heatmap]
    ax.plot(opt_m_curve, gamma_range_heatmap, 'r-', linewidth=2, label='Optimal m*')

    ax.set_xlabel('m (neighbors on each side)', fontsize=12)
    ax.set_ylabel('γ (Lipschitz constant)', fontsize=12)
    ax.set_title(f'Total Error Heatmap (σ={sigma}, δ={delta})\n' +
                 'Darker = Lower Error, Red Line = Optimal m*', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Total Error (SE + DE)', fontsize=11)

    plt.tight_layout()
    plt.savefig('scripts/output/weight_tradeoff_heatmap.png', dpi=150, bbox_inches='tight')
    print("Saved: scripts/output/weight_tradeoff_heatmap.png")

    # ==========================================================================
    # Print summary table
    # ==========================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY: Optimal m* for Various Lipschitz Constants")
    print("=" * 70)
    print(f"\n  Parameters: σ = {sigma}, δ = {delta}")
    print(f"  SE formula: sqrt(σ² * log(2/δ) / (2m))")
    print(f"  DE formula: γ * (m+1) / 2")
    print()
    print(f"  {'γ':>8} | {'m* (emp)':>10} | {'m* (theory)':>12} | {'SE':>8} | {'DE':>8} | {'Total':>8}")
    print("  " + "-" * 66)

    for gamma in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        opt_m, opt_error = find_optimal_m(gamma, max_m=1000, sigma=sigma, delta=delta)
        theory_m = theoretical_optimal_m(gamma, sigma, delta)
        se = compute_se(opt_m, sigma, delta)
        de = compute_de(opt_m, gamma)
        print(f"  {gamma:>8.2f} | {opt_m:>10d} | {theory_m:>12.1f} | {se:>8.4f} | {de:>8.4f} | {opt_error:>8.4f}")

    print()
    print("  Key insight: As γ increases, optimal m* decreases (use fewer, closer neighbors)")
    print("               to keep DE manageable, at the cost of higher SE.")
    print()
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    import os
    os.makedirs("scripts/output", exist_ok=True)
    main()
