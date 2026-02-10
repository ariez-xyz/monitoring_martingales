from typing import List, Optional
from abc import ABC, abstractmethod
import math
import warnings
from .adapters import DynamicalSystemAdapter


class WeightingStrategy(ABC):
    """Abstract base class for computing weights over the trajectory history."""

    @abstractmethod
    def __call__(self, adapter: DynamicalSystemAdapter, t: int) -> List[float]:
        """
        Compute weights for estimating E[R] at target timestep t.

        Args:
            adapter: The dynamical system adapter with state history.
            t: Target timestep to estimate (0-indexed).

        Returns:
            List of weights of length len(adapter.state_history), summing to 1.
            Non-zero weights are around index t (exact indices depend on strategy).
        """
        pass


class UniformWeights(WeightingStrategy):
    """Uniform weights over [0, t]: w_i = 1/(t+1) for i <= t, 0 otherwise."""

    def __call__(self, adapter: DynamicalSystemAdapter, t: int) -> List[float]:
        history_len = len(adapter.state_history)
        weights = [0.0] * history_len
        for i in range(t + 1):
            weights[i] = 1.0 / (t + 1)
        return weights


class RecentWeights(WeightingStrategy):
    """Uniform weights over the k most recent states up to t."""

    def __init__(self, k: int = 10):
        self.k = k

    def __call__(self, adapter: DynamicalSystemAdapter, t: int) -> List[float]:
        history_len = len(adapter.state_history)
        k = min(self.k, t + 1)
        start_idx = t - k + 1

        weights = [0.0] * history_len
        for i in range(start_idx, t + 1):
            weights[i] = 1.0 / k
        return weights


class OptimalTemporalWeights(WeightingStrategy):
    """Optimal uniform weights over 2m temporal neighbors (m on each side).

    Computes optimal m that minimizes total error = DE + SE where:
    - DE (discretization error) = γ · (m+1)/2, increases with m
    - SE (statistical error) = c₁ / √m, decreases with m

    Optimal m* = (c₁/γ)^(2/3)

    Where:
    - c₁ = σ · √(log(2/δ)/2)
    - γ = Lipschitz constant for expected rewards
    - σ = sub-Gaussian parameter (for bounded rewards, σ = (b-a)/2)
    - δ = confidence level
    """

    def __init__(self, delta: float = 0.05, min_m: int = 1, max_m: Optional[int] = None):
        """
        Args:
            delta: Confidence level (default 0.05 for 95% confidence)
            min_m: Minimum number of neighbors on each side (default 1)
            max_m: Maximum number of neighbors on each side (None = no limit)
        """
        self.delta = delta
        self.min_m = min_m
        self.max_m = max_m
        self._cached_m: Optional[int] = None

    def compute_optimal_m(self, adapter: DynamicalSystemAdapter) -> int:
        """Compute optimal m based on adapter parameters."""
        # Get parameters from adapter
        gamma = adapter.get_lipschitz_constant()
        reward_lo, reward_hi = adapter.get_reward_bounds()

        # Sub-Gaussian parameter for bounded rewards
        sigma = (reward_hi - reward_lo) / 2.0

        # c1 = σ · √(log(2/δ)/2)
        c1 = sigma * math.sqrt(math.log(2.0 / self.delta) / 2.0)

        # Optimal m* = (c1/γ)^(2/3)
        if gamma <= 0:
            # If gamma is 0 or negative, use max possible m
            m_star = float('inf')
        else:
            m_star = (c1 / gamma) ** (2.0 / 3.0)

        # Round to integer and apply bounds
        m = max(self.min_m, int(round(m_star)))
        if self.max_m is not None:
            m = min(m, self.max_m)

        return m

    def __call__(self, adapter: DynamicalSystemAdapter, t: int) -> List[float]:
        """Compute weights using optimal m temporal neighbors on each side.

        Args:
            adapter: The dynamical system adapter with state history.
            t: Target timestep to estimate (0-indexed).

        Returns:
            List of weights with length = len(adapter.state_history).
            Non-zero weights are centered around t, using lookahead if available.
        """
        # Get available history length from adapter
        history_len = len(adapter.state_history)
        T = history_len - 1  # Latest available index

        # Compute optimal m
        m = self.compute_optimal_m(adapter)

        # Symmetric neighbors around t, bounded by available history
        if t - m < 0 or t + m >= T:
            warnings.warn("Sequence truncated due to boundary conditions", UserWarning)
        start_idx = max(0, t - m)
        end_idx = min(T, t + m)

        num_weighted = end_idx - start_idx + 1

        weights = [0.0] * history_len
        for i in range(start_idx, end_idx + 1):
            weights[i] = 1.0 / num_weighted

        return weights

    def get_optimal_m(self, adapter: DynamicalSystemAdapter) -> int:
        """Get the computed optimal m value (useful for debugging/logging)."""
        return self.compute_optimal_m(adapter)
