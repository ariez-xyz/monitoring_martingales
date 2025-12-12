from typing import List
from abc import ABC, abstractmethod
from .adapters import DynamicalSystemAdapter


class WeightingStrategy(ABC):
    """Abstract base class for computing weights over the trajectory history."""

    @abstractmethod
    def __call__(self, adapter: DynamicalSystemAdapter, t: int) -> List[float]:
        """
        Compute weights for timestep t.

        Args:
            adapter: The dynamical system adapter with state history.
            t: Current timestep (0-indexed).

        Returns:
            List of weights [w_0, w_1, ..., w_t] summing to 1.
        """
        pass


class UniformWeights(WeightingStrategy):
    """Uniform weights over all history: w_i = 1/(t+1)."""

    def __call__(self, adapter: DynamicalSystemAdapter, t: int) -> List[float]:
        n = t + 1
        return [1.0 / n] * n


class RecentWeights(WeightingStrategy):
    """Uniform weights over the k most recent states."""

    def __init__(self, k: int = 10):
        self.k = k

    def __call__(self, adapter: DynamicalSystemAdapter, t: int) -> List[float]:
        n = t + 1
        k = min(self.k, n)
        weights = [0.0] * n
        for i in range(n - k, n):
            weights[i] = 1.0 / k
        return weights
