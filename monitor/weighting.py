from typing import Optional
from abc import ABC, abstractmethod
from math import log, sqrt

import torch
from .adapters import DynamicalSystemAdapter


class WeightingStrategy(ABC):
    """Abstract base class for computing weights over the observed drift history."""

    @abstractmethod
    def get_radius(self) -> int:
        pass

    @abstractmethod
    def __call__(self, drift_history: torch.Tensor, target: int) -> Optional[torch.Tensor]:
        """
        Compute weights for estimating the expected drift at drift index `target`.

        Args:
            drift_history: Tensor of realized one-step drifts.
            target: Target drift index to estimate (0-indexed).

        Returns:
            Tensor of weights with the same length as `drift_history`, or `None`
            if the required centered window is not yet available.
        """
        pass

    @abstractmethod
    def DE(self, adapter: DynamicalSystemAdapter) -> float:
        pass

    @abstractmethod
    def SE(self, adapter: DynamicalSystemAdapter, delta: float) -> float:
        pass



class UniformWeights(WeightingStrategy):
    """Sliding window of 2radius+1 uniform weights centered around target."""

    def __init__(self, radius: int):
        self.r = radius

    def get_radius(self) -> int:
        return self.r

    def __call__(self, drift_history: torch.Tensor, target: int) -> Optional[torch.Tensor]:
        if target - self.r < 0 or target + self.r >= len(drift_history):
            return None

        weights = torch.zeros_like(drift_history)
        start_idx = max(0, target - self.r)
        end_idx = min(len(drift_history) - 1, target + self.r)
        window_length = end_idx - start_idx + 1
        weights[start_idx:end_idx + 1] = 1.0 / window_length
        return weights

    def DE(self, adapter: DynamicalSystemAdapter) -> float:
        gamma = adapter.get_drift_bound()
        rho = adapter.get_transition_wasserstein_lipschitz()
        m = 2 * self.r + 1
        return gamma * (rho + 1) * (m**2 -1)/(4*m)

    def SE(self, adapter: DynamicalSystemAdapter, delta: float) -> float:
        gamma = adapter.get_drift_bound()
        m = 2 * self.r + 1
        return gamma * sqrt((2*log(2/delta)) / m)

class OptimalTemporalWeights(WeightingStrategy):
    """Centered uniform window with radius chosen from the discrete-time guide.

    The optimal odd window length m* for the centered estimator is given by

        m* ≈ (2 * sqrt(2 * log(2 / delta)) / (rho + 1))^(2/3)

    where rho is the one-step transition-kernel Lipschitz constant in W1.
    We convert that optimal window length to a centered window radius r by
    choosing the nearest integer to r = (m* - 1) / 2.
    """

    def __init__(self, adapter: DynamicalSystemAdapter, delta: float):
        """
        Args:
            adapter: Adapter supplying the transition-kernel Lipschitz constant.
            delta: Confidence level.
        """
        rho = adapter.get_transition_wasserstein_lipschitz()
        optimal_window_length = ((2 * sqrt(2 * log(2/delta)))/(rho + 1)) ** (2/3)
        optimal_radius = round((optimal_window_length-1)/2)
        self.weights = UniformWeights(radius=optimal_radius)

    def get_radius(self) -> int:
        return self.weights.get_radius()

    def __call__(self, drift_history: torch.Tensor, target: int) -> Optional[torch.Tensor]:
        return self.weights(drift_history, target)

    def DE(self, adapter: DynamicalSystemAdapter) -> float:
        return self.weights.DE(adapter)

    def SE(self, adapter: DynamicalSystemAdapter, delta: float) -> float:
        return self.weights.SE(adapter, delta)
