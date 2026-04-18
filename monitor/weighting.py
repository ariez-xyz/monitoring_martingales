import math
from typing import Optional
from abc import ABC, abstractmethod
from math import log, sqrt

import torch
from .adapters import DynamicalSystemAdapter
from .calibration import LipschitzConstantProvider


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
    def DE(self, adapter: DynamicalSystemAdapter, continuous: bool = False) -> float:
        pass

    @abstractmethod
    def SE(self, adapter: DynamicalSystemAdapter, delta: float, continuous: bool = False) -> float:
        pass

    @abstractmethod
    def AE(self, adapter: DynamicalSystemAdapter, continuous: bool = False) -> float:
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

    def DE(self, adapter: DynamicalSystemAdapter, continuous: bool = False) -> float:
        if continuous and "dt" not in adapter.bound_key().keys():
            raise KeyError(f"Can't compute continuous DE for {adapter.__class__} without dt. Got bound key {adapter.bound_key()}.")

        gamma = LipschitzConstantProvider.get_drift_bound(adapter)
        rho = LipschitzConstantProvider.get_transition_wasserstein_lipschitz(adapter, continuous)
        h = adapter.bound_key().get('dt', 1)
        m = 2 * self.r + 1 # window size

        if continuous: 
            rho_term = math.e ** (rho * h) - 1
            h_term = h
        else:
            rho_term = rho + 1
            h_term = 1

        DE = gamma * h_term * rho_term * (m**2 - 1)/(4*m)
        return DE

    def SE(self, adapter: DynamicalSystemAdapter, delta: float, continuous: bool = False) -> float:
        if continuous and "dt" not in adapter.bound_key().keys():
            raise KeyError(f"Can't compute continuous DE for {adapter.__class__} without dt. Got bound key {adapter.bound_key()}.")

        h = adapter.bound_key().get('dt', 1)
        gamma = LipschitzConstantProvider.get_drift_bound(adapter)
        m = 2 * self.r + 1 # window size

        if continuous: 
            h_term = h
        else:
            h_term = 1

        SE = gamma * sqrt((2*log(2/delta)) / m) * h_term
        return SE

    def AE(self, adapter: DynamicalSystemAdapter, continuous: bool = False) -> float:
        if not continuous: return 0

        if continuous and "dt" not in adapter.bound_key().keys():
            raise KeyError(f"Can't compute continuous DE for {adapter.__class__} without dt. Got bound key {adapter.bound_key()}.")

        gamma = LipschitzConstantProvider.get_drift_bound(adapter)
        rho = LipschitzConstantProvider.get_transition_wasserstein_lipschitz(adapter, continuous)
        h = adapter.bound_key().get('dt', 1)

        return rho * h * gamma / 2


class OptimalTemporalWeights(WeightingStrategy):
    """Centered uniform window with radius chosen from the discrete-time guide.

    The optimal odd window length m* for the centered estimator is given by

        m* ≈ (2 * sqrt(2 * log(2 / delta)) / (rho + 1))^(2/3)

    where rho is the one-step transition-kernel Lipschitz constant in W1.
    We convert that optimal window length to a centered window radius r by
    choosing the nearest integer to r = (m* - 1) / 2.
    """

    def __init__(self, adapter: DynamicalSystemAdapter, delta: float, continuous: bool = False):
        """
        Args:
            adapter: Adapter supplying the transition-kernel Lipschitz constant.
            delta: Confidence level.
        """
        if continuous and "dt" not in adapter.bound_key().keys():
            raise KeyError(f"Can't compute continuous DE for {adapter.__class__} without dt. Got bound key {adapter.bound_key()}.")

        rho = LipschitzConstantProvider.get_transition_wasserstein_lipschitz(adapter, continuous)
        h = adapter.bound_key().get('dt', 1)

        numerator = 2 * sqrt(2 * log(2/delta))

        if continuous:
            denominator = math.e ** (rho * h) - 1
        else: 
            denominator = rho + 1

        optimal_window_length = (numerator / denominator) ** (2/3)
        optimal_radius = round((optimal_window_length-1)/2)
        self.weights = UniformWeights(radius=optimal_radius)

    def get_radius(self) -> int:
        return self.weights.get_radius()

    def __call__(self, drift_history: torch.Tensor, target: int) -> Optional[torch.Tensor]:
        return self.weights(drift_history, target)

    def DE(self, adapter: DynamicalSystemAdapter, continuous: bool = False) -> float:
        return self.weights.DE(adapter, continuous)

    def SE(self, adapter: DynamicalSystemAdapter, delta: float, continuous: bool = False) -> float:
        return self.weights.SE(adapter, delta, continuous)

    def AE(self, adapter: DynamicalSystemAdapter, continuous: bool = False) -> float:
        return self.weights.AE(adapter, continuous)
