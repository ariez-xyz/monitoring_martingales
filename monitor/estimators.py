from typing import Any, Dict, Literal, Tuple
from .adapters import DynamicalSystemAdapter
from .weighting import OptimalTemporalWeights, UniformWeights, WeightingStrategy
from math import log, sqrt, pi
from abc import ABC, abstractmethod
import torch


class Estimator(ABC):
    @abstractmethod
    def __call__(
        self, adapter: DynamicalSystemAdapter
    ) -> Tuple[Literal["T","F","?"], float, float, Dict[str, Any]]:
        """
        Estimate E[R] for the adapter's current state and return a confidence interval.

        The reward R = V(Y) - V(x) - α(V(x)) where Y ~ P(x) is the next state.
        E[R] ≤ 0 indicates the certificate condition is satisfied (safe).

        Args:
            adapter: The dynamical system adapter at its current state.

        Returns:
            A tuple (safety, lower, upper, info) where:
            - safety: "T" if upper < 0 (certified safe),
                      "F" if lower > 0 (certified unsafe),
                      "?" if CI straddles zero (inconclusive)
            - lower: Lower bound of confidence interval for E[R]
            - upper: Upper bound of confidence interval for E[R]
            - info: Estimator-specific metadata dict
        """
        pass


class HistoryEstimator(Estimator):
    """
    Estimates the local expected one-step drift from the observed drift history.

    This implements the centered sliding-window logic for uniform windows of 
    odd length m = 2r + 1. The estimator targets the center index of the window, 
    so it incurs a delay of r drift observations.

    The confidence interval radius is fixed to be

        R = B * sqrt(2 * log(2 / delta) / m)
            + B * (rho + 1) * (m^2 - 1) / (4m)

    where B is the one-step drift bound and rho is the one-step transition
    Wasserstein Lipschitz constant.

    This CI will have to be updated for continuous-time weights.
    """

    def __init__(self, weighting: WeightingStrategy, delta: float):
        """
        Args:
            weighting: Centered uniform weighting strategy over drift history.
            delta: Confidence level for the CI.
        """
        self.weighting = weighting
        self.delta = delta
        if not isinstance(weighting, UniformWeights) and not isinstance(weighting, OptimalTemporalWeights):
            raise ValueError("SE and DE are tuned for uniform weights (move their implementation to weighting classes for new types of weights)")

    def __call__(self, adapter):
        drift_history = adapter.get_drift_history()

        current_drift_index = len(drift_history) - 1
        weighting_radius = self.weighting.get_radius()
        target = current_drift_index - weighting_radius

        maybeWeights = self.weighting(drift_history, target)

        if maybeWeights is None:
            info = { 
                "current_drift_index": current_drift_index, 
                "target": target, 
                "delay": weighting_radius, 
                "reason": "insufficient history" 
            }
            return "?", float('-inf'), float('inf'), info

        # Aliases for paper notation
        gamma = adapter.get_drift_bound()
        m = 2 * weighting_radius + 1 # compute length of sliding window
        rho = adapter.get_transition_wasserstein_lipschitz()

        # \hat{d}_c^{(m)}
        weighted_mean = float(torch.dot(drift_history, maybeWeights))

        SE = gamma * sqrt((2*log(2/self.delta)) / m)
        DE = gamma * (rho + 1) * (m**2 -1)/(4*m)

        # R_{DT}^{ctr}(m)
        error = SE + DE

        # Final CI
        lower = weighted_mean - error
        upper = weighted_mean + error

        safety = "T" if upper <= 0 else "F" if lower > 0 else "?"
        info = { 
            "current_drift_index": current_drift_index,
            "target": target,
            "delay": weighting_radius,
            "weighted_mean": weighted_mean,
            "DE": DE,
            "SE": SE,
        }
        return safety, lower, upper, info


class SamplingEstimator(Estimator):
    """
    Estimates E[R] by sampling next states and using Hoeffding's inequality.

    Progressively samples more states until the CI is conclusive or
    the maximum sample count is reached.
    """

    def __init__(self, delta):
        self.delta = delta
        self.hoeffding_cache = {}

    def __call__(self, adapter, max_extra: int = 4096):
        sampled_states = adapter.sample(n_samples=512)
        rewards = adapter.get_reward(sampled_states)

        for n_extra in [0, 512, 1024, 2048, 4096]:
            if n_extra:
                sampled_states = adapter.sample(n_samples=n_extra)
                extra_rewards = adapter.get_reward(sampled_states)
                rewards = torch.cat([rewards, extra_rewards], dim=0)
            ci = self._hoeffding_ci(adapter, rewards.shape[0])
            mean = rewards.mean().item()
            lower, upper = (mean-ci, mean+ci)
            if upper < 0 or lower > 0 or n_extra >= max_extra:
                break

        safety = "T" if upper < 0 else "F" if lower > 0 else "?" # type: ignore[possibly-unbound]
        info = {"n_samples": int(rewards.shape[0])}
        return safety, lower, upper, info                        # type: ignore[possibly-unbound]

    def _hoeffding_ci(self, adapter, n):
        reward_bounds = adapter.get_reward_bounds()
        cache_key = (n, reward_bounds)
        cached = self.hoeffding_cache.get(cache_key)
        if cached: return cached
        range_squared = (reward_bounds[1] - reward_bounds[0]) ** 2
        log_term = log(pi / sqrt(6) * log(n))
        val = sqrt((1.064 * range_squared * 2 * log_term + log(2/self.delta)) / n)
        self.hoeffding_cache[cache_key] = val
        return val


class AnalyticEstimator(Estimator):
    """
    Computes E[R] ≈ V(E[Y]) - V(x) - α(V(x)) using analytic expected next state.

    This exploits the fact that for many systems, Jensen's gap is small:
    V(E[Y]) ≈ E[V(Y)], so we can compute expected reward without Monte Carlo.

    Requires the adapter to implement get_expected_next_state().
    Returns a degenerate CI (point estimate) since there's no sampling variance.
    """

    def __call__(self, adapter):
        expected_next_state = adapter.get_expected_next_state()
        # get_reward expects batched input
        expected_next_state = expected_next_state.unsqueeze(0)
        reward = float(adapter.get_reward(expected_next_state)) # V(E[Y])

        # Degenerate CI: point estimate with zero width
        lower, upper = reward, reward
        safety = "T" if reward < 0 else "F" if reward > 0 else "?"
        return safety, lower, upper, {}
