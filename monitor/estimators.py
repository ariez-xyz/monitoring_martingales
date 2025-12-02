from typing import Any, Literal, Tuple
from .adapters import DynamicalSystemAdapter
from math import log, sqrt, pi
from abc import ABC, abstractmethod
import torch


class Estimator(ABC):
    @abstractmethod
    def __call__(self, adapter: DynamicalSystemAdapter) -> Tuple[Literal["T","F","?"], float, float, Any]:
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
            - info: Estimator-specific metadata (e.g., number of samples used)
        """
        pass


class HistoryEstimator(Estimator):
    """
    Estimates E[R] using only the observed trajectory history (no sampling).

    Uses the Lipschitz assumption to bound discretization error and
    concentration inequalities for the statistical error.
    """

    def __init__(self):
        pass

    def __call__(self, adapter):
        # TODO: Implement history-based estimation
        raise NotImplementedError("HistoryEstimator not yet implemented")


class SamplingEstimator(Estimator):
    """
    Estimates E[R] by sampling next states and using Hoeffding's inequality.

    Progressively samples more states until the CI is conclusive or
    the maximum sample count is reached.
    """

    def __init__(self, delta: float = 0.01):
        self.delta = delta
        self.hoeffding_cache = {}

    def __call__(self, adapter):
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
            if upper < 0 or lower > 0:
                break

        safety = "T" if upper < 0 else "F" if lower > 0 else "?" # type: ignore[possibly-unbound]
        return safety, lower, upper, rewards.shape[0]            # type: ignore[possibly-unbound]

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
        return safety, lower, upper, None

