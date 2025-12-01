from typing import Any, Literal, Tuple
from .adapters import DynamicalSystemAdapter
from math import log, sqrt, pi
from abc import ABC, abstractmethod
import torch

DELTA = 0.01


class Estimator(ABC):
    @abstractmethod
    def __call__(self, adapter) -> Tuple[Literal["T","F","?"], float, float, Any]:
        pass


class NeuralCertificateMonitor:
    def __init__(self, adapter: DynamicalSystemAdapter, estimator: Estimator):
        self.adapter = adapter
        self.estimator = estimator

    def run(self):
        try:
            while not self.adapter.done():
                print(self.estimator(self.adapter))
                self.adapter.step()
        except KeyboardInterrupt:
            pass


class HistoryEstimator(Estimator):
    def __init__(self):
        pass

    def __call__(self, adapter):
        safety = "T" if upper < 0 else "F" if lower > 0 else "?" # type: ignore[possibly-unbound]
        return safety, lower, upper, None                        # type: ignore[possibly-unbound]


class SamplingEstimator(Estimator):
    def __init__(self):
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
        range_sq = (reward_bounds[1] - reward_bounds[0]) ** 2
        log_term = log(pi / sqrt(6) * log(n))
        val = sqrt((1.064 * range_sq * 2 * log_term + log(2/DELTA)) / n)
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
        reward = float(adapter.get_reward(expected_next_state))

        # Degenerate CI: point estimate with zero width
        lower, upper = reward, reward
        safety = "T" if reward < 0 else "F" if reward > 0 else "?"
        return safety, lower, upper, None

