from typing import Any, List, Literal, Optional, Tuple
from .adapters import DynamicalSystemAdapter
from .weighting import WeightingStrategy, RecentWeights
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

    The CI is: weighted_mean ± (DE + SE) where:
    - DE = γ * Σ w_i * d(x_t, x_i)  (discretization error)
    - SE = sqrt(3.3 * V_t * (2*log(log(max(1, V_t))) + log(2/δ)))  (statistical error)
    - V_t = Σ (w_i * σ)²  (weighted variance, with σ = (b-a)/2)
    """

    def __init__(self, weighting: Optional[WeightingStrategy] = None, delta: float = 0.01):
        self.weighting = weighting or RecentWeights(k=10)
        self.delta = delta

    def __call__(self, adapter):
        history = adapter.state_history
        t = len(history) - 1

        # Need at least one prior state to have a reward
        if t < 1:
            return "?", float('-inf'), float('inf'), {"t": t, "reason": "insufficient history"}

        # Compute observed rewards R_i for each transition i -> i+1
        get_reward_at = lambda i: float(adapter.get_reward(history[i+1].unsqueeze(0), cur_state=history[i]))
        rewards = [get_reward_at(i) for i in range(t)]

        # Get weights for current timestep
        # We have t rewards (for transitions 0->1, ..., (t-1)->t)
        weights = self.weighting(adapter, t - 1)  # t weights for t rewards

        # Weighted mean of observed rewards
        weighted_mean = sum(w * r for w, r in zip(weights, rewards))

        # Discretization error: DE = γ * Σ w_i * d(x_t, x_i)
        gamma = adapter.get_lipschitz_constant()
        current_state = history[t]
        DE = 0.0
        for i, w in enumerate(weights):
            dist = adapter.distance(current_state, history[i])
            DE += w * dist
        DE *= gamma

        # Statistical error
        reward_bounds = adapter.get_reward_bounds()
        sigma = (reward_bounds[1] - reward_bounds[0]) / 2  # sub-Gaussian norm from bounded rewards

        # V_t = Σ (w_i * σ)²
        V_t = sum((w * sigma) ** 2 for w in weights)

        # SE = sqrt(3.3 * V_t * (2*log(log(max(1, V_t))) + log(2/δ)))
        # TODO: The formula has issues when V_t is small:
        #   - log(1) = 0, so log(log(1)) is undefined (log(0))
        #   - For small inner_log, 2*log(inner_log) can be negative enough to make
        #     the entire sqrt argument negative (depends on delta)
        #   Using max(e, V_t) ensures inner_log >= 1, so log(inner_log) >= 0, avoiding both issues.
        #   This is conservative but may not match the theoretical formula exactly.
        if V_t > 0:
            inner_log = log(max(2.718281828, V_t))  # >= 1
            SE = sqrt(3.3 * V_t * (2 * log(inner_log) + log(2 / self.delta)))
        else:
            SE = 0.0

        # Final CI
        lower = weighted_mean - DE - SE
        upper = weighted_mean + DE + SE

        safety = "T" if upper < 0 else "F" if lower > 0 else "?"
        return safety, lower, upper, {"t": t, "DE": DE, "SE": SE, "V_t": V_t}


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

