"""Reusable test fixtures for fast, deterministic monitor and estimator tests."""

from typing import List, Optional, Tuple
import torch
from monitor.adapters.interface import DynamicalSystemAdapter


class NormalIncrementAdapter(DynamicalSystemAdapter):
    """A 1D stochastic adapter with additive normal one-step dynamics.

    State dynamics:
        X_{t+1} = X_t + Z_t
        Z_t ~ Normal(mean, sigma)

    Reward:
        R_t = X_{t+1} - X_t

    This fixture is intended for fast estimator tests where the next-state
    distribution is known analytically and rollout randomness is controlled by a
    local seeded RNG.
    """

    def __init__(
        self,
        mean: float = -0.1,
        sigma: float = 0.05,
        initial_value: float = 10,
        clamp_at_sigma: float = 3,
        max_steps: int = 1000,
        base_seed: int = 0,
    ) -> None:
        if sigma < 0: raise ValueError("sigma must be nonnegative")
        if clamp_at_sigma < 0: raise ValueError("clamp_at_sigma must be nonnegative")
        self.mean = float(mean)
        self.sigma = float(sigma)
        self.initial_value = float(initial_value)
        self.max_steps = int(max_steps)
        self.base_seed = int(base_seed)
        self.clamp_at_sigma = clamp_at_sigma
        self._done = False
        self.reset(seed=self.base_seed)

    def reset(self, seed: Optional[int] = None):
        if seed is None:
            seed = self.base_seed
        self._rng = torch.Generator()
        self._rng.manual_seed(int(seed))
        self.step_count = 0
        self.state = torch.tensor([float(self.initial_value)], dtype=torch.float32)
        self.state_history: List[torch.Tensor] = [self.state.clone()]
        self.drift_history: List[float] = []

    def done(self) -> bool:
        return self.step_count >= self.max_steps or self._done

    def _draw_increments(self, n: int) -> torch.Tensor:
        mean = torch.tensor(float(self.mean), dtype=torch.float32)
        std = torch.tensor(float(self.sigma), dtype=torch.float32)
        increments = mean + std * torch.randn(n, generator=self._rng)
        if self.sigma > 0:
            lower = self.mean - self.clamp_at_sigma * self.sigma
            upper = self.mean + self.clamp_at_sigma * self.sigma
            increments = torch.clamp(increments, lower, upper)
        return increments

    def step(self) -> torch.Tensor:
        increment = self._draw_increments(1)[0]
        if self.state + increment < 0: 
            self._done = True

        if self.done():
            return self.state.clone()

        self.step_count += 1
        self.state = self.state + increment
        self.state_history.append(self.state.clone())
        self.drift_history.append(float(increment))
        return self.state.clone()

    def get_drift(
        self, next_state: torch.Tensor, cur_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if cur_state is None:
            cur_state = self.state
        if next_state.dim() == 1:
            next_state = next_state.unsqueeze(0)
        if cur_state.dim() == 1:
            cur_state = cur_state.unsqueeze(0)
        return next_state[..., 0] - cur_state[..., 0]

    def get_certificate_value(self, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        if state is None:
            state = self.state
        if state.dim() == 1:
            return state[0]
        return state[..., 0]

    def get_state_history(self) -> torch.Tensor:
        return torch.stack(self.state_history)

    def get_drift_history(self) -> torch.Tensor:
        return torch.tensor(self.drift_history, dtype=torch.float32)

    def sample(self, state: Optional[torch.Tensor] = None, n_samples: int = 1) -> torch.Tensor:
        if state is None:
            state = self.state
        if state.dim() == 1:
            state = state.unsqueeze(0)
        increments = self._draw_increments(n_samples).unsqueeze(-1)
        return state[0].unsqueeze(0) + increments

    def get_state_dim(self) -> int:
        return 1

    def distance(self, state1: torch.Tensor, state2: torch.Tensor) -> float:
        return float(torch.abs(state1[0] - state2[0]))

    def get_drift_bound(self) -> float:
        lo, hi = (
            self.mean - self.clamp_at_sigma * self.sigma,
            self.mean + self.clamp_at_sigma * self.sigma,
        )
        return max(abs(lo), abs(hi))

    def get_transition_wasserstein_lipschitz(self) -> float:
        # Under additive state-independent noise, P(x) is a translated copy of
        # the same increment law, so W1(P(x), P(x')) = |x - x'|.
        return 1.0

    def get_expected_next_state(self, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        if state is None:
            state = self.state
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return state[0] + float(self.mean)

def check_close(a: torch.Tensor, b: torch.Tensor, msg: str) -> None:
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-6), msg
