"""Tests for NeuralCertificateMonitor scheduling behavior."""

from typing import Any, List, Optional, Tuple

import torch

from monitor import NeuralCertificateMonitor
from monitor.adapters.interface import DynamicalSystemAdapter
from monitor.estimators import Estimator


class DummyAdapter(DynamicalSystemAdapter):
    """Minimal deterministic adapter for monitor scheduling tests."""

    def __init__(self, max_steps: int):
        self.max_steps = max_steps
        self.step_count = 0
        self.state_history = [torch.tensor([0.0])]
        self.state = self.state_history[-1]

    def done(self) -> bool:
        return self.step_count >= self.max_steps

    def step(self) -> torch.Tensor:
        self.step_count += 1
        self.state = torch.tensor([float(self.step_count)])
        self.state_history.append(self.state)
        return self.state

    def get_reward_bounds(self) -> Tuple[float, float]:
        return (-1.0, 1.0)

    def get_reward(
        self, next_state: torch.Tensor, cur_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _ = cur_state
        return next_state.squeeze() * 0.0

    def get_certificate_value(self, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        if state is None:
            state = self.state
        return state.squeeze()

    def get_state_history(self) -> torch.Tensor:
        return torch.stack(self.state_history)

    def sample(self, state: Optional[torch.Tensor] = None, n_samples: int = 1) -> torch.Tensor:
        if state is None:
            state = self.state
        return state.unsqueeze(0).expand(n_samples, -1)

    def get_state_dim(self) -> int:
        return 1

    def distance(self, state1: torch.Tensor, state2: torch.Tensor) -> float:
        return float(torch.linalg.norm(state1 - state2))


class DummyEstimator(Estimator):
    def __init__(self):
        self.calls: List[int] = []

    def __call__(self, adapter: DummyAdapter):
        self.calls.append(adapter.step_count)
        return "?", 0.0, 0.0, {"step_count": adapter.step_count}


def test_monitor_stride_default_every_step():
    adapter = DummyAdapter(max_steps=5)
    estimator = DummyEstimator()
    monitor = NeuralCertificateMonitor(adapter, estimator)

    results = list(monitor.run())
    assert len(results) == 5
    assert estimator.calls == [0, 1, 2, 3, 4]
    assert adapter.step_count == 5


def test_monitor_stride_skips_observations_but_not_steps():
    adapter = DummyAdapter(max_steps=5)
    estimator = DummyEstimator()
    monitor = NeuralCertificateMonitor(adapter, estimator, monitor_stride=2)

    results = list(monitor.run())
    assert len(results) == 3
    assert estimator.calls == [0, 2, 4]
    assert adapter.step_count == 5


def test_monitor_stride_must_be_positive():
    adapter = DummyAdapter(max_steps=1)
    estimator = DummyEstimator()
    try:
        _ = NeuralCertificateMonitor(adapter, estimator, monitor_stride=0)
        assert False, "Expected ValueError for monitor_stride=0"
    except ValueError:
        pass
