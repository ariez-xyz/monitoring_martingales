"""Tests for the unified reward sign convention across adapters.

Contract:
- reward <= 0 means the certificate condition is satisfied.
"""

import torch

from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum
from monitor.adapters.sablas import SablasDrone


def _check_close(a: torch.Tensor, b: torch.Tensor, msg: str) -> None:
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-6), msg


def test_pendulum_reward_sign_convention():
    """Pendulum uses CLF residual: V(y) - V(x), so negative means safe."""
    adapter = NeuralCLBFPendulum(dt=0.01, noise_level=0.0)

    cur_state = adapter.state.clone()
    cur_v = adapter.get_certificate_value(cur_state)
    next_states = adapter.sample(cur_state, n_samples=32)
    next_v = adapter.get_certificate_value(next_states)

    # Independent residual formula from CLF condition.
    residual = next_v - cur_v
    reward = adapter.get_reward(next_states, cur_state)

    _check_close(
        reward,
        residual,
        "Pendulum reward must equal CLF residual V(y) - V(x)",
    )
    assert ((reward <= 0) == (residual <= 0)).all(), "Sign convention mismatch for pendulum"


def test_sablas_reward_sign_convention():
    """Sablas uses CBF residual: (h(x) - h(y)) - alpha(h(x)), so non-positive means safe."""
    adapter = SablasDrone(dt=0.1, noise_level=0.1)

    cur_state = adapter.state_history[-1].clone()
    cur_v = adapter.get_certificate_value(cur_state)
    next_states = adapter.sample(cur_state, n_samples=32)
    next_v = adapter.get_certificate_value(next_states)

    # Independent residual formula from CBF condition in adapter docs.
    alpha_cur = float(adapter.alpha(float(cur_v)))
    residual = (float(cur_v) - next_v) - alpha_cur
    reward = adapter.get_reward(next_states, cur_state)

    _check_close(
        reward,
        residual,
        "Sablas reward must equal CBF residual (h(x)-h(y)) - alpha(h(x))",
    )
    assert ((reward <= 0) == (residual <= 0)).all(), "Sign convention mismatch for sablas"

