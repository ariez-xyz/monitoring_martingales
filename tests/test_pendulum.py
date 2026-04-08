"""Tests for the neural_clbf pendulum adapter."""
import torch
from monitor.calibration import LipschitzConstantProvider

from tests.fixtures import check_close

def test_pendulum_reset_seed_is_deterministic():
    """Resetting with the same seed should reproduce the same initial state."""
    from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum

    adapter = NeuralCLBFPendulum(dt=0.01, noise_level=0.0)

    adapter.reset(seed=123)
    first = adapter.state.clone()
    adapter.reset(seed=123)
    second = adapter.state.clone()
    adapter.reset(seed=456)
    third = adapter.state.clone()

    assert torch.allclose(first, second)
    assert not torch.allclose(first, third)


def test_pendulum_flip_fault_affects_zero_noise_sampling():
    """Flip fault mode should be reflected by zero-noise one-step sampling."""
    from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum

    torch.manual_seed(0)
    adapter = NeuralCLBFPendulum(dt=0.01, noise_level=0.0)
    state = torch.tensor([1.0, 0.5])
    adapter.reset(initial_state=state)

    nominal_next = adapter.sample(state, n_samples=1, noise_level=0.0).squeeze(0)
    nominal_samples = adapter.sample(state, n_samples=4)

    adapter.flip_inputs_state = True
    flipped_next = adapter.sample(state, n_samples=1, noise_level=0.0).squeeze(0)
    flipped_samples = adapter.sample(state, n_samples=4)

    assert nominal_samples.shape == (4, 2)
    assert flipped_samples.shape == (4, 2)
    assert torch.allclose(nominal_samples, nominal_next.unsqueeze(0).expand_as(nominal_samples))
    assert torch.allclose(flipped_samples, flipped_next.unsqueeze(0).expand_as(flipped_samples))
    assert not torch.allclose(nominal_next, flipped_next), (
        "Expected flip fault to change one-step dynamics used by sample()"
    )


def test_pendulum_explicit_state_queries_do_not_reuse_held_control():
    """Explicit state queries should recompute control instead of reusing the held one."""
    from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum

    adapter = NeuralCLBFPendulum(dt=0.01, noise_level=0.0)
    current_state = torch.tensor([1.0, 0.5])
    other_state = torch.tensor([-1.0, -0.5])
    adapter.reset(initial_state=current_state)

    # Prime the held-control cache using the live state.
    _ = adapter.sample(n_samples=1, noise_level=0.0)

    live_next = adapter.sample(n_samples=1, noise_level=0.0).squeeze(0)
    queried_next = adapter.sample(other_state, n_samples=1, noise_level=0.0).squeeze(0)
    queried_samples = adapter.sample(other_state, n_samples=3)

    assert queried_samples.shape == (3, 2)
    assert torch.allclose(
        queried_samples,
        queried_next.unsqueeze(0).expand_as(queried_samples),
    )
    assert not torch.allclose(live_next, queried_next), (
        "Explicit state queries should not silently reuse the held control from the live state"
    )


def test_pendulum_sample_with_zero_noise_matches_expected_next_state():
    from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum

    adapter = NeuralCLBFPendulum(dt=0.01, noise_level=0.1)
    state = torch.tensor([1.0, 0.5])
    adapter.reset(initial_state=state)

    expected_next = adapter.sample(state, n_samples=1, noise_level=0.0).squeeze(0)
    sampled_next = adapter.sample(state, n_samples=4, noise_level=0.0)

    assert sampled_next.shape == (4, 2)
    assert torch.allclose(
        sampled_next,
        expected_next.unsqueeze(0).expand_as(sampled_next),
    )


def test_pendulum_held_control_rejects_batched_queries():
    """Held-control path is only defined for the single live state."""
    import pytest
    from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum

    adapter = NeuralCLBFPendulum(dt=0.01, noise_level=0.0)
    batched_states = torch.tensor([[1.0, 0.5], [-1.0, -0.5]])

    with pytest.raises(ValueError, match="single-state batch"):
        adapter._get_control(batched_states, use_hold=True)


def test_pendulum_drift_bound_orders_widely_separated_timesteps():
    """
    Test that the pendulum drift bound scales sensibly across far-apart dt values.

    Verifies:
    1. The precomputed bound is positive and cached
    2. A much smaller dt gives a smaller bound
    3. A much larger dt gives a larger bound
    """
    import time
    from monitor.calibration import LipschitzConstantSampler
    from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum

    print("\n--- Pendulum Drift Bound Ordering Test ---")

    # Test bound lookup works
    adapter = NeuralCLBFPendulum(dt=0.01)
    start = time.time()
    gamma1 = LipschitzConstantProvider.get_drift_bound(adapter)
    time1 = time.time() - start

    assert gamma1 > 0, f"gamma should be positive, got {gamma1}"
    assert gamma1 < 10, f"gamma should be reasonable, got {gamma1}"
    print(f"  dt=0.01: gamma = {gamma1:.4f} (estimated in {time1:.2f}s)")

    # Test caching - second call should be instant
    start = time.time()
    gamma1_cached = LipschitzConstantProvider.get_drift_bound(adapter)
    time_cached = time.time() - start

    assert gamma1_cached == gamma1, "Cached value should match"
    assert time_cached < 0.01, f"Cached call should be instant, took {time_cached:.3f}s"
    print(f"  Cached call: {time_cached*1000:.2f}ms")

    # Compare a smaller precomputed dt.
    adapter_small_dt = NeuralCLBFPendulum(dt=0.001)
    gamma_small = LipschitzConstantProvider.get_drift_bound(adapter_small_dt)

    # Estimate a larger dt value inline through the new sampler path.
    sampler = LipschitzConstantSampler()
    gamma_large_samples = sampler.sample_drift_bounds(
        lambda: NeuralCLBFPendulum(dt=0.1, noise_level=0.0),
        n_episodes=20,
        max_steps=100, # large dt doesn't need many steps
        samples_per_step=4,
    )
    gamma_large = max(sample.value for sample in gamma_large_samples)

    print(f"  dt=0.001: gamma = {gamma_small:.4f}")
    print(f"  dt=0.1:  gamma = {gamma_large:.4f}")

    # Smaller dt should give smaller per-step drift; larger dt should give larger drift.
    assert gamma_small < gamma1, f"Smaller dt should give smaller gamma: {gamma_small} vs {gamma1}"
    assert gamma_large > gamma1, f"Larger dt should give larger gamma: {gamma_large} vs {gamma1}"

    print("  Drift bound ordering works correctly!")


def test_pendulum_seeded_bounds_cover_observed_step_drift():
    """
    Validate that seeded pendulum B_k-style bounds cover observed one-step drift.

    For each seeded (dt, noise_level) configuration, this test runs short
    rollouts via the adapter API and checks:
    |V(x_{t+1}) - V(x_t)| <= get_drift_bound().
    """
    from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum
    from random import random

    seeded_bounds = {
        (0.001, 0.0): 0.20,
        (0.001, 0.1): 0.20,
        (0.001, 1.0): 0.20,
        (0.001, 10.0): 0.25,
        (0.01, 0.0): 1.95,
        (0.01, 0.1): 1.90,
        (0.01, 1.0): 2.00,
        (0.01, 10.0): 2.50,
    }
    tol = 1e-6

    for (dt, noise_level), expected_bound in seeded_bounds.items():
        adapter = NeuralCLBFPendulum(dt=dt, noise_level=noise_level)
        gamma = LipschitzConstantProvider.get_drift_bound(adapter)
        assert abs(gamma - expected_bound) < tol, (
            f"Unexpected cached bound for dt={dt}, noise={noise_level}: {gamma} vs {expected_bound}"
        )

        # 50 resets with 10 steps to keep runtime short.
        for _ in range(50):
            observed_max = 0.0
            seed = int(random() * 999999)
            adapter.reset(seed=seed)

            for _ in range(10):
                v_cur = float(adapter.get_certificate_value())
                adapter.step()
                v_next = float(adapter.get_certificate_value())
                observed_drift = abs(v_next - v_cur)
                observed_max = max(observed_max, observed_drift)
                if adapter.done():
                    break

            assert observed_max <= gamma + tol, (
                f"Observed step drift exceeds cached bound for dt={dt}, noise={noise_level}: "
                f"observed_max={observed_max:.6f}, gamma={gamma:.6f}"
                f"seed={seed}"
            )

def test_pendulum_drift_sign_convention():
    """Pendulum uses CLF residual: V(y) - V(x), so negative means safe."""
    from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum
    adapter = NeuralCLBFPendulum(dt=0.01, noise_level=0.0)

    cur_state = adapter.state.clone()
    cur_v = adapter.get_certificate_value(cur_state)
    next_states = adapter.sample(cur_state, n_samples=32)
    next_v = adapter.get_certificate_value(next_states)

    # Independent residual formula from CLF condition.
    residual = next_v - cur_v
    drift = adapter.get_drift(next_states, cur_state)

    check_close(
        drift,
        residual,
        "Pendulum drift must equal CLF residual V(y) - V(x)",
    )
    assert ((drift <= 0) == (residual <= 0)).all(), "Sign convention mismatch for pendulum"


def test_pendulum_empirical_drift_bound_estimation_does_not_mutate_live_state():
    """Empirical drift-bound estimation should not disturb the live adapter rollout."""
    from monitor.calibration import LipschitzConstantSampler
    from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum

    adapter = NeuralCLBFPendulum(dt=0.1, noise_level=0.0)
    adapter.reset(seed=123)
    adapter.step()
    adapter.step()

    state_before = adapter.state.clone()
    state_history_before = adapter.get_state_history().clone()
    drift_history_before = adapter.get_drift_history().clone()
    control_history_before = list(adapter.control_history)
    applied_control_history_before = list(adapter.applied_control_history)
    clf_history_before = list(adapter.clf_history)
    step_count_before = adapter._step_count
    done_before = adapter.is_done
    cached_control_before = (
        None if adapter._cached_control is None else adapter._cached_control.clone()
    )

    sampler = LipschitzConstantSampler()
    gamma_samples = sampler.sample_drift_bounds(
        lambda: NeuralCLBFPendulum(dt=0.1, noise_level=0.0),
        n_episodes=1,
        max_steps=2,
    )
    gamma = max(sample.value for sample in gamma_samples)

    assert gamma > 0 # type:ignore
    assert torch.allclose(adapter.state, state_before)
    assert torch.allclose(adapter.get_state_history(), state_history_before)
    assert torch.allclose(adapter.get_drift_history(), drift_history_before)
    assert adapter.control_history == control_history_before
    assert adapter.applied_control_history == applied_control_history_before
    assert adapter.clf_history == clf_history_before
    assert adapter._step_count == step_count_before
    assert adapter.is_done == done_before
    if cached_control_before is None:
        assert adapter._cached_control is None
    else:
        assert adapter._cached_control is not None
        assert torch.allclose(adapter._cached_control, cached_control_before)


def test_pendulum_sample_with_extremes_returns_endpoints_plus_samples():
    """Including extremes should prepend endpoint successors to sampled-noise successors."""
    from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum

    torch.manual_seed(0)
    adapter = NeuralCLBFPendulum(dt=0.1, noise_level=0.1)

    current_state = adapter.get_state()
    next_states = adapter.sample(n_samples=4, include_extremes=True)

    assert current_state.shape == (adapter.dynamics.n_dims,)
    assert next_states.shape == (6, adapter.dynamics.n_dims)


def test_pendulum_step_appends_exactly_one_drift():
    """Each pendulum step should append exactly one realized drift."""
    from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum

    adapter = NeuralCLBFPendulum(dt=0.1, noise_level=0.0)
    initial_state_len = len(adapter.get_state_history())
    initial_drift_len = len(adapter.get_drift_history())

    prev_cert = float(adapter.get_certificate_value())

    extra_steps = 10
    for step_idx in range(1000):
        if adapter.done():
            extra_steps -= 1
            if extra_steps <= 0: break
        adapter.step()
        cur_state_len = len(adapter.get_state_history())
        cur_drift_len = len(adapter.get_drift_history())
        cur_cert = float(adapter.get_certificate_value())
        assert cur_state_len == initial_state_len + step_idx + 1
        assert cur_drift_len == initial_drift_len + step_idx + 1
        assert cur_state_len == cur_drift_len + 1
        assert abs(float(adapter.get_drift_history()[-1]) - (cur_cert - prev_cert)) < 1e-6
        prev_cert = cur_cert
