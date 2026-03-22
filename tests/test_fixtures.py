from tests.fixtures import NormalIncrementAdapter
import torch
import pytest

def test_normal_increment_adapter_is_seed_reproducible():
    adapter = NormalIncrementAdapter(mean=-0.2, sigma=0.1, initial_value=3.0)

    adapter.reset(seed=123)
    first_samples = adapter.sample(n_samples=5)
    first_step = adapter.step()

    adapter.reset(seed=123)
    second_samples = adapter.sample(n_samples=5)
    second_step = adapter.step()

    adapter.reset(seed=456)
    different_samples = adapter.sample(n_samples=5)

    assert torch.allclose(first_samples, second_samples)
    assert torch.allclose(first_step, second_step)
    assert not torch.allclose(first_samples, different_samples)


def test_normal_increment_adapter_respects_reward_bounds_and_expectation():
    adapter = NormalIncrementAdapter(mean=-0.1, sigma=0.05, initial_value=2.0, clamp_at_sigma=2.0)

    cur_state = adapter.state.clone()
    next_states = adapter.sample(n_samples=512)
    rewards = adapter.get_reward(next_states, cur_state)
    lower, upper = adapter.get_reward_bounds()

    expected_rewards = next_states[:, 0] - cur_state[0]
    assert torch.allclose(rewards, expected_rewards)
    assert float(rewards.min()) >= lower - 1e-6
    assert float(rewards.max()) <= upper + 1e-6

    expected_next = adapter.get_expected_next_state(cur_state)
    assert torch.allclose(expected_next, cur_state + adapter.mean)

    with pytest.raises(ValueError, match="sigma must be nonnegative"):
        NormalIncrementAdapter(mean=0.0, sigma=-1.0, initial_value=1.0)

def test_normal_increment_adapter_terminates():
    adapter = NormalIncrementAdapter(mean=-0.1, sigma=0.5, initial_value=20, clamp_at_sigma=3.0)
    while not adapter.done():
        adapter.step()
    assert 1.5 > adapter.get_certificate_value() >= 0

def test_normal_increment_adapter_0sigma_as_expected():
    adapter = NormalIncrementAdapter(mean=-1, sigma=0, initial_value=200)
    while not adapter.done():
        adapter.step()
    assert adapter.get_reward_bounds() == (-1, -1)
    assert adapter.get_lipschitz_constant() == 1
    assert len(adapter.get_state_history()) == 201
    assert float(adapter.get_state_history()[0]) == 200
    assert float(adapter.get_state_history()[50]) == 150
    assert float(adapter.get_state_history()[175]) == 25
    assert float(adapter.get_state_history()[-1]) == 0
