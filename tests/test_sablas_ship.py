import torch
import numpy as np
from monitor.calibration import LipschitzConstantProvider
from monitor.adapters import SablasShip
from sablas.envs.env_ship import Ship
from sablas.modules.network import NNController
from sablas.modules import utils
from tests.fixtures import check_close

def test_ship_interfaces():
    """Tests the ship adapter's basic shapes and types."""
    ship = SablasShip(dt=0.1, noise_level=0.0)
    state_dim = ship.get_state_dim()
    assert state_dim == 6
    assert ship.state.shape == (state_dim,)

    next_state = ship.step()
    assert isinstance(next_state, torch.Tensor)
    assert next_state.shape == (state_dim,)
    assert next_state.dtype == torch.float32

    samples = ship.sample(n_samples=10)
    assert isinstance(samples, torch.Tensor)
    assert samples.shape == (10, state_dim)

    v = ship.get_certificate_value()
    assert isinstance(v, torch.Tensor)
    assert v.ndim == 1


def test_ship_cross_validation_with_script():
    """Cross-validate one ship step against sablas/scripts/test_ship.py logic."""
    seed = 7
    dt = 0.1
    k_obs = 8

    torch.manual_seed(seed)
    np.random.seed(seed)
    adapter = SablasShip(dt=dt, noise_level=0.0)
    adapter.reset(seed=seed)
    adapter_state_0 = adapter.state.copy()
    adapter_state_1 = adapter.step().numpy()

    torch.manual_seed(seed)
    np.random.seed(seed)
    env = Ship(dt=dt, k_obstacle=k_obs)
    preprocess = lambda x: utils.angle_to_sin_cos_torch(x, [2])
    controller = NNController(
        n_state=7,
        k_obstacle=k_obs,
        m_control=2,
        preprocess_func=preprocess,
        output_scale=1.1,
    )
    controller.load_state_dict(torch.load("./sablas/data/ship_controller_weights.pth", map_location="cpu"))
    controller.eval()

    script_state_0, obstacle, goal = env.reset()
    state_error = np.zeros((6,), dtype=np.float32)
    u_nominal = env.nominal_controller(script_state_0, goal)
    u_tensor = controller(
        torch.from_numpy(script_state_0.reshape(1, 6).astype(np.float32)),
        torch.from_numpy(obstacle.reshape(1, k_obs, 6).astype(np.float32)),
        torch.from_numpy(u_nominal.reshape(1, 2).astype(np.float32)),
        torch.from_numpy(state_error.reshape(1, 6).astype(np.float32)),
    )
    u = np.squeeze(u_tensor.detach().cpu().numpy())
    script_state_1, _, _, _, _ = env.step(u)

    np.testing.assert_allclose(adapter_state_0, script_state_0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(adapter_state_1, script_state_1, rtol=1e-6, atol=1e-6)


def test_ship_control_period():
    """Ship adapter should keep control updates on the original 0.1s cadence."""
    adapter = SablasShip()
    try:
        LipschitzConstantProvider.get_drift_bound(adapter)
        assert False, "Expected Sablas ship drift-bound lookup to be unavailable"
    except KeyError:
        pass

    assert adapter.control_period == 0.1
    assert adapter.update_control_every == 1

    adapter_fine = SablasShip(dt=0.05)
    assert adapter_fine.control_period == 0.1
    assert adapter_fine.update_control_every == 2


def test_ship_reset_seed_is_deterministic_and_clear():
    """Ship reset(seed=...) should be reproducible and maintain clearance."""
    adapter = SablasShip(dt=0.1, noise_level=0.0)

    adapter.reset(seed=123)
    state_a = adapter.state.copy()
    goal_a = adapter.goal.copy()
    obstacle_a = adapter.obstacle.copy()

    adapter.reset(seed=123)
    state_b = adapter.state.copy()
    goal_b = adapter.goal.copy()
    obstacle_b = adapter.obstacle.copy()

    adapter.reset(seed=456)
    state_c = adapter.state.copy()

    assert np.allclose(state_a, state_b)
    assert np.allclose(goal_a, goal_b)
    assert np.allclose(obstacle_a, obstacle_b)
    assert not np.allclose(state_a, state_c)

    state_clearance = np.min(np.linalg.norm(adapter.env.obstacle - adapter.state[:2], axis=1))
    goal_clearance = np.min(np.linalg.norm(adapter.env.obstacle - adapter.goal[:2], axis=1))
    assert state_clearance >= adapter.env.safe_dist
    assert goal_clearance >= adapter.env.safe_dist


def test_ship_drift_sign_convention():
    """Ship uses the same CBF residual sign convention as sablas drone."""
    adapter = SablasShip(dt=0.1, noise_level=0.05)

    cur_state = adapter.get_state_history()[-1].clone()
    cur_v = adapter.get_certificate_value(cur_state)
    next_states = adapter.sample(cur_state, n_samples=32)
    next_v = adapter.get_certificate_value(next_states)

    alpha_cur = float(adapter.alpha(float(cur_v)))
    residual = (float(cur_v) - next_v) - alpha_cur
    drift = adapter.get_drift(next_states, cur_state)

    check_close(
        drift,
        residual,
        "Sablas ship drift must equal CBF residual (h(x)-h(y)) - alpha(h(x))",
    )
    assert ((drift <= 0) == (residual <= 0)).all(), "Sign convention mismatch for sablas ship"
