import torch
import numpy as np
from monitor.adapters import SablasDrone
from sablas.envs.env_drone import Drone
from sablas.modules.network import NNController

def test_drone_interfaces():
    """
    Tests the basic shapes and types of the adapter outputs.
    """
    drone = SablasDrone(dt=0.1)
    state_dim = drone.get_state_dim()
    assert state_dim == 8

    # Test initial state
    assert drone.state.shape == (state_dim,)

    # Test step()
    next_state = drone.step()
    assert isinstance(next_state, torch.Tensor)
    assert next_state.shape == (state_dim,)
    assert next_state.dtype == torch.float32

    # Test sample()
    n_samples = 10
    samples = drone.sample(n_samples=n_samples)
    assert isinstance(samples, torch.Tensor)
    assert samples.shape == (n_samples, state_dim)

    # Test get_certificate_value()
    v = drone.get_certificate_value()
    assert isinstance(v, torch.Tensor)
    assert v.ndim == 1

def test_certificate_change_on_step():
    """
    Tests the core logic by taking a step and observing the change in the certificate value.
    """
    drone = SablasDrone(dt=0.1)

    # 1. Get the certificate value for the current state
    v_current = drone.get_certificate_value()
    print(f"\nInitial State Position: {drone.state[:3]}")
    print(f"Initial Certificate V(x): {v_current.item():.4f}")

    # 2. Take one simulation step
    next_state = drone.step()

    # 3. Get the certificate value for the new state
    v_next = drone.get_certificate_value(state=next_state)
    print(f"Next State Position:    {next_state.numpy()[:3]}")
    print(f"Next Certificate V(y): {v_next.item():.4f}")

    # 4. Calculate and report the change
    delta_v = v_next - v_current
    print(f"Change in Certificate (V(y) - V(x)): {delta_v.item():.4f}")
    
    # A working Control Barrier Function (CBF) should result in V staying
    # positive and ideally increasing as the controller actively avoids obstacles.
    # We won't assert a specific behavior yet, but observing this value is the key output.

def test_cross_validation_with_script():
    """
    Cross-validates the adapter's logic against the original script's logic
    by running one step in parallel under a fixed random seed.
    """
    seed = 42
    dt = 0.1
    k_obs = 8

    # --- 1. Run the adapter for one step ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    adapter = SablasDrone(dt=dt)
    adapter_state_0 = adapter.state.copy()
    adapter_state_1 = adapter.step().numpy()

    # --- 2. Manually run the script logic for one step ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialization logic from script
    env = Drone(k_obstacle=k_obs, dt=dt)
    controller = NNController(n_state=8, k_obstacle=8, m_control=3)
    controller.load_state_dict(torch.load('./sablas/data/drone_controller_weights.pth'))
    controller.eval()
    
    # Reset and get initial state
    script_state_0, obstacle, goal = env.reset()
    state_error = torch.zeros(1, 8, dtype=torch.float32)

    # Step logic from script
    u_nominal = env.nominal_controller(script_state_0, goal)
    u_tensor = controller(
        torch.from_numpy(script_state_0.reshape(1, 8)).float(),
        torch.from_numpy(obstacle.reshape(1, k_obs, 8)).float(),
        torch.from_numpy(u_nominal.reshape(1, 3)).float(),
        state_error
    )
    u = np.squeeze(u_tensor.detach().cpu().numpy())
    script_state_1, _, _, _, _ = env.step(u)

    # --- 3. Compare the results ---
    # Assert that the initial states are identical
    np.testing.assert_allclose(adapter_state_0, script_state_0, rtol=1e-6, atol=1e-6)
    
    # Assert that the next states after one step are identical
    np.testing.assert_allclose(adapter_state_1, script_state_1, rtol=1e-6, atol=1e-6)

    print("\nCross-validation successful: Adapter state matches script state for one step.")
