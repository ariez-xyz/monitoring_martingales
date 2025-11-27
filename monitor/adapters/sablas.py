from .interface import DynamicalSystemAdapter
from sablas.envs.env_drone import Drone
from sablas.modules.network import CBF, NNController
from typing import Callable, Optional
import torch
import numpy as np

class SablasDrone(DynamicalSystemAdapter):
    def __init__(self, k_obstacle: int = 8, use_estimated_param: bool = False, dt: float = 0.1):
        self.k_obs = k_obstacle
        self.is_done = False
        
        estimated_param = None
        if use_estimated_param:
            estimated_param = np.load(open('./sablas/data/estimated_model_drone.npz', 'rb'))

        self.env = Drone(dt=dt, k_obstacle=self.k_obs, estimated_param=estimated_param)
        
        self.controller = NNController(n_state=8, k_obstacle=self.k_obs, m_control=3)
        self.controller.load_state_dict(torch.load('./sablas/data/drone_controller_weights.pth'))
        self.controller.eval()

        self.cbf = CBF(n_state=8, k_obstacle=self.k_obs, m_control=3)
        self.cbf.load_state_dict(torch.load('./sablas/data/drone_cbf_weights.pth'))
        self.cbf.eval()

        self.reset()

    def reset(self):
        """Resets the environment and the adapter's internal state."""
        self.state, self.obstacle, self.goal = self.env.reset()
        self.state_error = torch.zeros(1, self.get_state_dim(), dtype=torch.float32)

    def _get_control_input(self, state: np.ndarray) -> np.ndarray:
        """Helper function to compute the control input 'u'."""
        u_nominal = self.env.nominal_controller(state, self.goal)
        u_tensor = self.controller(
            torch.from_numpy(state.reshape(1, 8)).float(), 
            torch.from_numpy(self.obstacle.reshape(1, self.k_obs, 8)).float(),
            torch.from_numpy(u_nominal.reshape(1, 3)).float(),
            self.state_error)
        return np.squeeze(u_tensor.detach().cpu().numpy())

    def done(self) -> bool:
        return self.is_done

    def get_reward(self, next_state: torch.Tensor, alpha: Callable[[float], float] = lambda x: x) -> torch.Tensor:
        cur_reward = float(self.get_certificate_value())
        next_rewards = self.get_certificate_value(next_state)
        deltas = cur_reward - next_rewards  # V(x) - V(Y)
        return deltas - alpha(cur_reward)   # V(x) - V(Y) - α(V(x)) ≤ 0 means CBF condition satisfied

    def get_certificate_value(self, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculates V(x) for the given or current state.
        V(x) is the minimum of the CBF values over all obstacles.

        Uses the current obstacle set (self.obstacle) for all evaluations to ensure
        consistent comparisons when computing V(Y) - V(x) for sampled next states.
        """
        resolved_state = self.resolve_state(state)  # float32 tensor
        n_states = 1 if resolved_state.dim() == 1 else len(resolved_state)
        state_tensor = resolved_state.reshape(n_states, 8)
        # Use current obstacle set for all states (don't recompute per sample)
        obstacle_tensor = torch.from_numpy(self.obstacle).float()
        obstacle_tensor = obstacle_tensor.unsqueeze(0).expand(n_states, -1, -1)
        cbf_values = self.cbf(state_tensor, obstacle_tensor)
        min_cbf, _ = torch.min(cbf_values, dim=1)
        return min_cbf

    def step(self) -> torch.Tensor:
        """
        Take one simulation step using the NN controller.
        """
        u = self._get_control_input(self.state)
        state_next, state_nominal_next, obstacle_next, goal_next, done = self.env.step(u)

        # Update state error for the next controller input
        self.state_error = ((torch.from_numpy(state_next) - torch.from_numpy(state_nominal_next)).unsqueeze(0) / self.env.dt).float()

        # Update internal state
        self.state = state_next
        self.obstacle = obstacle_next
        self.goal = goal_next
        self.is_done = bool(done)
        
        return torch.from_numpy(self.state).float()

    def sample(self, state: Optional[torch.Tensor] = None, n_samples: int = 1) -> torch.Tensor:
        """
        Samples 'n_samples' next states for the given or current state.
        """
        resolved_state_np = self.resolve_state(state).numpy()
        u = self._get_control_input(resolved_state_np)
        
        next_states = []
        for _ in range(n_samples):
            dsdt = self.env.uncertain_dynamics(resolved_state_np, u)
            noise = self.env.get_noise()
            next_state_np = resolved_state_np + (dsdt + noise) * self.env.dt
            # Clipping from env logic
            next_state_np[3:6] = np.clip(next_state_np[3:6], -self.env.max_speed, self.env.max_speed)
            next_state_np[6:] = np.clip(next_state_np[6:], -self.env.max_theta, self.env.max_theta)
            next_states.append(torch.from_numpy(next_state_np))

        return torch.stack(next_states).float()

    def get_state_dim(self) -> int:
        """Returns the dimension of the state space."""
        return 8


