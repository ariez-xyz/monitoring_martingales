from .interface import DynamicalSystemAdapter
from sablas.envs.env_drone import Drone
from sablas.modules.network import CBF, NNController
from typing import Any, Callable, Dict, Optional, Tuple
import torch
import numpy as np


class SablasDrone(DynamicalSystemAdapter):
    def __init__(
        self,
        k_obstacle: int = 8,
        use_estimated_param: bool = False,
        dt: float = 0.1,
        noise_level: float = 0.1,
        vis_every: int = 0,
        vis_block: bool = False,
    ):
        self.k_obs = k_obstacle
        self.is_done = False
        if noise_level < 0:
            raise ValueError("noise_level must be non-negative")
        self.noise_level = float(noise_level)
        self.control_period = 0.1

        self.vis_every = vis_every
        self.vis_block = vis_block
        self._vis_fig = None
        self._vis_ax = None
        self._step_count = 0

        self._alpha: Callable[[float], float] = lambda h: h
        self._drift_bounds_cache: Optional[Tuple[float, float]] = None

        self.certificate_bounds: Tuple[float, float] = (-2.0, 0.86)
        self.lipschitz_constant: float = 1.0

        estimated_param = None
        if use_estimated_param:
            estimated_param = np.load(open("./sablas/data/estimated_model_drone.npz", "rb"))

        self.env = Drone(
            dt=dt,
            k_obstacle=self.k_obs,
            noise_std=self.noise_level,
            estimated_param=estimated_param,
        )
        self.update_control_every = max(1, int(round(self.control_period / self.env.dt)))
        self._cached_control: Optional[np.ndarray] = None

        baseline_dt = self.control_period
        self.env.max_steps = max(1, int(round(self.env.max_steps * baseline_dt / self.env.dt)))

        self.controller = NNController(n_state=8, k_obstacle=self.k_obs, m_control=3)
        self.controller.load_state_dict(torch.load("./sablas/data/drone_controller_weights.pth"))
        self.controller.eval()

        self.cbf = CBF(n_state=8, k_obstacle=self.k_obs, m_control=3)
        self.cbf.load_state_dict(torch.load("./sablas/data/drone_cbf_weights.pth"))
        self.cbf.eval()

        self.reset()

    @property
    def alpha(self) -> Callable[[float], float]:
        return self._alpha

    @alpha.setter
    def alpha(self, value: Callable[[float], float]):
        self._alpha = value
        self._drift_bounds_cache = None

    def reset(self, seed: Optional[int] = None):
        self.state, self.obstacle, self.goal = self.env.reset(seed=seed)
        self.state_error = torch.zeros(1, self.get_state_dim(), dtype=torch.float32)
        self.state_history = [torch.from_numpy(self.state).float()]
        self.drift_history = []
        self._cached_control = None
        self.is_done = False
        self._step_count = 0

    def _compute_control_input(self, state: np.ndarray) -> np.ndarray:
        u_nominal = self.env.nominal_controller(state, self.goal)
        u_tensor = self.controller(
            torch.from_numpy(state.reshape(1, 8)).float(),
            torch.from_numpy(self.obstacle.reshape(1, self.k_obs, 8)).float(),
            torch.from_numpy(u_nominal.reshape(1, 3)).float(),
            self.state_error,
        )
        return np.squeeze(u_tensor.detach().cpu().numpy())

    def _get_control_input(self, state: np.ndarray) -> np.ndarray:
        need_update = self._cached_control is None or self._step_count % self.update_control_every == 0
        if need_update:
            self._cached_control = self._compute_control_input(state)
        return self._cached_control  # type: ignore

    def _peek_control_input(self, state: np.ndarray) -> np.ndarray:
        need_update = self._cached_control is None or self._step_count % self.update_control_every == 0
        if need_update:
            return self._compute_control_input(state)
        return self._cached_control  # type: ignore

    def done(self) -> bool:
        return self.is_done

    def get_drift(self, next_state: torch.Tensor, cur_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        cur_v = float(self.get_certificate_value(cur_state))
        next_v = self.get_certificate_value(next_state)
        return (cur_v - next_v) - self.alpha(cur_v)

    def get_certificate_value(self, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        resolved_state = self.resolve_state(state)
        n_states = 1 if resolved_state.dim() == 1 else len(resolved_state)
        state_tensor = resolved_state.reshape(n_states, 8)
        obstacle_tensor = torch.from_numpy(self.obstacle).float()
        obstacle_tensor = obstacle_tensor.unsqueeze(0).expand(n_states, -1, -1)
        cbf_values = self.cbf(state_tensor, obstacle_tensor)
        min_cbf, _ = torch.min(cbf_values, dim=1)
        lo, hi = self.certificate_bounds
        return torch.clamp(min_cbf, lo, hi)

    def step(self) -> torch.Tensor:
        cur_state_tensor = torch.from_numpy(self.state).float()
        u = self._get_control_input(self.state)
        state_next, state_nominal_next, obstacle_next, goal_next, done = self.env.step(u)

        self.state_error = (
            (torch.from_numpy(state_next) - torch.from_numpy(state_nominal_next)).unsqueeze(0) / self.env.dt
        ).float()
        self.state = state_next
        self.obstacle = obstacle_next
        self.goal = goal_next
        self.is_done = bool(done)
        next_state_tensor = torch.from_numpy(self.state).float()
        self.state_history.append(next_state_tensor)
        self.drift_history.append(float(self.get_drift(next_state_tensor, cur_state_tensor)))

        self._step_count += 1
        if self.vis_every > 0 and self._step_count % self.vis_every == 0:
            self._visualize()

        return torch.from_numpy(self.state).float()

    def sample(
        self,
        state: Optional[torch.Tensor] = None,
        n_samples: int = 1,
        include_extremes: bool = False,
        noise_level: float = -1.0,
    ) -> torch.Tensor:
        resolved_state_np = self.resolve_state(state).numpy()
        if state is None:
            u = self._peek_control_input(resolved_state_np)
        else:
            u = self._compute_control_input(resolved_state_np)
        dsdt = self.env.uncertain_dynamics(resolved_state_np, u)
        effective_noise_level = self.noise_level if noise_level < 0 else float(noise_level)

        if noise_level >= 0: raise ValueError("sablas drone has its own noise, cannot override noise level")

        next_states = []
        if include_extremes and effective_noise_level > 0:
            for sign in (1.0, -1.0):
                extreme_noise = sign * np.ones((8,), dtype=resolved_state_np.dtype) * effective_noise_level
                extreme_noise[:3] = 0
                next_state_np = resolved_state_np + (dsdt + extreme_noise) * self.env.dt
                next_state_np[3:6] = np.clip(next_state_np[3:6], -self.env.max_speed, self.env.max_speed)
                next_state_np[6:] = np.clip(next_state_np[6:], -self.env.max_theta, self.env.max_theta)
                next_states.append(torch.from_numpy(next_state_np))

        for _ in range(n_samples):
            if effective_noise_level <= 0:
                noise = np.zeros((8,), dtype=resolved_state_np.dtype)
            elif np.random.uniform() < 0.05:
                noise = np.random.normal(size=(8,)) * effective_noise_level
            else:
                scale = 0.0 if self.noise_level <= 0 else (effective_noise_level / self.noise_level)
                noise = self.env.noise.copy() * scale
            noise[:3] = 0
            next_state_np = resolved_state_np + (dsdt + noise) * self.env.dt
            next_state_np[3:6] = np.clip(next_state_np[3:6], -self.env.max_speed, self.env.max_speed)
            next_state_np[6:] = np.clip(next_state_np[6:], -self.env.max_theta, self.env.max_theta)
            next_states.append(torch.from_numpy(next_state_np))

        if not next_states:
            return torch.empty((0, self.get_state_dim()), dtype=torch.float32)

        return torch.stack(next_states).float()

    def get_state_dim(self) -> int:
        return 8

    def get_state_history(self) -> torch.Tensor:
        return torch.stack(self.state_history)

    def get_drift_history(self) -> torch.Tensor:
        return torch.tensor(self.drift_history, dtype=torch.float32)

    def distance(self, state1: torch.Tensor, state2: torch.Tensor) -> float:
        return float(torch.linalg.norm(state1 - state2))

    def bound_key(self) -> Dict[str, Any]:
        return {
            "adapter_class": type(self).__name__,
            "dt": float(self.env.dt),
            "noise_level": float(self.noise_level),
            "k_obstacle": int(self.k_obs),
        }

    def noisy_transitions(self, samples: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        current_state = self.resolve_state().clone()
        return current_state, self.sample(n_samples=samples, include_extremes=True)

    def get_expected_next_state(self, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        resolved_state_np = self.resolve_state(state).numpy()
        if state is None:
            u = self._peek_control_input(resolved_state_np)
        else:
            u = self._compute_control_input(resolved_state_np)
        dsdt = self.env.uncertain_dynamics(resolved_state_np, u)

        expected_noise = 0.95 * self.env.noise.copy()
        expected_noise[:3] = 0

        expected_next = resolved_state_np + (dsdt + expected_noise) * self.env.dt
        expected_next[3:6] = np.clip(expected_next[3:6], -self.env.max_speed, self.env.max_speed)
        expected_next[6:] = np.clip(expected_next[6:], -self.env.max_theta, self.env.max_theta)

        return torch.from_numpy(expected_next).float()

    def _visualize(self):
        import matplotlib.pyplot as plt

        if self._vis_fig is None or self._vis_ax is None:
            plt.ion()
            self._vis_fig = plt.figure(figsize=(10, 10))
            self._vis_ax = self._vis_fig.add_subplot(111, projection="3d")

        self._vis_ax.clear()
        self._vis_ax.set_xlim(0, 20)
        self._vis_ax.set_ylim(0, 20)
        self._vis_ax.set_zlim(0, 20)  # type: ignore

        drone_pos = self.state[:3]
        self._vis_ax.scatter(*drone_pos, color="darkred", label="drone")
        self._vis_ax.scatter(self.goal[0], self.goal[1], self.goal[2], color="darkorange", label="goal")
        self._vis_ax.scatter(self.obstacle[:, 0], self.obstacle[:, 1], self.obstacle[:, 2], color="grey", label="obstacles")

        dang_dist, safe_dist = 0.6, 3.0
        for obs in self.obstacle:
            obs_pos = obs[:3]
            dist = np.linalg.norm(obs_pos - drone_pos)
            t = np.clip((dist - dang_dist) / (safe_dist - dang_dist), 0, 1)
            color = (1 - 0.5 * t, 0.5 * t, 0.5 * t)
            alpha = 1.0 - 0.8 * t
            self._vis_ax.plot(
                [drone_pos[0], obs_pos[0]],
                [drone_pos[1], obs_pos[1]],
                [drone_pos[2], obs_pos[2]],
                color=color,
                alpha=alpha,
                linewidth=0.8,
            )

        self._vis_fig.canvas.draw()
        if self.vis_block:
            plt.waitforbuttonpress()
        else:
            plt.pause(0.01)
