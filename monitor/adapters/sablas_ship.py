from .interface import DynamicalSystemAdapter
from sablas.envs.env_ship import Ship, River
from sablas.modules.network import CBF, NNController
from sablas.modules import utils
from typing import Any, Callable, Dict, Optional, Tuple
import torch
import numpy as np
import random

class SablasShip(DynamicalSystemAdapter):
    def __init__(
        self,
        env_name: str = "ship",
        k_obstacle: int = 8,
        use_estimated_param: bool = False,
        dt: float = 0.1,
        noise_level: float = 0.0,
        vis_every: int = 0,
        vis_block: bool = False,
    ):
        if env_name not in {"ship", "river"}:
            raise ValueError("env_name must be one of {'ship', 'river'}")
        if noise_level < 0:
            raise ValueError("noise_level must be non-negative")

        self.env_name = env_name
        self.k_obs = int(k_obstacle)
        self.noise_level = float(noise_level)
        self.is_done = False
        self.control_period = 0.1
        self.vis_every = vis_every
        self.vis_block = vis_block
        self._vis_fig = None
        self._vis_ax = None
        self._step_count = 0
        self._alpha: Callable[[float], float] = lambda h: h
        self._preprocess = lambda x: utils.angle_to_sin_cos_torch(x, [2])

        # Ship CBF forward pass is also unbounded. Keep bounds wide and only
        # clamp catastrophic outliers.
        self.certificate_bounds: Tuple[float, float] = (-10.0, 10.0)

        estimated_param = None
        if use_estimated_param:
            estimated_param = "./sablas/data/estimated_model_ship.pth"

        if env_name == "ship":
            self.env = Ship(
                dt=dt,
                k_obstacle=self.k_obs,
                estimated_param=estimated_param,
            )
        else:
            self.env = River(
                dt=dt,
                k_obstacle=self.k_obs,
                estimated_param=estimated_param,
            )

        self.update_control_every = max(1, int(round(self.control_period / self.env.dt)))

        baseline_dt = self.control_period
        self.env.max_steps = max(1, int(round(self.env.max_steps * baseline_dt / self.env.dt)))

        self.controller = NNController(
            n_state=7,
            k_obstacle=self.k_obs,
            m_control=2,
            preprocess_func=self._preprocess,
            output_scale=1.1,
        )
        self.controller.load_state_dict(
            torch.load("./sablas/data/ship_controller_weights.pth", map_location=torch.device("cpu"))
        )
        self.controller.eval()

        self.cbf = CBF(
            n_state=7,
            k_obstacle=self.k_obs,
            m_control=2,
            preprocess_func=self._preprocess,
        )
        self.cbf.load_state_dict(
            torch.load("./sablas/data/ship_cbf_weights.pth", map_location=torch.device("cpu"))
        )
        self.cbf.eval()

        self._cached_control: Optional[np.ndarray] = None
        self.reset()

    @property
    def alpha(self) -> Callable[[float], float]:
        return self._alpha

    @alpha.setter
    def alpha(self, value: Callable[[float], float]):
        self._alpha = value

    def _reset_env_with_seed(self, seed: Optional[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if seed is None:
            return self.env.reset()

        np_state = np.random.get_state()
        py_state = random.getstate()
        np.random.seed(int(seed))
        random.seed(int(seed))
        try:
            return self.env.reset()
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)

    def reset(self, seed: Optional[int] = None):
        self.state, self.obstacle, self.goal = self._reset_env_with_seed(seed)
        self.state_error = torch.zeros(1, self.get_state_dim(), dtype=torch.float32)
        self.state_history = [torch.from_numpy(self.state).float()]
        self.drift_history = []
        self._cached_control = None
        self.is_done = False
        self._step_count = 0

    def _compute_control_input(self, state: np.ndarray) -> np.ndarray:
        u_nominal = self.env.nominal_controller(state, self.goal)
        u_tensor = self.controller(
            torch.from_numpy(state.reshape(1, 6).astype(np.float32)),
            torch.from_numpy(self.obstacle.reshape(1, self.k_obs, 6).astype(np.float32)),
            torch.from_numpy(u_nominal.reshape(1, 2).astype(np.float32)),
            torch.from_numpy(self.state_error.reshape(1, 6).astype(np.float32)),
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

    def _sample_process_noise(self) -> np.ndarray:
        if self.noise_level <= 0:
            return np.zeros((6,), dtype=np.float32)
        noise = np.random.normal(loc=0.0, scale=self.noise_level, size=(6,)).astype(np.float32)
        noise[:3] = 0.0
        return noise

    def _propagate(self, state: np.ndarray, u: np.ndarray, noise: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        dsdt = self.env.uncertain_dynamics(state, u)
        if noise is None:
            noise = np.zeros((6,), dtype=np.float32)
        next_state = state + (dsdt + noise) * self.env.dt
        next_state[3:] = np.clip(next_state[3:], -self.env.max_speed, self.env.max_speed)

        dsdt_nominal = self.env.nominal_dynamics(state, u)
        nominal_next = state + dsdt_nominal * self.env.dt
        nominal_next[3:] = np.clip(nominal_next[3:], -self.env.max_speed, self.env.max_speed)
        return next_state, nominal_next

    def done(self) -> bool:
        return self.is_done

    def get_drift(self, next_state: torch.Tensor, cur_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        cur_v = float(self.get_certificate_value(cur_state))
        next_v = self.get_certificate_value(next_state)
        return (cur_v - next_v) - self.alpha(cur_v)

    def get_certificate_value(self, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        resolved_state = self.resolve_state(state)
        n_states = 1 if resolved_state.dim() == 1 else len(resolved_state)
        state_tensor = resolved_state.reshape(n_states, 6)
        obstacle_tensor = torch.from_numpy(self.obstacle).float()
        obstacle_tensor = obstacle_tensor.unsqueeze(0).expand(n_states, -1, -1)
        cbf_values = self.cbf(state_tensor, obstacle_tensor)
        min_cbf, _ = torch.min(cbf_values, dim=1)
        lo, hi = self.certificate_bounds
        return torch.clamp(min_cbf, lo, hi)

    def step(self) -> torch.Tensor:
        cur_state_tensor = torch.from_numpy(self.state).float()
        u = self._get_control_input(self.state)
        noise = self._sample_process_noise()
        state_next, state_nominal_next = self._propagate(self.state, u, noise=noise)

        self.state_error = (
            (torch.from_numpy(state_next) - torch.from_numpy(state_nominal_next)).unsqueeze(0) / self.env.dt
        ).float()
        self.state = state_next
        self.obstacle = self.env.get_obstacle(state_next)
        self.goal = self.env.get_goal(state_next)
        self.env.state = state_next
        self.env.num_steps = self.env.num_steps + 1
        self.is_done = bool(
            np.linalg.norm(state_next[:2] - self.goal[:2]) < self.env.safe_dist
            or self.env.num_steps > self.env.max_steps
        )
        next_state_tensor = torch.from_numpy(self.state).float()
        self.state_history.append(next_state_tensor)
        self.drift_history.append(float(self.get_drift(next_state_tensor, cur_state_tensor)))

        self._step_count += 1
        if self.vis_every > 0 and self._step_count % self.vis_every == 0:
            self._visualize()

        return next_state_tensor

    def sample(self, state: Optional[torch.Tensor] = None, n_samples: int = 1) -> torch.Tensor:
        resolved_state_np = self.resolve_state(state).numpy()
        if state is None:
            u = self._peek_control_input(resolved_state_np)
        else:
            u = self._compute_control_input(resolved_state_np)

        next_states = []
        for _ in range(n_samples):
            noise = self._sample_process_noise()
            next_state_np, _ = self._propagate(resolved_state_np, u, noise=noise)
            next_states.append(torch.from_numpy(next_state_np))
        return torch.stack(next_states).float()

    def get_state_dim(self) -> int:
        return 6

    def get_state_history(self) -> torch.Tensor:
        return torch.stack(self.state_history)

    def get_drift_history(self) -> torch.Tensor:
        return torch.tensor(self.drift_history, dtype=torch.float32)

    def distance(self, state1: torch.Tensor, state2: torch.Tensor) -> float:
        return float(torch.linalg.norm(state1 - state2))

    def bound_key(self) -> Dict[str, Any]:
        return {
            "adapter_class": type(self).__name__,
            "env_name": self.env_name,
            "dt": float(self.env.dt),
            "noise_level": float(self.noise_level),
            "k_obstacle": int(self.k_obs),
        }

    def noisy_transitions(self, samples: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        current_state = torch.from_numpy(self.state).float()
        return current_state, self.sample(n_samples=samples)

    def successor_distribution_for(self, state: torch.Tensor):
        raise NotImplementedError("Successor distribution representation is not implemented yet for SablasShip")

    def get_expected_next_state(self, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        resolved_state_np = self.resolve_state(state).numpy()
        if state is None:
            u = self._peek_control_input(resolved_state_np)
        else:
            u = self._compute_control_input(resolved_state_np)

        expected_next, _ = self._propagate(resolved_state_np, u, noise=np.zeros((6,), dtype=np.float32))
        return torch.from_numpy(expected_next).float()

    def _visualize(self):
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Polygon

        if self._vis_fig is None or self._vis_ax is None:
            plt.ion()
            self._vis_fig = plt.figure(figsize=(10, 10))
            self._vis_ax = self._vis_fig.add_subplot(111)

        self._vis_ax.clear()
        self._vis_ax.scatter(self.obstacle[:, 0], self.obstacle[:, 1], color="grey")
        self._vis_ax.scatter(self.state[0], self.state[1], color="darkred")
        self._vis_ax.scatter(self.goal[0], self.goal[1], color="darkorange")

        rect = np.array([[0.6, 0.4], [-0.6, 0.4], [-0.6, -0.4], [0.6, -0.4]])
        yaw = self.state[2]
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rect = rect.dot(rotation.T) + self.state[:2]
        patch = PatchCollection([Polygon(rect, closed=True)], alpha=0.1, color="darkred")
        self._vis_ax.add_collection(patch)

        if self.env_name == "river":
            self._vis_ax.set_xlim(0, self.env.env_size)
            self._vis_ax.set_ylim(0, self.env.env_size)
        else:
            self._vis_ax.set_xlim(0, self.env.env_size)
            self._vis_ax.set_ylim(0, self.env.env_size)

        self._vis_fig.canvas.draw()
        if self.vis_block:
            plt.waitforbuttonpress()
        else:
            plt.pause(0.01)
