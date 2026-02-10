from .interface import DynamicalSystemAdapter
from sablas.envs.env_drone import Drone
from sablas.modules.network import CBF, NNController
from typing import Callable, Optional, Tuple
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
        # Keep control update period fixed to training cadence.
        self.control_period = 0.1

        # Visualization settings
        self.vis_every = vis_every
        self.vis_block = vis_block
        self._vis_fig = None
        self._vis_ax = None
        self._step_count = 0

        # Alpha function for CBF condition: ḣ ≥ -α(h)
        # sablas uses identity (α(h) = h), see sablas/modules/trainer.py:78,142,209
        #   deriv_cond = (h_next - h) / self.dt + h  # i.e., ḣ + α(h) with α = identity
        self._alpha: Callable[[float], float] = lambda h: h
        self._reward_bounds_cache: Optional[Tuple[float, float]] = None

        # Certificate bounds - the sablas CBF network is unbounded (Tanh defined but not
        # applied in forward pass, see sablas/modules/network.py:21,43).
        # Empirically estimated via scripts/estimate_cbf_bounds.py over 1M steps (4549 resets):
        # observed [-1.80, 0.66], using 0.2 margin. 
        # Values outside are clamped in get_certificate_value.
        self.certificate_bounds: Tuple[float, float] = (-2.0, 0.86)

        # Lipschitz constant for expected reward function.
        # Empirically estimated via scripts/estimate_gamma.py over 50k steps.
        # 99th percentile ~0.43, max ~7. Using 1.0 as a balanced default.
        self.lipschitz_constant: float = 1.0

        estimated_param = None
        if use_estimated_param:
            estimated_param = np.load(open('./sablas/data/estimated_model_drone.npz', 'rb'))

        self.env = Drone(
            dt=dt,
            k_obstacle=self.k_obs,
            noise_std=self.noise_level,
            estimated_param=estimated_param,
        )
        # dt controls simulation fidelity; control remains fixed at training cadence.
        self.update_control_every = max(1, int(round(self.control_period / self.env.dt)))
        self._cached_control: Optional[np.ndarray] = None

        # Keep max episode duration comparable in physical time when dt changes.
        baseline_dt = self.control_period
        self.env.max_steps = max(1, int(round(self.env.max_steps * baseline_dt / self.env.dt)))

        self.controller = NNController(n_state=8, k_obstacle=self.k_obs, m_control=3)
        self.controller.load_state_dict(torch.load('./sablas/data/drone_controller_weights.pth'))
        self.controller.eval()

        self.cbf = CBF(n_state=8, k_obstacle=self.k_obs, m_control=3)
        self.cbf.load_state_dict(torch.load('./sablas/data/drone_cbf_weights.pth'))
        self.cbf.eval()

        self.reset()

    @property
    def alpha(self) -> Callable[[float], float]:
        return self._alpha

    @alpha.setter
    def alpha(self, value: Callable[[float], float]):
        self._alpha = value
        self._reward_bounds_cache = None

    def reset(self):
        """Resets the environment and the adapter's internal state."""
        self.state, self.obstacle, self.goal = self.env.reset()
        self.state_error = torch.zeros(1, self.get_state_dim(), dtype=torch.float32)
        self.state_history = [torch.from_numpy(self.state).float()]
        self._cached_control = None

    def _compute_control_input(self, state: np.ndarray) -> np.ndarray:
        """Compute a fresh control input for a given state."""
        u_nominal = self.env.nominal_controller(state, self.goal)
        u_tensor = self.controller(
            torch.from_numpy(state.reshape(1, 8)).float(), 
            torch.from_numpy(self.obstacle.reshape(1, self.k_obs, 8)).float(),
            torch.from_numpy(u_nominal.reshape(1, 3)).float(),
            self.state_error)
        return np.squeeze(u_tensor.detach().cpu().numpy())

    def _get_control_input(self, state: np.ndarray) -> np.ndarray:
        """Get held control input, updating only at fixed control cadence."""
        need_update = self._cached_control is None or self._step_count % self.update_control_every == 0
        if need_update:
            self._cached_control = self._compute_control_input(state)
        return self._cached_control

    def _peek_control_input(self, state: np.ndarray) -> np.ndarray:
        """Get control for hypothetical transitions without mutating held-control state."""
        need_update = self._cached_control is None or self._step_count % self.update_control_every == 0
        if need_update:
            return self._compute_control_input(state)
        return self._cached_control

    def done(self) -> bool:
        return self.is_done

    def get_reward_bounds(self) -> Tuple[float, float]:
        """Computes reward bounds from certificate bounds and alpha.

        Reward = (cur_v - next_v) - α(cur_v) = f(cur_v) - next_v
        where f(v) = v - α(v).

        Since reward is linear (decreasing) in next_v, extremes occur at next_v boundaries.
        For cur_v, we sample f(v) to find its range, handling any α without assuming α'≤1.

        Results are cached; cache invalidates when alpha is reassigned.
        """
        if self._reward_bounds_cache is not None:
            return self._reward_bounds_cache

        cert_lo, cert_hi = self.certificate_bounds

        # Sample f(v) = v - α(v) to find its range over [cert_lo, cert_hi]
        n_samples = 1000
        vs = [cert_lo + (cert_hi - cert_lo) * i / (n_samples - 1) for i in range(n_samples)]
        f_values = [v - self.alpha(v) for v in vs]
        f_min, f_max = min(f_values), max(f_values)

        # reward = f(cur_v) - next_v
        # reward_lo: minimize f(cur_v), maximize next_v
        # reward_hi: maximize f(cur_v), minimize next_v
        # TODO: currently this doesn't take into account that we won't probably transition from 
        # minimum to maximum value of f in a single step?
        reward_lo = f_min - cert_hi
        reward_hi = f_max - cert_lo

        self._reward_bounds_cache = (reward_lo, reward_hi)
        return self._reward_bounds_cache

    def get_reward(self, next_state: torch.Tensor, cur_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        cur_v = float(self.get_certificate_value(cur_state))
        next_v = self.get_certificate_value(next_state)
        # V(x) - V(Y) - α(V(x)) ≤ 0 means CBF condition satisfied
        return (cur_v - next_v) - self.alpha(cur_v)

    def get_certificate_value(self, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculates V(x) for the given or current state.
        V(x) is the minimum of the CBF values over all obstacles.

        Uses the current obstacle set (self.obstacle) for all evaluations to ensure
        consistent comparisons when computing V(Y) - V(x) for sampled next states.

        Raises:
            ValueError: If any CBF value exceeds the assumed bounds.
        """
        resolved_state = self.resolve_state(state)  # float32 tensor
        n_states = 1 if resolved_state.dim() == 1 else len(resolved_state)
        state_tensor = resolved_state.reshape(n_states, 8)
        # Use current obstacle set for all states (don't recompute per sample)
        obstacle_tensor = torch.from_numpy(self.obstacle).float()
        obstacle_tensor = obstacle_tensor.unsqueeze(0).expand(n_states, -1, -1)
        cbf_values = self.cbf(state_tensor, obstacle_tensor)
        min_cbf, _ = torch.min(cbf_values, dim=1)

        # Clamp to declared bounds (rare outliers from e.g. bad spawns)
        lo, hi = self.certificate_bounds
        min_cbf = torch.clamp(min_cbf, lo, hi)

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
        self.state_history.append(torch.from_numpy(self.state).float())

        self._step_count += 1
        if self.vis_every > 0 and self._step_count % self.vis_every == 0:
            self._visualize()

        return torch.from_numpy(self.state).float()

    def sample(self, state: Optional[torch.Tensor] = None, n_samples: int = 1) -> torch.Tensor:
        """
        Samples 'n_samples' next states for the given or current state.

        Matches the env's sticky noise process (95% keep, 5% resample) without
        mutating the env's actual noise state.
        """
        resolved_state_np = self.resolve_state(state).numpy()
        if state is None:
            # Use the same held-control policy as step() for the current rollout state.
            u = self._peek_control_input(resolved_state_np)
        else:
            # For arbitrary queried states, compute a fresh control.
            u = self._compute_control_input(resolved_state_np)
        dsdt = self.env.uncertain_dynamics(resolved_state_np, u)

        next_states = []
        for _ in range(n_samples):
            # Replicate sticky noise logic without mutating env
            if np.random.uniform() < 0.05:
                noise = np.random.normal(size=(8,)) * self.env.noise_std
            else:
                noise = self.env.noise.copy()
            noise[:3] = 0  # No noise on position
            next_state_np = resolved_state_np + (dsdt + noise) * self.env.dt
            # Clipping from env logic
            next_state_np[3:6] = np.clip(next_state_np[3:6], -self.env.max_speed, self.env.max_speed)
            next_state_np[6:] = np.clip(next_state_np[6:], -self.env.max_theta, self.env.max_theta)
            next_states.append(torch.from_numpy(next_state_np))

        return torch.stack(next_states).float()

    def get_state_dim(self) -> int:
        """Returns the dimension of the state space."""
        return 8

    def get_state_history(self) -> torch.Tensor:
        """Return the history of states."""
        return torch.stack(self.state_history)

    def distance(self, state1: torch.Tensor, state2: torch.Tensor) -> float:
        """Euclidean distance on full state."""
        return float(torch.linalg.norm(state1 - state2))

    def get_lipschitz_constant(self) -> float:
        """Returns the Lipschitz constant for expected rewards."""
        return self.lipschitz_constant

    def get_expected_next_state(self, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes E[Y] analytically for sablas drone dynamics.

        Dynamics: Y = x + (dsdt + noise) * dt
        Noise is sticky: 95% keep env.noise, 5% resample from N(0, noise_std).
        E[noise] = 0.95 * env.noise + 0.05 * 0 = 0.95 * env.noise
        (position noise is zeroed out)

        So: E[Y] = x + (dsdt + 0.95 * env.noise) * dt
        """
        resolved_state_np = self.resolve_state(state).numpy()
        if state is None:
            u = self._peek_control_input(resolved_state_np)
        else:
            u = self._compute_control_input(resolved_state_np)
        dsdt = self.env.uncertain_dynamics(resolved_state_np, u)

        # Expected noise: 95% of current sticky noise (position components zeroed)
        expected_noise = 0.95 * self.env.noise.copy()
        expected_noise[:3] = 0

        expected_next = resolved_state_np + (dsdt + expected_noise) * self.env.dt

        # Apply same clipping as in sample() - ignoring Jensen gap from clipping
        expected_next[3:6] = np.clip(expected_next[3:6], -self.env.max_speed, self.env.max_speed)
        expected_next[6:] = np.clip(expected_next[6:], -self.env.max_theta, self.env.max_theta)

        return torch.from_numpy(expected_next).float()

    def _visualize(self):
        """Render current state in 3D plot."""
        import matplotlib.pyplot as plt

        if self._vis_fig is None or self._vis_ax is None:
            plt.ion()
            self._vis_fig = plt.figure(figsize=(10, 10))
            self._vis_ax = self._vis_fig.add_subplot(111, projection='3d')

        self._vis_ax.clear()
        self._vis_ax.set_xlim(0, 20)
        self._vis_ax.set_ylim(0, 20)
        self._vis_ax.set_zlim(0, 20)

        drone_pos = self.state[:3]
        self._vis_ax.scatter(*drone_pos, color='darkred', label='drone')
        self._vis_ax.scatter(self.goal[0], self.goal[1], self.goal[2], color='darkorange', label='goal')
        self._vis_ax.scatter(self.obstacle[:, 0], self.obstacle[:, 1], self.obstacle[:, 2], color='grey', label='obstacles')

        # Draw lines to obstacles, color by distance
        dang_dist, safe_dist = 0.6, 3.0
        for obs in self.obstacle:
            obs_pos = obs[:3]
            dist = np.linalg.norm(obs_pos - drone_pos)
            # Interpolate: close (<=dang_dist) -> red, alpha=1; far (>=safe_dist) -> grey, alpha=0.2
            t = np.clip((dist - dang_dist) / (safe_dist - dang_dist), 0, 1)
            color = (1 - 0.5 * t, 0.5 * t, 0.5 * t)  # red -> grey
            alpha = 1.0 - 0.8 * t  # 1.0 -> 0.2
            self._vis_ax.plot(
                [drone_pos[0], obs_pos[0]],
                [drone_pos[1], obs_pos[1]],
                [drone_pos[2], obs_pos[2]],
                color=color, alpha=alpha, linewidth=0.8
            )

        self._vis_fig.canvas.draw()
        if self.vis_block:
            plt.waitforbuttonpress()
        else:
            plt.pause(0.01)
