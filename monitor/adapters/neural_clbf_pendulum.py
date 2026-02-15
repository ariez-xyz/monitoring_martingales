"""Adapter for neural_clbf inverted pendulum system."""
from .interface import DynamicalSystemAdapter
from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np

# Import neural_clbf (this also sets up the cvxpylayers stub)
from neural_clbf.controllers import NeuralCLBFController


class NeuralCLBFPendulum(DynamicalSystemAdapter):
    """Adapter for the neural_clbf inverted pendulum with learned CLF."""

    def __init__(
        self,
        checkpoint_path: str = "neural_clbf/saved_models/review/inverted_pendulum_clf.ckpt",
        dt: Optional[float] = None,
        noise_level: float = 0.0,
        vis_every: int = 0,
        vis_block: bool = False,
    ):
        """
        Args:
            checkpoint_path: Path to the pretrained checkpoint
            dt: Timestep for simulation. If None, uses the model's default (0.01)
            noise_level: Control-noise magnitude for this adapter.
                        Applies additive uniform-in-ball noise in control space.
                        For pendulum (1D control), this is equivalent to uniform
                        noise in [-noise_level, noise_level] added to torque.
            vis_every: Visualize every N steps (0 to disable)
            vis_block: If True, wait for keypress after each visualization
        """
        # Load pretrained controller
        self.controller = NeuralCLBFController.load_from_checkpoint(checkpoint_path)
        self.controller.eval()
        self.dynamics = self.controller.dynamics_model

        # Use model's dt or override
        self.dt = dt if dt is not None else self.dynamics.dt
        # Keep control update period fixed to the model/training timestep.
        self.control_period = float(self.dynamics.dt)
        if noise_level < 0:
            raise ValueError("noise_level must be non-negative")

        self.noise_level = float(noise_level)

        # Control updates are held at the training cadence; dt only affects simulation fidelity.
        self.update_control_every = max(1, int(round(self.control_period / self.dt)))
        self._cached_control: Optional[torch.Tensor] = None

        # Visualization settings
        self.vis_every = vis_every
        self.vis_block = vis_block
        self._vis_fig = None
        self._vis_axes = None

        # Certificate bounds - estimated via scripts/estimate_pendulum_bounds.py
        # Grid over [-2,2]²: V_max ≈ 20.1 at corner (-1.72, -2.0)
        # V_min ≈ 0 at origin (goal state)
        self.certificate_bounds: Tuple[float, float] = (0.0, 22.0)

        # Temporal Lipschitz constant γ = |ΔR|/k (reward change per timestep)
        # Cached by (dt, noise_level) since both affect reward variation.
        # Used by OptimalTemporalWeights for m* = (c1/γ)^(2/3)
        self._lipschitz_cache: Dict[Tuple[float, float], float] = {}

        self.reset()

    def reset(self, initial_state: Optional[torch.Tensor] = None):
        """Reset the simulation."""
        if initial_state is not None:
            self.state = initial_state.clone()
        else:
            # Random initial state within bounds
            upper, lower = self.dynamics.state_limits
            self.state = lower + torch.rand(self.dynamics.n_dims) * (upper - lower)

        self.state_history = [self.state.clone()]
        self.control_history = []  # Nominal control history (for visualization)
        self.applied_control_history = []  # Noisy control history (for diagnostics)
        self.clf_history = [float(self.get_certificate_value())]
        self._step_count = 0
        self._cached_control = None
        self.is_done = False

    def done(self) -> bool:
        """Check if simulation is complete (goal reached)."""
        # Goal: near origin
        if self.dynamics.goal_mask(self.state.unsqueeze(0)).item():
            return True
        return self.is_done

    def get_reward_bounds(self) -> Tuple[float, float]:
        """Compute reward bounds from certificate bounds.

        Reward = V(x) - V(Y), so bounds are [cert_lo - cert_hi, cert_hi - cert_lo].
        """
        cert_lo, cert_hi = self.certificate_bounds
        return (cert_lo - cert_hi, cert_hi - cert_lo)

    def get_reward(self, next_state: torch.Tensor, cur_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute reward: V(Y) - V(x).

        For CLF, we want V to decrease, so negative reward means CLF condition satisfied.
        This matches the interface contract: E[reward] <= 0 indicates safety.
        """
        cur_v = self.get_certificate_value(cur_state)
        next_v = self.get_certificate_value(next_state)
        return next_v - cur_v

    def get_certificate_value(self, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute V(x) using the learned CLF."""
        resolved_state = self.resolve_state(state)
        if resolved_state.dim() == 1:
            resolved_state = resolved_state.unsqueeze(0)

        with torch.no_grad():
            V = self.controller.V(resolved_state)

        # Clamp to declared bounds
        lo, hi = self.certificate_bounds
        V = torch.clamp(V, lo, hi)

        return V.squeeze()

    def step(self) -> torch.Tensor:
        """Take one simulation step using nominal controller."""
        state_batch = self.state.unsqueeze(0)

        with torch.no_grad():
            # Get control-affine dynamics
            f, g = self.dynamics.control_affine_dynamics(state_batch)

            # Update nominal control only at specified frequency (zero-order hold).
            u_nominal = self._get_control(state_batch)
            u = u_nominal + self._sample_control_noise(n_samples=1)

            # Compute state derivative: ẋ = f + g @ u
            xdot = f.squeeze(-1) + (g @ u.unsqueeze(-1)).squeeze(-1)

            # Euler integration
            next_state = self.state + self.dt * xdot.squeeze(0)

        self.state = next_state
        self.state_history.append(self.state.clone())
        self.control_history.append(float(u_nominal.squeeze()))
        self.applied_control_history.append(float(u.squeeze()))
        self.clf_history.append(float(self.get_certificate_value()))
        self._step_count += 1

        # Visualization
        if self.vis_every > 0 and self._step_count % self.vis_every == 0:
            self._visualize()

        return self.state.clone()

    def sample(self, state: Optional[torch.Tensor] = None, n_samples: int = 1) -> torch.Tensor:
        """Sample n next states (with noise if configured)."""
        resolved_state = self.resolve_state(state)
        state_batch = resolved_state.unsqueeze(0)

        with torch.no_grad():
            f, g = self.dynamics.control_affine_dynamics(state_batch)
            # Sample path is side-effect free: compute nominal control directly.
            u_nominal = self.dynamics.u_nominal(state_batch)

            # Draw control noise independently per sample.
            u_samples = u_nominal.expand(n_samples, -1) + self._sample_control_noise(
                n_samples=n_samples
            )

            f_samples = f.squeeze(-1).expand(n_samples, -1)
            g_samples = g.expand(n_samples, -1, -1)
            xdot_samples = f_samples + (g_samples @ u_samples.unsqueeze(-1)).squeeze(-1)
            next_states = resolved_state.unsqueeze(0) + self.dt * xdot_samples

        return next_states

    def _sample_control_noise(self, n_samples: int) -> torch.Tensor:
        """Sample additive control noise in control space."""
        if self.noise_level <= 0:
            return torch.zeros(n_samples, self.dynamics.n_controls)

        # Inverted pendulum has 1D control (torque), so sample directly from
        # uniform[-noise_level, noise_level] instead of generic ball sampling.
        if self.dynamics.n_controls == 1:
            return (2.0 * torch.rand(n_samples, 1) - 1.0) * self.noise_level

        # Generic fallback for multi-input systems.
        dim = self.dynamics.n_controls
        direction = torch.randn(n_samples, dim)
        direction = direction / torch.norm(direction, dim=1, keepdim=True)
        radius = torch.rand(n_samples, 1) ** (1.0 / dim)
        return (direction * radius) * self.noise_level

    def _get_control(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Get control using fixed-period zero-order hold tied to training dt."""
        need_update = self._cached_control is None or self._step_count % self.update_control_every == 0
        if need_update:
            u = self.dynamics.u_nominal(state_batch)
            self._cached_control = u
            return u
        # _cached_control is guaranteed non-None when no update is needed.
        return self._cached_control  # type: ignore[return-value]

    def get_state_dim(self) -> int:
        """Returns state dimension (2 for inverted pendulum)."""
        return self.dynamics.n_dims

    def get_state_history(self) -> torch.Tensor:
        """Return history of states."""
        return torch.stack(self.state_history)

    def distance(self, state1: torch.Tensor, state2: torch.Tensor) -> float:
        """Euclidean distance on full state."""
        return float(torch.linalg.norm(state1 - state2))

    def get_lipschitz_constant(self) -> float:
        """Returns an empirical per-step drift bound proxy for current config.

        For the test-based monitor we use this as a proxy for B_k at fixed step
        size. It is estimated conservatively from one-step |Delta V| samples and
        cached by (dt, noise_level).
        """
        cache_key = (float(self.dt), float(self.noise_level))

        # Empirical preseeds (conservative rounded maxima) obtained with:
        #   python scripts/check_pendulum_lipschitz_variance.py \
        #     --fast-private --trials 10 --same-adapter-repeats 0 \
        #     --n-episodes 100 --max-steps 20000 --percentile 99.999 \
        #     --dt {0.001,0.01} --noise-level {0,0.1,1,10}
        preseed = {
            (0.001, 0.0): 0.20,
            (0.001, 0.1): 0.20,
            (0.001, 1.0): 0.20,
            (0.001, 10.0): 0.25,
            (0.01, 0.0): 1.95,
            (0.01, 0.1): 1.90,
            (0.01, 1.0): 2.00,
            (0.01, 10.0): 2.50,
        }
        if cache_key in preseed and cache_key not in self._lipschitz_cache:
            self._lipschitz_cache[cache_key] = preseed[cache_key]

        if cache_key not in self._lipschitz_cache:
            self._lipschitz_cache[cache_key] = self._estimate_lipschitz_constant()
        return self._lipschitz_cache[cache_key]

    def _estimate_lipschitz_constant(
        self,
        n_episodes: int = 30,
        max_steps: int = 20000,
        percentile: float = 99.999,
        return_diffs: bool = False,
    ) -> Union[float, Tuple[float, List[float]]]:
        """Estimate conservative one-step drift bound from rollout samples.

        At each state, evaluates reward for endpoint control noises
        (+/- noise_level) and keeps max(|Delta V|). Returns a high percentile
        over all collected samples.

        Args:
            n_episodes: Number of episodes to run
            max_steps: Max steps per episode
            percentile: Percentile of |Delta V| samples to return

        Returns:
            Estimated bound proxy.
            If return_diffs=True, returns (bound, samples).
        """
        # Save current state
        saved_state = self.state.clone()
        saved_history = list(self.state_history)
        saved_control = list(self.control_history)
        saved_applied_control = list(self.applied_control_history)
        saved_clf = list(self.clf_history)
        saved_step = self._step_count
        saved_done = self.is_done

        # Per-step bound proxy samples for |Delta V| at each visited state.
        step_bound_samples: List[float] = []

        for _ in range(n_episodes):
            self.reset()
            for step in range(max_steps):
                # Compute a conservative one-step reward proxy at the current state:
                # evaluate both endpoint control noises (+/- noise_level) and keep
                # the larger absolute reward magnitude.
                state_batch = self.state.unsqueeze(0)
                with torch.no_grad():
                    f, g = self.dynamics.control_affine_dynamics(state_batch)
                    u_nominal = self._get_control(state_batch)
                    if self.noise_level > 0 and self.dynamics.n_controls == 1:
                        noise_plus = torch.full((1, 1), self.noise_level, dtype=u_nominal.dtype)
                        noise_minus = -noise_plus
                        u_plus = u_nominal + noise_plus
                        u_minus = u_nominal + noise_minus
                        xdot_plus = f.squeeze(-1) + (g @ u_plus.unsqueeze(-1)).squeeze(-1)
                        xdot_minus = f.squeeze(-1) + (g @ u_minus.unsqueeze(-1)).squeeze(-1)
                        next_state_plus = self.state + self.dt * xdot_plus.squeeze(0)
                        next_state_minus = self.state + self.dt * xdot_minus.squeeze(0)
                    else:
                        # Zero-noise case and generic multi-control fallback.
                        xdot_nominal = f.squeeze(-1) + (g @ u_nominal.unsqueeze(-1)).squeeze(-1)
                        next_state_plus = self.state + self.dt * xdot_nominal.squeeze(0)
                        next_state_minus = next_state_plus

                V_cur = float(self.get_certificate_value(self.state))
                reward_plus = float(self.get_certificate_value(next_state_plus)) - V_cur
                reward_minus = float(self.get_certificate_value(next_state_minus)) - V_cur
                step_bound = max(abs(reward_plus), abs(reward_minus))
                step_bound_samples.append(step_bound)

                # Roll out trajectory with a sampled noise draw to keep state coverage
                # broad while using endpoint rewards for conservative local variation.
                with torch.no_grad():
                    u_rollout = u_nominal + self._sample_control_noise(n_samples=1)
                    xdot_rollout = f.squeeze(-1) + (g @ u_rollout.unsqueeze(-1)).squeeze(-1)
                    next_state_rollout = self.state + self.dt * xdot_rollout.squeeze(0)

                self.state = next_state_rollout
                self._step_count += 1

                if self.dynamics.goal_mask(self.state.unsqueeze(0)).item():
                    break

        # Restore state
        self.state = saved_state
        self.state_history = saved_history
        self.control_history = saved_control
        self.applied_control_history = saved_applied_control
        self.clf_history = saved_clf
        self._step_count = saved_step
        self.is_done = saved_done

        if not step_bound_samples:
            gamma = 0.1  # Fallback
            if return_diffs:
                return gamma, step_bound_samples
            return gamma

        gamma = float(np.percentile(step_bound_samples, percentile))
        gamma = max(gamma, 1e-6)  # Avoid zero
        if return_diffs:
            return gamma, step_bound_samples
        return gamma

    def get_expected_next_state(self, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute E[Y] analytically.

        For deterministic dynamics with zero-mean noise:
        E[Y] = x + dt * (f + g @ u)
        """
        resolved_state = self.resolve_state(state)
        state_batch = resolved_state.unsqueeze(0)

        with torch.no_grad():
            f, g = self.dynamics.control_affine_dynamics(state_batch)
            u = self.dynamics.u_nominal(state_batch)
            xdot = f.squeeze(-1) + (g @ u.unsqueeze(-1)).squeeze(-1)
            expected_next = resolved_state + self.dt * xdot.squeeze(0)

        return expected_next

    def _visualize(self):
        """Render current state with multi-panel visualization."""
        import matplotlib.pyplot as plt

        if self._vis_fig is None or self._vis_axes is None:
            plt.ion()
            self._vis_fig, self._vis_axes = plt.subplots(2, 2, figsize=(12, 10))
            self._vis_fig.suptitle('Inverted Pendulum CLF Control', fontsize=14)

        ax_phase, ax_pendulum, ax_time, ax_clf = self._vis_axes.flatten()

        # Get trajectory data
        history = torch.stack(self.state_history).numpy()
        theta = history[:, 0]
        theta_dot = history[:, 1]
        t = np.arange(len(history)) * self.dt

        # --- Top Left: Phase Space (θ vs θ̇) ---
        ax_phase.clear()
        ax_phase.plot(theta, theta_dot, 'b-', alpha=0.6, linewidth=1, label='Trajectory')
        ax_phase.plot(theta[-1], theta_dot[-1], 'ro', markersize=10, label='Current')
        ax_phase.plot(0, 0, 'g*', markersize=15, label='Goal')
        ax_phase.set_xlabel('θ (rad)')
        ax_phase.set_ylabel('θ̇ (rad/s)')
        ax_phase.set_title('Phase Space')
        ax_phase.legend(loc='upper right')
        ax_phase.grid(True, alpha=0.3)
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-3, 3)
        ax_phase.axhline(y=0, color='k', linewidth=0.5)
        ax_phase.axvline(x=0, color='k', linewidth=0.5)

        # --- Top Right: Pendulum Animation ---
        ax_pendulum.clear()
        L = 1.0  # pendulum length for visualization
        x_end = L * np.sin(theta[-1])
        y_end = -L * np.cos(theta[-1])

        # Draw pendulum
        ax_pendulum.plot([0, x_end], [0, y_end], 'b-', linewidth=4)
        ax_pendulum.plot(0, 0, 'ko', markersize=10)  # pivot
        ax_pendulum.plot(x_end, y_end, 'ro', markersize=15)  # bob

        # Draw ghost trail (last few positions)
        n_ghost = min(20, len(theta))
        for i in range(n_ghost):
            idx = -(n_ghost - i)
            x_g = L * np.sin(theta[idx])
            y_g = -L * np.cos(theta[idx])
            alpha = 0.1 + 0.4 * (i / n_ghost)
            ax_pendulum.plot(x_g, y_g, 'o', color='blue', alpha=alpha, markersize=5)

        ax_pendulum.set_xlim(-1.5, 1.5)
        ax_pendulum.set_ylim(-1.5, 1.5)
        ax_pendulum.set_aspect('equal')
        ax_pendulum.set_title(f'Pendulum (θ={theta[-1]:.3f} rad)')
        ax_pendulum.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
        ax_pendulum.grid(True, alpha=0.3)

        # --- Bottom Left: States and Control over Time ---
        ax_time.clear()
        ax_time.plot(t, theta, 'b-', label='θ', linewidth=1.5)
        ax_time.plot(t, theta_dot, 'g-', label='θ̇', linewidth=1.5)
        if self.control_history:
            t_ctrl = t[1:]  # control starts at step 1
            ax_time.plot(t_ctrl, self.control_history, 'r-', label='u', linewidth=1.5, alpha=0.7)
        ax_time.set_xlabel('Time (s)')
        ax_time.set_ylabel('Value')
        ax_time.set_title('States & Control')
        ax_time.legend(loc='upper right')
        ax_time.grid(True, alpha=0.3)
        ax_time.axhline(y=0, color='k', linewidth=0.5)

        # --- Bottom Right: CLF Value ---
        ax_clf.clear()
        ax_clf.plot(t, self.clf_history, 'purple', linewidth=2)
        ax_clf.fill_between(t, 0, self.clf_history, alpha=0.3, color='purple')
        ax_clf.set_xlabel('Time (s)')
        ax_clf.set_ylabel('V(x)')
        ax_clf.set_title(f'CLF Value (V={self.clf_history[-1]:.4f})')
        ax_clf.grid(True, alpha=0.3)
        ax_clf.set_ylim(bottom=0)
        ax_clf.axhline(y=0, color='k', linewidth=0.5)

        plt.tight_layout()
        self._vis_fig.canvas.draw()

        if self.vis_block:
            plt.waitforbuttonpress()
        else:
            plt.pause(0.01)
