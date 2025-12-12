"""Adapter for neural_clbf inverted pendulum system."""
from .interface import DynamicalSystemAdapter
from typing import Optional, Tuple
import torch
import numpy as np

# Import neural_clbf (this also sets up the cvxpylayers stub)
from neural_clbf.controllers import NeuralCLBFController


def _sample_uniform_ball(n_samples: int, dim: int) -> torch.Tensor:
    """Sample uniformly from the unit ball in R^dim."""
    # Sample direction uniformly on sphere
    direction = torch.randn(n_samples, dim)
    direction = direction / torch.norm(direction, dim=1, keepdim=True)
    # Sample radius: for uniform in ball, use r ~ U[0,1]^(1/dim)
    radius = torch.rand(n_samples, 1) ** (1.0 / dim)
    return direction * radius


class NeuralCLBFPendulum(DynamicalSystemAdapter):
    """Adapter for the neural_clbf inverted pendulum with learned CLF."""

    def __init__(
        self,
        checkpoint_path: str = "neural_clbf/saved_models/review/inverted_pendulum_clf.ckpt",
        dt: Optional[float] = None,
        noise_scale: float = 0.0,
        vis_every: int = 0,
        vis_block: bool = False,
    ):
        """
        Args:
            checkpoint_path: Path to the pretrained checkpoint
            dt: Timestep for simulation. If None, uses the model's default (0.01)
            noise_scale: Noise radius = noise_scale * dt. Uniform ball distribution.
                         E.g., noise_scale=1.0 with dt=0.01 gives radius=0.01.
            vis_every: Visualize every N steps (0 to disable)
            vis_block: If True, wait for keypress after each visualization
        """
        # Load pretrained controller
        self.controller = NeuralCLBFController.load_from_checkpoint(checkpoint_path)
        self.controller.eval()
        self.dynamics = self.controller.dynamics_model

        # Use model's dt or override
        self.dt = dt if dt is not None else self.dynamics.dt
        self.noise_scale = noise_scale

        # Visualization settings
        self.vis_every = vis_every
        self.vis_block = vis_block
        self._vis_fig = None
        self._vis_axes = None

        # Certificate bounds - estimated via scripts/estimate_pendulum_bounds.py
        # Grid over [-2,2]²: V_max ≈ 20.1 at corner (-1.72, -2.0)
        # V_min ≈ 0 at origin (goal state)
        self.certificate_bounds: Tuple[float, float] = (0.0, 22.0)

        # Lipschitz constant - need to estimate empirically
        # TODO: Run estimation script
        self.lipschitz_constant: float = 1.0

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
        self.control_history = []
        self.clf_history = [float(self.get_certificate_value())]
        self._step_count = 0
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
        """Compute reward: V(x) - V(Y).

        For CLF, we want V to decrease, so positive reward means CLF condition satisfied.
        """
        cur_v = self.get_certificate_value(cur_state)
        next_v = self.get_certificate_value(next_state)
        return cur_v - next_v

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

            # Use nominal LQR controller (avoids QP solver issues)
            u = self.dynamics.u_nominal(state_batch)

            # Compute state derivative: ẋ = f + g @ u
            xdot = f.squeeze(-1) + (g @ u.unsqueeze(-1)).squeeze(-1)

            # Euler integration
            next_state = self.state + self.dt * xdot.squeeze(0)

            # Add noise if configured (uniform ball, radius = noise_scale * dt)
            if self.noise_scale > 0:
                noise_radius = self.noise_scale * self.dt
                noise = _sample_uniform_ball(1, self.dynamics.n_dims).squeeze(0) * noise_radius
                next_state = next_state + noise

        self.state = next_state
        self.state_history.append(self.state.clone())
        self.control_history.append(float(u.squeeze()))
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
            u = self.dynamics.u_nominal(state_batch)
            xdot = f.squeeze(-1) + (g @ u.unsqueeze(-1)).squeeze(-1)
            deterministic_next = resolved_state + self.dt * xdot.squeeze(0)

        if self.noise_scale > 0:
            noise_radius = self.noise_scale * self.dt
            noise = _sample_uniform_ball(n_samples, self.dynamics.n_dims) * noise_radius
            next_states = deterministic_next.unsqueeze(0) + noise
        else:
            # No noise: all samples are identical
            next_states = deterministic_next.unsqueeze(0).expand(n_samples, -1)

        return next_states

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
        """Returns Lipschitz constant for expected rewards."""
        return self.lipschitz_constant

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
