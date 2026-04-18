from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import torch
import numpy as np

class DynamicalSystemAdapter(ABC):
    """
    Base interface for monitor-compatible systems.

    Noise contract:
    - Adapters should accept a constructor argument named `noise_level`.
    - `noise_level=0` should yield completely deterministic transitions.
    - Noise should be zero-mean, such that `noise_level=0` gives the expected next state.
    - Larger `noise_level` values should produce larger transition perturbations.
    - Normalization: Adapters shouldn't have wildly different scales for `noise_level`. A value of 0.5 should produce similar-ish outcomes.
    - The exact noise process is adapter-specific.
    """
    state_history: List[torch.Tensor]
    @abstractmethod
    def done(self) -> bool:
        """
        Return True iff done (target region reached).
        """
        pass

    @abstractmethod
    def get_drift(self, next_state: torch.Tensor, cur_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculates the drift for monitoring the certificate condition.

        In the simplest case, this is just:

        get_certificate_value(next_state) - get_certificate_value(cur_state)

        HOWEVER:

        The drift should be computed such that E[drift] ≤ 0 indicates the
        certificate condition is satisfied. Adapters handle sign conventions
        and alpha functions internally.

        Args:
            next_state: Tensor of sampled next states, shape (n_samples, state_dim)
            cur_state: Optional current state tensor. If None, uses adapter's current state.

        Returns:
            Tensor of shape (n_samples,) representing drift for each sample.
        """
        pass

    @abstractmethod
    def get_drift_history(self) -> torch.Tensor:
        """
        Return the history of per-step drifts.

        Returns:
            Tensor of shape (n_steps) representing past drifts.
        """
        pass

    @abstractmethod
    def get_certificate_value(self, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculates V(x) for given or current state.

        Returns:
            Tensor of shape (batch_size, 1) representing raw Lyapunov/Barrier values.
        """
        pass

    @abstractmethod
    def step(self) -> torch.Tensor:
        """
        Take a simulation step.

        Each call to step() should append a drift to the tensor returned by get_drift_history().

        Returns:
            Tensor of shape (state_dim) representing next state.
        """
        pass

    @abstractmethod
    def reset(self, seed: Optional[int] = None):
        """
        Reset the adapter to a fresh rollout state.

        Args:
            seed: Optional seed for deterministic reset behavior.
        """
        pass

    @abstractmethod
    def get_state(self) -> torch.Tensor:
        """
        Return the current state.

        Returns:
            Tensor of shape (state_dim) representing current state.
        """
        pass

    @abstractmethod
    def get_state_history(self) -> torch.Tensor:
        """
        Return the history of states.

        Returns:
            Tensor of shape (n_steps, state_dim) representing past states.
        """
        pass

    @abstractmethod
    def sample(
        self,
        state: Optional[torch.Tensor] = None,
        n_samples: int = 1,
        include_extremes: bool = False,
        noise_level: float = -1.0,
    ) -> torch.Tensor:
        """
        Samples 'n_samples' next states for given or current state.

        Args:
            state: Tensor of shape (state_dim)
            n_samples: Number of stochastic samples to draw
            include_extremes: If True, include adapter-specific extreme successor
                states when the noise model has meaningful bounded extremes.
            noise_level: Per-call noise override. Negative values use the
                adapter's configured noise level.

        Returns:
            Tensor of shape `(batch_size, state_dim)`, where `batch_size` is
            typically `n_samples` and may be larger if `include_extremes=True`.
        """
        pass

    @abstractmethod
    def get_state_dim(self) -> int:
        """Returns the dimension of the state space."""
        pass

    @abstractmethod
    def distance(self, state1: torch.Tensor, state2: torch.Tensor) -> float:
        """
        Computes distance between two states for the Lipschitz assumption.

        The choice of metric is domain-specific. For example, a drone adapter
        might use Euclidean distance on positions only, ignoring velocities.

        Args:
            state1: Tensor of shape (state_dim,)
            state2: Tensor of shape (state_dim,)

        Returns:
            Non-negative distance value.
        """
        pass

    @abstractmethod
    def bound_key(self) -> Dict[str, Any]:
        """
        Return a stable metadata dictionary used to look up calibrated bounds.

        The returned dict should include all adapter configuration fields that
        materially affect the calibrated constants. It must also include the
        adapter class name so keys remain unique across adapter types.
        """
        pass

    def resolve_state(self, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns the provided state tensor, or falls back to self.state if no state is provided."""
        if state is not None: 
            return state
        return self.get_state()

