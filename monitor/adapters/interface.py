from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import torch
import numpy as np

class DynamicalSystemAdapter(ABC):
    state_history: List[torch.Tensor]
    @abstractmethod
    def done(self) -> bool:
        """
        Return True iff done (target region reached).
        """
        pass

    @abstractmethod
    def get_reward_bounds(self) -> Tuple[float, float]:
        """
        Returns (lower, upper) bounds on the reward returned by get_reward().

        These bounds are required for concentration inequalities (e.g., Hoeffding).
        Implementations should raise an exception if observed values exceed these bounds.
        """
        pass

    @abstractmethod
    def get_reward(self, next_state: torch.Tensor, cur_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculates the reward for monitoring the certificate condition.

        The reward should be computed such that E[reward] ≤ 0 indicates the
        certificate condition is satisfied. Adapters handle sign conventions
        and alpha functions internally.

        Args:
            next_state: Tensor of sampled next states, shape (n_samples, state_dim)
            cur_state: Optional current state tensor. If None, uses adapter's current state.

        Returns:
            Tensor of shape (n_samples,) representing reward for each sample.
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

        Returns:
            Tensor of shape (state_dim) representing next state.
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
    def sample(self, state: Optional[torch.Tensor] = None, n_samples: int = 1) -> torch.Tensor:
        """
        Samples 'n_samples' next states for given or current state.

        Args:
            state: Tensor of shape (state_dim)
            n_samples: Number of stochastic samples to draw

        Returns:
            Tensor of shape (n_samples, state_dim)
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

    def get_lipschitz_constant(self) -> float:
        """
        Returns the Lipschitz constant γ for the expected reward function.

        The Lipschitz assumption is: |E[R(x)] - E[R(x')]| ≤ γ · d(x, x')

        This is used by HistoryEstimator to compute discretization error.
        The constant depends on the (CBF, controller, dynamics) triple and
        should be estimated empirically or computed from network weights.

        Raises:
            NotImplementedError: If Lipschitz constant is not available.
        """
        raise NotImplementedError("Lipschitz constant not available for this adapter")

    def get_expected_next_state(self, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes E[Y] where Y ~ P(x) is the next state analytically.

        This requires knowledge of the transition dynamics. Not all adapters
        can implement this - the default raises NotImplementedError.

        When Jensen's gap is small (i.e., V is approximately linear over the
        noise distribution), V(E[Y]) ≈ E[V(Y)], allowing near-exact computation
        of expected rewards without Monte Carlo sampling.

        Args:
            state: Optional current state tensor. If None, uses adapter's current state.

        Returns:
            Tensor of shape (state_dim,) representing E[Y].

        Raises:
            NotImplementedError: If analytic computation is not available.
        """
        raise NotImplementedError("Analytic expected next state not available for this adapter")

    def resolve_state(self, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns the dimension of the state space."""
        if state is not None: 
            return state
        elif hasattr(self, 'state') and self.state is not None:  # type: ignore
            if isinstance(self.state, np.ndarray):               # type: ignore
                return torch.from_numpy(self.state).float()      # type: ignore
            return self.state                                    # type: ignore
        else:
            raise RuntimeError("No state provided and object has no valid state attribute")

