from abc import ABC, abstractmethod
from typing import Optional
import torch
import numpy as np

class DynamicalSystemAdapter(ABC):
    @abstractmethod
    def done(self) -> bool:
        """
        Return True iff done (target region reached).
        """
        pass

    @abstractmethod
    def get_reward(self, next_state: torch.Tensor) -> torch.Tensor:
        """
        Calculates R(x) for given or current state.

        Returns:
            Tensor of shape (batch_size, 1) representing:
                V(x') - V(x)                   Lyapunov
                V(x') - V(x) - alpha(V(x))     Barrier
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

