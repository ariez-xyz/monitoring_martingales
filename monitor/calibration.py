from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import torch

from .adapters import DynamicalSystemAdapter

AdapterFactory = Callable[[], DynamicalSystemAdapter]

class LipschitzConstantEstimator:
    """Estimate calibration constants from rollout data."""

    def estimate_drift_bound(
        self,
        adapter_factory: AdapterFactory,
        n_episodes: int = 500,
        max_steps: int = 20,
        percentile: float = 100.0,
        samples_per_step: int = 8,
    ) -> float:
        """Estimate a conservative one-step drift bound.

        Coverage note:
        For rollout-based calibration, prefer increasing `n_episodes` over
        increasing `max_steps`. Longer rollouts concentrate samples near the
        target/stable region and underexplore the broader state space, while
        more resets tend to improve coverage of the operating region.
        """
        step_bound_samples = []
        adapter = adapter_factory()

        for _ in range(n_episodes):
            adapter.reset()
            for _ in range(max_steps):
                current_state = adapter.get_state()
                next_states = adapter.sample(n_samples=samples_per_step, include_extremes=True)
                drift_batch = adapter.get_drift(next_states, current_state)
                step_bound_samples.append(float(torch.max(torch.abs(drift_batch))))
                adapter.step()
                if adapter.done():
                    break

        if not step_bound_samples:
            raise ValueError("No rollout samples were collected while estimating drift bound")

        gamma = float(np.percentile(step_bound_samples, percentile))
        return max(gamma, 1e-6)


    def estimate_transition_wasserstein_lipschitz(
        self,
        adapter_factory: AdapterFactory,
        n_episodes: int = 500,
        max_steps: int = 20,
        percentile: float = 100.0,
    ) -> float:
        """Estimate W1-Lipschitz bound for one-step kernels.

        Coverage note:
        For rollout-based calibration, prefer increasing `n_episodes` over
        increasing `max_steps`. Longer rollouts concentrate samples near the
        target/stable region and underexplore the broader state space, while
        more resets tend to improve coverage of the operating region.
        """
        wasserstein_distances = []
        adapter = adapter_factory()

        for _ in range(n_episodes):
            adapter.reset()
            for _ in range(max_steps):
                # Wasserstein distance for **ZERO-MEAN** distributions reduces to
                # ratio of distances, hence our procedure:

                # 1. Sample two nearby states
                s1, s2 = adapter.sample(n_samples=2, noise_level=0.1)

                # 2. Get expected successors (noise_level=0 is expectation for zero mean)
                t1 = adapter.sample(s1, noise_level=0)
                t2 = adapter.sample(s2, noise_level=0) 

                # 3. Compute distances
                state_distance = adapter.distance(s1, s2)
                succ_distance = adapter.distance(t1, t2)

                # 4. Compute empirical bound rho:
                #      W1(P(s1), P(s2))             <= rho * d(s1, s2)
                # <=>  W1(P(s1), P(s2)) / d(s1, s2) <= rho
                # <=>         d(t1, t2) / d(s1, s2) <= rho
                # (the latter simplification holds because noise laws are translated copies)
                wasserstein_distances.append(succ_distance / state_distance)

                adapter.step()
                if adapter.done():
                    break

        if not wasserstein_distances:
            raise ValueError("No non-zero state distances were collected while estimating Wasserstein Lipschitz bound")

        rho = float(np.percentile(wasserstein_distances, percentile))
        return max(rho, 1e-6)


class LipschitzConstantProvider:
    """Thin JSON-backed store for precomputed and runtime-calibrated constants."""

    _json_path = Path(__file__).resolve().parent.parent / "calibration_constants.json"
    _loaded = False
    _precomputed_drift_bounds: Dict[str, float] = {}
    _precomputed_transition_lipschitz: Dict[str, float] = {}
    _drift_bound_cache: Dict[str, float] = {}
    _transition_lipschitz_cache: Dict[str, float] = {}

    @classmethod
    def _encode_key(cls, key: Dict[str, Any]) -> str:
        if not isinstance(key, dict):
            raise TypeError(f"Expected bound_key() to return a dict, got {key!r}")
        return json.dumps(key, sort_keys=True, separators=(",", ":"))

    @classmethod
    def _load_if_needed(cls) -> None:
        if cls._loaded:
            return

        with cls._json_path.open() as f:
            data = json.load(f)

        cls._precomputed_drift_bounds = dict(data.get("drift_bounds", {}))
        cls._precomputed_transition_lipschitz = dict(data.get("transition_wasserstein_lipschitz", {}))
        cls._loaded = True

    @classmethod
    def _write_json(cls) -> None:
        data = {
            "drift_bounds": cls._precomputed_drift_bounds,
            "transition_wasserstein_lipschitz": cls._precomputed_transition_lipschitz,
        }
        with cls._json_path.open("w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")

    @classmethod
    def reload(cls) -> None:
        cls._loaded = False
        cls._load_if_needed()

    @classmethod
    def clear_cache(cls) -> None:
        cls._drift_bound_cache.clear()
        cls._transition_lipschitz_cache.clear()

    @classmethod
    def _validate_overwrite(
        cls,
        existing_value: float | None,
        new_value: float,
        force: bool,
        quantity: str,
        cache_key: str,
    ) -> None:
        if existing_value is not None and new_value < existing_value and not force:
            raise ValueError(
                f"Refusing to decrease {quantity} for key {cache_key!r} "
                f"from {existing_value} to {new_value} without force=True"
            )

    @classmethod
    def set_drift_bound(
        cls,
        adapter: DynamicalSystemAdapter,
        value: float,
        *,
        force: bool = False,
    ) -> None:
        cls._load_if_needed()
        cache_key = cls._encode_key(adapter.bound_key())
        existing_value = cls._precomputed_drift_bounds.get(cache_key)
        if existing_value is None:
            existing_value = cls._drift_bound_cache.get(cache_key)
        cls._validate_overwrite(existing_value, value, force, "drift bound", cache_key)
        cls._drift_bound_cache[cache_key] = value
        cls._precomputed_drift_bounds[cache_key] = value
        cls._write_json()

    @classmethod
    def set_transition_wasserstein_lipschitz(
        cls,
        adapter: DynamicalSystemAdapter,
        value: float,
        *,
        force: bool = False,
    ) -> None:
        cls._load_if_needed()
        cache_key = cls._encode_key(adapter.bound_key())
        existing_value = cls._precomputed_transition_lipschitz.get(cache_key)
        if existing_value is None:
            existing_value = cls._transition_lipschitz_cache.get(cache_key)
        cls._validate_overwrite(
            existing_value,
            value,
            force,
            "transition Wasserstein Lipschitz bound",
            cache_key,
        )
        cls._transition_lipschitz_cache[cache_key] = value
        cls._precomputed_transition_lipschitz[cache_key] = value
        cls._write_json()

    @classmethod
    def get_drift_bound(cls, adapter: DynamicalSystemAdapter) -> float:
        """Return a drift bound from precomputed values or cache."""
        cls._load_if_needed()
        cache_key = cls._encode_key(adapter.bound_key())
        if cache_key in cls._precomputed_drift_bounds:
            return cls._precomputed_drift_bounds[cache_key]
        if cache_key in cls._drift_bound_cache:
            return cls._drift_bound_cache[cache_key]
        raise KeyError(f"No drift bound registered for key {cache_key!r}")

    @classmethod
    def get_transition_wasserstein_lipschitz(
        cls,
        adapter: DynamicalSystemAdapter,
    ) -> float:
        """Return a transition-kernel Lipschitz bound from precomputed values or cache."""
        cls._load_if_needed()
        cache_key = cls._encode_key(adapter.bound_key())
        if cache_key in cls._precomputed_transition_lipschitz:
            return cls._precomputed_transition_lipschitz[cache_key]
        if cache_key in cls._transition_lipschitz_cache:
            return cls._transition_lipschitz_cache[cache_key]
        raise KeyError(f"No transition Wasserstein Lipschitz bound registered for key {cache_key!r}")
