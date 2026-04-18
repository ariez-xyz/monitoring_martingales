from abc import ABC, abstractmethod
from typing import Generator, Tuple, Literal, Any, List, Dict
from .adapters import DynamicalSystemAdapter
from .calibration import LipschitzConstantProvider
from .estimators import Estimator

class Monitor(ABC):
    @abstractmethod
    def __call__(self, adapter: DynamicalSystemAdapter) -> Generator[Tuple[Literal["T","F","?"], Dict[Any, Any]], None, None]:
        pass

    @abstractmethod
    def _viz_item(self, item: Tuple[Literal["T","F","?"], Dict[Any, Any]]):
        """
        For example, pretty print confidence intervals.
        """
        pass

    def viz(self, adapter: DynamicalSystemAdapter):
        for item in self(adapter):
            self._viz_item(item)


class EstimationMonitor(Monitor):
    def __init__(self, estimator: Estimator):
        self.estimator = estimator

    def __call__(self, adapter: DynamicalSystemAdapter, continuous: bool = False) -> Generator[Tuple[Literal["T","F","?"], Dict[Any, Any]], None, None]:
        step_idx = 0
        try:
            while not adapter.done():
                verdict, lower, upper, info = self.estimator(adapter, continuous)
                info = dict(info)
                info["ci"] = (lower, upper)
                yield verdict, info
                adapter.step()
                step_idx += 1
        except KeyboardInterrupt:
            pass

    def _viz_item(self, item: Tuple[Literal["T","F","?"], Dict[Any, Any]]):
        verdict, info = item
        print(f"{verdict} {info['ci']}")


class HypothesisTestingMonitor(Monitor):
    def __init__(self, delta: float):
        self.delta = delta
        self._reset()

    def _reset(self):
        self.last_verdict = "?"
        self.E_cache: List[float] = [1]
        self.S_cache: List[float] = [0]
        self.V_cache: List[float] = [0]
        self.Delta: List[float] = []

    def beta(self, n: int, adapter: DynamicalSystemAdapter) -> float:
        """
        Predictable plug-in bet
        """
        # Note: We assume adapter.dt never varies, so B_n = B for all n
        B = LipschitzConstantProvider.get_drift_bound(adapter)
        return min(1/B, max(0, self.S(n) / (self.V(n) + B**2)))

    def S(self, n: int) -> float:
        """
        S_n = sum_{i=0}^{n-1} Delta_i
        """
        assert len(self.Delta) >= n-1
        if n <= 0: return 0
        for k in range(len(self.S_cache), n+1):
            next_S = self.S_cache[-1] + self.Delta[k-1]
            self.S_cache.append(next_S)
        return self.S_cache[n]

    def V(self, n: int) -> float:
        """
        V_n = sum_{i=0}^{n-1} Delta_i ** 2
        """
        assert len(self.Delta) >= n-1
        if n <= 0: return 0
        for k in range(len(self.V_cache), n+1):
            next_V = self.V_cache[-1] + self.Delta[k-1] ** 2
            self.V_cache.append(next_V)
        return self.V_cache[n]

    def E(self, n: int, adapter: DynamicalSystemAdapter) -> float:
        assert len(self.Delta) >= n-1
        if n <= 0: return 1
        for k in range(len(self.E_cache), n+1):
            next_E = self.E_cache[-1] * (1 + self.beta(k-1, adapter) * self.Delta[k-1])
            self.E_cache.append(next_E)
        return self.E_cache[n]

    def __call__(self, adapter: DynamicalSystemAdapter) -> Generator[Tuple[Literal["T", "F","?"], Dict[Any, Any]], None, None]:
        self._reset()
        n = 0 # number of transitions taken

        try:
            while not adapter.done():
                adapter.step()
                self.Delta = adapter.get_drift_history().tolist()
                n += 1
                e_value = self.E(n, adapter)

                info = {
                    "n": n, 
                    "e_value": e_value,
                    "Delta_{n-1}": self.Delta[n-1],
                    "S_n": self.S(n),
                    "V_n": self.V(n),
                    "beta_{n-1}": self.beta(n-1, adapter),
                    "threshold": 1 / self.delta
                }

                if self.last_verdict == "F":
                    info["reason"] = "previous verdict is F"
                    yield "F", info
                elif e_value >= 1/self.delta:
                    self.last_verdict = "F"
                    info["reason"] = f"E_{n} >= {1/self.delta} (1/delta)"
                    yield "F", info
                else:
                    yield "?", info

        except KeyboardInterrupt:
            pass

    def _viz_item(self, item: Tuple[Literal["T","F","?"], Dict[Any, Any]]):
        verdict, info = item
        print(f"{verdict} {info['e_value']}")
