from typing import Generator, Tuple, Literal, Any, List, Dict
from .adapters import DynamicalSystemAdapter
from .estimators import Estimator


class NeuralCertificateMonitor:
    def __init__(
        self,
        adapter: DynamicalSystemAdapter,
        estimator: Estimator,
    ):
        self.adapter = adapter
        self.estimator = estimator

    def run(self) -> Generator[Tuple[Literal["T","F","?"], Dict[Any, Any]], None, None]:
        step_idx = 0
        try:
            while not self.adapter.done():
                verdict, lower, upper, info = self.estimator(self.adapter)
                info = dict(info)
                info["ci"] = (lower, upper)
                yield verdict, info
                self.adapter.step()
                step_idx += 1
        except KeyboardInterrupt:
            pass


class HypothesisTestingMonitor:
    def __init__(
        self,
        adapter: DynamicalSystemAdapter,
        delta: float,
    ):
        self.adapter = adapter
        self.delta = delta
        self.last_verdict = "?"
        self.E_cache: List[float] = [1]
        self.S_cache: List[float] = [0]
        self.V_cache: List[float] = [0]
        self.Delta: List[float] = []

    def cur_cert_value(self) -> float:
        return float(self.adapter.get_certificate_value())

    def beta(self, n) -> float:
        """
        Predictable plug-in bet
        """
        # Note: We assume adapter.dt never varies, so B_n = B for all n
        B = self.adapter.get_drift_bound() 
        return min(1/B, max(0, self.S(n-1) / (self.V(n-1) + B**2)))

    def S(self, n) -> float:
        assert len(self.Delta) >= n-1
        if n <= 0: return 0
        for k in range(len(self.S_cache), n+1):
            next_S = self.S_cache[-1] + self.Delta[k-1]
            self.S_cache.append(next_S)
        return self.S_cache[n]

    def V(self, n) -> float:
        assert len(self.Delta) >= n-1
        if n <= 0: return 0
        for k in range(len(self.V_cache), n+1):
            next_V = self.V_cache[-1] + self.Delta[k-1] ** 2
            self.V_cache.append(next_V)
        return self.V_cache[n]

    def E(self, n) -> float:
        assert len(self.Delta) >= n-1
        if n <= 0: return 1
        for k in range(len(self.E_cache), n+1):
            next_E = self.E_cache[-1] * (1 + self.beta(k-1) * self.Delta[k-1])
            self.E_cache.append(next_E)
        return self.E_cache[n]

    def run(self) -> Generator[Tuple[Literal["F","?"], Any], None, None]:
        n = 0 # number of transitions taken

        try:
            while not self.adapter.done():
                certificate_value = self.cur_cert_value()
                self.adapter.step()
                drift = self.cur_cert_value() - certificate_value
                self.Delta.append(drift)
                n += 1
                e_value = self.E(n)

                info = {
                    "n": n, 
                    "e_value": e_value,
                    "Delta_{n-1}": self.Delta[n - 1],
                    "S_{n-1}": self.S(n - 1),
                    "V_{n-1}": self.V(n - 1),
                    "beta{n-1}": self.beta(n - 1),
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
