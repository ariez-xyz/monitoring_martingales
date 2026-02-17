from typing import Generator, Tuple, Literal, Any, List
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

    def run(self) -> Generator[Tuple[Literal["T","F","?"], float, float, Any], None, None]:
        step_idx = 0
        try:
            while not self.adapter.done():
                yield self.estimator(self.adapter)
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
        self.G_cache: List[float] = [1]
        self.S_cache: List[float] = [0]
        self.V_cache: List[float] = [0]
        self.Delta: List[float] = []

    def cur_cert_value(self) -> float:
        return float(self.adapter.get_certificate_value())

    def eta(self, n) -> float:
        # Note: We assume adapter.dt never varies, so B_n = B for all n
        B = self.adapter.get_lipschitz_constant() 
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

    def G(self, n) -> float:
        assert len(self.Delta) >= n-1
        if n <= 0: return 1
        for k in range(len(self.G_cache), n+1):
            next_G = self.G_cache[-1] * (1 + self.eta(k-1) * self.Delta[k-1])
            self.G_cache.append(next_G)
        return self.G_cache[n]

    def run(self) -> Generator[Tuple[Literal["F","?"], Any], None, None]:
        n = 0

        try:
            while not self.adapter.done():
                certificate_value = self.cur_cert_value()
                self.adapter.step()
                # here: n is index of current transition, starting from 0
                reward = self.cur_cert_value() - certificate_value
                self.Delta.append(reward)

                if self.last_verdict == "F":
                    yield "F", { "reason": "previous verdict is F" }
                elif self.G(n) >= 1/self.delta:
                    self.last_verdict = "F"
                    yield "F", { "reason": f"G_{n} >= {1/self.delta} (1/delta)" }
                else:
                    yield "?", None

                n += 1

        except KeyboardInterrupt:
            pass
