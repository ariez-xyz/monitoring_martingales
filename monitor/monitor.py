from typing import Generator, Tuple, Literal, Any, List
from warnings import warn
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
        self.G: List[float] = [1, 1]
        self.S: List[float] = [0, 0]
        self.V: List[float] = [0, 0]
        self.deltaX: List[float] = [0]
        self.last_certificate_value: float = -1

    def cur_cert_value(self) -> float:
        return float(self.adapter.get_certificate_value())

    def eta(self, k) -> float:
        S, V, B = self.S, self.V, self.adapter.get_lipschitz_constant() # match paper notation
        # Note: We assume adapter.dt never varies, so B_k = B for all k
        return min(1/B, max(0, S[k-1] / (V[k-1] + B**2)))

    def run(self) -> Generator[Tuple[Literal["F","?"], Any], None, None]:
        n = 0 # Step index
        self.last_certificate_value = self.cur_cert_value()

        try:
            while not self.adapter.done():
                if self.last_verdict == "F":
                    yield "F", { "reason": "previous verdict is F" }
                    break

                certificate_value = self.cur_cert_value()
                reward = certificate_value - self.last_certificate_value
                self.deltaX.append(reward)

                if len(self.G) <= n: # update G,S,V on every index n>=2
                    self.last_certificate_value = certificate_value

                    G_n = self.G[n-1] * (1 + self.eta(n-1) * self.deltaX[n-1])
                    S_n = self.S[n-1] + self.deltaX[n-1]
                    V_n = self.V[n-1] + (self.deltaX[n-1] ** 2)

                    self.G.append(G_n)
                    self.S.append(S_n)
                    self.V.append(V_n)

                if self.G[n] >= 1/self.delta:
                    self.last_verdict = "F"
                    yield "F", { "reason": f"G[{n}] >= {1/self.delta} (1/delta)" }
                else:
                    yield "?", None

                self.adapter.step()
                n += 1

        except KeyboardInterrupt:
            pass
