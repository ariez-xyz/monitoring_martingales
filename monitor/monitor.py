from typing import Generator, Tuple, Literal, Any
from .adapters import DynamicalSystemAdapter
from .estimators import Estimator


class NeuralCertificateMonitor:
    def __init__(self, adapter: DynamicalSystemAdapter, estimator: Estimator):
        self.adapter = adapter
        self.estimator = estimator

    def run(self) -> Generator[Tuple[Literal["T","F","?"], float, float, Any], None, None]:
        try:
            while not self.adapter.done():
                yield self.estimator(self.adapter)
                self.adapter.step()
        except KeyboardInterrupt:
            pass
