from typing import Generator, Tuple, Literal, Any
from .adapters import DynamicalSystemAdapter
from .estimators import Estimator


class NeuralCertificateMonitor:
    def __init__(
        self,
        adapter: DynamicalSystemAdapter,
        estimator: Estimator,
        monitor_stride: int = 1,
    ):
        if monitor_stride < 1:
            raise ValueError("monitor_stride must be >= 1")
        self.adapter = adapter
        self.estimator = estimator
        self.monitor_stride = monitor_stride

    def run(self) -> Generator[Tuple[Literal["T","F","?"], float, float, Any], None, None]:
        step_idx = 0
        try:
            while not self.adapter.done():
                if step_idx % self.monitor_stride == 0:
                    yield self.estimator(self.adapter)
                self.adapter.step()
                step_idx += 1
        except KeyboardInterrupt:
            pass
