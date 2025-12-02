from .adapters import DynamicalSystemAdapter
from .estimators import Estimator


class NeuralCertificateMonitor:
    def __init__(self, adapter: DynamicalSystemAdapter, estimator: Estimator):
        self.adapter = adapter
        self.estimator = estimator

    def run(self):
        try:
            while not self.adapter.done():
                print(self.estimator(self.adapter))
                self.adapter.step()
        except KeyboardInterrupt:
            pass
