from .adapters import DynamicalSystemAdapter
import torch

class NeuralCertificateMonitor:
    def __init__(self, adapter: DynamicalSystemAdapter):
        self.adapter = adapter
        self.outputs = []

    def sampling_estimate_step(self, n_samples: int = 100):
        sampled_next_states = self.adapter.sample(n_samples=n_samples)
        rewards = self.adapter.get_reward(sampled_next_states)
        self.outputs.append(rewards.mean())

    def print_outputs(self):
        for i, v in enumerate(self.outputs):
            print(f"{i}\t{v}")

    def run_till_done(self):
        try:
            while not self.adapter.done():
                self.sampling_estimate_step()
                self.adapter.step()
        except KeyboardInterrupt:
            pass

