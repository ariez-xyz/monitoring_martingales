import pytest

from monitor.adapters import DynamicalSystemAdapter, NeuralCLBFPendulum, SablasDrone
from tests.fixtures import NormalIncrementAdapter


def _concrete_subclasses(cls: type[DynamicalSystemAdapter]) -> set[type[DynamicalSystemAdapter]]:
    subclasses: set[type[DynamicalSystemAdapter]] = set()
    for subcls in cls.__subclasses__():
        subclasses.add(subcls)
        subclasses.update(_concrete_subclasses(subcls))
    return subclasses


ADAPTER_FACTORIES = {
    NormalIncrementAdapter: lambda: NormalIncrementAdapter(mean=-0.1, sigma=0.05, initial_value=2.0),
    NeuralCLBFPendulum: lambda: NeuralCLBFPendulum(noise_level=0.1),
    SablasDrone: lambda: SablasDrone(noise_level=0.1),
}


@pytest.mark.parametrize(
    "adapter_cls",
    sorted(_concrete_subclasses(DynamicalSystemAdapter), key=lambda cls: cls.__name__),
)
def test_adapter_sample_returns_expected_batch_size(adapter_cls: type[DynamicalSystemAdapter]):
    adapter = ADAPTER_FACTORIES[adapter_cls]()
    n_samples = 5

    sampled = adapter.sample(n_samples=n_samples)
    assert sampled.shape == (n_samples, adapter.get_state_dim())

    sampled_with_extremes = adapter.sample(n_samples=n_samples, include_extremes=True)
    assert sampled_with_extremes.shape[0] >= n_samples
    assert sampled_with_extremes.shape[1] == adapter.get_state_dim()

    if adapter_cls is NeuralCLBFPendulum:
        assert sampled_with_extremes.shape[0] == n_samples + 2
