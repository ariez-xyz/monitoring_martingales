import json

from monitor.calibration import LipschitzConstantEstimator, LipschitzConstantProvider
from tests.fixtures import NormalIncrementAdapter


def _restore_fixture_bounds(adapter: NormalIncrementAdapter) -> None:
    # These tests intentionally overwrite the provider's persistent JSON-backed
    # values. Restore the fixture's canonical per-key constants afterward so
    # constructor auto-seeding doesn't raise exceptions because it's trying to
    # overwrite larger values on later test runs.
    LipschitzConstantProvider.set_drift_bound(adapter, adapter._drift_bound(), force=True)
    LipschitzConstantProvider.set_transition_wasserstein_lipschitz(adapter, 1.0, force=True)


def test_estimator_computes_drift_bound_from_transition_batches():
    estimator = LipschitzConstantEstimator()
    adapter_factory = lambda: NormalIncrementAdapter(mean=0.2, sigma=0.0, initial_value=10.0)

    gamma = estimator.estimate_drift_bound(
        adapter_factory,
        n_episodes=5,
        max_steps=30,
        percentile=100.0,
        samples_per_step=4,
    )

    assert abs(gamma - 0.2) < 1e-6


def test_estimator_computes_transition_wasserstein_lipschitz():
    estimator = LipschitzConstantEstimator()
    adapter_factory = lambda: NormalIncrementAdapter(mean=0.2, sigma=0.0, initial_value=10.0)

    try:
        estimator.estimate_transition_wasserstein_lipschitz(
            adapter_factory,
            n_episodes=5,
            max_steps=30,
            percentile=100.0,
            samples_per_step=4,
        )
        assert False, "Expected transition Wasserstein estimation to be unimplemented"
    except NotImplementedError:
        pass


def test_provider_reads_persisted_values_from_real_json():
    adapter = NormalIncrementAdapter(
        mean=0.31,
        sigma=0.07,
        clamp_at_sigma=4,
    )
    try:
        LipschitzConstantProvider.set_drift_bound(adapter, 0.9)
        LipschitzConstantProvider.reload()
        LipschitzConstantProvider.clear_cache()

        assert LipschitzConstantProvider.get_drift_bound(adapter) == 0.9
    finally:
        _restore_fixture_bounds(adapter)


def test_provider_persists_set_values_to_json():
    adapter = NormalIncrementAdapter(
        mean=0.29,
        sigma=0.11,
        clamp_at_sigma=5,
    )
    try:
        LipschitzConstantProvider.set_drift_bound(adapter, 3.5)
        LipschitzConstantProvider.set_transition_wasserstein_lipschitz(adapter, 1.75)

        assert LipschitzConstantProvider.get_drift_bound(adapter) == 3.5
        assert LipschitzConstantProvider.get_transition_wasserstein_lipschitz(adapter) == 1.75

        LipschitzConstantProvider.reload()
        assert LipschitzConstantProvider.get_drift_bound(adapter) == 3.5
        assert LipschitzConstantProvider.get_transition_wasserstein_lipschitz(adapter) == 1.75

        data = json.loads(LipschitzConstantProvider._json_path.read_text())
        drift_key = json.dumps(adapter.bound_key(), sort_keys=True, separators=(",", ":"))
        assert data["drift_bounds"][drift_key] == 3.5
        assert data["transition_wasserstein_lipschitz"][drift_key] == 1.75
    finally:
        _restore_fixture_bounds(adapter)


def test_provider_requires_force_to_decrease_existing_value():
    adapter = NormalIncrementAdapter(
        mean=0.271,
        sigma=0.019,
        clamp_at_sigma=6,
    )
    try:
        LipschitzConstantProvider.set_drift_bound(adapter, 2.0)

        try:
            LipschitzConstantProvider.set_drift_bound(adapter, 1.0)
            assert False, "Expected decreasing drift bound to require force=True"
        except ValueError:
            pass

        LipschitzConstantProvider.set_drift_bound(adapter, 1.0, force=True)
        assert LipschitzConstantProvider.get_drift_bound(adapter) == 1.0
    finally:
        _restore_fixture_bounds(adapter)


def test_provider_raises_on_missing_precomputed_transition_bound():
    from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum

    adapter = NeuralCLBFPendulum(dt=1234, noise_level=4321)
    LipschitzConstantProvider.reload()
    LipschitzConstantProvider.clear_cache()

    try:
        LipschitzConstantProvider.get_transition_wasserstein_lipschitz(adapter)
        assert False, "Expected a transition-Lipschitz cache miss"
    except KeyError:
        pass
