import json

from monitor.calibration import LipschitzConstantProvider
from monitor.cli import calibrate
from tests.fixtures import NormalIncrementAdapter


def test_calibration_cli_smoke_updates_temp_cache(tmp_path, monkeypatch, capsys):
    temp_json = tmp_path / "calibration_constants.json"
    temp_json.write_text(json.dumps({"drift_bounds": {}, "transition_wasserstein_lipschitz": {}}))
    temp_cache = tmp_path / "data" / "calibration"

    monkeypatch.setattr(LipschitzConstantProvider, "_json_path", temp_json)
    monkeypatch.setattr(LipschitzConstantProvider, "_loaded", False)
    monkeypatch.setattr(LipschitzConstantProvider, "_precomputed_drift_bounds", {})
    monkeypatch.setattr(LipschitzConstantProvider, "_precomputed_transition_lipschitz", {})
    monkeypatch.setattr(calibrate, "CACHE_DIR", temp_cache)
    monkeypatch.setitem(
        calibrate.ADAPTER_REGISTRY,
        "fixture",
        lambda args: NormalIncrementAdapter(
            mean=args.adapter_kwargs.get("mean", 0.2),
            sigma=args.adapter_kwargs.get("sigma", 0.0),
            initial_value=10.0,
        ),
    )

    exit_code = calibrate.main(
        [
            "--adapter",
            "fixture",
            "--estimate",
            "drift",
            "--adapter-kwargs",
            '{"mean": 0.2, "sigma": 0.0}',
            "--episodes",
            "2",
            "--max-steps",
            "5",
            "--percentile",
            "100",
            "--force",
        ]
    )

    assert exit_code == 0
    data = json.loads(temp_json.read_text())
    assert len(data["drift_bounds"]) == 1
    cache_files = list(temp_cache.glob("*.json"))
    assert len(cache_files) == 1
    cache_payload = json.loads(cache_files[0].read_text())
    assert cache_payload["estimate"] == "drift"
    assert cache_payload["sample_count"] > 0
    assert cache_payload["args"]["adapter_kwargs"] == {"mean": 0.2, "sigma": 0.0}
    assert "stdout" in cache_payload
    assert "stderr" in cache_payload
    assert cache_payload["stderr"] == ""
    assert list(cache_payload)[-1] == "samples"
    assert "value" in cache_payload["samples"][0]
    assert "from_state" in cache_payload["samples"][0]
    assert "to_state" in cache_payload["samples"][0]

    out = capsys.readouterr().out
    assert "adapter=fixture" in out
    assert 'adapter_kwargs={"mean":0.2,"sigma":0.0}' in out
    assert "drift_samples=" in out
    assert "drift_bound=" in out
    assert "drift_cache=" in out
