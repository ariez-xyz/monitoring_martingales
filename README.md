# Monitoring neural control certificate functions

This project implements runtime monitors for neural control systems using certificate functions such as Lyapunov and barrier functions. 

## Installation

The project uses a shared virtual environment for all submodules. 

Run from the repository root: 

```bash
# 1. Initialize Git submodules
git submodule update --init --recursive

# 2. Create venv
uv venv --python 3.9
source .venv/bin/activate  # adjust for shell

# 3. Install dependencies
uv pip install -r requirements.txt

# ⚠️ IMPORTANT ⚠️
# pytorch-lightning is REQUIRED for neural_clbf. It requires 
# pyyaml <= 5.4.1, which fails to build. However, pyyaml 6.0.3 
# builds and works fine. 
# Therefore, requirements.txt installs pyyaml==6.0.3, and 
# pytorch-lightning must be installed manually afterward
# with --no-deps.
uv pip install pytorch-lightning==1.3.8 --no-deps

# 4. Verify installation
pytest
```

## Usage

### Estimator-based monitor

The history estimator uses a centered temporal window and therefore reports with a delay equal to the weighting radius. It needs both a drift bound and a transition Wasserstein-Lipschitz bound (see [#Lipschitz constants](#Lipschitz constants)).

```python
from monitor import EstimationMonitor
from monitor.adapters import NeuralCLBFPendulum
from monitor.estimators import HistoryEstimator
from monitor.weighting import OptimalTemporalWeights

adapter = NeuralCLBFPendulum()
delta = 0.01
weighting = OptimalTemporalWeights(adapter, delta)
monitor = EstimationMonitor(HistoryEstimator(weighting, delta))

for verdict, info in monitor(adapter):
    print(verdict, info)
```

For the continuous-time error formulas, construct the weights with
`continuous=True` and pass the same flag when running the estimation monitor:

```python
weighting = OptimalTemporalWeights(adapter, delta, continuous=True)
monitor = EstimationMonitor(HistoryEstimator(weighting, delta))

for verdict, info in monitor(adapter, continuous=True):
    print(verdict, info)
```

### Hypothesis-testing monitor

This monitor implements a one-sided sequential test based on a betting e-process.

```python
from monitor import HypothesisTestingMonitor
from monitor.adapters import NeuralCLBFPendulum

adapter = NeuralCLBFPendulum()
monitor = HypothesisTestingMonitor(delta=0.01)

for verdict, info in monitor(adapter):
    print(verdict, info)
```

## Lipschitz constants

The required Lipschitz constants are persisted in `calibration_constants.json`. Lookups are keyed by `adapter.bound_key()`. 

Constants are not estimated automatically at monitor runtime. A configuration without a matching entry raises `KeyError`.

Use the calibration CLI to estimate Lipschitz constants empirically. Run data is stored under `data/calibration/`, and, unless `--dry-run` is used, the CLI also updates `calibration_constants.json`:

```bash
python -m monitor.cli.calibrate \
  --adapter pendulum \
  --estimate all \
  --dt 0.01 \
  --noise-level 0.0 \
  --episodes 100 \
  --max-steps 20 \
  --samples-per-step 8 \
  --percentile 100
```

## CI Sweep

To inspect the confidence-interval formulas over synthetic parameter grids:

```bash
python -m monitor.cli.ci_sweep \
  --gamma 0.1,0.5,1.0 \
  --rho 0.5,1.0,2.0 \
  --h 0.01,0.1 \
  --delta 0.01
```

Results are written under `data/ci_sweeps/`.

## API Reference

Monitor and estimator calls return verdicts as `"T"` (condition satisfied),
`"F"` (condition violated), or `"?"` (inconclusive).

```python
from monitor import EstimationMonitor, HypothesisTestingMonitor
from monitor.adapters import DynamicalSystemAdapter, NeuralCLBFPendulum, SablasDrone
from monitor.calibration import CalibrationSample, LipschitzConstantProvider, LipschitzConstantSampler
from monitor.estimators import AnalyticEstimator, HistoryEstimator, SamplingEstimator
from monitor.weighting import OptimalTemporalWeights, UniformWeights
```

### Monitors

#### `EstimationMonitor`

```python
EstimationMonitor(estimator: Estimator)
monitor(adapter: DynamicalSystemAdapter, continuous: bool = False)
monitor.viz(adapter: DynamicalSystemAdapter)
```

- `estimator`: Estimator used to produce a confidence interval at each step.
- `adapter`: System to monitor. The monitor advances it after each verdict.
- `continuous`: Use continuous-time confidence-interval terms. The estimator's
  weighting strategy must have been configured for the same mode.

Calling the monitor yields `(verdict, info)` pairs. `info["ci"]` contains the
`(lower, upper)` confidence interval.

#### `HypothesisTestingMonitor`

```python
HypothesisTestingMonitor(delta: float)
monitor(adapter: DynamicalSystemAdapter)
monitor.viz(adapter: DynamicalSystemAdapter)
```

- `delta`: Test level. The monitor rejects when the e-value reaches `1 / delta`.
- `adapter`: System to monitor. The monitor advances it before each verdict.

Calling the monitor yields `(verdict, info)` pairs. The info dictionary includes
`e_value`, `threshold`, `S_n`, `V_n`, and the current bet.

### Estimators

All estimators are called as:

```python
estimator(adapter: DynamicalSystemAdapter, continuous: bool = False)
```

They return `(verdict, lower, upper, info)`.

#### `HistoryEstimator`

```python
HistoryEstimator(weighting: WeightingStrategy, delta: float)
```

- `weighting`: A `UniformWeights` or `OptimalTemporalWeights` instance.
- `delta`: Confidence level used by the weighting strategy.

Uses observed drift history. It returns inconclusive infinite bounds until the
centered window is available. `info` includes the target index, delay, weighted
mean, and error terms.

#### `SamplingEstimator`

```python
SamplingEstimator(delta: float)
estimator(adapter, continuous: bool = False, max_extra: int = 4096)
```

- `delta`: Confidence level for the sampling interval.
- `max_extra`: Maximum number of additional samples after the initial batch of
  512. Sampling stops early when the interval is conclusive.

`info["n_samples"]` reports the number of sampled successor states.

#### `AnalyticEstimator`

```python
AnalyticEstimator()
```

Uses `adapter.sample(n_samples=1, noise_level=0.0)` and returns a point
interval. Use it only with adapters where zero noise represents the expected
successor state.

### Weighting Strategies

#### `UniformWeights`

```python
UniformWeights(radius: int)
weights(drift_history: torch.Tensor, target: int) -> Optional[torch.Tensor]
weights.get_radius() -> int
```

- `radius`: Number of observations on either side of the target. The full
  window size is `2 * radius + 1`.
- `drift_history`: One-dimensional tensor of observed drifts.
- `target`: Zero-based drift index at the center of the requested window.

Returns a tensor aligned with `drift_history`, or `None` when the full centered
window is unavailable.

#### `OptimalTemporalWeights`

```python
OptimalTemporalWeights(
    adapter: DynamicalSystemAdapter,
    delta: float,
    continuous: bool = False,
)
```

- `adapter`: Supplies the persisted transition Wasserstein-Lipschitz bound and,
  in continuous mode, `dt`.
- `delta`: Confidence level used to choose the window radius.
- `continuous`: Select continuous-time window and error formulas.

This class exposes the same `__call__()` and `get_radius()` interface as
`UniformWeights`.

### Adapters

#### `NeuralCLBFPendulum`

```python
NeuralCLBFPendulum(
    checkpoint_path: str = "neural_clbf/saved_models/review/inverted_pendulum_clf.ckpt",
    dt: Optional[float] = None,
    noise_level: float = 0.0,
    vis_every: int = 0,
    vis_block: bool = False,
    certificate_slope: float = 0.0,
    flip_inputs_prob_to: float = 0.0,
    flip_inputs_prob_from: float = 0.0,
)
```

- `checkpoint_path`: Neural-CLBF checkpoint to load.
- `dt`: Simulation timestep; `None` uses the checkpoint model's timestep.
- `noise_level`: Magnitude of additive uniform control noise.
- `vis_every`: Render every N steps; `0` disables rendering.
- `vis_block`: Wait for input after each render.
- `certificate_slope`: Per-step affine offset added to certificate values for
  stress testing.
- `flip_inputs_prob_to`: Per-step probability of entering the control-sign
  fault mode.
- `flip_inputs_prob_from`: Per-step probability of leaving that fault mode.

`reset(seed: Optional[int] = None, initial_state: Optional[torch.Tensor] = None)`
also accepts an explicit pendulum state.

#### `SablasDrone`

```python
SablasDrone(
    k_obstacle: int = 8,
    use_estimated_param: bool = False,
    dt: float = 0.1,
    noise_level: float = 0.1,
    vis_every: int = 0,
    vis_block: bool = False,
)
```

- `k_obstacle`: Number of obstacles represented in the environment.
- `use_estimated_param`: Load the SABLAS estimated dynamics parameters.
- `dt`: Simulation timestep.
- `noise_level`: Environment process-noise scale.
- `vis_every`: Render every N steps; `0` disables rendering.
- `vis_block`: Wait for input after each render.

The SABLAS adapter does not support per-call `noise_level` overrides in
`sample()`.


## Current Caveats

### Neural-CLBF and `cvxpylayers`

Currently, neural-clbf checkpoints are loaded without `cvxpylayers` through a compatibility shim in the submodule. This workaround is required because `cvxpylayers` fails to install on Apple Silicon. 

The pendulum adapter uses the nominal controller (`u_nominal`, effectively the LQR path for this setup), with zero-order hold and optional injected control noise. Due to the `cvxpylayers` issue, it does not run the original CLF-QP correction layer at runtime, so certificate increases can occur and are part of the monitored behavior.

### Empirical constants

The shipped pendulum drift and transition-kernel constants are persisted empirical calibration results, not formal global bounds.

Changing fields such as `dt`, `noise_level`, or injected-fault probabilities generally requires a new matching calibration entry.

The transition-bound sampler assumes that `noise_level=0` gives the expected next state and that transition laws are translated copies of a common zero-mean noise family. 

The SABLAS adapter does not currently satisfy the per-call noise override required by that sampler, and no SABLAS constants are shipped.

## Status

- Adapters: Neural-CLBF inverted pendulum and SABLAS drone.
- Estimators: analytic, sampled, and centered-history implementations.
- Monitors: confidence-interval estimation and sequential hypothesis testing.
- Tooling: persisted empirical calibration and confidence-interval sweeps.
