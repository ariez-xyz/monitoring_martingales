# Monitoring neural control certificate functions

This project implements runtime monitoring for neural control systems using certificate functions such as Lyapunov and barrier functions. The monitor detects whether the monitored certificate value is decreasing in expectation (i.e. whether it forms a supermartingale). It requires Lipschitz assumptions on the system dynamics and certificate functions, and provides statistical guarantees (confidence $\gt 1-\delta$). Several existing neural control systems (from MIT-REALM) are included as submodules and integrated via adapters providing a common interface for the monitor to run against.

## Installation

The project uses a shared virtual environment for all submodules.

```bash
# Create and activate virtual environment
uv venv --python 3.9
source .venv/bin/activate  # adjust for shell

# Install all dependencies (including submodules and core package)
uv pip install -r requirements.txt

# Demo
pytest -s tests/test_integration.py::test_monitor_demo

# Optional: run all tests
pytest -s tests/
```


## Usage

The monitor can be integrated into any control loop by writing an adapter (see `monitor/adapters/interface.py`). This repository currently includes adapters for `neural_clbf` and `sablas`.

The monitor assigns the following verdicts:

- `T`: Attests that the certificate is **valid** (decreasing). Used by estimator-based monitors when the upper confidence bound is below `0`.
- `F`: Attests that the certificate is **invalid**, i.e. detected violation / rejected null hypothesis
- `?`: Inconclusive.


### Estimator-based monitor

This monitor is able to attest validity, but it additionally requires an estimator and a weighting strategy.

```python
from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum
from monitor.estimators import HistoryEstimator
from monitor.weighting import OptimalTemporalWeights
from monitor.monitor import NeuralCertificateMonitor

# 1. Initialize the system and the monitor
adapter = NeuralCLBFPendulum()
weighting = OptimalTemporalWeights()
estimator = HistoryEstimator(weighting, delta=0.01)
monitor = NeuralCertificateMonitor(adapter, estimator)

# 2. Run the control/monitor loop
for verdict, info in monitor.run():
    print(verdict, info)
```

### Hypothesis-testing monitor

This monitor implements a one-sided sequential test based on a betting e-process. It can reject the supermartingale hypothesis, but it does not certify it as true.

```python
from monitor import HypothesisTestingMonitor
from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum

adapter = NeuralCLBFPendulum()
monitor = HypothesisTestingMonitor(adapter=adapter, delta=0.01)

for verdict, info in monitor.run():
    print(verdict, info)
```


## Methodology

The core quantity is the one-step reward

$R_t = V(X_{t+1}) - V(X_t)$

and the monitoring objective is to determine whether

$\mathbb{E}[R_t \mid X_t] \leq 0$

holds along the observed execution, i.e. $(R_i)_{i\in\mathbb{N}}$ is a supermartingale.

### Estimator-based monitors

Three estimator strategies are implemented in `monitor/estimators.py`:

### 1. Analytic Estimator
Computes $\mathbb{E}[R_t]$ directly using known system dynamics $P(x)$.
*   **Pros**: Fast, exact (if dynamics known).
*   **Cons**: Requires white-box access to dynamics; susceptible to Jensen gap if $V$ is highly nonlinear.
*   **Assumption**: Small Jensen gap $|V(\mathbb{E}[Y]) - \mathbb{E}[V(Y)]| \approx 0$.

### 2. Sampling-Based Estimator
Estimates $\mathbb{E}[R_t]$ by sampling $n$ next states from the simulator.
*   **Pros**: Works with black-box simulators; provides rigorous concentration bounds (Hoeffding).
*   **Cons**: Computationally expensive (requires multiple simulator steps per monitor tick).

### 3. History-Based Estimator
Estimates $\mathbb{E}[R_t]$ using past observed transitions $x_1, \dots, x_t$ without resetting the simulator.
*   **Pros**: No simulator resets required; pure runtime monitoring.
*   **Cons**: Requires Lipschitz assumptions on system and reward function.
*   **Key Insight**: Uses a weighted average of past rewards, balancing **Discretization Error** (distance between current state and past states) and **Statistical Error** (variance of the estimator).

### Hypothesis-testing monitor

`HypothesisTestingMonitor` implements a sequential test for violations of the supermartingale condition. It maintains a predictable betting process and rejects when the e-process exceeds `1 / delta`.

This path is useful when you want an online rejection test instead of a confidence interval for expected drift.

## Project Structure

*   `monitor/`: Core monitoring logic.
    *   `monitor.py`: Estimator-based monitor and hypothesis-testing monitor.
    *   `estimators.py`: Implementation of estimators (Analytic, Sampling, History).
    *   `adapters/`: Interfaces for specific environments (Sablas, Neural-CLBF).
    *   `weighting.py`: Weighting strategies for history-based estimation.
*   `neural_clbf/`: Submodule for Neural CLBF controllers. In this repo it is primarily used for evaluation/runtime experiments.
*   `sablas/`: Submodule for SABLAS environments.
*   `scripts/`: Analysis and estimation scripts.
*   `tests/`: Unit and integration tests.

## Current Caveats

### `neural_clbf` integration

The `neural_clbf` submodule is currently used in an evaluation-first configuration.

- Older checkpoints can be loaded without `cvxpylayers` through a compatibility shim in the submodule.
- This is sufficient for the monitoring and rollout code used here.
- Training workflows that require differentiating through the CLF-QP layer are not the focus of this repository.

### Pendulum adapter behavior

The pendulum adapter does not currently run the original CLF-QP correction layer at runtime. It uses the nominal controller (`u_nominal`, effectively the LQR path for this setup) plus optional injected control noise.

In plain terms:

- the controller can still stabilize the system,
- but the control input is not explicitly projected each step to enforce CLF decrease,
- so certificate increases can occur and should be treated as part of the monitored behavior rather than impossible by construction.

### Lipschitz constants are implementation-level proxies

Several monitor components rely on Lipschitz-style constants or one-step drift bounds. In the current codebase these should be read as practical estimates used for monitoring, not as formal analytic constants derived from the underlying system.

- In the pendulum adapter, `get_lipschitz_constant()` in [monitor/adapters/neural_clbf_pendulum.py](/Users/ariez/Projects/neural-control-monitoring/monitor/adapters/neural_clbf_pendulum.py) returns a per-step drift bound proxy.
- For common `(dt, noise_level)` settings, the adapter uses seeded conservative values.
- Otherwise it estimates the quantity empirically by rolling out trajectories, evaluating one-step certificate changes, and taking a high percentile of the observed absolute drifts.
- This value is then cached and reused by the monitor.

In plain terms, the monitor is currently using conservative data-driven bounds for the pendulum setup. That is good enough for experiments and smoke tests, but it is weaker than having a proof-level Lipschitz constant for the true closed-loop system.

## Status

*   **Adapters**: Implemented for Sablas and Neural-CLBF pendulum.
*   **Estimator monitors**: Analytic, Sampling, and History monitors are implemented and exercised by tests and demos.
*   **Hypothesis monitor**: Implemented and currently exercised through an integration-style smoke test on the pendulum adapter.
*   **Documentation split**: This README is intended to describe the implemented system; exploratory derivations and research notes are kept separately.
