# Monitoring neural control certificate functions

This project implements runtime monitoring for neural control systems using Certificate Functions (Lyapunov/Barrier functions). It provides statistical guarantees on the expected decrease of these functions to ensure safety and stability.

## Installation

The project uses a shared virtual environment for all submodules.

```bash
# Create and activate virtual environment
uv venv --python 3.9
source .venv/bin/activate  # or activate.fish

# Install all dependencies (including submodules and core package)
uv pip install -r requirements.txt

# Demo
pytest -s tests/test_integration.py::test_monitor_demo

# Optional: run all tests
pytest -s tests/
```


## Usage

The monitor can be integrated into any control loop by writing an adapter (see `monitor/adapters/interface.py`). This repository includes adapters for `neural_clbf` and `sablas`.

```python
from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum
from monitor.estimators import HistoryEstimator
from monitor.weighting import OptimalTemporalWeights
from monitor.monitor import NeuralCertificateMonitor

# 1. Initialize the system and the monitor
adapter = NeuralCLBFPendulum(dt=0.01)
weighting = OptimalTemporalWeights()
estimator = HistoryEstimator(weighting, delta=0.01)
monitor = NeuralCertificateMonitor(adapter, estimator)

# 2. Run the control/monitor loop
for safety, lower, upper, info in monitor.run():
    print(f"Safety Status: {safety} | CI: [{lower:+.4f}, {upper:+.4f}] | info: {info}")
```


## Methodology

The monitor estimates the expected reward $\mathbb{E}[R_t]$ (change in certificate value) to verify:

$\mathbb{E}[R_t] \leq 0$

Three strategies are implemented in `monitor/estimators.py`:

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

## Project Structure

*   `monitor/`: Core monitoring logic.
    *   `monitor.py`: Simple convenience class looping the simulation until done.
    *   `estimators.py`: Implementation of Estimators (Analytic, Sampling, History).
    *   `adapters/`: Interfaces for specific environments (Sablas, Neural-CLBF).
    *   `weighting.py`: Weighting strategies for history-based estimation.
*   `neural_clbf/`: Submodule for Neural CLBF controllers (Evaluation only).
*   `sablas/`: Submodule for SABLAS environments.
*   `scripts/`: Analysis and estimation scripts.
*   `tests/`: Unit and integration tests.

## Status

*   **Adapters**: Implemented for Sablas (Drone) and Neural-CLBF (Inverted Pendulum).
*   **Estimators**: All three strategies implemented and tested.
*   **Verification**: CI coverage tests confirm statistical guarantees hold empirically.
