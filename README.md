# Monitoring neural control certificate functions

This project implements runtime monitoring for neural control systems using Certificate Functions (Lyapunov/Barrier functions). It provides statistical guarantees on the expected decrease of these functions to ensure safety and stability.

## Installation

The project uses a shared virtual environment for all submodules.

```bash
# Create and activate virtual environment
uv venv --python 3.9
source .venv/bin/activate  # or activate.fish

# Install core dependencies
uv pip install "pyyaml>=6.0"
uv pip install pytorch-lightning==1.3.8 --no-deps

# Clone and install neural_clbf (Apple Silicon compatible fork)
git clone https://github.com/ariez-xyz/neural_clbf.git
cd neural_clbf
uv pip install -e .
uv pip install -r requirements.txt --no-deps
cd ..

# Clone and install sablas (Fork)
git clone https://github.com/ariez-xyz/sablas.git
cd sablas
# (Add specific installation steps if any, or rely on root requirements)
cd ..

# Install this package in editable mode if necessary, or just run from root
```

## Usage

### Running Tests

```bash
# Run all tests
pytest -s tests/

# Run specific tests
pytest -s tests/test_monitor.py
```

### Scripts

*   `scripts/check_jensen_gap.py`: Validates the Jensen gap assumption for the analytic estimator.
*   `scripts/estimate_cbf_bounds.py`: Empirically estimates the bounds of the certificate function.
*   `scripts/estimate_gamma.py`: Estimates the Lipschitz constant for the reward function.

## Methodology

The monitor estimates the expected reward $\mathbb{E}[R_t]$ (change in certificate value) to verify:

$\mathbb{E}[R_t] \leq 0$

Three strategies are implemented:

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
    *   `monitor.py`: Implementation of Estimators (Analytic, Sampling, History).
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
