#!/usr/bin/env bash
# Finds the constants required by the basic integration test.

set -euo pipefail

source .venv/bin/activate

python3 -m monitor.cli.calibrate \
  --adapter pendulum \
  --estimate all \
  --dt 0.001 \
  --noise-level 0.0 \
  --adapter-kwargs '{"flip_inputs_prob_to": 0.0, "flip_inputs_prob_from": 0.0}' \
  --time-budget-sec 900 \
  --max-steps 40 \
  --samples-per-step 8 \
  --percentile 100.0

python3 -m monitor.cli.calibrate \
  --adapter pendulum \
  --estimate all \
  --dt 0.001 \
  --noise-level 0.0 \
  --adapter-kwargs '{"flip_inputs_prob_to": 1.0, "flip_inputs_prob_from": 0.0}' \
  --time-budget-sec 900 \
  --max-steps 40 \
  --samples-per-step 8 \
  --percentile 100.0
