#!/usr/bin/env python3
"""Measure variance of pendulum Lipschitz estimates across repeated runs.

This script reports:
1) Same-adapter repeated calls to get_lipschitz_constant() (cache behavior)
2) Fresh-adapter calls to get_lipschitz_constant() (estimation variance)

By default it uses the public adapter API. An optional fast mode calls the
private estimator directly with configurable rollout settings.
"""

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum


def summarize(values: List[float], label: str) -> None:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        print(f"\n{label}")
        print("-" * len(label))
        print("count=0")
        print("No data collected for this section.")
        return

    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    cv = std / mean if mean != 0 else float("inf")
    p05 = float(np.percentile(arr, 5))
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))

    print(f"\n{label}")
    print("-" * len(label))
    print(f"count={len(arr)}")
    print(f"mean={mean:.6f}")
    print(f"std={std:.6f}")
    print(f"cv={cv:.4f}")
    print(f"min={arr.min():.6f}, p05={p05:.6f}, median={p50:.6f}, p95={p95:.6f}, max={arr.max():.6f}")


def call_public_gamma(adapter: NeuralCLBFPendulum) -> float:
    return float(adapter.get_lipschitz_constant())


def call_private_gamma_with_diffs(
    adapter: NeuralCLBFPendulum,
    n_episodes: int,
    max_steps: int,
    percentile: float,
) -> Tuple[float, List[float]]:
    gamma, diffs = adapter._estimate_lipschitz_constant(
        n_episodes=n_episodes,
        max_steps=max_steps,
        percentile=percentile,
        return_diffs=True,
    )
    return float(gamma), [float(x) for x in diffs]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check variance of pendulum Lipschitz estimates")
    parser.add_argument("--dt", type=float, default=0.01, help="Simulation dt")
    parser.add_argument("--noise-level", type=float, default=0.0, help="Control-noise level for adapter")
    parser.add_argument("--trials", type=int, default=10, help="Number of fresh-adapter trials")
    parser.add_argument("--same-adapter-repeats", type=int, default=5, help="Repeated calls on one adapter")
    parser.add_argument(
        "--fast-private",
        action="store_true",
        help="Use private estimator directly (faster configurable sweeps, bypasses cache)",
    )
    parser.add_argument("--n-episodes", type=int, default=10, help="Private estimator episodes")
    parser.add_argument("--max-steps", type=int, default=1000, help="Private estimator max steps")
    parser.add_argument("--percentile", type=float, default=99.9, help="Private estimator percentile")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base seed. If set, each trial uses seed+i.",
    )
    parser.add_argument(
        "--save-diffs-json",
        type=str,
        default=None,
        help="Optional output path to save per-run reward diffs and metadata as JSON.",
    )
    args = parser.parse_args()

    mode = "private estimator" if args.fast_private else "public get_lipschitz_constant()"
    print("=" * 80)
    print("Pendulum Lipschitz Variance Check")
    print("=" * 80)
    print(f"dt={args.dt}")
    print(f"noise_level={args.noise_level}")
    print(f"mode={mode}")
    print(f"trials={args.trials}, same_adapter_repeats={args.same_adapter_repeats}")
    if args.fast_private:
        print(
            f"private params: n_episodes={args.n_episodes}, "
            f"max_steps={args.max_steps}, percentile={args.percentile}"
        )
    if args.seed is not None:
        print(f"base_seed={args.seed}")
    if args.save_diffs_json is not None and not args.fast_private:
        print("Warning: diff export requires --fast-private; diffs will be omitted in public mode.")

    export_payload: Dict[str, Any] = {
        "meta": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dt": args.dt,
            "noise_level": args.noise_level,
            "mode": mode,
            "sample_quantity": "abs_step_drift_bound_proxy",
            "trials": args.trials,
            "same_adapter_repeats": args.same_adapter_repeats,
            "fast_private": args.fast_private,
            "n_episodes": args.n_episodes,
            "max_steps": args.max_steps,
            "percentile": args.percentile,
            "seed": args.seed,
        },
        "same_adapter_runs": [],
        "fresh_adapter_runs": [],
    }

    # 1) Same-adapter repeated calls (cache behavior for public API)
    adapter = NeuralCLBFPendulum(dt=args.dt, noise_level=args.noise_level)
    same_adapter_values: List[float] = []

    t0 = time.time()
    for i in range(args.same_adapter_repeats):
        run_seed: Optional[int] = None
        if args.seed is not None:
            run_seed = args.seed + i
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)

        diffs: Optional[List[float]] = None
        if args.fast_private:
            g, diffs = call_private_gamma_with_diffs(
                adapter, args.n_episodes, args.max_steps, args.percentile
            )
        else:
            g = call_public_gamma(adapter)

        same_adapter_values.append(g)
        print(f"same-adapter call {i+1}/{args.same_adapter_repeats}: gamma={g:.6f}")
        export_payload["same_adapter_runs"].append(
            {
                "index": i + 1,
                "seed": run_seed,
                "gamma": g,
                "n_diffs": 0 if diffs is None else len(diffs),
                "reward_diffs": diffs,
                "step_bound_samples": diffs,
            }
        )
    same_elapsed = time.time() - t0

    # 2) Fresh-adapter trials (re-estimation variance)
    fresh_values: List[float] = []
    trial_times: List[float] = []

    for i in range(args.trials):
        run_seed: Optional[int] = None
        if args.seed is not None:
            run_seed = args.seed + 10000 + i
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)

        trial_start = time.time()
        trial_adapter = NeuralCLBFPendulum(dt=args.dt, noise_level=args.noise_level)

        diffs: Optional[List[float]] = None
        if args.fast_private:
            g, diffs = call_private_gamma_with_diffs(
                trial_adapter, args.n_episodes, args.max_steps, args.percentile
            )
        else:
            g = call_public_gamma(trial_adapter)

        elapsed = time.time() - trial_start
        fresh_values.append(g)
        trial_times.append(elapsed)
        print(f"trial {i+1}/{args.trials}: gamma={g:.6f}, time={elapsed:.2f}s")
        export_payload["fresh_adapter_runs"].append(
            {
                "index": i + 1,
                "seed": run_seed,
                "gamma": g,
                "elapsed_s": elapsed,
                "n_diffs": 0 if diffs is None else len(diffs),
                "reward_diffs": diffs,
                "step_bound_samples": diffs,
            }
        )

    print("\n" + "=" * 80)
    summarize(same_adapter_values, "Same Adapter Repeats")
    print(f"elapsed={same_elapsed:.2f}s")

    summarize(fresh_values, "Fresh Adapter Trials")
    if trial_times:
        print(
            f"time per trial: mean={statistics.mean(trial_times):.2f}s, "
            f"median={statistics.median(trial_times):.2f}s"
        )
    else:
        print("time per trial: no trials run")

    if not args.fast_private:
        print("\nNote: public API caches per adapter+(dt, noise_level), so same-adapter variance should be ~0.")

    if args.save_diffs_json is not None:
        export_payload["summary"] = {
            "same_adapter": {
                "count": len(same_adapter_values),
                "mean": float(np.mean(same_adapter_values)) if same_adapter_values else None,
                "std": (
                    float(np.std(same_adapter_values, ddof=1))
                    if len(same_adapter_values) > 1
                    else (0.0 if len(same_adapter_values) == 1 else None)
                ),
            },
            "fresh_adapter": {
                "count": len(fresh_values),
                "mean": float(np.mean(fresh_values)) if fresh_values else None,
                "std": (
                    float(np.std(fresh_values, ddof=1))
                    if len(fresh_values) > 1
                    else (0.0 if len(fresh_values) == 1 else None)
                ),
            },
        }
        with open(args.save_diffs_json, "w", encoding="utf-8") as f:
            json.dump(export_payload, f, indent=2)
        print(f"\nSaved diff samples + metadata to: {args.save_diffs_json}")


if __name__ == "__main__":
    main()
