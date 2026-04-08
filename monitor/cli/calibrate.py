from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from monitor.adapters import NeuralCLBFPendulum, SablasDrone
from monitor.adapters.interface import DynamicalSystemAdapter
from monitor.calibration import LipschitzConstantProvider, LipschitzConstantSampler


AdapterBuilder = Callable[[argparse.Namespace], DynamicalSystemAdapter]


def _adapter_kwargs(args: argparse.Namespace) -> dict:
    return dict(args.adapter_kwargs or {})


ADAPTER_REGISTRY: dict[str, AdapterBuilder] = {
    "pendulum": lambda args: NeuralCLBFPendulum(
        dt=args.dt,
        noise_level=args.noise_level,
        **_adapter_kwargs(args),
    ),
    "sablas_drone": lambda args: SablasDrone(
        dt=args.dt,
        noise_level=args.noise_level,
        **_adapter_kwargs(args),
    ),
}

CACHE_DIR = Path("data") / "calibration"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample and persist calibration constants.")
    parser.add_argument("--adapter", required=True, choices=sorted(ADAPTER_REGISTRY))
    parser.add_argument("--estimate", choices=("drift", "transition", "all"), default="all")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument(
        "--adapter-kwargs",
        type=json.loads,
        default={},
        help="JSON object of additional adapter-specific constructor kwargs",
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--samples-per-step", type=int, default=8)
    parser.add_argument("--percentile", type=float, default=100.0)
    parser.add_argument("--time-budget-sec", type=float, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _accumulate_episode_samples(
    sampler: LipschitzConstantSampler,
    adapter_factory: Callable[[], DynamicalSystemAdapter],
    estimate: str,
    max_steps: int,
    samples_per_step: int,
) -> list[float]:
    if estimate == "drift":
        return sampler.sample_drift_bounds(
            adapter_factory,
            n_episodes=1,
            max_steps=max_steps,
            samples_per_step=samples_per_step,
        )
    if estimate == "transition":
        return sampler.sample_transition_wasserstein_ratios(
            adapter_factory,
            n_episodes=1,
            max_steps=max_steps,
        )
    raise ValueError(f"Unsupported estimate kind: {estimate}")


def _cache_run(
    *,
    path: Path,
    adapter_name: str,
    bound_key: dict,
    estimate: str,
    args: argparse.Namespace,
    samples: list[float],
    result: float,
    stdout_text: str,
    stderr_text: str,
) -> Path:
    payload = {
        "adapter": adapter_name,
        "bound_key": bound_key,
        "estimate": estimate,
        "args": {
            "dt": args.dt,
            "noise_level": args.noise_level,
            "adapter_kwargs": args.adapter_kwargs,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "samples_per_step": args.samples_per_step,
            "percentile": args.percentile,
            "time_budget_sec": args.time_budget_sec,
        },
        "sample_count": len(samples),
        "summary": {
            "min": min(samples) if samples else None,
            "max": max(samples) if samples else None,
            "mean": float(np.mean(samples)) if samples else None,
            "percentile": args.percentile,
            "result": result,
        },
        "stdout": stdout_text,
        "stderr": stderr_text,
        "samples": samples,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _cache_path(adapter_name: str, estimate: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return CACHE_DIR / f"{timestamp}_{adapter_name}_{estimate}.json"


def _run_estimate(
    *,
    estimate: str,
    args: argparse.Namespace,
    builder: AdapterBuilder,
    log_stdout: Callable[[str], None],
) -> tuple[float, list[float]]:
    sampler = LipschitzConstantSampler()
    samples: list[float] = []
    started = time.monotonic()
    episodes_run = 0

    while True:
        if args.episodes is not None and episodes_run >= args.episodes:
            break
        if args.time_budget_sec is not None and (time.monotonic() - started) >= args.time_budget_sec:
            break
        episode_samples = _accumulate_episode_samples(
            sampler,
            lambda: builder(args),
            estimate,
            max_steps=args.max_steps,
            samples_per_step=args.samples_per_step,
        )
        samples.extend(episode_samples)
        episodes_run += 1

    if not samples:
        raise ValueError(f"No {estimate} samples were collected")

    result = max(float(np.percentile(samples, args.percentile)), 1e-6)
    log_stdout(f"{estimate}_samples={len(samples)}")
    return result, samples


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.episodes is None and args.time_budget_sec is None:
        parser.error("Set --episodes, --time-budget-sec, or both")

    builder = ADAPTER_REGISTRY[args.adapter]
    adapter = builder(args)
    bound_key = adapter.bound_key()
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def log_stdout(message: str) -> None:
        stdout_lines.append(message)
        print(message)

    def log_stderr(message: str) -> None:
        stderr_lines.append(message)
        print(message, file=sys.stderr)

    log_stdout(f"adapter={args.adapter}")
    log_stdout(f"adapter_kwargs={json.dumps(args.adapter_kwargs, sort_keys=True, separators=(',', ':'))}")
    log_stdout(f"bound_key={json.dumps(bound_key, sort_keys=True, separators=(',', ':'))}")

    if args.estimate in ("drift", "all"):
        drift, drift_samples = _run_estimate(
            estimate="drift",
            args=args,
            builder=builder,
            log_stdout=log_stdout,
        )
        log_stdout(f"drift_bound={drift}")
        drift_cache_path = _cache_path(args.adapter, "drift")
        if not args.dry_run:
            try:
                LipschitzConstantProvider.set_drift_bound(adapter, drift, force=args.force)
            except ValueError as e:
                log_stderr(str(e))
        log_stdout(f"drift_cache={drift_cache_path}")
        _cache_run(
            path=drift_cache_path,
            adapter_name=args.adapter,
            bound_key=bound_key,
            estimate="drift",
            args=args,
            samples=drift_samples,
            result=drift,
            stdout_text="\n".join(stdout_lines) + "\n",
            stderr_text="\n".join(stderr_lines) + ("\n" if stderr_lines else ""),
        )

    if args.estimate in ("transition", "all"):
        transition, transition_samples = _run_estimate(
            estimate="transition",
            args=args,
            builder=builder,
            log_stdout=log_stdout,
        )
        log_stdout(f"transition_wasserstein_lipschitz={transition}")
        transition_cache_path = _cache_path(args.adapter, "transition")
        if not args.dry_run:
            try:
                LipschitzConstantProvider.set_transition_wasserstein_lipschitz(adapter, transition, force=args.force)
            except ValueError as e:
                log_stderr(str(e))
        log_stdout(f"transition_cache={transition_cache_path}")
        _cache_run(
            path=transition_cache_path,
            adapter_name=args.adapter,
            bound_key=bound_key,
            estimate="transition",
            args=args,
            samples=transition_samples,
            result=transition,
            stdout_text="\n".join(stdout_lines) + "\n",
            stderr_text="\n".join(stderr_lines) + ("\n" if stderr_lines else ""),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
