#!/usr/bin/env python3
"""Plot reward-diff histogram from Lipschitz variance JSON export.

Expected input is produced by scripts/check_pendulum_lipschitz_variance.py
with --save-diffs-json.
"""

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def collect_diffs(payload: dict, include_same: bool, include_fresh: bool) -> np.ndarray:
    diffs: List[float] = []

    if include_same:
        for run in payload.get("same_adapter_runs", []):
            diffs.extend(run.get("reward_diffs") or [])

    if include_fresh:
        for run in payload.get("fresh_adapter_runs", []):
            diffs.extend(run.get("reward_diffs") or [])

    return np.array(diffs, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot log-log histogram of reward diffs from JSON")
    parser.add_argument("json_path", type=str, help="Path to JSON from check_pendulum_lipschitz_variance.py")
    parser.add_argument("--output", type=str, default=None, help="Output plot path (.png).")
    parser.add_argument("--bins", type=int, default=60, help="Number of log-spaced bins")
    parser.add_argument("--no-same", action="store_true", help="Exclude same-adapter runs")
    parser.add_argument("--no-fresh", action="store_true", help="Exclude fresh-adapter runs")
    args = parser.parse_args()

    include_same = not args.no_same
    include_fresh = not args.no_fresh

    if not include_same and not include_fresh:
        raise ValueError("At least one of same/fresh runs must be included")

    json_path = Path(args.json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    diffs = collect_diffs(payload, include_same=include_same, include_fresh=include_fresh)
    if diffs.size == 0:
        raise ValueError("No reward_diffs found in selected runs")

    positive_diffs = diffs[diffs > 0.0]
    n_nonpositive = int(diffs.size - positive_diffs.size)
    if positive_diffs.size == 0:
        raise ValueError("All reward diffs are <= 0; cannot plot log-scale x-axis")

    # Log-spaced bins for log-log histogram.
    x_min = float(np.min(positive_diffs))
    x_max = float(np.max(positive_diffs))
    if x_min == x_max:
        # Single-value fallback: make tiny range around point
        x_min = x_min * 0.9
        x_max = x_max * 1.1
    bins = np.logspace(np.log10(x_min), np.log10(x_max), args.bins + 1)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(positive_diffs, bins=bins, color="#4c78a8", alpha=0.85, edgecolor="black", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    meta = payload.get("meta", {})
    dt = meta.get("dt", "?")
    mode = meta.get("mode", "?")
    ax.set_title("Reward-Diff Histogram (log-log)")
    ax.set_xlabel("|delta reward|")
    ax.set_ylabel("count")
    ax.grid(True, which="both", alpha=0.3)

    text = (
        f"samples={diffs.size}\n"
        f"positive={positive_diffs.size}\n"
        f"nonpositive_dropped={n_nonpositive}\n"
        f"dt={dt}\n"
        f"mode={mode}"
    )
    ax.text(
        0.98,
        0.98,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    if args.output is None:
        default_name = json_path.stem + "_reward_diff_hist_loglog.png"
        output_path = json_path.with_name(default_name)
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print(f"Loaded {diffs.size} diffs ({positive_diffs.size} positive, {n_nonpositive} non-positive dropped)")
    print(f"Saved log-log histogram to: {output_path}")


if __name__ == "__main__":
    main()
