from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "monitoring_martingales_matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from monitor.weighting import OptimalTemporalWeights, UniformWeights


OUTPUT_DIR = Path("data") / "ci_sweeps"


@dataclass(frozen=True)
class CITerms:
    gamma: float
    rho: float
    h: float
    delta: float
    radius: int
    m: int
    continuous: bool
    statistical_error: float
    discretization_error: float
    atom_error: float
    total_error: float


def _parse_float_values(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float")
    return values


def optimal_radius_for(
    *,
    rho: float,
    h: float,
    delta: float,
    continuous: bool,
) -> int:
    window_length = OptimalTemporalWeights.compute_window_length(
        rho=rho,
        h=h,
        delta=delta,
        continuous=continuous,
    )
    return round((window_length - 1) / 2)


def compute_ci_terms(
    *,
    gamma: float,
    rho: float,
    h: float,
    delta: float,
    continuous: bool,
) -> CITerms:
    radius = optimal_radius_for(
        rho=rho,
        h=h,
        delta=delta,
        continuous=continuous,
    )
    m = 2 * radius + 1
    statistical_error = UniformWeights.SE_fla(
        gamma=gamma,
        h=h,
        m=m,
        delta=delta,
        continuous=continuous,
    )
    discretization_error = UniformWeights.DE_fla(
        rho=rho,
        h=h,
        gamma=gamma,
        m=m,
        continuous=continuous,
    )
    atom_error = UniformWeights.AE_fla(
        rho=rho,
        h=h,
        gamma=gamma,
        continuous=continuous,
    )
    return CITerms(
        gamma=gamma,
        rho=rho,
        h=h,
        delta=delta,
        radius=radius,
        m=m,
        continuous=continuous,
        statistical_error=statistical_error,
        discretization_error=discretization_error,
        atom_error=atom_error,
        total_error=statistical_error + discretization_error + atom_error,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep CI formula terms over synthetic parameters.")
    parser.add_argument("--gamma", type=_parse_float_values, required=True, help="Comma-separated gamma values")
    parser.add_argument("--rho", type=_parse_float_values, required=True, help="Comma-separated rho values")
    parser.add_argument("--h", type=_parse_float_values, default=[1.0], help="Comma-separated step sizes")
    parser.add_argument("--delta", type=_parse_float_values, default=[0.01], help="Comma-separated delta values")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-plot", action="store_true")
    return parser


def _output_path(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output_dir / f"{timestamp}_ci_sweep.json"


def _plot_path(json_path: Path) -> Path:
    return json_path.with_suffix(".png")


def _grid_for(rows: list[CITerms], *, h: float, metric: str, gamma_values: list[float], rho_values: list[float]) -> np.ndarray:
    by_key = {
        (row.gamma, row.rho, row.h): getattr(row, metric)
        for row in rows
    }
    grid = np.full((len(gamma_values), len(rho_values)), np.nan)
    for i, gamma in enumerate(gamma_values):
        for j, rho in enumerate(rho_values):
            grid[i, j] = by_key[(gamma, rho, h)]
    return grid


def _tick_positions(values: list[float]) -> tuple[list[int], list[str]]:
    if len(values) <= 8:
        positions = list(range(len(values)))
    else:
        positions = sorted(set(np.linspace(0, len(values) - 1, 8, dtype=int).tolist()))
    labels = [f"{values[pos]:g}" for pos in positions]
    return positions, labels


def write_plot(path: Path, rows: list[CITerms], *, gamma_values: list[float], rho_values: list[float], h_values: list[float]) -> Path:
    metrics = [
        ("total_error", "total"),
        ("statistical_error", "SE"),
        ("discretization_error", "DE"),
        ("atom_error", "AE"),
        ("radius", "radius"),
    ]
    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=len(h_values),
        figsize=(4.2 * len(h_values), 3.2 * len(metrics)),
        squeeze=False,
        constrained_layout=True,
    )

    x_ticks, x_labels = _tick_positions(rho_values)
    y_ticks, y_labels = _tick_positions(gamma_values)

    for row_idx, (metric, metric_label) in enumerate(metrics):
        metric_grids = [
            _grid_for(rows, h=h, metric=metric, gamma_values=gamma_values, rho_values=rho_values)
            for h in h_values
        ]
        vmin = min(float(np.nanmin(grid)) for grid in metric_grids)
        vmax = max(float(np.nanmax(grid)) for grid in metric_grids)

        for col_idx, h in enumerate(h_values):
            ax = axes[row_idx][col_idx]
            image = ax.imshow(
                metric_grids[col_idx],
                origin="lower",
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
            )
            if row_idx == 0:
                ax.set_title(f"h={h:g}")
            if col_idx == 0:
                ax.set_ylabel(f"{metric_label}\ngamma")
            else:
                ax.set_ylabel("")
            if row_idx == len(metrics) - 1:
                ax.set_xlabel("rho")
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("CI Sweep Heatmaps", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    rows: list[CITerms] = []
    if len(args.delta) != 1:
        parser.error("Plotting expects exactly one --delta value for now")

    for gamma in args.gamma:
        for rho in args.rho:
            for h in args.h:
                for delta in args.delta:
                    rows.append(
                        compute_ci_terms(
                            gamma=gamma,
                            rho=rho,
                            h=h,
                            delta=delta,
                            continuous=args.continuous,
                        )
                    )

    path = _output_path(args.output_dir)
    payload = {
        "args": {
            "gamma": args.gamma,
            "rho": args.rho,
            "h": args.h,
            "delta": args.delta,
            "continuous": args.continuous,
            "radius": "optimal",
        },
        "row_count": len(rows),
        "rows": [asdict(row) for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    plot_path = None
    if not args.no_plot:
        plot_path = write_plot(
            _plot_path(path),
            rows,
            gamma_values=args.gamma,
            rho_values=args.rho,
            h_values=args.h,
        )

    print(f"ci_sweep_rows={len(rows)}")
    print(f"ci_sweep_output={path}")
    if plot_path is not None:
        print(f"ci_sweep_plot={plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
