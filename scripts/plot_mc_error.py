#!/usr/bin/env python3
"""
Plot Monte Carlo prediction-error API JSON: mean signed error vs time and optional scatter.

Usage:
  python3 scripts/plot_mc_error.py data/prediction_error_response.json --output data/mc_error.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot prediction-error JSON from API.")
    parser.add_argument("json_path", type=Path, help="JSON file from POST .../prediction-error")
    parser.add_argument("--output", "-o", type=Path, default=Path("mc_prediction_error.png"))
    args = parser.parse_args()

    with args.json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    t = data["time"]
    mean = data["signed_error"]["mean"]
    p05 = data["signed_error"]["p05"]
    p95 = data["signed_error"]["p95"]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(t, p05, p95, alpha=0.25, label="P05-P95 signed error")
    ax.plot(t, mean, color="C0", lw=2, label="Mean signed error")

    st = data.get("scatter_time")
    se = data.get("scatter_signed_error")
    if st and se:
        ax.scatter(st, se, s=4, alpha=0.15, c="gray", label="Samples (subsampled)")

    ax.axhline(0.0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("predicted_apogee − reference_apogee [m]")
    ref = data.get("reference_apogee_m", 0.0)
    ax.set_title(
        f"Prediction error vs reference apogeum ({ref:.1f} m); "
        f"MAE full={data['mean_abs_error_full_flight_m']:.3f} m"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
