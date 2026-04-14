"""
Visualization module - fixed 2x2 panel layout for simulation results.

When a `noisy_result` is supplied every panel shows two curves:
  · solid  line = clean / ideal simulation
  · dashed line = noisy simulation

Provides:
  - plot_simulation()         - matplotlib Figure (PNG-ready)

Usage::

    from src.simulation import FlightProfile, SimulationEngine
    from src.noise import GaussianNoiseModel
    from src.visualization.plots import plot_simulation

    profile = FlightProfile.synthetic()
    clean   = SimulationEngine(profile).run()
    noisy   = SimulationEngine(profile, noise_model=GaussianNoiseModel(seed=42)).run()

    fig = plot_simulation(clean, noisy_result=noisy)
    fig.savefig("output.png", dpi=150)
"""

from __future__ import annotations

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from src.simulation.result import SimulationResult

_C = {
    "altitude":         {"clean": "#1f77b4", "noisy": "#6baed6"},   # blue
    "predicted_apogee": {"clean": "#d62728", "noisy": "#fc8d59"},   # red / orange
    "speed":            {"clean": "#ff7f0e", "noisy": "#fdae6b"},   # orange
    "static_pressure":  {"clean": "#2ca02c", "noisy": "#74c476"},   # green
    "dynamic_pressure": {"clean": "#9467bd", "noisy": "#c5b0d5"},   # purple
}

_CLEAN_STYLE  = {"linewidth": 2.0, "linestyle": "-",  "alpha": 1.0}
_NOISY_STYLE  = {"linewidth": 1.2, "linestyle": "--", "alpha": 0.85}


def plot_simulation(
    result: SimulationResult,
    noisy_result: SimulationResult | None = None,
    title: str = "Rocket Flight Simulation",
    figsize: tuple[float, float] = (14, 8),
) -> matplotlib.figure.Figure:
    """
    Generate a 2x2 matplotlib figure.

    Panel layout:
        [0,0] Altitude + Predicted Apogee  [0,1] Speed
        [1,0] Static Pressure              [1,1] Dynamic Pressure

    Args:
        result:       Clean / ideal simulation result.
        noisy_result: Optional noisy simulation result to overlay (dashed lines).
        title:        Figure super-title.
        figsize:      Figure size in inches (width, height).

    Returns:
        matplotlib.figure.Figure - caller is responsible for show() / savefig().
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    t_clean = result.time
    t_noisy = noisy_result.time if noisy_result else None
    has_noise = noisy_result is not None

    ax = axes[0, 0]
    ax.plot(t_clean, result.altitude,
            color=_C["altitude"]["clean"], label="Altitude (clean)", **_CLEAN_STYLE)
    ax.plot(t_clean, result.predicted_apogee,
            color=_C["predicted_apogee"]["clean"], label="Predicted Apogee (clean)", **_CLEAN_STYLE)
    if has_noise:
        ax.plot(t_noisy, noisy_result.altitude,
                color=_C["altitude"]["noisy"], label="Altitude (noisy)", **_NOISY_STYLE)
        ax.plot(t_noisy, noisy_result.predicted_apogee,
                color=_C["predicted_apogee"]["noisy"], label="Predicted Apogee (noisy)", **_NOISY_STYLE)
    ax.set_title("Altitude & Predicted Apogee")
    ax.set_ylabel("Altitude [m]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t_clean, result.speed,
            color=_C["speed"]["clean"], label="Speed (clean)", **_CLEAN_STYLE)
    if has_noise:
        ax.plot(t_noisy, noisy_result.speed,
                color=_C["speed"]["noisy"], label="Speed (noisy)", **_NOISY_STYLE)
        ax.legend(fontsize=8)
    ax.set_title("Airspeed")
    ax.set_ylabel("Speed [m/s]")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t_clean, result.static_pressure,
            color=_C["static_pressure"]["clean"], label="Static pressure (clean)", **_CLEAN_STYLE)
    if has_noise:
        ax.plot(t_noisy, noisy_result.static_pressure,
                color=_C["static_pressure"]["noisy"], label="Static pressure (noisy)", **_NOISY_STYLE)
        ax.legend(fontsize=8)
    ax.set_title("Static Pressure")
    ax.set_ylabel("Pressure [Pa]")
    ax.set_xlabel("Time [s]")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t_clean, result.dynamic_pressure,
            color=_C["dynamic_pressure"]["clean"], label="Dynamic pressure (clean)", **_CLEAN_STYLE)
    if has_noise:
        ax.plot(t_noisy, noisy_result.dynamic_pressure,
                color=_C["dynamic_pressure"]["noisy"], label="Dynamic pressure (noisy)", **_NOISY_STYLE)
        ax.legend(fontsize=8)
    ax.set_title("Dynamic Pressure")
    ax.set_ylabel("Pressure [Pa]")
    ax.set_xlabel("Time [s]")
    ax.grid(True, alpha=0.3)

    for ax in [axes[0, 0], axes[0, 1]]:
        ax.set_xlabel("Time [s]")

    fig.tight_layout()
    return fig