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


def _detect_events(result: SimulationResult) -> tuple[float | None, float]:
    """
    Detect burnout and apogee times from simulation data.

    Burnout  ≈ time of maximum airspeed (thrust ends → drag decelerates rocket).
    Apogee   = time of maximum altitude.

    Returns:
        (t_burnout, t_apogee) – t_burnout is None if no clear speed peak found.
    """
    apogee_idx = int(np.argmax(result.altitude))
    t_apogee = float(result.time[apogee_idx])

    speed_ascent = result.speed[: apogee_idx + 1]
    burnout_idx = int(np.argmax(speed_ascent))
    t_burnout = float(result.time[burnout_idx]) if burnout_idx > 0 else None

    return t_burnout, t_apogee


def _add_event_lines(ax: plt.Axes, t_burnout: float | None, t_apogee: float) -> None:
    """Draw vertical event markers on a single axes panel."""
    if t_burnout is not None:
        ax.axvline(
            t_burnout, color="#ff7f0e", linestyle=":", linewidth=1.8, alpha=0.9,
            label=f"Burnout ({t_burnout:.1f} s)",
        )
    ax.axvline(
        t_apogee, color="#d62728", linestyle=":", linewidth=1.8, alpha=0.9,
        label=f"Apogee ({t_apogee:.1f} s)",
    )


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

    t_burnout, t_apogee = _detect_events(result)

    # Trim all arrays to [0, t_apogee]
    apogee_idx = int(np.argmax(result.altitude))

    def _trim(r: SimulationResult) -> tuple[np.ndarray, SimulationResult]:
        idx = np.searchsorted(r.time, t_apogee, side="right")
        trimmed = SimulationResult(
            time=r.time[:idx],
            altitude=r.altitude[:idx],
            speed=r.speed[:idx],
            static_pressure=r.static_pressure[:idx],
            dynamic_pressure=r.dynamic_pressure[:idx],
            total_pressure=r.total_pressure[:idx],
            predicted_apogee=r.predicted_apogee[:idx],
            dt=r.dt,
            noise_enabled=r.noise_enabled,
            noise_type=r.noise_type,
        )
        return trimmed.time, trimmed

    t_clean, result = _trim(result)
    has_noise = noisy_result is not None
    if has_noise:
        t_noisy, noisy_result = _trim(noisy_result)
    else:
        t_noisy = None

    dev: dict[str, dict[str, float]] = (
        result.deviation_from(noisy_result) if has_noise else {}
    )

    def _badge(ax: plt.Axes, channel: str, unit: str) -> None:
        """Annotate an axes with max and mean deviation for one channel."""
        d = dev[channel]
        ax.annotate(
            f"Max |Δ| = {d['max']:.2f} {unit}\nMean |Δ| = {d['mean']:.2f} {unit}",
            xy=(0.98, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=7.5,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f5f5f5",
                  "edgecolor": "#aaaaaa", "alpha": 0.92},
        )

    # ------------------------------------------------------------------
    # [0, 0]  Altitude + Predicted Apogee
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    ax.plot(t_clean, result.altitude,
            color=_C["altitude"]["clean"], label="Altitude (clean)", **_CLEAN_STYLE)
    ax.plot(t_clean, result.predicted_apogee,
            color=_C["predicted_apogee"]["clean"], label="Pred. Apogee (clean)", **_CLEAN_STYLE)
    if has_noise:
        ax.plot(t_noisy, noisy_result.altitude,
                color=_C["altitude"]["noisy"], label="Altitude (noisy)", **_NOISY_STYLE)
        ax.plot(t_noisy, noisy_result.predicted_apogee,
                color=_C["predicted_apogee"]["noisy"], label="Pred. Apogee (noisy)", **_NOISY_STYLE)
        # Badge shows the larger of the two deviations (altitude dominates scale)
        worst = max(dev["altitude"]["max"], dev["predicted_apogee"]["max"])
        worst_mean = max(dev["altitude"]["mean"], dev["predicted_apogee"]["mean"])
        ax.annotate(
            f"Max |Δ| = {worst:.2f} m\nMean |Δ| = {worst_mean:.2f} m",
            xy=(0.98, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=7.5,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f5f5f5",
                  "edgecolor": "#aaaaaa", "alpha": 0.92},
        )
    _add_event_lines(ax, t_burnout, t_apogee)
    ax.set_title("Altitude & Predicted Apogee")
    ax.set_ylabel("Altitude [m]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # [0, 1]  Speed
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    ax.plot(t_clean, result.speed,
            color=_C["speed"]["clean"], label="Speed (clean)", **_CLEAN_STYLE)
    if has_noise:
        ax.plot(t_noisy, noisy_result.speed,
                color=_C["speed"]["noisy"], label="Speed (noisy)", **_NOISY_STYLE)
        _badge(ax, "speed", "m/s")
    _add_event_lines(ax, t_burnout, t_apogee)
    ax.set_title("Airspeed")
    ax.set_ylabel("Speed [m/s]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # [1, 0]  Static Pressure
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    ax.plot(t_clean, result.static_pressure,
            color=_C["static_pressure"]["clean"], label="Static pressure (clean)", **_CLEAN_STYLE)
    if has_noise:
        ax.plot(t_noisy, noisy_result.static_pressure,
                color=_C["static_pressure"]["noisy"], label="Static pressure (noisy)", **_NOISY_STYLE)
        _badge(ax, "static_pressure", "Pa")
    _add_event_lines(ax, t_burnout, t_apogee)
    ax.set_title("Static Pressure")
    ax.set_ylabel("Pressure [Pa]")
    ax.set_xlabel("Time [s]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # [1, 1]  Dynamic Pressure
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    ax.plot(t_clean, result.dynamic_pressure,
            color=_C["dynamic_pressure"]["clean"], label="Dynamic pressure (clean)", **_CLEAN_STYLE)
    if has_noise:
        ax.plot(t_noisy, noisy_result.dynamic_pressure,
                color=_C["dynamic_pressure"]["noisy"], label="Dynamic pressure (noisy)", **_NOISY_STYLE)
        _badge(ax, "dynamic_pressure", "Pa")
    _add_event_lines(ax, t_burnout, t_apogee)
    ax.set_title("Dynamic Pressure")
    ax.set_ylabel("Pressure [Pa]")
    ax.set_xlabel("Time [s]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    for ax in [axes[0, 0], axes[0, 1]]:
        ax.set_xlabel("Time [s]")

    fig.tight_layout()
    return fig