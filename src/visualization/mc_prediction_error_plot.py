"""PNG figure for Monte Carlo prediction-error results."""

from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.api.schemas import MonteCarloPredictionErrorResponse

_C = {
    "mean_error":        "#c51b8a",   # vivid magenta — raw noisy mean
    "mc_samples":        "#f7b0dc",   # light pink — raw scatter
    "p05_p95_band":      "#a5f595",   # light green — raw 90% band
    "mean_error_kf":     "#1f77b4",   # blue — KF-filtered mean
    "mc_samples_kf":     "#aec7e8",   # light blue — KF scatter
    "p05_p95_band_kf":   "#c6e9f7",   # very light blue — KF 90% band
    "mean_error_pf":     "#2ca02c",   # green — EMA pressure-filter mean
    "mc_samples_pf":     "#98df8a",   # light green — EMA scatter
    "p05_p95_band_pf":   "#d4f7d4",   # very light green — EMA 90% band
    "burnout":           "#ff7f0e",   # orange
    "apogee":            "#d62728",   # red
    "zero_line":         "#4d4d4d",   # dark gray
}


def prediction_error_figure_png_bytes(
    data: MonteCarloPredictionErrorResponse,
    scatter_time: list[float] | None,
    scatter_signed_error: list[float] | None,
    scatter_signed_error_filtered: list[float] | None,
    scatter_signed_error_pf: list[float] | None,
    title: str,
    dpi: int,
) -> bytes:
    """Build a prediction-error panel up to apogee; return PNG bytes.

    When ``data.signed_error_filtered`` is present, overlays the KF-filtered
    prediction error in blue. When ``data.signed_error_pf`` is present, overlays
    the EMA pressure-filter prediction error in green.
    """
    t = np.asarray(data.time, dtype=np.float64)
    mean = np.asarray(data.signed_error.mean, dtype=np.float64)
    p05 = np.asarray(data.signed_error.p05, dtype=np.float64)
    p95 = np.asarray(data.signed_error.p95, dtype=np.float64)

    has_kf = data.signed_error_filtered is not None
    if has_kf:
        mean_f = np.asarray(data.signed_error_filtered.mean, dtype=np.float64)
        p05_f = np.asarray(data.signed_error_filtered.p05, dtype=np.float64)
        p95_f = np.asarray(data.signed_error_filtered.p95, dtype=np.float64)

    has_pf = data.signed_error_pf is not None
    if has_pf:
        mean_pf = np.asarray(data.signed_error_pf.mean, dtype=np.float64)
        p05_pf = np.asarray(data.signed_error_pf.p05, dtype=np.float64)
        p95_pf = np.asarray(data.signed_error_pf.p95, dtype=np.float64)

    t_burnout = data.burnout_time_used_s
    t_apogee = data.apogee_time_used_s

    fig, ax = plt.subplots(1, 1, figsize=(11, 4.8))

    pre_mask = np.ones_like(t, dtype=bool) if t_apogee is None else (t <= t_apogee)
    if not np.any(pre_mask):
        pre_mask = np.ones_like(t, dtype=bool)

    t_panel = t[pre_mask]

    # --- Raw noisy band and mean ---
    ax.fill_between(
        t_panel,
        p05[pre_mask],
        p95[pre_mask],
        color=_C["p05_p95_band"],
        alpha=0.35,
        label="P05–P95 noisy (90% runs)",
    )
    ax.plot(
        t_panel,
        mean[pre_mask],
        color=_C["mean_error"],
        lw=2,
        label="Mean error (noisy)",
    )

    if scatter_time and scatter_signed_error:
        s_t = np.asarray(scatter_time, dtype=np.float64)
        s_e = np.asarray(scatter_signed_error, dtype=np.float64)
        s_mask = (s_t >= t_panel[0]) & (s_t <= t_panel[-1])
        if np.any(s_mask):
            ax.scatter(
                s_t[s_mask],
                s_e[s_mask],
                s=4,
                alpha=0.25,
                c=_C["mc_samples"],
                label="MC samples noisy (subsampled)",
            )

    # --- KF-filtered band and mean (overlay) ---
    if has_kf:
        ax.fill_between(
            t_panel,
            p05_f[pre_mask],
            p95_f[pre_mask],
            color=_C["p05_p95_band_kf"],
            alpha=0.40,
            label="P05–P95 KF-filtered (90% runs)",
        )
        ax.plot(
            t_panel,
            mean_f[pre_mask],
            color=_C["mean_error_kf"],
            lw=2,
            linestyle="--",
            label="Mean error (KF-filtered)",
        )

        if scatter_time and scatter_signed_error_filtered:
            s_t = np.asarray(scatter_time, dtype=np.float64)
            s_ef = np.asarray(scatter_signed_error_filtered, dtype=np.float64)
            s_mask = (s_t >= t_panel[0]) & (s_t <= t_panel[-1])
            if np.any(s_mask):
                ax.scatter(
                    s_t[s_mask],
                    s_ef[s_mask],
                    s=4,
                    alpha=0.20,
                    c=_C["mc_samples_kf"],
                    label="MC samples KF-filtered (subsampled)",
                )

    # --- EMA pressure-filter band and mean (overlay) ---
    if has_pf:
        ax.fill_between(
            t_panel,
            p05_pf[pre_mask],
            p95_pf[pre_mask],
            color=_C["p05_p95_band_pf"],
            alpha=0.40,
            label="P05–P95 EMA-filtered (90% runs)",
        )
        ax.plot(
            t_panel,
            mean_pf[pre_mask],
            color=_C["mean_error_pf"],
            lw=2,
            linestyle="-.",
            label="Mean error (EMA pressure-filtered)",
        )

        if scatter_time and scatter_signed_error_pf:
            s_t = np.asarray(scatter_time, dtype=np.float64)
            s_epf = np.asarray(scatter_signed_error_pf, dtype=np.float64)
            s_mask = (s_t >= t_panel[0]) & (s_t <= t_panel[-1])
            if np.any(s_mask):
                ax.scatter(
                    s_t[s_mask],
                    s_epf[s_mask],
                    s=4,
                    alpha=0.20,
                    c=_C["mc_samples_pf"],
                    label="MC samples EMA-filtered (subsampled)",
                )

    # --- Reference lines ---
    if t_burnout is not None and t_panel[0] <= t_burnout <= t_panel[-1]:
        ax.axvline(
            t_burnout,
            color=_C["burnout"],
            lw=1.8,
            linestyle=":",
            alpha=0.9,
            label="Burnout",
        )
    if t_apogee is not None and t_panel[0] <= t_apogee <= t_panel[-1]:
        ax.axvline(
            t_apogee,
            color=_C["apogee"],
            lw=1.8,
            linestyle=":",
            alpha=0.9,
            label="Apogee",
        )

    ax.axhline(0.0, color=_C["zero_line"], lw=0.7, linestyle="--", alpha=0.75)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("predicted_apogee − reference [m]")
    ax.set_title("Up to apogee", fontsize=10)
    ax.grid(True, alpha=0.3)

    subtitle = (
        f"h_ref={data.reference_apogee_m:.1f} m | "
        f"MAE→apogee={data.mean_abs_error_full_flight_m:.3f} m | "
        f"RMSE→apogee={data.rmse_full_flight_m:.3f} m"
    )
    if data.mean_abs_error_post_burnout_m is not None:
        subtitle += f" | MAE burnout→apogee={data.mean_abs_error_post_burnout_m:.3f} m"
    if has_kf and data.mean_abs_error_full_flight_filtered_m is not None:
        subtitle += (
            f"\nKF MAE→apogee={data.mean_abs_error_full_flight_filtered_m:.3f} m | "
            f"KF RMSE→apogee={data.rmse_full_flight_filtered_m:.3f} m"
        )
        if data.mean_abs_error_post_burnout_filtered_m is not None:
            subtitle += (
                f" | KF MAE burnout→apogee={data.mean_abs_error_post_burnout_filtered_m:.3f} m"
            )
    if has_pf and data.mean_abs_error_full_flight_pf_m is not None:
        subtitle += (
            f"\nEMA MAE→apogee={data.mean_abs_error_full_flight_pf_m:.3f} m | "
            f"EMA RMSE→apogee={data.rmse_full_flight_pf_m:.3f} m"
        )
        if data.mean_abs_error_post_burnout_pf_m is not None:
            subtitle += (
                f" | EMA MAE burnout→apogee={data.mean_abs_error_post_burnout_pf_m:.3f} m"
            )

    fig.suptitle(f"{title}\n{subtitle}", fontsize=9)

    handles, labels = ax.get_legend_handles_labels()
    dedup: dict[str, object] = {}
    for h, lbl in zip(handles, labels):
        if lbl not in dedup:
            dedup[lbl] = h
    ax.legend(dedup.values(), dedup.keys(), loc="best", fontsize=8)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
