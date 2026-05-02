#!/usr/bin/env python3
"""
Kalman Filter Tuning — Monte Carlo Grid Search.

Searches over (sigma_a, sigma_h, sigma_v) on a 3-D grid.
Each point runs N_MC independent BinczarNoiseModel + KalmanFilter simulations
and evaluates:

  - MAE_post  : mean absolute prediction error from burnout to apogee  [m]
  - MAE_full  : mean absolute prediction error from launch to apogee   [m]
  - RMSE_post : RMSE from burnout to apogee                           [m]

Output — three separate PNG files derived from --output stem:
  <stem>_heatmaps.png   – MAE_post heatmaps for each parameter pair
  <stem>_ranking.png    – top-15 configurations bar chart
  <stem>_timeseries.png – signed prediction-error time series with
                          cloud of ALL grid configs + highlighted best / ref KF

Usage:
  poetry run python3 scripts/tune_kalman.py
  poetry run python3 scripts/tune_kalman.py \\
      --csv data/or_flight.csv \\
      --n-mc 20 --base-seed 42 \\
      --sigma-a 5 15 45 120 \\
      --sigma-h 5 15 45 120 \\
      --sigma-v 2 7 20 60 \\
      --ref-sigma-a 30 --ref-sigma-h 20 --ref-sigma-v 30 \\
      --output results/kf_tune
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filters.kalman import KalmanFilter
from src.models.physics import ModelParams
from src.noise.noise_model import BinczarNoiseModel, NoNoiseModel
from src.simulation.engine import SimulationEngine
from src.simulation.flight_profile import FlightProfile
from src.simulation.result import SimulationResult

# ── colour palette ─────────────────────────────────────────────────────────────
_C = {
    "raw":          "#c51b8a",   # magenta  – raw noisy
    "raw_band":     "#f7cce9",   # pale pink band
    "best_kf":      "#1f77b4",   # blue     – best KF
    "best_band":    "#aec7e8",   # light blue band
    "ref_kf":       "#e65c00",   # orange   – user's reference KF
    "ref_band":     "#ffd5aa",   # pale orange band
    "cloud":        "#4db86c",   # medium green – all-other-configs cloud
    "burnout":      "#ff7f0e",
    "apogee":       "#d62728",
}


# ══════════════════════════════════════════════════════════════════════════════
# Core MC helpers
# ══════════════════════════════════════════════════════════════════════════════

def _clean_run(profile: FlightProfile, params: ModelParams) -> SimulationResult:
    return SimulationEngine(profile, params=params, noise_model=NoNoiseModel()).run()


def _burnout_apogee_times(clean: SimulationResult) -> tuple[float | None, float | None]:
    t = clean.time
    return float(t[int(np.argmax(clean.speed))]), float(t[int(np.argmax(clean.altitude))])


def _run_mc(
    profile: FlightProfile,
    params: ModelParams,
    sigma_a: float,
    sigma_h: float,
    sigma_v: float,
    n_mc: int,
    base_seed: int,
) -> list[SimulationResult]:
    def _one(seed: int) -> SimulationResult:
        noise = BinczarNoiseModel(seed=seed)
        kf    = KalmanFilter(sigma_a=sigma_a, sigma_h=sigma_h, sigma_v=sigma_v)
        return SimulationEngine(profile, params=params, noise_model=noise, filter_model=kf).run()

    with concurrent.futures.ThreadPoolExecutor() as ex:
        return list(ex.map(_one, range(base_seed, base_seed + n_mc)))


def _run_mc_nofilter(
    profile: FlightProfile,
    params: ModelParams,
    n_mc: int,
    base_seed: int,
) -> list[SimulationResult]:
    def _one(seed: int) -> SimulationResult:
        return SimulationEngine(
            profile, params=params, noise_model=BinczarNoiseModel(seed=seed)
        ).run()

    with concurrent.futures.ThreadPoolExecutor() as ex:
        return list(ex.map(_one, range(base_seed, base_seed + n_mc)))


def _compute_metrics(
    results: list[SimulationResult],
    clean: SimulationResult,
    t_burnout: float | None,
    t_apogee: float | None,
    use_filtered: bool = True,
) -> dict[str, float]:
    time_1d = results[0].time
    clean_pred = np.interp(time_1d, clean.time, clean.predicted_apogee)
    pred_stack = np.stack(
        [r.predicted_apogee_filtered if use_filtered else r.predicted_apogee for r in results],
        axis=0,
    )
    errors = pred_stack - clean_pred[None, :]

    full_mask = time_1d <= t_apogee if t_apogee else np.ones_like(time_1d, dtype=bool)
    if not np.any(full_mask):
        full_mask = np.ones_like(time_1d, dtype=bool)

    post_mask: np.ndarray | None = None
    if t_burnout is not None:
        pm = time_1d >= t_burnout
        if t_apogee:
            pm &= time_1d <= t_apogee
        if np.any(pm):
            post_mask = pm

    def _stats(mask: np.ndarray) -> tuple[float, float]:
        e = errors[:, mask]
        return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e**2)))

    mae_full, _ = _stats(full_mask)
    mae_post, rmse_post = (
        _stats(post_mask) if post_mask is not None else (float("nan"), float("nan"))
    )
    return {"mae_full": mae_full, "mae_post": mae_post, "rmse_post": rmse_post}


def _mean_signed_error(
    results: list[SimulationResult],
    clean: SimulationResult,
    use_filtered: bool = True,
) -> np.ndarray:
    """Return the mean signed-error array (n_steps,)."""
    time_1d = results[0].time
    clean_pred = np.interp(time_1d, clean.time, clean.predicted_apogee)
    pred_stack = np.stack(
        [r.predicted_apogee_filtered if use_filtered else r.predicted_apogee for r in results],
        axis=0,
    )
    return np.mean(pred_stack - clean_pred[None, :], axis=0)


def _signed_error_stats(
    results: list[SimulationResult],
    clean: SimulationResult,
    use_filtered: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (time, mean, p05, p95) of signed error across MC runs."""
    time_1d = results[0].time
    clean_pred = np.interp(time_1d, clean.time, clean.predicted_apogee)
    pred_stack = np.stack(
        [r.predicted_apogee_filtered if use_filtered else r.predicted_apogee for r in results],
        axis=0,
    )
    errors = pred_stack - clean_pred[None, :]
    return (
        time_1d,
        np.mean(errors, axis=0),
        np.percentile(errors, 5, axis=0),
        np.percentile(errors, 95, axis=0),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Grid search
# ══════════════════════════════════════════════════════════════════════════════

def grid_search(
    profile: FlightProfile,
    params: ModelParams,
    clean: SimulationResult,
    sigma_a_vals: list[float],
    sigma_h_vals: list[float],
    sigma_v_vals: list[float],
    n_mc: int,
    base_seed: int,
) -> tuple[list[dict], list[np.ndarray]]:
    """
    Evaluate every (σ_a, σ_h, σ_v) combination.

    Returns:
        records          – list of metric dicts, sorted by mae_post ascending
        mean_error_curves – mean signed-error array (n_steps,) per combo,
                           in the same order as records (after sorting)
    """
    t_burnout, t_apogee = _burnout_apogee_times(clean)
    combos = list(itertools.product(sigma_a_vals, sigma_h_vals, sigma_v_vals))
    total  = len(combos)
    records: list[dict] = []
    curves:  list[np.ndarray] = []

    print(f"\nGrid search: {total} combinations × {n_mc} MC runs each")
    t0 = time.perf_counter()

    for i, (sa, sh, sv) in enumerate(combos, 1):
        results = _run_mc(profile, params, sa, sh, sv, n_mc, base_seed)
        m       = _compute_metrics(results, clean, t_burnout, t_apogee, use_filtered=True)
        curve   = _mean_signed_error(results, clean, use_filtered=True)
        records.append({"sigma_a": sa, "sigma_h": sh, "sigma_v": sv, **m})
        curves.append(curve)

        elapsed = time.perf_counter() - t0
        eta     = elapsed / i * (total - i)
        bar     = "█" * int(30 * i / total) + "░" * (30 - int(30 * i / total))
        print(
            f"\r  [{bar}] {i}/{total}  "
            f"σ_a={sa:6.1f} σ_h={sh:6.1f} σ_v={sv:6.1f}  "
            f"MAE_post={m['mae_post']:7.2f} m  ETA {eta:5.0f}s",
            end="", flush=True,
        )

    print(f"\n  Done in {time.perf_counter() - t0:.1f}s")

    # sort both by mae_post
    order  = sorted(range(len(records)), key=lambda i: records[i]["mae_post"])
    records = [records[i] for i in order]
    curves  = [curves[i]  for i in order]
    return records, curves


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — heatmaps
# ══════════════════════════════════════════════════════════════════════════════

def _heatmap(
    ax: plt.Axes,
    x_vals: list[float],
    y_vals: list[float],
    z: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    """Filled heatmap with per-cell MAE annotations. No star marker."""
    vmin, vmax = np.nanmin(z), np.nanmax(z)
    im = ax.imshow(
        z,
        origin="lower",
        aspect="auto",
        cmap="RdYlGn_r",
        vmin=vmin, vmax=vmax,
        extent=[-0.5, len(x_vals) - 0.5, -0.5, len(y_vals) - 0.5],
    )
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([f"{v:.0f}" for v in x_vals], fontsize=8)
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels([f"{v:.0f}" for v in y_vals], fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, label="MAE_post [m]", pad=0.02)

    threshold = vmin + 0.65 * (vmax - vmin)
    for yi in range(len(y_vals)):
        for xi in range(len(x_vals)):
            val = z[yi, xi]
            if not np.isnan(val):
                ax.text(
                    xi, yi, f"{val:.1f}",
                    ha="center", va="center", fontsize=6.5,
                    color="white" if val > threshold else "black",
                )


def save_heatmaps(
    records: list[dict],
    sigma_a_vals: list[float],
    sigma_h_vals: list[float],
    sigma_v_vals: list[float],
    best: dict,
    raw_metrics: dict,
    out_path: Path,
    dpi: int,
    csv_name: str,
    n_mc: int,
) -> None:
    import pandas as pd
    df = pd.DataFrame(records)

    best_sa = best["sigma_a"]
    best_sh = best["sigma_h"]
    best_sv = best["sigma_v"]

    def _lookup(row_vals, col_vals, row_key, col_key, fix_key, fix_val):
        z = np.full((len(row_vals), len(col_vals)), np.nan)
        sub = df[np.isclose(df[fix_key], fix_val)]
        for ri, rv in enumerate(row_vals):
            for ci, cv in enumerate(col_vals):
                cell = sub[np.isclose(sub[row_key], rv) & np.isclose(sub[col_key], cv)]
                if not cell.empty:
                    z[ri, ci] = cell["mae_post"].values[0]
        return z

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#f8f8f8")

    z1 = _lookup(sigma_h_vals, sigma_a_vals, "sigma_h", "sigma_a", "sigma_v", best_sv)
    _heatmap(axes[0], sigma_a_vals, sigma_h_vals, z1,
             "σ_a [m/s²]", "σ_h [m]",
             f"MAE_post: σ_a × σ_h  (σ_v = {best_sv:.0f} m/s fixed at optimal)")

    z2 = _lookup(sigma_v_vals, sigma_a_vals, "sigma_v", "sigma_a", "sigma_h", best_sh)
    _heatmap(axes[1], sigma_a_vals, sigma_v_vals, z2,
             "σ_a [m/s²]", "σ_v [m/s]",
             f"MAE_post: σ_a × σ_v  (σ_h = {best_sh:.0f} m fixed at optimal)")

    z3 = _lookup(sigma_v_vals, sigma_h_vals, "sigma_v", "sigma_h", "sigma_a", best_sa)
    _heatmap(axes[2], sigma_h_vals, sigma_v_vals, z3,
             "σ_h [m]", "σ_v [m/s]",
             f"MAE_post: σ_h × σ_v  (σ_a = {best_sa:.0f} m/s² fixed at optimal)")

    n_combos = len(sigma_a_vals) * len(sigma_h_vals) * len(sigma_v_vals)
    fig.suptitle(
        f"KF Tuning — MAE_post heatmaps | {csv_name} · {n_combos} combos × {n_mc} MC runs\n"
        f"Optimal: σ_a={best_sa:.0f}  σ_h={best_sh:.0f}  σ_v={best_sv:.0f}"
        f" → MAE_post = {best['mae_post']:.2f} m"
        f"  (raw-noise baseline: {raw_metrics['mae_post']:.2f} m)",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — ranking
# ══════════════════════════════════════════════════════════════════════════════

def save_ranking(
    records: list[dict],
    raw_metrics: dict,
    out_path: Path,
    dpi: int,
    csv_name: str,
    n_mc: int,
    top_n: int = 15,
    ref_sa: float | None = None,
    ref_sh: float | None = None,
    ref_sv: float | None = None,
) -> None:
    from matplotlib.patches import Patch

    top = records[:top_n]
    labels   = [f"σ_a={r['sigma_a']:.0f}  σ_h={r['sigma_h']:.0f}  σ_v={r['sigma_v']:.0f}" for r in top]
    mae_vals = [r["mae_post"] for r in top]

    colors = []
    for r in top:
        is_ref = (
            ref_sa is not None
            and np.isclose(r["sigma_a"], ref_sa)
            and np.isclose(r["sigma_h"], ref_sh)
            and np.isclose(r["sigma_v"], ref_sv)
        )
        colors.append(_C["ref_kf"] if is_ref else _C["best_kf"])
    colors[0] = "gold"

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.45 + 1.5)))
    fig.patch.set_facecolor("#f8f8f8")

    bars = ax.barh(range(len(top)), mae_vals, color=colors, edgecolor="gray", linewidth=0.5)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("MAE_post [m]", fontsize=10)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="x", alpha=0.3)

    x_max = max(mae_vals) * 1.15
    for bar, val in zip(bars, mae_vals):
        ax.text(val + x_max * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f} m", va="center", fontsize=8)
    ax.set_xlim(0, x_max)

    # raw baseline line
    ax.axvline(raw_metrics["mae_post"], color=_C["raw"], lw=1.8, linestyle="--",
               alpha=0.85, label=f"Raw-noise baseline ({raw_metrics['mae_post']:.2f} m)")

    legend_elements = [
        Patch(facecolor="gold",        label="Best found"),
        Patch(facecolor=_C["best_kf"], label="Other top-N"),
    ]
    if ref_sa is not None:
        legend_elements.append(Patch(facecolor=_C["ref_kf"], label="User's reference KF"))
    legend_elements.append(
        plt.Line2D([0], [0], color=_C["raw"], lw=1.8, linestyle="--",
                   label=f"Raw-noise baseline ({raw_metrics['mae_post']:.2f} m)")
    )
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    n_combos = len(records)
    ax.set_title(
        f"KF Tuning — Top-{top_n} configurations (MAE burnout→apogee)\n"
        f"{csv_name} · {n_combos} combos × {n_mc} MC runs",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — signed-error time series with all-configs cloud
# ══════════════════════════════════════════════════════════════════════════════

def save_timeseries(
    records: list[dict],
    all_curves: list[np.ndarray],   # mean error per grid config, same order as records
    clean: SimulationResult,
    raw_results: list[SimulationResult],
    best_results: list[SimulationResult],
    ref_results: list[SimulationResult] | None,
    t_burnout: float | None,
    t_apogee: float | None,
    best: dict,
    raw_metrics: dict,
    ref_sa: float | None,
    ref_sh: float | None,
    ref_sv: float | None,
    out_path: Path,
    dpi: int,
    csv_name: str,
    n_mc: int,
) -> None:
    from matplotlib.lines import Line2D

    t_arr = raw_results[0].time
    mask  = t_arr <= t_apogee if t_apogee else np.ones_like(t_arr, dtype=bool)
    if not np.any(mask):
        mask = np.ones_like(t_arr, dtype=bool)
    t_plot = t_arr[mask]

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#f8f8f8")

    # ── cloud: ALL grid configurations ───────────────────────────────────────
    # identify best and ref indices to exclude from the grey cloud
    best_idx = 0   # records already sorted → best is first
    ref_idx  = next(
        (
            i for i, r in enumerate(records)
            if ref_sa is not None
            and np.isclose(r["sigma_a"], ref_sa)
            and np.isclose(r["sigma_h"], ref_sh)
            and np.isclose(r["sigma_v"], ref_sv)
        ),
        None,
    )

    first_cloud = True
    for i, (rec, curve) in enumerate(zip(records, all_curves)):
        if i in (best_idx, ref_idx):
            continue
        label = "_cloud" if not first_cloud else "All other KF configs (mean)"
        ax.plot(
            t_plot, curve[mask],
            color=_C["cloud"], lw=0.6, alpha=0.18,
            label=label, zorder=1,
        )
        first_cloud = False

    # ── helper to draw mean + P05/P95 band ───────────────────────────────────
    def _draw(results, label_mean, label_band, color, band_color, filtered: bool, ls="-", zorder=3):
        _, mean, p05, p95 = _signed_error_stats(results, clean, use_filtered=filtered)
        ax.fill_between(t_plot, p05[mask], p95[mask],
                        color=band_color, alpha=0.35, label=label_band, zorder=zorder)
        ax.plot(t_plot, mean[mask],
                color=color, lw=2.2, linestyle=ls, label=label_mean, zorder=zorder + 1)

    # ── raw noisy ─────────────────────────────────────────────────────────────
    _draw(raw_results,
          f"raw noisy — mean error (MAE_post={raw_metrics['mae_post']:.2f} m)",
          "raw noisy — P05–P95",
          _C["raw"], _C["raw_band"], filtered=False, zorder=4)

    best_m = _compute_metrics(best_results, clean, t_burnout, t_apogee, use_filtered=True)
    _draw(best_results,
          (f"best KF σ_a={best['sigma_a']:.0f} σ_h={best['sigma_h']:.0f} "
           f"σ_v={best['sigma_v']:.0f} — mean error (MAE_post={best_m['mae_post']:.2f} m)"),
          "best KF — P05–P95",
          _C["best_kf"], _C["best_band"], filtered=True, zorder=6)

    if ref_results is not None:
        ref_m = _compute_metrics(ref_results, clean, t_burnout, t_apogee, use_filtered=True)
        _draw(ref_results,
              (f"ref KF σ_a={ref_sa:.0f} σ_h={ref_sh:.0f} σ_v={ref_sv:.0f} "
               f"— mean error (MAE_post={ref_m['mae_post']:.2f} m)"),
              "ref KF — P05–P95",
              _C["ref_kf"], _C["ref_band"], filtered=True, ls="--", zorder=5)

    # ── reference lines ───────────────────────────────────────────────────────
    ax.axhline(0.0, color="#4d4d4d", lw=0.8, linestyle="--", alpha=0.65, zorder=2)
    if t_burnout is not None and t_plot[0] <= t_burnout <= t_plot[-1]:
        ax.axvline(t_burnout, color=_C["burnout"], lw=1.6, linestyle=":", alpha=0.9,
                   label=f"Burnout (t={t_burnout:.2f}s)", zorder=2)
    if t_apogee is not None:
        ax.axvline(t_apogee, color=_C["apogee"], lw=1.6, linestyle=":", alpha=0.9,
                   label=f"Apogee (t={t_apogee:.2f}s)", zorder=2)

    # ── axes decorations ─────────────────────────────────────────────────────
    ax.set_xlabel("Time [s]", fontsize=10)
    ax.set_ylabel("predicted_apogee − reference [m]", fontsize=10)
    ax.grid(True, alpha=0.25)

    # legend: deduplicate, keep cloud entry first
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, object] = {}
    for h, lbl in zip(handles, labels):
        if lbl not in seen:
            seen[lbl] = h
    ax.legend(seen.values(), seen.keys(), fontsize=8, loc="best", ncol=1,
              framealpha=0.85)

    n_combos = len(records)
    improvement = (1 - best["mae_post"] / raw_metrics["mae_post"]) * 100
    if improvement >= 0:
        verdict = f"KF reduces MAE_post by {improvement:.1f}%"
    else:
        verdict = f"[!] KF degrades MAE_post by {-improvement:.1f}% vs raw (large dt / small noise)"

    ax.set_title(
        f"Signed prediction error: raw vs KF configurations\n"
        f"{csv_name} · {n_combos} combos × {n_mc} MC runs | {verdict}",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tune Kalman filter via Monte Carlo grid search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", default="data/or_flight.csv",
                   help="Flight-profile CSV (time, static_pressure, total_pressure).")
    p.add_argument("--n-mc", type=int, default=20,
                   help="MC runs per grid point.")
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--sigma-a", type=float, nargs="+", default=[5, 15, 45, 120],
                   help="σ_a grid values [m/s²].")
    p.add_argument("--sigma-h", type=float, nargs="+", default=[5, 15, 45, 120],
                   help="σ_h grid values [m].")
    p.add_argument("--sigma-v", type=float, nargs="+", default=[2, 7, 20, 60],
                   help="σ_v grid values [m/s].")
    p.add_argument("--ref-sigma-a", type=float, default=30.0)
    p.add_argument("--ref-sigma-h", type=float, default=20.0)
    p.add_argument("--ref-sigma-v", type=float, default=30.0)
    p.add_argument("--output", default="results/kf_tune",
                   help="Output stem. Three PNGs: <stem>_heatmaps.png, _ranking.png, _timeseries.png")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    stem = Path(args.output)
    stem.parent.mkdir(parents=True, exist_ok=True)

    sigma_a_vals = sorted(args.sigma_a)
    sigma_h_vals = sorted(args.sigma_h)
    sigma_v_vals = sorted(args.sigma_v)
    n_mc         = args.n_mc
    base_seed    = args.base_seed
    ref_sa, ref_sh, ref_sv = args.ref_sigma_a, args.ref_sigma_h, args.ref_sigma_v

    # ── 1. Load profile ───────────────────────────────────────────────────────
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 1

    print(f"Loading flight profile from {csv_path} …")
    profile = FlightProfile.from_csv(str(csv_path))
    params  = ModelParams()
    print(f"  {profile.n_steps} steps, dt={profile.dt:.4f}s, duration={profile.time[-1]:.2f}s")

    # ── 2. Clean reference run ────────────────────────────────────────────────
    print("Running clean reference simulation …")
    clean = _clean_run(profile, params)
    t_burnout, t_apogee = _burnout_apogee_times(clean)
    print(f"  Reference apogee: {clean.max_altitude:.1f} m  "
          f"burnout: {t_burnout:.2f}s  apogee: {t_apogee:.2f}s")

    # ── 3. Raw-noise baseline (no KF) ─────────────────────────────────────────
    print(f"\nRaw-noise baseline ({n_mc} runs, no KF) …")
    raw_results = _run_mc_nofilter(profile, params, n_mc, base_seed)
    raw_metrics = _compute_metrics(raw_results, clean, t_burnout, t_apogee, use_filtered=False)
    print(f"  MAE_post={raw_metrics['mae_post']:.2f} m  "
          f"MAE_full={raw_metrics['mae_full']:.2f} m  "
          f"RMSE_post={raw_metrics['rmse_post']:.2f} m")

    # ── 4. Grid search ────────────────────────────────────────────────────────
    records, all_curves = grid_search(
        profile, params, clean,
        sigma_a_vals, sigma_h_vals, sigma_v_vals,
        n_mc, base_seed,
    )
    best = records[0]
    improvement_pct = (1 - best["mae_post"] / raw_metrics["mae_post"]) * 100
    print(f"\nBest configuration:")
    print(f"  σ_a={best['sigma_a']:.1f}  σ_h={best['sigma_h']:.1f}  σ_v={best['sigma_v']:.1f}")
    print(f"  MAE_post={best['mae_post']:.2f} m  (raw baseline: {raw_metrics['mae_post']:.2f} m)")
    if improvement_pct >= 0:
        print(f"  KF improvement: {improvement_pct:.1f}%")
    else:
        print(f"  KF is {-improvement_pct:.1f}% WORSE than raw — filter adds lag at this dt/noise level.")

    # ── 5. Best-KF final MC run ───────────────────────────────────────────────
    final_n = max(n_mc, 40)
    print(f"\nFinal MC run with best KF ({final_n} runs) …")
    best_results = _run_mc(
        profile, params,
        best["sigma_a"], best["sigma_h"], best["sigma_v"],
        final_n, base_seed,
    )

    # ── 6. Reference-KF MC run ────────────────────────────────────────────────
    ref_results: list[SimulationResult] | None = None
    if not (
        np.isclose(ref_sa, best["sigma_a"])
        and np.isclose(ref_sh, best["sigma_h"])
        and np.isclose(ref_sv, best["sigma_v"])
    ):
        print(f"Reference KF MC run (σ_a={ref_sa}, σ_h={ref_sh}, σ_v={ref_sv}, {final_n} runs) …")
        ref_results = _run_mc(profile, params, ref_sa, ref_sh, ref_sv, final_n, base_seed)
        ref_m = _compute_metrics(ref_results, clean, t_burnout, t_apogee, use_filtered=True)
        print(f"  MAE_post={ref_m['mae_post']:.2f} m")
    else:
        print("Reference KF == best KF; skipping separate run.")

    # ── 7. Figures ────────────────────────────────────────────────────────────
    print("\nBuilding figures …")
    csv_name = csv_path.name

    save_heatmaps(
        records, sigma_a_vals, sigma_h_vals, sigma_v_vals,
        best, raw_metrics,
        stem.parent / (stem.name + "_heatmaps.png"),
        args.dpi, csv_name, n_mc,
    )

    save_ranking(
        records, raw_metrics,
        stem.parent / (stem.name + "_ranking.png"),
        args.dpi, csv_name, n_mc, top_n=15,
        ref_sa=ref_sa, ref_sh=ref_sh, ref_sv=ref_sv,
    )

    save_timeseries(
        records, all_curves,
        clean, raw_results, best_results, ref_results,
        t_burnout, t_apogee,
        best, raw_metrics,
        ref_sa, ref_sh, ref_sv,
        stem.parent / (stem.name + "_timeseries.png"),
        args.dpi, csv_name, n_mc,
    )

    # ── 8. Summary table ──────────────────────────────────────────────────────
    print("\n── Top-10 configurations ──────────────────────────────────────")
    print(f"  {'σ_a':>6}  {'σ_h':>6}  {'σ_v':>6}  {'MAE_post':>10}  {'MAE_full':>10}  {'RMSE_post':>10}")
    print("  " + "─" * 62)
    for r in records[:10]:
        print(f"  {r['sigma_a']:>6.1f}  {r['sigma_h']:>6.1f}  {r['sigma_v']:>6.1f}  "
              f"{r['mae_post']:>10.3f}  {r['mae_full']:>10.3f}  {r['rmse_post']:>10.3f}")
    print(f"\n  Raw noisy baseline:  MAE_post={raw_metrics['mae_post']:.3f} m  "
          f"MAE_full={raw_metrics['mae_full']:.3f} m  RMSE_post={raw_metrics['rmse_post']:.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
