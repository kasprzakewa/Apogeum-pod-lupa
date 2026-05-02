#!/usr/bin/env python3
"""
EMA Pressure-Filter Tuning — Monte Carlo Grid Search.

Searches over (tau_static, tau_dynamic) on a 2-D grid.
Each point runs N_MC independent BinczarNoiseModel + PressureEMAFilter simulations
and evaluates:

  - MAE_post  : mean absolute prediction error from burnout to apogee  [m]
  - MAE_full  : mean absolute prediction error from launch to apogee   [m]
  - RMSE_post : RMSE from burnout to apogee                           [m]

Output — three separate PNG files derived from --output stem:
  <stem>_heatmap.png    – MAE_post heatmap (tau_static × tau_dynamic)
  <stem>_ranking.png    – top-15 configurations bar chart
  <stem>_timeseries.png – signed prediction-error time series with
                          cloud of ALL grid configs + highlighted best / ref EMA

Usage:
  poetry run python3 scripts/tune_ema.py
  poetry run python3 scripts/tune_ema.py \\
      --csv data/or_flight.csv \\
      --n-mc 20 --base-seed 42 \\
      --tau-static  0.01 0.05 0.15 0.5 \\
      --tau-dynamic 0.005 0.02 0.08 0.3 \\
      --ref-tau-static 0.05 --ref-tau-dynamic 0.02 \\
      --output results/ema_tune
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

from src.filters.pressure_filter import PressureEMAFilter
from src.models.physics import ModelParams
from src.noise.noise_model import BinczarNoiseModel, NoNoiseModel
from src.simulation.engine import SimulationEngine
from src.simulation.flight_profile import FlightProfile
from src.simulation.result import SimulationResult

# ── colour palette ─────────────────────────────────────────────────────────────
_C = {
    "raw":          "#c51b8a",   # magenta  – raw noisy
    "raw_band":     "#f7cce9",   # pale pink band
    "best":         "#2ca02c",   # green    – best EMA
    "best_band":    "#b5e8b5",   # light green band
    "ref":          "#e65c00",   # orange   – user's reference EMA
    "ref_band":     "#ffd5aa",   # pale orange band
    "cloud":        "#1f77b4",   # blue     – all-other-configs cloud
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


def _run_mc_ema(
    profile: FlightProfile,
    params: ModelParams,
    tau_static: float,
    tau_dynamic: float,
    n_mc: int,
    base_seed: int,
) -> list[SimulationResult]:
    def _one(seed: int) -> SimulationResult:
        noise = BinczarNoiseModel(seed=seed)
        pf    = PressureEMAFilter(tau_static=tau_static, tau_dynamic=tau_dynamic)
        return SimulationEngine(
            profile, params=params, noise_model=noise, pressure_filter=pf
        ).run()

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
    use_pf: bool = True,
) -> dict[str, float]:
    time_1d = results[0].time
    clean_pred = np.interp(time_1d, clean.time, clean.predicted_apogee)
    pred_stack = np.stack(
        [r.predicted_apogee_pf if use_pf else r.predicted_apogee for r in results],
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
    use_pf: bool = True,
) -> np.ndarray:
    time_1d = results[0].time
    clean_pred = np.interp(time_1d, clean.time, clean.predicted_apogee)
    pred_stack = np.stack(
        [r.predicted_apogee_pf if use_pf else r.predicted_apogee for r in results],
        axis=0,
    )
    return np.mean(pred_stack - clean_pred[None, :], axis=0)


def _signed_error_stats(
    results: list[SimulationResult],
    clean: SimulationResult,
    use_pf: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time_1d = results[0].time
    clean_pred = np.interp(time_1d, clean.time, clean.predicted_apogee)
    pred_stack = np.stack(
        [r.predicted_apogee_pf if use_pf else r.predicted_apogee for r in results],
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
    tau_static_vals: list[float],
    tau_dynamic_vals: list[float],
    n_mc: int,
    base_seed: int,
) -> tuple[list[dict], list[np.ndarray]]:
    """
    Evaluate every (tau_static, tau_dynamic) combination.

    Returns:
        records          – list of metric dicts, sorted by mae_post ascending
        mean_error_curves – mean signed-error array (n_steps,) per combo,
                           in the same order as records (after sorting)
    """
    t_burnout, t_apogee = _burnout_apogee_times(clean)
    combos = list(itertools.product(tau_static_vals, tau_dynamic_vals))
    total  = len(combos)
    records: list[dict] = []
    curves:  list[np.ndarray] = []

    print(f"\nGrid search: {total} combinations × {n_mc} MC runs each")
    t0 = time.perf_counter()

    for i, (ts, td) in enumerate(combos, 1):
        results = _run_mc_ema(profile, params, ts, td, n_mc, base_seed)
        m       = _compute_metrics(results, clean, t_burnout, t_apogee, use_pf=True)
        curve   = _mean_signed_error(results, clean, use_pf=True)
        records.append({"tau_static": ts, "tau_dynamic": td, **m})
        curves.append(curve)

        elapsed = time.perf_counter() - t0
        eta     = elapsed / i * (total - i)
        bar     = "█" * int(30 * i / total) + "░" * (30 - int(30 * i / total))
        print(
            f"\r  [{bar}] {i}/{total}  "
            f"τ_s={ts:.4f} τ_d={td:.4f}  "
            f"MAE_post={m['mae_post']:7.2f} m  ETA {eta:5.0f}s",
            end="", flush=True,
        )

    print(f"\n  Done in {time.perf_counter() - t0:.1f}s")

    order  = sorted(range(len(records)), key=lambda i: records[i]["mae_post"])
    records = [records[i] for i in order]
    curves  = [curves[i]  for i in order]
    return records, curves


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — heatmap
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
    ax.set_xticklabels([f"{v:.4g}" for v in x_vals], fontsize=8)
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels([f"{v:.4g}" for v in y_vals], fontsize=8)
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
                    xi, yi, f"{val:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if val > threshold else "black",
                )


def save_heatmap(
    records: list[dict],
    tau_static_vals: list[float],
    tau_dynamic_vals: list[float],
    best: dict,
    raw_metrics: dict,
    out_path: Path,
    dpi: int,
    csv_name: str,
    n_mc: int,
) -> None:
    import pandas as pd
    df = pd.DataFrame(records)

    z = np.full((len(tau_dynamic_vals), len(tau_static_vals)), np.nan)
    for ri, td in enumerate(tau_dynamic_vals):
        for ci, ts in enumerate(tau_static_vals):
            cell = df[np.isclose(df["tau_static"], ts) & np.isclose(df["tau_dynamic"], td)]
            if not cell.empty:
                z[ri, ci] = cell["mae_post"].values[0]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.patch.set_facecolor("#f8f8f8")

    _heatmap(
        ax, tau_static_vals, tau_dynamic_vals, z,
        "τ_static [s]", "τ_dynamic [s]",
        "MAE_post (burnout→apogee) [m]",
    )

    n_combos = len(tau_static_vals) * len(tau_dynamic_vals)
    fig.suptitle(
        f"EMA Tuning — MAE_post heatmap | {csv_name} · {n_combos} combos × {n_mc} MC runs\n"
        f"Optimal: τ_s={best['tau_static']:.4g}  τ_d={best['tau_dynamic']:.4g}"
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
    ref_ts: float | None = None,
    ref_td: float | None = None,
) -> None:
    from matplotlib.patches import Patch

    top = records[:top_n]
    labels   = [f"τ_s={r['tau_static']:.4g}  τ_d={r['tau_dynamic']:.4g}" for r in top]
    mae_vals = [r["mae_post"] for r in top]

    colors = []
    for r in top:
        is_ref = (
            ref_ts is not None
            and np.isclose(r["tau_static"], ref_ts)
            and np.isclose(r["tau_dynamic"], ref_td)
        )
        colors.append(_C["ref"] if is_ref else _C["best"])
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

    ax.axvline(raw_metrics["mae_post"], color=_C["raw"], lw=1.8, linestyle="--",
               alpha=0.85, label=f"Raw-noise baseline ({raw_metrics['mae_post']:.2f} m)")

    legend_elements = [
        Patch(facecolor="gold",      label="Best found"),
        Patch(facecolor=_C["best"],  label="Other top-N"),
    ]
    if ref_ts is not None:
        legend_elements.append(Patch(facecolor=_C["ref"], label="User's reference EMA"))
    legend_elements.append(
        plt.Line2D([0], [0], color=_C["raw"], lw=1.8, linestyle="--",
                   label=f"Raw-noise baseline ({raw_metrics['mae_post']:.2f} m)")
    )
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    n_combos = len(records)
    ax.set_title(
        f"EMA Tuning — Top-{top_n} configurations (MAE burnout→apogee)\n"
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
    all_curves: list[np.ndarray],
    clean: SimulationResult,
    raw_results: list[SimulationResult],
    best_results: list[SimulationResult],
    ref_results: list[SimulationResult] | None,
    t_burnout: float | None,
    t_apogee: float | None,
    best: dict,
    raw_metrics: dict,
    ref_ts: float | None,
    ref_td: float | None,
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

    best_idx = 0
    ref_idx  = next(
        (
            i for i, r in enumerate(records)
            if ref_ts is not None
            and np.isclose(r["tau_static"], ref_ts)
            and np.isclose(r["tau_dynamic"], ref_td)
        ),
        None,
    )

    first_cloud = True
    for i, (rec, curve) in enumerate(zip(records, all_curves)):
        if i in (best_idx, ref_idx):
            continue
        label = "_cloud" if not first_cloud else "All other EMA configs (mean)"
        ax.plot(
            t_plot, curve[mask],
            color=_C["cloud"], lw=0.6, alpha=0.18,
            label=label, zorder=1,
        )
        first_cloud = False

    def _draw(results, label_mean, label_band, color, band_color, use_pf: bool, ls="-", zorder=3):
        _, mean, p05, p95 = _signed_error_stats(results, clean, use_pf=use_pf)
        ax.fill_between(t_plot, p05[mask], p95[mask],
                        color=band_color, alpha=0.35, label=label_band, zorder=zorder)
        ax.plot(t_plot, mean[mask],
                color=color, lw=2.2, linestyle=ls, label=label_mean, zorder=zorder + 1)

    _draw(raw_results,
          f"raw noisy — mean error (MAE_post={raw_metrics['mae_post']:.2f} m)",
          "raw noisy — P05–P95",
          _C["raw"], _C["raw_band"], use_pf=False, zorder=4)

    best_m = _compute_metrics(best_results, clean, t_burnout, t_apogee, use_pf=True)
    _draw(best_results,
          (f"best EMA τ_s={best['tau_static']:.4g} τ_d={best['tau_dynamic']:.4g} "
           f"— mean error (MAE_post={best_m['mae_post']:.2f} m)"),
          "best EMA — P05–P95",
          _C["best"], _C["best_band"], use_pf=True, zorder=6)

    if ref_results is not None:
        ref_m = _compute_metrics(ref_results, clean, t_burnout, t_apogee, use_pf=True)
        _draw(ref_results,
              (f"ref EMA τ_s={ref_ts:.4g} τ_d={ref_td:.4g} "
               f"— mean error (MAE_post={ref_m['mae_post']:.2f} m)"),
              "ref EMA — P05–P95",
              _C["ref"], _C["ref_band"], use_pf=True, ls="--", zorder=5)

    ax.axhline(0.0, color="#4d4d4d", lw=0.8, linestyle="--", alpha=0.65, zorder=2)
    if t_burnout is not None and t_plot[0] <= t_burnout <= t_plot[-1]:
        ax.axvline(t_burnout, color=_C["burnout"], lw=1.6, linestyle=":", alpha=0.9,
                   label=f"Burnout (t={t_burnout:.2f}s)", zorder=2)
    if t_apogee is not None:
        ax.axvline(t_apogee, color=_C["apogee"], lw=1.6, linestyle=":", alpha=0.9,
                   label=f"Apogee (t={t_apogee:.2f}s)", zorder=2)

    ax.set_xlabel("Time [s]", fontsize=10)
    ax.set_ylabel("predicted_apogee − reference [m]", fontsize=10)
    ax.grid(True, alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, object] = {}
    for h, lbl in zip(handles, labels):
        if lbl not in seen:
            seen[lbl] = h
    ax.legend(seen.values(), seen.keys(), fontsize=8, loc="best", ncol=1, framealpha=0.85)

    n_combos = len(records)
    improvement = (1 - best["mae_post"] / raw_metrics["mae_post"]) * 100
    if improvement >= 0:
        verdict = f"EMA reduces MAE_post by {improvement:.1f}%"
    else:
        verdict = f"[!] EMA degrades MAE_post by {-improvement:.1f}% vs raw"

    ax.set_title(
        f"Signed prediction error: raw vs EMA configurations\n"
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
        description="Tune EMA pressure filter via Monte Carlo grid search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", default="data/or_flight.csv",
                   help="Flight-profile CSV (time, static_pressure, total_pressure).")
    p.add_argument("--n-mc", type=int, default=20,
                   help="MC runs per grid point.")
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--tau-static", type=float, nargs="+", default=[0.01, 0.05, 0.15, 0.5],
                   help="tau_static grid values [s].")
    p.add_argument("--tau-dynamic", type=float, nargs="+", default=[0.005, 0.02, 0.08, 0.3],
                   help="tau_dynamic grid values [s].")
    p.add_argument("--ref-tau-static",  type=float, default=0.05)
    p.add_argument("--ref-tau-dynamic", type=float, default=0.02)
    p.add_argument("--output", default="results/ema_tune",
                   help="Output stem. Three PNGs: <stem>_heatmap.png, _ranking.png, _timeseries.png")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    stem = Path(args.output)
    stem.parent.mkdir(parents=True, exist_ok=True)

    tau_static_vals  = sorted(args.tau_static)
    tau_dynamic_vals = sorted(args.tau_dynamic)
    n_mc      = args.n_mc
    base_seed = args.base_seed
    ref_ts    = args.ref_tau_static
    ref_td    = args.ref_tau_dynamic

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

    # ── 3. Raw-noise baseline (no filter) ─────────────────────────────────────
    print(f"\nRaw-noise baseline ({n_mc} runs, no EMA) …")
    raw_results = _run_mc_nofilter(profile, params, n_mc, base_seed)
    raw_metrics = _compute_metrics(raw_results, clean, t_burnout, t_apogee, use_pf=False)
    print(f"  MAE_post={raw_metrics['mae_post']:.2f} m  "
          f"MAE_full={raw_metrics['mae_full']:.2f} m  "
          f"RMSE_post={raw_metrics['rmse_post']:.2f} m")

    # ── 4. Grid search ────────────────────────────────────────────────────────
    records, all_curves = grid_search(
        profile, params, clean,
        tau_static_vals, tau_dynamic_vals,
        n_mc, base_seed,
    )
    best = records[0]
    improvement_pct = (1 - best["mae_post"] / raw_metrics["mae_post"]) * 100
    print(f"\nBest configuration:")
    print(f"  τ_static={best['tau_static']:.4g}  τ_dynamic={best['tau_dynamic']:.4g}")
    print(f"  MAE_post={best['mae_post']:.2f} m  (raw baseline: {raw_metrics['mae_post']:.2f} m)")
    if improvement_pct >= 0:
        print(f"  EMA improvement: {improvement_pct:.1f}%")
    else:
        print(f"  EMA is {-improvement_pct:.1f}% WORSE than raw.")

    # ── 5. Best-EMA final MC run ──────────────────────────────────────────────
    final_n = max(n_mc, 40)
    print(f"\nFinal MC run with best EMA ({final_n} runs) …")
    best_results = _run_mc_ema(
        profile, params,
        best["tau_static"], best["tau_dynamic"],
        final_n, base_seed,
    )

    # ── 6. Reference-EMA MC run ───────────────────────────────────────────────
    ref_results: list[SimulationResult] | None = None
    if not (np.isclose(ref_ts, best["tau_static"]) and np.isclose(ref_td, best["tau_dynamic"])):
        print(f"Reference EMA MC run (τ_s={ref_ts}, τ_d={ref_td}, {final_n} runs) …")
        ref_results = _run_mc_ema(profile, params, ref_ts, ref_td, final_n, base_seed)
        ref_m = _compute_metrics(ref_results, clean, t_burnout, t_apogee, use_pf=True)
        print(f"  MAE_post={ref_m['mae_post']:.2f} m")
    else:
        print("Reference EMA == best EMA; skipping separate run.")

    # ── 7. Figures ────────────────────────────────────────────────────────────
    print("\nBuilding figures …")
    csv_name = csv_path.name

    save_heatmap(
        records, tau_static_vals, tau_dynamic_vals,
        best, raw_metrics,
        stem.parent / (stem.name + "_heatmap.png"),
        args.dpi, csv_name, n_mc,
    )

    save_ranking(
        records, raw_metrics,
        stem.parent / (stem.name + "_ranking.png"),
        args.dpi, csv_name, n_mc, top_n=15,
        ref_ts=ref_ts, ref_td=ref_td,
    )

    save_timeseries(
        records, all_curves,
        clean, raw_results, best_results, ref_results,
        t_burnout, t_apogee,
        best, raw_metrics,
        ref_ts, ref_td,
        stem.parent / (stem.name + "_timeseries.png"),
        args.dpi, csv_name, n_mc,
    )

    # ── 8. Summary table ──────────────────────────────────────────────────────
    print("\n── Top-10 configurations ──────────────────────────────────────")
    print(f"  {'τ_static':>10}  {'τ_dynamic':>10}  {'MAE_post':>10}  {'MAE_full':>10}  {'RMSE_post':>10}")
    print("  " + "─" * 58)
    for r in records[:10]:
        print(f"  {r['tau_static']:>10.4g}  {r['tau_dynamic']:>10.4g}  "
              f"{r['mae_post']:>10.3f}  {r['mae_full']:>10.3f}  {r['rmse_post']:>10.3f}")
    print(f"\n  Raw noisy baseline:  MAE_post={raw_metrics['mae_post']:.3f} m  "
          f"MAE_full={raw_metrics['mae_full']:.3f} m  RMSE_post={raw_metrics['rmse_post']:.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
