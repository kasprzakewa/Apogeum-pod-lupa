"""
FastAPI route handlers for the simulation API.
"""

from __future__ import annotations

import concurrent.futures
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from src.api.schemas import (
    ChannelDeviation,
    DeviationStats,
    HealthResponse,
    MonteCarloPredictionErrorRequest,
    MonteCarloPredictionErrorResponse,
    MonteCarloSeriesStats,
    PlotRequest,
    SimulationRequest,
    SimulationResponse,
)
from src.filters.kalman import KalmanFilter
from src.filters.pressure_filter import PressureEMAFilter
from src.models.physics import ModelParams
from src.noise.noise_model import NoiseModel, create_noise_model
from src.simulation.engine import SimulationEngine
from src.simulation.flight_profile import FlightProfile
from src.simulation.result import SimulationResult  # noqa: F401 – used in type hint below
from src.visualization.mc_prediction_error_plot import prediction_error_figure_png_bytes
from src.visualization.plots import plot_simulation

router = APIRouter()

_VERSION = "0.1.0"


def _build_profile(request: SimulationRequest) -> FlightProfile:
    if request.csv_path is not None:
        try:
            return FlightProfile.from_csv(request.csv_path)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"CSV file not found: {request.csv_path}",
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    sp = request.synthetic_profile
    return FlightProfile.synthetic(
        duration=sp.duration,
        dt=sp.dt,
        max_altitude=sp.max_altitude,
        max_speed=sp.max_speed,
        burnout_time=sp.burnout_time,
    )


def _build_params(request: SimulationRequest) -> ModelParams:
    mp = request.model_params
    return ModelParams(
        reference_pressure=mp.reference_pressure,
        reference_air_density=mp.reference_air_density,
        g=mp.g,
        cross_section=mp.cross_section,
        drag_coefficient=mp.drag_coefficient,
        mass=mp.mass,
    )


def _build_noise(request: SimulationRequest) -> NoiseModel:
    nc = request.noise_config
    try:
        return create_noise_model(nc.noise_type, **nc.params)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))


def _build_filter(request: SimulationRequest) -> KalmanFilter | None:
    fc = request.filter_config
    if not fc.enabled:
        return None
    return KalmanFilter(
        sigma_a=fc.sigma_a,
        sigma_h=fc.sigma_h,
        sigma_v=fc.sigma_v,
        init_p=fc.init_p,
    )


def _build_pressure_filter(request: SimulationRequest) -> PressureEMAFilter | None:
    pfc = request.pressure_filter_config
    if not pfc.enabled:
        return None
    return PressureEMAFilter(
        tau_static=pfc.tau_static,
        tau_dynamic=pfc.tau_dynamic,
    )


def _run(request: SimulationRequest) -> SimulationResult:
    profile = _build_profile(request)
    params = _build_params(request)
    noise = _build_noise(request)
    kf = _build_filter(request)
    pf = _build_pressure_filter(request)
    return SimulationEngine(
        profile,
        params=params,
        noise_model=noise,
        pressure_filter=pf,
        filter_model=kf,
    ).run()


def _build_mc_simulation_factory(sim_request: SimulationRequest):
    profile = _build_profile(sim_request)
    params = _build_params(sim_request)
    noise_type = sim_request.noise_config.noise_type
    noise_params = dict(sim_request.noise_config.params)
    filter_cfg = sim_request.filter_config
    pf_cfg = sim_request.pressure_filter_config

    def simulation_factory(seed: int) -> SimulationEngine:
        noise_model_params = dict(noise_params)
        if noise_type != "none":
            noise_model_params["seed"] = seed
        noise = create_noise_model(noise_type, **noise_model_params)

        kf: KalmanFilter | None = None
        if filter_cfg.enabled:
            kf = KalmanFilter(
                sigma_a=filter_cfg.sigma_a,
                sigma_h=filter_cfg.sigma_h,
                sigma_v=filter_cfg.sigma_v,
                init_p=filter_cfg.init_p,
            )

        pf: PressureEMAFilter | None = None
        if pf_cfg.enabled:
            pf = PressureEMAFilter(
                tau_static=pf_cfg.tau_static,
                tau_dynamic=pf_cfg.tau_dynamic,
            )

        return SimulationEngine(
            profile,
            params=params,
            noise_model=noise,
            pressure_filter=pf,
            filter_model=kf,
        )

    return simulation_factory


def _series_stats(data: np.ndarray) -> MonteCarloSeriesStats:
    return MonteCarloSeriesStats(
        mean=np.mean(data, axis=0).tolist(),
        std=np.std(data, axis=0).tolist(),
        p05=np.percentile(data, 5, axis=0).tolist(),
        p95=np.percentile(data, 95, axis=0).tolist(),
    )


def _clean_reference_run(sim_request: SimulationRequest) -> SimulationResult:
    """Single noise-free simulation used for MC reference values."""
    clean = sim_request.model_copy(
        update={
            "noise_config": sim_request.noise_config.model_copy(
                update={"noise_type": "none", "params": {}}
            ),
        }
    )
    return _run(clean)


def _reference_apogee_burnout_and_time(
    clean_result: SimulationResult,
) -> tuple[float, float | None, float | None]:
    """
    Extract reference values from one clean simulation.

    - Reference apogee: max altitude on clean trajectory.
    - Burnout time proxy: timestamp of max speed on clean trajectory.
    """
    h_ref = float(np.max(clean_result.altitude))
    if clean_result.n_steps == 0:
        return h_ref, None, None
    burnout_idx = int(np.argmax(clean_result.speed))
    apogee_idx = int(np.argmax(clean_result.altitude))
    return h_ref, float(clean_result.time[burnout_idx]), float(clean_result.time[apogee_idx])


def _subsample_scatter_flat(
    time_1d: np.ndarray,
    errors_runs_steps: np.ndarray,
    max_points: int,
) -> tuple[list[float], list[float]]:
    """Flatten (n_runs, n_steps) errors with aligned time; subsample to max_points."""
    n_runs, n_steps = errors_runs_steps.shape
    if n_steps != len(time_1d):
        raise ValueError("Time length must match error matrix width.")
    total = n_runs * n_steps
    t_flat = np.tile(time_1d, n_runs)
    e_flat = errors_runs_steps.ravel(order="C")
    if total <= max_points:
        return t_flat.tolist(), e_flat.tolist()
    idx = np.linspace(0, total - 1, num=max_points, dtype=np.int64)
    return t_flat[idx].tolist(), e_flat[idx].tolist()


@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health_check() -> HealthResponse:
    """Return API health status and version."""
    return HealthResponse(status="ok", version=_VERSION)


@router.post("/simulate", response_model=SimulationResponse, tags=["simulation"])
def run_simulation(request: SimulationRequest) -> SimulationResponse:
    """
    Run a full software-in-the-loop simulation.

    - If **csv_path** is provided, the flight profile is loaded from that CSV file.
    - Otherwise, a synthetic profile is generated using **synthetic_profile** parameters.
    - Noise is applied according to **noise_config**.
    - Physical model uses **model_params**.

    Returns the complete time-series of all computed channels plus summary metadata.
    """
    result = _run(request)
    data = result.to_dict()
    meta = data["metadata"]

    deviations = None
    if result.noise_enabled:
        clean_req = request.model_copy(
            update={"noise_config": request.noise_config.model_copy(
                update={"noise_type": "none", "params": {}}
            )}
        )
        clean_result = _run(clean_req)
        dev = clean_result.deviation_from(result)
        deviations = DeviationStats(
            **{ch: ChannelDeviation(**vals) for ch, vals in dev.items()}
        )

    return SimulationResponse(
        time=data["time"],
        static_pressure=data["static_pressure"],
        total_pressure=data["total_pressure"],
        dynamic_pressure=data["dynamic_pressure"],
        altitude=data["altitude"],
        speed=data["speed"],
        predicted_apogee=data["predicted_apogee"],
        altitude_filtered=data.get("altitude_filtered"),
        speed_filtered=data.get("speed_filtered"),
        predicted_apogee_filtered=data.get("predicted_apogee_filtered"),
        altitude_pf=data.get("altitude_pf"),
        speed_pf=data.get("speed_pf"),
        predicted_apogee_pf=data.get("predicted_apogee_pf"),
        metadata=meta,
        deviations=deviations,
    )


@router.post(
    "/plot",
    response_class=Response,
    tags=["visualization"],
    responses={200: {"content": {"image/png": {}}, "description": "PNG plot image"}},
)
def plot_png(request: PlotRequest) -> Response:
    """
    Run the simulation and return a **PNG image**.

    Returns `image/png`.
    """
    sim_req = request.simulation
    result = _run(sim_req)

    clean_result: SimulationResult | None = None
    if request.overlay_noise and sim_req.noise_config.noise_type != "none":
        clean_req = sim_req.model_copy(
            update={"noise_config": sim_req.noise_config.model_copy(update={"noise_type": "none"})}
        )
        clean_result = _run(clean_req)

    fig = plot_simulation(
        result=result if clean_result is None else clean_result,
        noisy_result=result if clean_result is not None else None,
        title=request.title,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=request.dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return Response(content=buf.read(), media_type="image/png")


def _mae_rmse(errors: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Return (MAE, RMSE) for a (n_runs, n_steps) error matrix over a boolean mask."""
    e = errors[:, mask]
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e**2)))


def _compute_monte_carlo_prediction_error(
    request: MonteCarloPredictionErrorRequest,
) -> tuple[MonteCarloPredictionErrorResponse, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Run Monte Carlo; return JSON payload, raw signed-error matrix (n_runs, n_steps),
    optionally KF-filtered signed-error matrix, and optionally pressure-filtered signed-error matrix.
    """
    clean_ref = _clean_reference_run(request.simulation)
    h_ref, t_burnout, t_apogee = _reference_apogee_burnout_and_time(clean_ref)

    simulation_factory = _build_mc_simulation_factory(request.simulation)
    seeds = range(request.base_seed, request.base_seed + request.n_runs)

    def run_single(seed: int) -> SimulationResult:
        return simulation_factory(seed).run()

    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(run_single, seeds))
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if not results:
        raise HTTPException(status_code=422, detail="No Monte Carlo runs were executed.")

    time_1d = results[0].time
    clean_pred_on_mc_time = np.interp(
        time_1d,
        clean_ref.time,
        clean_ref.predicted_apogee,
    )

    # --- Raw signed errors ---
    pred_stack = np.stack([r.predicted_apogee for r in results], axis=0)
    signed_errors = pred_stack - clean_pred_on_mc_time[None, :]

    # --- Ascent / post-burnout masks ---
    full_mask = time_1d <= t_apogee if t_apogee is not None else np.ones_like(time_1d, dtype=bool)
    if not np.any(full_mask):
        full_mask = np.ones_like(time_1d, dtype=bool)

    post_mask: np.ndarray | None = None
    if t_burnout is not None:
        pm = time_1d >= t_burnout
        if t_apogee is not None:
            pm = pm & (time_1d <= t_apogee)
        if np.any(pm):
            post_mask = pm

    signed_stats = _series_stats(signed_errors)
    mae_full, rmse_full = _mae_rmse(signed_errors, full_mask)
    mae_post: float | None = None
    rmse_post: float | None = None
    if post_mask is not None:
        mae_post, rmse_post = _mae_rmse(signed_errors, post_mask)

    scatter_t: list[float] | None = None
    scatter_e: list[float] | None = None
    if request.include_all_runs:
        scatter_t, scatter_e = _subsample_scatter_flat(
            time_1d, signed_errors, request.scatter_max_points
        )

    # --- Kalman-filtered signed errors (only when KF was active) ---
    filter_active = request.simulation.filter_config.enabled and all(
        r.predicted_apogee_filtered is not None for r in results
    )
    signed_errors_filtered: np.ndarray | None = None
    signed_stats_f = mae_full_f = rmse_full_f = mae_post_f = rmse_post_f = None
    scatter_e_f: list[float] | None = None

    if filter_active:
        pred_stack_f = np.stack([r.predicted_apogee_filtered for r in results], axis=0)
        signed_errors_filtered = pred_stack_f - clean_pred_on_mc_time[None, :]
        signed_stats_f = _series_stats(signed_errors_filtered)
        mae_full_f, rmse_full_f = _mae_rmse(signed_errors_filtered, full_mask)
        if post_mask is not None:
            mae_post_f, rmse_post_f = _mae_rmse(signed_errors_filtered, post_mask)
        if request.include_all_runs:
            _, scatter_e_f = _subsample_scatter_flat(
                time_1d, signed_errors_filtered, request.scatter_max_points
            )

    # --- Pressure-filtered signed errors (only when pressure filter was active) ---
    pf_active = request.simulation.pressure_filter_config.enabled and all(
        r.predicted_apogee_pf is not None for r in results
    )
    signed_errors_pf: np.ndarray | None = None
    signed_stats_pf = mae_full_pf = rmse_full_pf = mae_post_pf = rmse_post_pf = None
    scatter_e_pf: list[float] | None = None

    if pf_active:
        pred_stack_pf = np.stack([r.predicted_apogee_pf for r in results], axis=0)
        signed_errors_pf = pred_stack_pf - clean_pred_on_mc_time[None, :]
        signed_stats_pf = _series_stats(signed_errors_pf)
        mae_full_pf, rmse_full_pf = _mae_rmse(signed_errors_pf, full_mask)
        if post_mask is not None:
            mae_post_pf, rmse_post_pf = _mae_rmse(signed_errors_pf, post_mask)
        if request.include_all_runs:
            _, scatter_e_pf = _subsample_scatter_flat(
                time_1d, signed_errors_pf, request.scatter_max_points
            )

    response = MonteCarloPredictionErrorResponse(
        reference_apogee_m=h_ref,
        burnout_time_used_s=t_burnout,
        apogee_time_used_s=t_apogee,
        n_runs=request.n_runs,
        base_seed=request.base_seed,
        time=time_1d.tolist(),
        signed_error=signed_stats,
        mean_abs_error_full_flight_m=mae_full,
        rmse_full_flight_m=rmse_full,
        mean_abs_error_post_burnout_m=mae_post,
        rmse_post_burnout_m=rmse_post,
        scatter_time=scatter_t,
        scatter_signed_error=scatter_e,
        signed_error_filtered=signed_stats_f,
        mean_abs_error_full_flight_filtered_m=mae_full_f,
        rmse_full_flight_filtered_m=rmse_full_f,
        mean_abs_error_post_burnout_filtered_m=mae_post_f,
        rmse_post_burnout_filtered_m=rmse_post_f,
        scatter_signed_error_filtered=scatter_e_f,
        signed_error_pf=signed_stats_pf,
        mean_abs_error_full_flight_pf_m=mae_full_pf,
        rmse_full_flight_pf_m=rmse_full_pf,
        mean_abs_error_post_burnout_pf_m=mae_post_pf,
        rmse_post_burnout_pf_m=rmse_post_pf,
        scatter_signed_error_pf=scatter_e_pf,
    )
    return response, signed_errors, signed_errors_filtered, signed_errors_pf


@router.post(
    "/monte-carlo/prediction-error",
    tags=["simulation"],
    response_model=None,
    responses={
        200: {
            "content": {
                "application/json": {},
                "image/png": {},
            },
            "description": "JSON statistics or PNG plot (see response_format).",
        }
    },
)
def run_monte_carlo_prediction_error(
    request: MonteCarloPredictionErrorRequest,
) -> MonteCarloPredictionErrorResponse | Response:
    """
    Monte Carlo: at each timestep, error = predicted_apogee(noisy) − predicted_apogee(clean),
    where clean is a noise-free simulation on the same profile and model parameters.

    Set **response_format** to ``png`` to return a figure instead of JSON.
    """
    response, signed_errors, signed_errors_filtered, signed_errors_pf = (
        _compute_monte_carlo_prediction_error(request)
    )

    if request.response_format == "png":
        t_arr = np.asarray(response.time, dtype=np.float64)
        plot_t, plot_e = _subsample_scatter_flat(
            t_arr, signed_errors, request.scatter_max_points
        )
        plot_e_f: list[float] | None = None
        if signed_errors_filtered is not None:
            _, plot_e_f = _subsample_scatter_flat(
                t_arr, signed_errors_filtered, request.scatter_max_points
            )
        plot_e_pf: list[float] | None = None
        if signed_errors_pf is not None:
            _, plot_e_pf = _subsample_scatter_flat(
                t_arr, signed_errors_pf, request.scatter_max_points
            )
        png = prediction_error_figure_png_bytes(
            response,
            plot_t,
            plot_e,
            plot_e_f,
            plot_e_pf,
            request.figure_title,
            request.figure_dpi,
        )
        return Response(content=png, media_type="image/png")

    return response


