"""
FastAPI route handlers for the simulation API.
"""

from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from src.api.schemas import (
    HealthResponse,
    PlotRequest,
    SimulationRequest,
    SimulationResponse,
)
from src.models.physics import ModelParams
from src.noise.noise_model import NoNoiseModel
from src.simulation.engine import SimulationEngine
from src.simulation.flight_profile import FlightProfile
from src.simulation.result import SimulationResult  # noqa: F401 – used in type hint below
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


def _build_noise(_request: SimulationRequest) -> NoNoiseModel:
    return NoNoiseModel()


def _run(request: SimulationRequest) -> SimulationResult:
    profile = _build_profile(request)
    params = _build_params(request)
    noise = _build_noise(request)
    return SimulationEngine(profile, params=params, noise_model=noise).run()


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

    return SimulationResponse(
        time=data["time"],
        static_pressure=data["static_pressure"],
        total_pressure=data["total_pressure"],
        dynamic_pressure=data["dynamic_pressure"],
        altitude=data["altitude"],
        speed=data["speed"],
        predicted_apogee=data["predicted_apogee"],
        metadata=meta,
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
        # Run a second, noise-free simulation with the same profile/params
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


