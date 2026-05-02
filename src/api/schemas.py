"""
Pydantic schemas for the FastAPI simulation API.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelParamsSchema(BaseModel):
    """Configurable physical and aerodynamic model parameters."""

    reference_pressure: float = Field(
        101325.0, description="Reference sea-level pressure [Pa]"
    )
    reference_air_density: float = Field(
        1.225, description="Reference sea-level air density [kg/m³]"
    )
    g: float = Field(9.81, description="Gravitational acceleration [m/s²]")
    cross_section: float = Field(
        0.0181458, description="Rocket cross-sectional area A [m²]"
    )
    drag_coefficient: float = Field(
        0.6, description="Aerodynamic drag coefficient C_D [-]"
    )
    mass: float = Field(50.0, description="Rocket mass [kg]")


class NoiseConfigSchema(BaseModel):
    """
    Noise model configuration.

    `noise_type` must match a key in NOISE_REGISTRY (default: "none").
    `params` is forwarded as keyword arguments to the model constructor,
    so adding a new noise model to the registry requires no schema changes.
    """

    noise_type: str = Field(
        "none",
        description=(
            "Name of the noise model to use. Must be registered in NOISE_REGISTRY. "
            "Examples: 'none', 'binczar', 'gaussian', 'ideal_gaussian'."
        ),
    )
    params: dict = Field(
        default_factory=dict,
        description=(
            "Constructor keyword arguments forwarded to the noise model. "
            "Use `{}` when the model needs no extra args (e.g. `binczar` with defaults). "
        ),
        examples=[{}],
    )


class KalmanConfigSchema(BaseModel):
    """
    Linear Kalman Filter configuration.

    The filter operates on the (altitude, velocity) state derived from noisy
    pressure readings and returns smoothed estimates of both channels plus a
    re-computed apogee prediction based on the filtered state.

    Set ``enabled = False`` (the default) to skip filtering entirely.
    """

    enabled: bool = Field(
        False,
        description="Enable the Kalman filter on sensor-derived altitude and velocity.",
    )
    sigma_a: float = Field(
        30.0,
        gt=0.0,
        description=(
            "Process acceleration noise std [m/s²]. "
            "Higher → more responsive to thrust/drag transients; "
            "lower → smoother but slower to track rapid changes."
        ),
    )
    sigma_h: float = Field(
        20.0,
        gt=0.0,
        description="Measurement noise std for altitude [m]. Match to static-pressure sensor noise.",
    )
    sigma_v: float = Field(
        5.0,
        gt=0.0,
        description="Measurement noise std for velocity [m/s]. Match to pitot sensor noise.",
    )
    init_p: float = Field(
        500.0,
        gt=0.0,
        description=(
            "Initial state covariance diagonal value [m² / (m/s)²]. "
            "Large value → first measurement is trusted completely."
        ),
    )


class PressureFilterConfigSchema(BaseModel):
    """
    Pressure-domain EMA filter configuration.

    The filter smooths raw sensor pressures (static and dynamic) **before**
    the barometric and Bernoulli conversions, so any lag stays expressed in
    Pascals rather than being amplified through the v² term in the apogee
    formula.

    Set ``enabled = True`` to activate.  Results appear in the ``*_pf``
    response channels (``altitude_pf``, ``speed_pf``, ``predicted_apogee_pf``).

    The two channels use independent time constants:
      - ``tau_static``  — longer, because static pressure changes slowly and
        strong smoothing is safe (altitude errors stay small).
      - ``tau_dynamic`` — shorter, because the dynamic-pressure peak at burnout
        is brief and a large lag would underestimate velocity.
    """

    enabled: bool = Field(
        False,
        description="Enable the pressure-domain EMA filter.",
    )
    tau_static: float = Field(
        0.05,
        gt=0.0,
        description="EMA time constant for the static-pressure channel [s].",
    )
    tau_dynamic: float = Field(
        0.02,
        gt=0.0,
        description="EMA time constant for the dynamic-pressure channel [s].",
    )


class SyntheticProfileSchema(BaseModel):
    """Parameters for synthetic flight profile generation."""

    duration: float = Field(60.0, description="Total flight duration [s]")
    dt: float = Field(0.01, description="Time step [s]")
    max_altitude: float = Field(3000.0, description="Target apogee altitude [m]")
    max_speed: float = Field(300.0, description="Peak airspeed at burnout [m/s]")
    burnout_time: float = Field(5.0, description="Motor burnout time [s]")


class SimulationRequest(BaseModel):
    """
    Request body for POST /simulate.

    Either provide a CSV file path (csv_path) or use synthetic profile generation.
    If both are omitted, a default synthetic profile is used.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "csv_path": "data/or_flight.csv",
                    "model_params": {
                        "reference_pressure": 101325.0,
                        "reference_air_density": 1.225,
                        "g": 9.81,
                        "cross_section": 0.00816714,
                        "drag_coefficient": 0.6625,
                        "mass": 19.453,
                    },
                    "noise_config": {
                        "noise_type": "none",
                        "params": {},
                    },
                }
            ]
        }
    )

    csv_path: str | None = Field(
        None,
        description="Path to CSV flight profile on the server filesystem. "
        "CSV must have columns: time, static_pressure, total_pressure.",
        examples=["data/or_flight.csv"],
    )
    synthetic_profile: SyntheticProfileSchema = Field(
        default_factory=SyntheticProfileSchema,
        description="Parameters for synthetic profile generation (used when csv_path is None).",
    )
    model_params: ModelParamsSchema = Field(
        default_factory=ModelParamsSchema,
        description="Physical and aerodynamic model parameters.",
    )
    noise_config: NoiseConfigSchema = Field(
        default_factory=NoiseConfigSchema,
        description="Sensor noise configuration.",
    )
    filter_config: KalmanConfigSchema = Field(
        default_factory=KalmanConfigSchema,
        description=(
            "Kalman filter configuration. Set enabled=true to apply the filter "
            "to noisy altitude and velocity readings."
        ),
    )
    pressure_filter_config: PressureFilterConfigSchema = Field(
        default_factory=PressureFilterConfigSchema,
        description=(
            "Pressure-domain EMA filter configuration. Set enabled=true to smooth "
            "raw sensor pressures before the barometric/Bernoulli conversions."
        ),
    )


class SimulationMetadata(BaseModel):
    """Summary statistics from a completed simulation run."""

    dt: float
    n_steps: int
    max_altitude: float
    max_speed: float
    final_apogee_prediction: float
    noise_enabled: bool
    noise_type: str
    filter_enabled: bool = False
    filter_type: str = "none"
    max_altitude_filtered: float | None = Field(
        None,
        description="Peak filtered altitude [m]. Null when filter is disabled.",
    )
    final_apogee_prediction_filtered: float | None = Field(
        None,
        description="Final apogee prediction from filtered state [m]. Null when filter is disabled.",
    )
    pressure_filter_enabled: bool = False
    pressure_filter_type: str = "none"
    max_altitude_pf: float | None = Field(
        None,
        description="Peak pressure-filtered altitude [m]. Null when pressure filter is disabled.",
    )
    final_apogee_prediction_pf: float | None = Field(
        None,
        description=(
            "Final apogee prediction from pressure-filtered state [m]. "
            "Null when pressure filter is disabled."
        ),
    )


class ChannelDeviation(BaseModel):
    """Deviation statistics for a single output channel."""

    max: float = Field(description="Maximum absolute deviation [channel unit]")
    mean: float = Field(description="Mean absolute deviation [channel unit]")


class DeviationStats(BaseModel):
    """Per-channel deviation between clean and noisy simulation runs."""

    altitude: ChannelDeviation
    speed: ChannelDeviation
    static_pressure: ChannelDeviation
    dynamic_pressure: ChannelDeviation
    predicted_apogee: ChannelDeviation


class SimulationResponse(BaseModel):
    """Full response from POST /simulate."""

    time: list[float]
    static_pressure: list[float]
    total_pressure: list[float]
    dynamic_pressure: list[float]
    altitude: list[float]
    speed: list[float]
    predicted_apogee: list[float]
    altitude_filtered: list[float] | None = Field(
        None,
        description="Kalman-filtered altitude [m]. Null when filter is disabled.",
    )
    speed_filtered: list[float] | None = Field(
        None,
        description="Kalman-filtered velocity [m/s]. Null when filter is disabled.",
    )
    predicted_apogee_filtered: list[float] | None = Field(
        None,
        description=(
            "Apogee prediction recomputed from filtered state [m]. "
            "Null when filter is disabled."
        ),
    )
    altitude_pf: list[float] | None = Field(
        None,
        description=(
            "Altitude derived from EMA pressure-filtered static pressure [m]. "
            "Null when pressure filter is disabled."
        ),
    )
    speed_pf: list[float] | None = Field(
        None,
        description=(
            "Velocity derived from EMA pressure-filtered dynamic pressure [m/s]. "
            "Null when pressure filter is disabled."
        ),
    )
    predicted_apogee_pf: list[float] | None = Field(
        None,
        description=(
            "Apogee prediction recomputed from pressure-filtered altitude and velocity [m]. "
            "Null when pressure filter is disabled."
        ),
    )
    metadata: SimulationMetadata
    deviations: DeviationStats | None = Field(
        None,
        description="Max and mean absolute deviation per channel vs. clean run. "
                    "Null when noise_type is 'none'.",
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


class PlotRequest(BaseModel):
    """
    Request body for POST /plot.
    """

    simulation: SimulationRequest = Field(
        default_factory=SimulationRequest,
        description="Full simulation configuration (profile, model params, noise).",
    )
    overlay_noise: bool = Field(
        False,
        description=(
            "When True and noise_type != 'none', run a second clean simulation "
            "and overlay it alongside the noisy one for comparison."
        ),
    )
    title: str = Field(
        "Rocket Flight Simulation",
        description="Figure title.",
    )
    dpi: int = Field(
        150,
        ge=72,
        le=600,
        description="Resolution of the PNG output [dots per inch]. Ignored for interactive plots.",
    )


class MonteCarloSeriesStats(BaseModel):
    """Pointwise statistics over Monte Carlo trajectories."""

    mean: list[float]
    std: list[float]
    p05: list[float]
    p95: list[float]


class MonteCarloPredictionErrorRequest(BaseModel):
    """
    Monte Carlo with per-timestep prediction error vs clean reference trajectory.

    The per-step error is computed as:
    ``predicted_apogee(noisy, t) - predicted_apogee(clean, t)``,
    where ``clean`` is one noise-free simulation on the same profile and
    physical parameters (computed server-side).

    Burnout time used for post-burnout metrics is estimated from the same
    noise-free reference simulation as the timestamp of maximum speed.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "n_runs": 100,
                    "base_seed": 42,
                    "burnout_time": 5.0,
                    "response_format": "json",
                    "include_all_runs": False,
                    "scatter_max_points": 50000,
                    "simulation": {
                        "synthetic_profile": {
                            "duration": 60.0,
                            "dt": 0.1,
                            "max_altitude": 3000.0,
                            "max_speed": 300.0,
                            "burnout_time": 5.0,
                        },
                        "noise_config": {
                            "noise_type": "binczar",
                            "params": {},
                        },
                    },
                }
            ]
        }
    )

    simulation: SimulationRequest = Field(
        default_factory=SimulationRequest,
        description="Monte Carlo noise configuration (noise ≠ none for stochastic runs).",
    )
    n_runs: int = Field(1000, ge=1, le=100000)
    base_seed: int = Field(42)
    burnout_time: float | None = Field(
        None,
        ge=0.0,
        description=(
            "Optional legacy override for burnout time [s]. "
            "The API currently derives burnout_time_used_s from the clean reference run."
        ),
    )
    include_all_runs: bool = Field(
        False,
        description=(
            "If true, include subsampled scatter arrays (scatter_time, scatter_signed_error) "
            "for plotting all runs up to scatter_max_points total samples."
        ),
    )
    scatter_max_points: int = Field(
        50_000,
        ge=1_000,
        le=5_000_000,
        description="Upper bound on scatter point count when include_all_runs is true.",
    )
    response_format: Literal["json", "png"] = Field(
        "json",
        description='Return JSON statistics or a PNG plot (set to "png").',
    )
    figure_title: str = Field(
        "Monte Carlo: prediction error vs clean reference",
        description="Figure title when response_format is png.",
    )
    figure_dpi: int = Field(
        150,
        ge=72,
        le=600,
        description="PNG resolution when response_format is png.",
    )


class MonteCarloPredictionErrorResponse(BaseModel):
    """Prediction error vs clean reference trajectory, plus scalar MAE/RMSE.

    When ``filter_config.enabled`` was set in the request, all ``*_filtered``
    fields are populated with the analogous statistics computed from the
    Kalman-filtered apogee prediction channel, allowing direct comparison
    between raw-noisy and KF-smoothed prediction accuracy.
    """

    reference_apogee_m: float = Field(
        description="max(altitude) [m] from the noise-free reference simulation.",
    )
    burnout_time_used_s: float | None = Field(
        description=(
            "Burnout time used for post-burnout windows, estimated from clean reference "
            "simulation as time of maximum speed."
        ),
    )
    apogee_time_used_s: float | None = Field(
        description=(
            "Reference apogee timestamp from clean reference simulation "
            "(time of maximum altitude)."
        ),
    )
    n_runs: int
    base_seed: int
    time: list[float]

    # --- Raw (noisy) prediction error ---
    signed_error: MonteCarloSeriesStats = Field(
        description=(
            "Per-step statistics over runs of "
            "(predicted_apogee_noisy(t) - predicted_apogee_clean(t)), in metres."
        ),
    )
    mean_abs_error_full_flight_m: float = Field(
        description=(
            "MAE over ascent up to apogee [m]: from start of run to "
            "apogee_time_used_s."
        ),
    )
    rmse_full_flight_m: float = Field(
        description=(
            "RMSE over ascent up to apogee [m]: from start of run to "
            "apogee_time_used_s."
        ),
    )
    mean_abs_error_post_burnout_m: float | None = Field(
        description=(
            "MAE over the post-burnout ascent window [m]: "
            "from burnout_time_used_s to apogee_time_used_s."
        ),
    )
    rmse_post_burnout_m: float | None = Field(
        description=(
            "RMSE over the post-burnout ascent window [m]: "
            "from burnout_time_used_s to apogee_time_used_s."
        ),
    )
    scatter_time: list[float] | None = None
    scatter_signed_error: list[float] | None = None

    # --- Kalman-filtered prediction error (populated only when filter enabled) ---
    signed_error_filtered: MonteCarloSeriesStats | None = Field(
        None,
        description=(
            "Per-step statistics over runs of "
            "(predicted_apogee_filtered(t) - predicted_apogee_clean(t)), in metres. "
            "Null when filter_config.enabled is false."
        ),
    )
    mean_abs_error_full_flight_filtered_m: float | None = Field(
        None,
        description=(
            "MAE of filtered prediction over ascent up to apogee [m]. "
            "Null when filter_config.enabled is false."
        ),
    )
    rmse_full_flight_filtered_m: float | None = Field(
        None,
        description=(
            "RMSE of filtered prediction over ascent up to apogee [m]. "
            "Null when filter_config.enabled is false."
        ),
    )
    mean_abs_error_post_burnout_filtered_m: float | None = Field(
        None,
        description=(
            "MAE of filtered prediction over post-burnout ascent window [m]. "
            "Null when filter_config.enabled is false."
        ),
    )
    rmse_post_burnout_filtered_m: float | None = Field(
        None,
        description=(
            "RMSE of filtered prediction over post-burnout ascent window [m]. "
            "Null when filter_config.enabled is false."
        ),
    )
    scatter_signed_error_filtered: list[float] | None = None

    # --- Pressure-filtered prediction error (populated only when pressure filter enabled) ---
    signed_error_pf: MonteCarloSeriesStats | None = Field(
        None,
        description=(
            "Per-step statistics over runs of "
            "(predicted_apogee_pf(t) - predicted_apogee_clean(t)), in metres. "
            "Null when pressure_filter_config.enabled is false."
        ),
    )
    mean_abs_error_full_flight_pf_m: float | None = Field(
        None,
        description=(
            "MAE of pressure-filtered prediction over ascent up to apogee [m]. "
            "Null when pressure_filter_config.enabled is false."
        ),
    )
    rmse_full_flight_pf_m: float | None = Field(
        None,
        description=(
            "RMSE of pressure-filtered prediction over ascent up to apogee [m]. "
            "Null when pressure_filter_config.enabled is false."
        ),
    )
    mean_abs_error_post_burnout_pf_m: float | None = Field(
        None,
        description=(
            "MAE of pressure-filtered prediction over post-burnout ascent window [m]. "
            "Null when pressure_filter_config.enabled is false."
        ),
    )
    rmse_post_burnout_pf_m: float | None = Field(
        None,
        description=(
            "RMSE of pressure-filtered prediction over post-burnout ascent window [m]. "
            "Null when pressure_filter_config.enabled is false."
        ),
    )
    scatter_signed_error_pf: list[float] | None = None
