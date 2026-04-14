"""
Pydantic schemas for the FastAPI simulation API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


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
            "Currently available: 'none'."
        ),
    )
    params: dict = Field(
        default_factory=dict,
        description=(
            "Constructor keyword arguments forwarded to the noise model. "
            "E.g. for a GaussianNoiseModel: {\"sigma_static\": 10.0, \"sigma_total\": 15.0}."
        ),
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

    csv_path: str | None = Field(
        None,
        description="Path to CSV flight profile on the server filesystem. "
        "CSV must have columns: time, static_pressure, total_pressure.",
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


class SimulationMetadata(BaseModel):
    """Summary statistics from a completed simulation run."""

    dt: float
    n_steps: int
    max_altitude: float
    max_speed: float
    final_apogee_prediction: float
    noise_enabled: bool
    noise_type: str


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
