"""
Core physics functions for the 1D rocket flight simulation.
"""

import math
from dataclasses import dataclass, field

from src.models.constants import (
    BARO_EXPONENT,
    DEFAULT_CROSS_SECTION,
    DEFAULT_DRAG_COEFFICIENT,
    DEFAULT_MASS,
    DENSITY_EXPONENT,
    G,
    REFERENCE_AIR_DENSITY,
    REFERENCE_PRESSURE,
    SCALE_HEIGHT,
)


@dataclass
class ModelParams:
    """Configurable physical and aerodynamic parameters for the simulation."""

    reference_pressure: float = REFERENCE_PRESSURE
    reference_air_density: float = REFERENCE_AIR_DENSITY
    scale_height: float = SCALE_HEIGHT
    baro_exponent: float = BARO_EXPONENT
    density_exponent: float = DENSITY_EXPONENT
    g: float = G
    cross_section: float = DEFAULT_CROSS_SECTION       # A  [m²]
    drag_coefficient: float = DEFAULT_DRAG_COEFFICIENT  # C_D
    mass: float = DEFAULT_MASS                          # m  [kg]

    @property
    def drag_area(self) -> float:
        """A * C_D product used repeatedly in apogee prediction."""
        return self.cross_section * self.drag_coefficient


def calculate_altitude(pressure: float, params: ModelParams | None = None) -> float:
    """
    Convert static pressure to altitude using the barometric formula (ISA model).

    Args:
        pressure: Static pressure [Pa].
        params:   Physical model parameters; uses defaults when None.

    Returns:
        Altitude above sea level [m].
    """
    p = params or ModelParams()
    return p.scale_height * (1.0 - (pressure / p.reference_pressure) ** p.baro_exponent)


def calculate_air_density(altitude: float, params: ModelParams | None = None) -> float:
    """
    Estimate air density at a given altitude using a power-law approximation.

    Args:
        altitude: Altitude [m].
        params:   Physical model parameters; uses defaults when None.

    Returns:
        Air density [kg/m³].
    """
    p = params or ModelParams()
    factor = 1.0 - altitude / p.scale_height
    factor = max(factor, 0.0)  # clamp to avoid negative values at extreme altitudes
    return p.reference_air_density * (factor ** p.density_exponent)


def calculate_speed(
    diff_pressure: float,
    altitude: float,
    params: ModelParams | None = None,
) -> float:
    """
    Derive airspeed from differential (dynamic) pressure using Bernoulli's equation.

    Args:
        diff_pressure: Differential pressure (total – static) [Pa].
        altitude:      Current altitude [m], used to compute local air density.
        params:        Physical model parameters; uses defaults when None.

    Returns:
        Airspeed [m/s].
    """
    if abs(diff_pressure) <= 0.0:
        return 0.0

    p = params or ModelParams()
    rho = calculate_air_density(altitude, p)
    if rho <= 0.0:
        return 0.0

    return math.sqrt(2.0 * abs(diff_pressure) / rho)


def predict_apogee(
    speed: float,
    altitude: float,
    dynamic_pressure: float,
    params: ModelParams | None = None,
) -> float:
    """
    Predict apogee altitude from current speed, altitude, and aerodynamic drag.

    Two cases:
      - Fd > 0: drag-corrected kinematic formula (log term accounts for drag deceleration)
      - Fd = 0: simple kinematic apogee (v² / 2g above current altitude)

    Args:
        speed:            Current airspeed [m/s].
        altitude:         Current altitude [m].
        dynamic_pressure: Dynamic pressure q = 0.5 * rho * v² [Pa].
        params:           Physical model parameters; uses defaults when None.

    Returns:
        Predicted apogee altitude [m].
    """
    p = params or ModelParams()
    fd = dynamic_pressure * p.drag_area  # aerodynamic drag force [N]

    if fd > 0.0:
        ratio = fd / (p.mass * p.g)
        return altitude + (p.mass * speed**2 * math.log(1.0 + ratio)) / (2.0 * fd)
    else:
        return altitude + speed**2 / (2.0 * p.g)
