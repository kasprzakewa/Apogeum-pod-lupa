"""
SimulationEngine - discrete-time simulation loop.

At each time step the engine:
  1. Applies optional noise to raw pressure readings.
  2. Computes altitude from static pressure.
  3. Computes airspeed from differential pressure and altitude.
  4. Computes dynamic pressure.
  5. Predicts apogee from current state.
  6. Stores all values in the result arrays.
"""

from __future__ import annotations

import numpy as np

from src.models.physics import (
    ModelParams,
    calculate_altitude,
    calculate_speed,
    predict_apogee,
)
from src.simulation.flight_profile import FlightProfile
from src.simulation.result import SimulationResult


class SimulationEngine:
    """
    Runs a discrete-time software-in-the-loop simulation.

    Args:
        profile:     Input pressure data (time, static_pressure, total_pressure).
        params:      Physical/aerodynamic model parameters. Uses defaults when None.
        noise_model: Optional noise model applied to raw sensor readings.
                     Import from src.noise.noise_model; pass None for ideal simulation.
    """

    def __init__(
        self,
        profile: FlightProfile,
        params: ModelParams | None = None,
        noise_model=None,
    ) -> None:
        self.profile = profile
        self.params = params or ModelParams()
        self.noise_model = noise_model

    def run(self) -> SimulationResult:
        """
        Execute the full simulation over the flight profile.

        Returns:
            SimulationResult with all time-series arrays populated.
        """
        n = self.profile.n_steps
        profile = self.profile

        # Synchronize dt-dependent parameters (e.g. lag filters) with the profile
        if self.noise_model is not None:
            self.noise_model.configure(profile.dt)

        static_pressure = np.empty(n, dtype=np.float64)
        total_pressure = np.empty(n, dtype=np.float64)
        dynamic_pressure = np.empty(n, dtype=np.float64)
        altitude = np.empty(n, dtype=np.float64)
        speed = np.empty(n, dtype=np.float64)
        predicted_apogee = np.empty(n, dtype=np.float64)

        noise_enabled = self.noise_model is not None and self.noise_model.enabled
        noise_type = type(self.noise_model).__name__ if self.noise_model else "none"

        for i in range(n):
            p_static = profile.static_pressure[i]
            p_total = profile.total_pressure[i]

            # Apply noise to sensor readings
            if noise_enabled:
                p_static, p_total = self.noise_model.apply(p_static, p_total)

            p_diff = p_total - p_static

            alt = calculate_altitude(p_static, self.params)
            spd = calculate_speed(p_diff, alt, self.params)
            dyn_p = 0.5 * _air_density(alt, self.params) * spd**2
            apogee = predict_apogee(spd, alt, dyn_p, self.params)

            static_pressure[i] = p_static
            total_pressure[i] = p_total
            dynamic_pressure[i] = dyn_p
            altitude[i] = alt
            speed[i] = spd
            predicted_apogee[i] = apogee

        return SimulationResult(
            time=profile.time.copy(),
            static_pressure=static_pressure,
            total_pressure=total_pressure,
            dynamic_pressure=dynamic_pressure,
            altitude=altitude,
            speed=speed,
            predicted_apogee=predicted_apogee,
            dt=profile.dt,
            noise_enabled=noise_enabled,
            noise_type=noise_type,
        )


def _air_density(altitude: float, params: ModelParams) -> float:
    from src.models.physics import calculate_air_density
    return calculate_air_density(altitude, params)
