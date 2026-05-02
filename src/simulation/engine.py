"""
SimulationEngine - discrete-time simulation loop.

At each time step the engine:
  1. Applies optional noise to raw pressure readings.
  2. Optionally filters the noisy pressures with a PressureEMAFilter
     (pressure-domain filter, applied before any physical conversion).
  3. Computes altitude from static pressure.
  4. Computes airspeed from differential pressure and altitude.
  5. Optionally filters (altitude, velocity) with a KalmanFilter
     (state-space filter, applied after physical conversion).
  6. Computes dynamic pressure and predicts apogee for each active channel.
  7. Stores all values in the result arrays.

Output channels
---------------
  altitude / speed / predicted_apogee
      Always populated from raw (possibly noisy) pressures.

  altitude_pf / speed_pf / predicted_apogee_pf
      Populated when pressure_filter is active.  Derived from EMA-smoothed
      pressures — lag stays expressed in Pa, not amplified through v².

  altitude_filtered / speed_filtered / predicted_apogee_filtered
      Populated when filter_model (KalmanFilter) is active.  Applied to the
      raw-derived altitude and speed.

Both pressure_filter and filter_model can be active simultaneously.
"""

from __future__ import annotations

import numpy as np

from src.models.physics import (
    ModelParams,
    calculate_air_density,
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
        profile:          Input pressure data (time, static_pressure, total_pressure).
        params:           Physical/aerodynamic model parameters. Uses defaults when None.
        noise_model:      Optional noise model applied to raw sensor readings.
        pressure_filter:  Optional PressureEMAFilter applied to noisy pressures before
                          the barometric / Bernoulli conversions.  Produces the
                          ``*_pf`` output channels.
        filter_model:     Optional KalmanFilter applied to (altitude, velocity) derived
                          from noisy sensor readings.  Produces the ``*_filtered``
                          output channels.
    """

    def __init__(
        self,
        profile: FlightProfile,
        params: ModelParams | None = None,
        noise_model=None,
        pressure_filter=None,
        filter_model=None,
    ) -> None:
        self.profile = profile
        self.params = params or ModelParams()
        self.noise_model = noise_model
        self.pressure_filter = pressure_filter
        self.filter_model = filter_model

    def run(self) -> SimulationResult:
        """Execute the full simulation over the flight profile."""
        n = self.profile.n_steps
        profile = self.profile

        if self.noise_model is not None:
            self.noise_model.configure(profile.dt)

        pf_enabled = self.pressure_filter is not None
        if pf_enabled:
            self.pressure_filter.configure(profile.dt)
            self.pressure_filter.reset()

        kf_enabled = self.filter_model is not None
        if kf_enabled:
            self.filter_model.configure(profile.dt)
            self.filter_model.reset()

        # --- Raw channel arrays (always populated) ---
        static_pressure  = np.empty(n, dtype=np.float64)
        total_pressure   = np.empty(n, dtype=np.float64)
        dynamic_pressure = np.empty(n, dtype=np.float64)
        altitude         = np.empty(n, dtype=np.float64)
        speed            = np.empty(n, dtype=np.float64)
        predicted_apogee = np.empty(n, dtype=np.float64)

        # --- Pressure-filtered channel arrays ---
        if pf_enabled:
            altitude_pf         = np.empty(n, dtype=np.float64)
            speed_pf            = np.empty(n, dtype=np.float64)
            predicted_apogee_pf = np.empty(n, dtype=np.float64)
        else:
            altitude_pf = speed_pf = predicted_apogee_pf = None

        # --- State-space KF channel arrays ---
        if kf_enabled:
            altitude_filtered         = np.empty(n, dtype=np.float64)
            speed_filtered            = np.empty(n, dtype=np.float64)
            predicted_apogee_filtered = np.empty(n, dtype=np.float64)
        else:
            altitude_filtered = speed_filtered = predicted_apogee_filtered = None

        noise_enabled  = self.noise_model is not None and self.noise_model.enabled
        noise_type     = type(self.noise_model).__name__ if self.noise_model else "none"
        pf_type        = type(self.pressure_filter).__name__ if pf_enabled else "none"
        kf_type        = type(self.filter_model).__name__    if kf_enabled else "none"

        for i in range(n):
            p_static = profile.static_pressure[i]
            p_total  = profile.total_pressure[i]

            if noise_enabled:
                p_static, p_total = self.noise_model.apply(p_static, p_total)

            p_diff = p_total - p_static

            # 2. Raw channel: altitude / speed / apogee from noisy pressures
            alt    = calculate_altitude(p_static, self.params)
            spd    = calculate_speed(p_diff, alt, self.params)
            dyn_p  = 0.5 * _air_density(alt, self.params) * spd ** 2
            apogee = predict_apogee(spd, alt, dyn_p, self.params)

            static_pressure[i]  = p_static
            total_pressure[i]   = p_total
            dynamic_pressure[i] = dyn_p
            altitude[i]         = alt
            speed[i]            = spd
            predicted_apogee[i] = apogee

            # 3. Pressure-filtered channel: EMA on pressures, then recompute
            if pf_enabled:
                p_s_f, p_d_f    = self.pressure_filter.update(p_static, p_diff)
                alt_pf          = calculate_altitude(p_s_f, self.params)
                spd_pf          = calculate_speed(p_d_f, alt_pf, self.params)
                dyn_p_pf        = 0.5 * _air_density(alt_pf, self.params) * spd_pf ** 2
                apogee_pf       = predict_apogee(spd_pf, alt_pf, dyn_p_pf, self.params)
                altitude_pf[i]         = alt_pf
                speed_pf[i]            = spd_pf
                predicted_apogee_pf[i] = apogee_pf

            # 4. State-space KF channel: KF on raw altitude / speed
            if kf_enabled:
                alt_f, spd_f    = self.filter_model.update(alt, spd)
                dyn_p_f         = 0.5 * _air_density(alt_f, self.params) * spd_f ** 2
                apogee_f        = predict_apogee(spd_f, alt_f, dyn_p_f, self.params)
                altitude_filtered[i]         = alt_f
                speed_filtered[i]            = spd_f
                predicted_apogee_filtered[i] = apogee_f

        return SimulationResult(
            time=profile.time.copy(),
            static_pressure=static_pressure,
            total_pressure=total_pressure,
            dynamic_pressure=dynamic_pressure,
            altitude=altitude,
            speed=speed,
            predicted_apogee=predicted_apogee,
            altitude_pf=altitude_pf,
            speed_pf=speed_pf,
            predicted_apogee_pf=predicted_apogee_pf,
            altitude_filtered=altitude_filtered,
            speed_filtered=speed_filtered,
            predicted_apogee_filtered=predicted_apogee_filtered,
            dt=profile.dt,
            noise_enabled=noise_enabled,
            noise_type=noise_type,
            pressure_filter_enabled=pf_enabled,
            pressure_filter_type=pf_type,
            filter_enabled=kf_enabled,
            filter_type=kf_type,
        )


def _air_density(altitude: float, params: ModelParams) -> float:
    return calculate_air_density(altitude, params)
