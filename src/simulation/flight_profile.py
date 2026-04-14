from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.constants import REFERENCE_PRESSURE, SCALE_HEIGHT, BARO_EXPONENT


@dataclass
class FlightProfile:
    """Container for the raw pressure time-series used as simulation input."""

    time: np.ndarray            # [s]
    static_pressure: np.ndarray # [Pa]
    total_pressure: np.ndarray  # [Pa]

    @property
    def n_steps(self) -> int:
        return len(self.time)

    @property
    def dt(self) -> float:
        """Mean time step derived from the time array."""
        if self.n_steps < 2:
            return 0.0
        return float(np.mean(np.diff(self.time)))

    @classmethod
    def from_csv(cls, path: str | Path) -> "FlightProfile":
        """
        Load a flight profile from a CSV file.

        Expected columns (case-insensitive):
            time, static_pressure, total_pressure

        Args:
            path: Path to the CSV file.

        Returns:
            FlightProfile instance.
        """
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]

        required = {"time", "static_pressure", "total_pressure"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        return cls(
            time=df["time"].to_numpy(dtype=np.float64),
            static_pressure=df["static_pressure"].to_numpy(dtype=np.float64),
            total_pressure=df["total_pressure"].to_numpy(dtype=np.float64),
        )

    @classmethod
    def synthetic(
        cls,
        duration: float = 60.0,
        dt: float = 0.01,
        max_altitude: float = 3000.0,
        max_speed: float = 300.0,
        burnout_time: float = 5.0,
    ) -> "FlightProfile":
        """
        Generate a synthetic pressure profile for testing.

        The trajectory is modelled as:
          - Motor phase (0 → burnout_time): speed ramps up linearly
          - Coast phase (burnout_time → apogee): speed decreases, altitude rises
          - Descent (apogee → end): simple free-fall approximation

        Args:
            duration:     Total flight duration [s].
            dt:           Time step [s].
            max_altitude: Target apogee altitude [m].
            max_speed:    Peak airspeed at burnout [m/s].
            burnout_time: Motor burnout time [s].

        Returns:
            FlightProfile with synthetic pressure data.
        """
        time = np.arange(0.0, duration, dt)
        n = len(time)

        altitude = np.zeros(n)
        speed = np.zeros(n)

        # Simple parabolic altitude profile
        apogee_time = duration * 0.35
        for i, t in enumerate(time):
            if t <= apogee_time:
                frac = t / apogee_time
                altitude[i] = max_altitude * (2 * frac - frac**2)
                speed[i] = max_speed * (1.0 - frac) if t >= burnout_time else max_speed * (t / burnout_time)
            else:
                descent_frac = (t - apogee_time) / (duration - apogee_time)
                altitude[i] = max_altitude * (1.0 - descent_frac**2)
                speed[i] = max_speed * 0.3 * descent_frac

        altitude = np.clip(altitude, 0.0, None)

        # Static pressure from altitude via barometric formula (inverted)
        static_pressure = REFERENCE_PRESSURE * (
            1.0 - altitude / SCALE_HEIGHT
        ) ** (1.0 / BARO_EXPONENT)

        # Total pressure = static + dynamic, with density varying with altitude
        rho_factor = np.maximum(1.0 - altitude / SCALE_HEIGHT, 0.0)
        air_density = 1.225 * rho_factor**4.256
        dynamic_pressure = 0.5 * air_density * speed**2
        total_pressure = static_pressure + dynamic_pressure

        return cls(
            time=time,
            static_pressure=static_pressure,
            total_pressure=total_pressure,
        )
