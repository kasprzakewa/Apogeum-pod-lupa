from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimulationResult:
    """
    Holds the complete output of one simulation run.

    Two optional filtered output families are available:

    **State-space Kalman filter** (``filter_enabled = True``)
        ``altitude_filtered``, ``speed_filtered``, ``predicted_apogee_filtered``

    **Pressure-domain EMA filter** (``pressure_filter_enabled = True``)
        ``altitude_pf``, ``speed_pf``, ``predicted_apogee_pf``
        — EMA applied to raw (static, dynamic) pressures before the
          barometric / Bernoulli conversions.

    Both families can be active simultaneously.
    """

    time: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    static_pressure: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    total_pressure: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    dynamic_pressure: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )

    altitude: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    speed: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    predicted_apogee: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )

    # --- State-space KF outputs ---
    altitude_filtered: np.ndarray | None = None
    speed_filtered: np.ndarray | None = None
    predicted_apogee_filtered: np.ndarray | None = None

    # --- Pressure-domain EMA outputs ---
    altitude_pf: np.ndarray | None = None
    speed_pf: np.ndarray | None = None
    predicted_apogee_pf: np.ndarray | None = None

    dt: float = 0.01
    noise_enabled: bool = False
    noise_type: str = "none"
    filter_enabled: bool = False
    filter_type: str = "none"
    pressure_filter_enabled: bool = False
    pressure_filter_type: str = "none"

    @property
    def n_steps(self) -> int:
        return len(self.time)

    @property
    def max_altitude(self) -> float:
        return float(np.max(self.altitude)) if self.n_steps > 0 else 0.0

    @property
    def max_speed(self) -> float:
        return float(np.max(self.speed)) if self.n_steps > 0 else 0.0

    @property
    def final_apogee_prediction(self) -> float:
        return float(self.predicted_apogee[-1]) if self.n_steps > 0 else 0.0

    @property
    def max_altitude_filtered(self) -> float | None:
        if self.altitude_filtered is not None and len(self.altitude_filtered) > 0:
            return float(np.max(self.altitude_filtered))
        return None

    @property
    def final_apogee_prediction_filtered(self) -> float | None:
        if self.predicted_apogee_filtered is not None and len(self.predicted_apogee_filtered) > 0:
            return float(self.predicted_apogee_filtered[-1])
        return None

    @property
    def max_altitude_pf(self) -> float | None:
        if self.altitude_pf is not None and len(self.altitude_pf) > 0:
            return float(np.max(self.altitude_pf))
        return None

    @property
    def final_apogee_prediction_pf(self) -> float | None:
        if self.predicted_apogee_pf is not None and len(self.predicted_apogee_pf) > 0:
            return float(self.predicted_apogee_pf[-1])
        return None

    def deviation_from(self, other: "SimulationResult") -> dict[str, dict[str, float]]:
        """
        Compute max and mean absolute deviation between this result and another.

        Typical usage: clean.deviation_from(noisy)

        Returns:
            Dict keyed by channel name, each value a dict with "max" and "mean".
        """
        channels = {
            "altitude":         (self.altitude,         other.altitude),
            "speed":            (self.speed,             other.speed),
            "static_pressure":  (self.static_pressure,  other.static_pressure),
            "dynamic_pressure": (self.dynamic_pressure, other.dynamic_pressure),
            "predicted_apogee": (self.predicted_apogee, other.predicted_apogee),
        }
        return {
            name: {
                "max":  float(np.max(np.abs(a - b))),
                "mean": float(np.mean(np.abs(a - b))),
            }
            for name, (a, b) in channels.items()
        }

    def to_dict(self) -> dict:
        data: dict = {
            "time": self.time.tolist(),
            "static_pressure": self.static_pressure.tolist(),
            "total_pressure": self.total_pressure.tolist(),
            "dynamic_pressure": self.dynamic_pressure.tolist(),
            "altitude": self.altitude.tolist(),
            "speed": self.speed.tolist(),
            "predicted_apogee": self.predicted_apogee.tolist(),
            "metadata": {
                "dt": self.dt,
                "n_steps": self.n_steps,
                "max_altitude": self.max_altitude,
                "max_speed": self.max_speed,
                "final_apogee_prediction": self.final_apogee_prediction,
                "noise_enabled": self.noise_enabled,
                "noise_type": self.noise_type,
                "filter_enabled": self.filter_enabled,
                "filter_type": self.filter_type,
                "pressure_filter_enabled": self.pressure_filter_enabled,
                "pressure_filter_type": self.pressure_filter_type,
            },
        }

        if self.filter_enabled:
            data["altitude_filtered"] = (
                self.altitude_filtered.tolist() if self.altitude_filtered is not None else []
            )
            data["speed_filtered"] = (
                self.speed_filtered.tolist() if self.speed_filtered is not None else []
            )
            data["predicted_apogee_filtered"] = (
                self.predicted_apogee_filtered.tolist()
                if self.predicted_apogee_filtered is not None
                else []
            )
            data["metadata"]["max_altitude_filtered"] = self.max_altitude_filtered
            data["metadata"]["final_apogee_prediction_filtered"] = (
                self.final_apogee_prediction_filtered
            )

        if self.pressure_filter_enabled:
            data["altitude_pf"] = (
                self.altitude_pf.tolist() if self.altitude_pf is not None else []
            )
            data["speed_pf"] = (
                self.speed_pf.tolist() if self.speed_pf is not None else []
            )
            data["predicted_apogee_pf"] = (
                self.predicted_apogee_pf.tolist()
                if self.predicted_apogee_pf is not None
                else []
            )
            data["metadata"]["max_altitude_pf"] = self.max_altitude_pf
            data["metadata"]["final_apogee_prediction_pf"] = self.final_apogee_prediction_pf

        return data
