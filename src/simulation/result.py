from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimulationResult:
    """
    Holds the complete output of one simulation run.
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

    dt: float = 0.01                  # time step [s]
    noise_enabled: bool = False
    noise_type: str = "none"

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

    def to_dict(self) -> dict:
        return {
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
            },
        }
