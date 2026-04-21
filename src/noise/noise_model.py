"""
Noise models for sensor simulation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class NoiseModel(ABC):
    """
    Abstract base class for all sensor noise models.

    Subclasses must implement :meth:`apply`.
    The :attr:`enabled` flag allows the simulation engine to skip noise
    application without changing the engine code.
    """

    def __init__(self, enabled: bool = True, seed: int | None = None) -> None:
        self.enabled = enabled
        self._rng = np.random.default_rng(seed)

    @abstractmethod
    def apply(
        self, static_pressure: float, total_pressure: float
    ) -> tuple[float, float]:
        """
        Add noise to raw sensor readings.

        Args:
            static_pressure: Clean static pressure reading [Pa].
            total_pressure:  Clean total (pitot) pressure reading [Pa].

        Returns:
            Tuple of (noisy_static_pressure, noisy_total_pressure) [Pa].
        """

    def configure(self, dt: float) -> None:
        """
        Synchronize time-step dependent parameters with the simulation engine.
        """

    def reset(self, seed: int | None = None) -> None:
        """Re-seed the RNG (useful for Monte Carlo repeatability)."""
        self._rng = np.random.default_rng(seed)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(enabled={self.enabled})"


class NoNoiseModel(NoiseModel):
    """Pass-through noise model - returns sensor readings unchanged."""

    def __init__(self) -> None:
        super().__init__(enabled=False)

    def apply(
        self, static_pressure: float, total_pressure: float
    ) -> tuple[float, float]:
        return static_pressure, total_pressure


class BinczarNoiseModel(NoiseModel):
    """
    Realistic rocket pressure sensor model.

    Simulates the following error sources:
      - Pneumatic lag     - low-pass filter on pressure readings (tau_lag)
      - White noise       - Gaussian noise on each channel (rms_static, rms_dynamic)
      - Vibration pickup  - proportional to longitudinal acceleration (vib_sens)
      - Temperature drift - linear offset from 20 °C reference (temp_coeff)
      - ADC quantization  - rounding to sensor resolution (res)

    Parameters are drawn randomly at construction (Monte Carlo style), so each
    instance represents a unique hardware/flight combination.

    Args:
        config:  Dict with sensor parameter distributions. Uses defaults when None.
                   'static_rms_base'  : (mean, std)  [Pa]
                   'dynamic_rms_base' : (mean, std)  [Pa]
                   'tau_lag_range'    : (min, max)   [s]
                   'vib_sens_range'   : (min, max)   [Pa/g]
                   'temp_drift_range' : (min, max)   [Pa/°C]
                   'resolution'       : float        [Pa]
        accel_g: Constant longitudinal acceleration [g]. Default 0 (no vibration).
        temp_c:  Constant sensor temperature [°C]. Default 20 (no drift).
        seed:    RNG seed for reproducibility.

    Note:
        The lag filter coefficient (alpha) is computed automatically from the
        flight profile's dt by SimulationEngine before the simulation starts.
        You never need to pass dt manually.
    """

    _DEFAULT_CONFIG = {
        "static_rms_base":  (1.5, 0.5),
        "dynamic_rms_base": (4.0, 1.2),
        "tau_lag_range":    (0.02, 0.06),
        "vib_sens_range":   (0.1, 1.5),
        "temp_drift_range": (-0.3, 0.3),
        "resolution":       1.2,
    }

    def __init__(
        self,
        config: dict | None = None,
        accel_g: float = 0.0,
        temp_c: float = 20.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(enabled=True, seed=seed)
        cfg = config if config is not None else self._DEFAULT_CONFIG

        self.rms_static  = max(0.1, self._rng.normal(*cfg["static_rms_base"]))
        self.rms_dynamic = max(0.1, self._rng.normal(*cfg["dynamic_rms_base"]))
        self.tau_lag     = float(self._rng.uniform(*cfg["tau_lag_range"]))
        self.vib_sens    = float(self._rng.uniform(*cfg["vib_sens_range"]))
        self.temp_coeff  = float(self._rng.uniform(*cfg["temp_drift_range"]))
        self.res         = cfg["resolution"]

        self.accel_g = accel_g
        self.temp_c  = temp_c

        self._alpha: float = 1.0

        self._last_p_stat: float | None = None
        self._last_p_dyn:  float | None = None

    def configure(self, dt: float) -> None:
        """Recompute the lag coefficient using the engine's actual time step."""
        self._alpha = dt / (self.tau_lag + dt)

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._last_p_stat = None
        self._last_p_dyn  = None

    def apply(
        self, static_pressure: float, total_pressure: float
    ) -> tuple[float, float]:
        """
        Apply the full sensor error chain to one time-step.
        """
        p_dyn_true = total_pressure - static_pressure

        # 1. Pneumatic lag (first-order EMA, alpha pre-computed in configure())
        if self._last_p_stat is None:
            self._last_p_stat = static_pressure
            self._last_p_dyn  = p_dyn_true

        p_s_lag = self._last_p_stat + self._alpha * (static_pressure - self._last_p_stat)
        p_d_lag = self._last_p_dyn  + self._alpha * (p_dyn_true      - self._last_p_dyn)
        self._last_p_stat, self._last_p_dyn = p_s_lag, p_d_lag

        # 2. White noise + vibration + temperature drift
        n_s       = self._rng.normal(0.0, self.rms_static)  + self.accel_g * self.vib_sens * 0.5
        n_d       = self._rng.normal(0.0, self.rms_dynamic) + self.accel_g * self.vib_sens
        p_s_noisy = p_s_lag + n_s + (self.temp_c - 20.0) * self.temp_coeff
        p_d_noisy = p_d_lag + n_d

        # 3. ADC quantization
        p_s_final = np.round(p_s_noisy / self.res) * self.res
        p_d_final = np.round(p_d_noisy / self.res) * self.res

        return p_s_final, p_s_final + p_d_final
        

class GaussianNoiseModel(NoiseModel):
    """
    Simple Gaussian noise model for Monte Carlo simulations.
    
    Adds independent Gaussian noise to static and total pressure readings.
    
    Args:
        sigma_static: Standard deviation of noise on static pressure [Pa].
        sigma_total:  Standard deviation of noise on total pressure [Pa].
        seed:         RNG seed for reproducibility.
    """
    
    def __init__(
        self, 
        sigma_static: float, 
        sigma_total: float, 
        seed: int | None = None
    ) -> None:
        super().__init__(enabled=True, seed=seed)
        self.sigma_static = sigma_static
        self.sigma_total = sigma_total
    
    def apply(
        self, static_pressure: float, total_pressure: float
    ) -> tuple[float, float]:
        noise_static = self._rng.normal(0.0, self.sigma_static)
        noise_total = self._rng.normal(0.0, self.sigma_total)
        return static_pressure + noise_static, total_pressure + noise_total


NOISE_REGISTRY: dict[str, type[NoiseModel]] = {
    "none":   NoNoiseModel,
    "binczar": BinczarNoiseModel,
    "gaussian": GaussianNoiseModel,
}


def create_noise_model(noise_type: str, **params) -> NoiseModel:
    """
    Instantiate a noise model by name using the registry.

    Args:
        noise_type: Key from NOISE_REGISTRY (e.g. "none", "binczar").
        **params:   Forwarded as keyword arguments to the model constructor.

    Raises:
        ValueError: If noise_type is not in NOISE_REGISTRY.
    """
    cls = NOISE_REGISTRY.get(noise_type.lower())
    if cls is None:
        available = list(NOISE_REGISTRY.keys())
        raise ValueError(
            f"Unknown noise type '{noise_type}'. Available: {available}"
        )
    if noise_type.lower() == "none":
        return NoNoiseModel()
    return cls(**params)
