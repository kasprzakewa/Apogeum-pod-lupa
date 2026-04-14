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

    def reset(self, seed: int | None = None) -> None:
        """Re-seed the RNG (useful for Monte Carlo repeatability)."""
        self._rng = np.random.default_rng(seed)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(enabled={self.enabled})"


class NoNoiseModel(NoiseModel):
    """
    Pass-through noise model -returns sensor readings unchanged.
    """

    def __init__(self) -> None:
        super().__init__(enabled=False)

    def apply(
        self, static_pressure: float, total_pressure: float
    ) -> tuple[float, float]:
        return static_pressure, total_pressure
