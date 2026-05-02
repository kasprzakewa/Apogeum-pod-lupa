"""
Pressure-domain EMA (Exponential Moving Average) filter.

Filters raw sensor pressure readings **before** the barometric and Bernoulli
conversions, so any filtering lag stays expressed in Pascals rather than in
metres/second.  This matters because the apogee-prediction formula scales with
v², so a lag of Δv m/s at burnout causes an error ≈ v·Δv/g ≈ 27·Δv metres,
while the same lag expressed as a pressure error ΔP causes only
  Δv ≈ ΔP / (ρ·v)  →  Δapogee ≈ ΔP / (ρ·g) ≈ 0.09 m per Pa.

Two independent first-order IIR low-pass filters are applied:
  - one on static pressure  (longer time constant — changes slowly)
  - one on dynamic pressure (shorter time constant — needs faster response
                              during the brief burnout velocity peak)

Typical defaults that work well with BinczarNoiseModel at dt ≈ 0.01–0.35 s:
  tau_static  = 0.05 s
  tau_dynamic = 0.02 s
"""

from __future__ import annotations


class PressureEMAFilter:
    """
    Dual-channel EMA filter on (static pressure, dynamic pressure).

    Args:
        tau_static:  Time constant for the static-pressure channel [s].
                     Larger → stronger noise suppression, more lag on altitude.
        tau_dynamic: Time constant for the dynamic-pressure channel [s].
                     Smaller than tau_static recommended so velocity tracking
                     stays responsive during the motor-burn transient.
    """

    def __init__(
        self,
        tau_static: float = 0.03,
        tau_dynamic: float = 0.02,
    ) -> None:
        if tau_static <= 0.0:
            raise ValueError(f"tau_static must be positive, got {tau_static}")
        if tau_dynamic <= 0.0:
            raise ValueError(f"tau_dynamic must be positive, got {tau_dynamic}")

        self.tau_static  = float(tau_static)
        self.tau_dynamic = float(tau_dynamic)

        self._alpha_s: float = 1.0   # recomputed in configure()
        self._alpha_d: float = 1.0

        self._prev_s: float | None = None
        self._prev_d: float | None = None

    def configure(self, dt: float) -> None:
        """
        Precompute EMA coefficients for a given simulation time step.

        Must be called before the first :meth:`update` whenever ``dt`` changes.
        """
        self._alpha_s = dt / (self.tau_static  + dt)
        self._alpha_d = dt / (self.tau_dynamic + dt)

    def reset(self) -> None:
        """Clear filter memory (call before reusing for a new flight)."""
        self._prev_s = None
        self._prev_d = None

    def update(self, p_static: float, p_dynamic: float) -> tuple[float, float]:
        """
        Apply one EMA step to both pressure channels.

        On the very first call the filter is initialised with the raw reading
        (zero initial lag).

        Args:
            p_static:  Noisy static pressure reading [Pa].
            p_dynamic: Noisy dynamic pressure  (p_total − p_static) [Pa].

        Returns:
            ``(p_static_filtered, p_dynamic_filtered)`` in Pa.
        """
        if self._prev_s is None:
            self._prev_s = p_static
            self._prev_d = p_dynamic
            return p_static, p_dynamic

        p_s_out = self._prev_s + self._alpha_s * (p_static  - self._prev_s)
        p_d_out = self._prev_d + self._alpha_d * (p_dynamic - self._prev_d)

        self._prev_s = p_s_out
        self._prev_d = p_d_out

        return p_s_out, p_d_out

    @property
    def alpha_static(self) -> float:
        """EMA coefficient for the static-pressure channel (after configure())."""
        return self._alpha_s

    @property
    def alpha_dynamic(self) -> float:
        """EMA coefficient for the dynamic-pressure channel (after configure())."""
        return self._alpha_d

    def __repr__(self) -> str:
        return (
            f"PressureEMAFilter(tau_static={self.tau_static}, "
            f"tau_dynamic={self.tau_dynamic})"
        )
