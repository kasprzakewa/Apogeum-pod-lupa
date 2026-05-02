"""
Linear Kalman Filter for 1-D rocket flight state estimation.

State vector  x = [altitude, velocity]^T  (2 × 1)
Measurements  z = [altitude_meas, velocity_meas]^T  (2 × 1)

    x_{k+1} = F x_k + w_k,   w_k ~ N(0, Q)
    z_k     = H x_k + v_k,   v_k ~ N(0, R)

    F = [[1, dt],     H = I_2
         [0,  1]]

    Q = sigma_a^2 * [[dt^4/4, dt^3/2],
                     [dt^3/2, dt^2  ]]

    R = diag(sigma_h^2, sigma_v^2)
"""

from __future__ import annotations

import numpy as np


class KalmanFilter:
    """
    Linear Kalman Filter for rocket altitude and velocity estimation.

    The filter is called once per time step via :meth:`update`, which executes
    both the predict and measurement-update steps.  Before the first simulation
    call :meth:`configure` with the flight-profile dt so that the
    time-invariant matrices (F, Q) are precomputed.

    Args:
        sigma_a:   Std dev of process acceleration noise [m/s²].
        sigma_h:   Measurement noise std for altitude [m].
        sigma_v:   Measurement noise std for velocity [m/s].
        init_p:    Initial state covariance diagonal [m² or (m/s)²].
    """

    def __init__(
        self,
        sigma_a: float = 30.0,
        sigma_h: float = 20.0,
        sigma_v: float = 5.0,
        init_p: float = 500.0,
    ) -> None:
        self.sigma_a = float(sigma_a)
        self.sigma_h = float(sigma_h)
        self.sigma_v = float(sigma_v)
        self.init_p = float(init_p)

        self._x: np.ndarray | None = None   # state [h, v], shape (2,)
        self._P: np.ndarray | None = None   # covariance (2 × 2)
        self._dt: float = 0.0
        self._F: np.ndarray = np.eye(2)
        self._Q: np.ndarray = np.zeros((2, 2))
        self._R: np.ndarray = np.zeros((2, 2))

    def configure(self, dt: float) -> None:
        """Pre-compute time-invariant matrices for a given simulation time step."""
        self._dt = float(dt)
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

        self._F = np.array([[1.0, dt],
                             [0.0, 1.0]])

        self._Q = self.sigma_a ** 2 * np.array([
            [dt4 / 4.0, dt3 / 2.0],
            [dt3 / 2.0, dt2],
        ])

        self._R = np.diag([self.sigma_h ** 2, self.sigma_v ** 2])

    def reset(self) -> None:
        """Clear internal state (call before re-using the filter for a new flight)."""
        self._x = None
        self._P = None

    def update(self, altitude_meas: float, velocity_meas: float) -> tuple[float, float]:
        """
        Execute one predict-then-update cycle and return the filtered state.

        On the very first call the filter is initialised with the measurement
        directly (infinite initial uncertainty → Kalman gain = I).

        Args:
            altitude_meas: Altitude derived from noisy static pressure [m].
            velocity_meas: Velocity derived from noisy differential pressure [m/s].

        Returns:
            ``(filtered_altitude, filtered_velocity)`` tuple.
        """
        z = np.array([altitude_meas, velocity_meas])

        if self._x is None:
            self._x = z.copy()
            self._P = np.eye(2) * self.init_p
            return float(self._x[0]), float(self._x[1])

        x_pred = self._F @ self._x
        P_pred = self._F @ self._P @ self._F.T + self._Q

        # H = I, so innovation covariance S = P_pred + R
        S = P_pred + self._R
        K = P_pred @ np.linalg.inv(S)
        self._x = x_pred + K @ (z - x_pred)
        self._P = (np.eye(2) - K) @ P_pred

        return float(self._x[0]), float(self._x[1])

    @property
    def state(self) -> np.ndarray | None:
        """Current state estimate [altitude, velocity], or None before first call."""
        return self._x.copy() if self._x is not None else None

    @property
    def covariance(self) -> np.ndarray | None:
        """Current error covariance matrix (2 × 2), or None before first call."""
        return self._P.copy() if self._P is not None else None

    def __repr__(self) -> str:
        return (
            f"KalmanFilter(sigma_a={self.sigma_a}, "
            f"sigma_h={self.sigma_h}, sigma_v={self.sigma_v})"
        )
