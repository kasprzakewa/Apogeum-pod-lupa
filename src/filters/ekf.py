"""
Extended Kalman Filter (EKF) - placeholder module.

This module defines the intended interface for the EKF that will be used
to estimate rocket state (altitude, velocity) and predict apogee with
improved accuracy compared to the instantaneous physics model.

Planned state vector:
    x = [altitude [m], velocity [m/s], drag_coefficient [-]]

Planned measurement vector:
    z = [static_pressure [Pa], dynamic_pressure [Pa]]

TODO:
    1. Implement state transition function f(x) using the 1D flight model.
    2. Implement measurement function h(x) mapping state → pressures.
    3. Implement Jacobian F = ∂f/∂x (or use numerical differentiation).
    4. Implement Jacobian H = ∂h/∂x.
    5. Tune process noise Q and measurement noise R covariance matrices.
    6. Validate against Monte Carlo simulation results.
"""

from __future__ import annotations

import numpy as np


class ExtendedKalmanFilter:
    """
    EKF for 1D rocket apogee prediction.

    State vector  x ∈ ℝ³: [altitude, velocity, drag_coefficient]
    Measurement   z ∈ ℝ²: [static_pressure, dynamic_pressure]

    Args:
        x0: Initial state estimate, shape (3,).
        P0: Initial state covariance matrix, shape (3, 3).
        Q:  Process noise covariance matrix, shape (3, 3).
        R:  Measurement noise covariance matrix, shape (2, 2).
        dt: Time step [s].
    """

    def __init__(
        self,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        dt: float = 0.01,
    ) -> None:
        self.dt = dt
        self.x = x0 if x0 is not None else np.zeros(3)
        self.P = P0 if P0 is not None else np.eye(3)
        self.Q = Q if Q is not None else np.eye(3) * 1e-3
        self.R = R if R is not None else np.eye(2) * 1e1

    def predict(self) -> None:
        """
        EKF prediction step (time update).

        Propagates the state estimate and covariance forward by dt using
        the nonlinear flight model and its linearised Jacobian.

        Raises:
            NotImplementedError: Until implemented.
        """
        raise NotImplementedError(
            "EKF predict() is not yet implemented. "
            "Implement the state transition f(x) and Jacobian F = ∂f/∂x."
        )

    def update(self, z: np.ndarray) -> None:
        """
        EKF update step (measurement update).

        Corrects the state estimate using a new pressure measurement vector z.

        Args:
            z: Measurement vector [static_pressure, dynamic_pressure] [Pa].

        Raises:
            NotImplementedError: Until implemented.
        """
        raise NotImplementedError(
            "EKF update() is not yet implemented. "
            "Implement the measurement function h(x) and Jacobian H = ∂h/∂x."
        )

    def step(self, z: np.ndarray) -> np.ndarray:
        """
        Run one predict-update cycle.

        Args:
            z: Measurement vector [static_pressure, dynamic_pressure] [Pa].

        Returns:
            Current state estimate x after the update.

        Raises:
            NotImplementedError: Until implemented.
        """
        self.predict()
        self.update(z)
        return self.x.copy()

    def estimated_apogee(self) -> float:
        """
        Compute predicted apogee from current EKF state estimate.

        Returns:
            Predicted apogee altitude [m].

        Raises:
            NotImplementedError: Until implemented.
        """
        raise NotImplementedError(
            "estimated_apogee() requires predict() and update() to be implemented first."
        )
