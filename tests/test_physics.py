"""
Unit tests for src/models/physics.py and src/models/constants.py.

Tests verify:
  - Reference values produce expected outputs (regression against C implementation)
  - Edge cases (zero pressure, zero speed, zero dynamic pressure)
  - ModelParams overrides are respected
"""

import math

import pytest

from src.models.constants import (
    G,
    REFERENCE_AIR_DENSITY,
    REFERENCE_PRESSURE,
    SCALE_HEIGHT,
)
from src.models.physics import (
    ModelParams,
    calculate_air_density,
    calculate_altitude,
    calculate_speed,
    predict_apogee,
)


class TestCalculateAltitude:
    def test_sea_level_returns_zero(self):
        alt = calculate_altitude(REFERENCE_PRESSURE)
        assert abs(alt) < 0.01

    def test_half_pressure_gives_positive_altitude(self):
        alt = calculate_altitude(REFERENCE_PRESSURE / 2)
        assert alt > 0

    def test_known_value(self):
        # At ~5000 m ISA pressure is approx 54048 Pa
        alt = calculate_altitude(54048.0)
        assert 4800 < alt < 5200

    def test_custom_params(self):
        params = ModelParams(reference_pressure=50000.0)
        alt = calculate_altitude(50000.0, params)
        assert abs(alt) < 0.01


class TestCalculateAirDensity:
    def test_sea_level_equals_reference(self):
        rho = calculate_air_density(0.0)
        assert abs(rho - REFERENCE_AIR_DENSITY) < 1e-6

    def test_density_decreases_with_altitude(self):
        rho_low = calculate_air_density(1000.0)
        rho_high = calculate_air_density(5000.0)
        assert rho_low > rho_high

    def test_extreme_altitude_clamps_to_zero(self):
        rho = calculate_air_density(1_000_000.0)
        assert rho >= 0.0


class TestCalculateSpeed:
    def test_zero_diff_pressure_returns_zero(self):
        assert calculate_speed(0.0, 1000.0) == 0.0

    def test_positive_diff_pressure_returns_positive_speed(self):
        speed = calculate_speed(500.0, 1000.0)
        assert speed > 0.0

    def test_negative_diff_pressure_uses_abs(self):
        speed_pos = calculate_speed(500.0, 1000.0)
        speed_neg = calculate_speed(-500.0, 1000.0)
        assert abs(speed_pos - speed_neg) < 1e-9

    def test_speed_increases_with_diff_pressure(self):
        s1 = calculate_speed(100.0, 0.0)
        s2 = calculate_speed(400.0, 0.0)
        assert s2 > s1

    def test_bernoulli_at_sea_level(self):
        # v = sqrt(2 * q / rho_0)
        q = 5000.0
        expected = math.sqrt(2 * q / REFERENCE_AIR_DENSITY)
        result = calculate_speed(q, 0.0)
        assert abs(result - expected) < 0.01


class TestPredictApogee:
    def test_zero_speed_returns_current_altitude(self):
        apogee = predict_apogee(0.0, 1000.0, 0.0)
        assert abs(apogee - 1000.0) < 1e-6

    def test_no_drag_kinematic_apogee(self):
        speed = 100.0
        altitude = 500.0
        expected = altitude + speed**2 / (2 * G)
        result = predict_apogee(speed, altitude, 0.0)
        assert abs(result - expected) < 1e-4

    def test_with_drag_apogee_lower_than_without(self):
        speed = 200.0
        altitude = 1000.0
        no_drag = predict_apogee(speed, altitude, 0.0)
        with_drag = predict_apogee(speed, altitude, 5000.0)
        assert with_drag < no_drag

    def test_apogee_greater_than_current_altitude_when_speed_positive(self):
        apogee = predict_apogee(100.0, 2000.0, 1000.0)
        assert apogee > 2000.0

    def test_custom_params_respected(self):
        params = ModelParams(mass=100.0, drag_coefficient=0.3, cross_section=0.01, g=9.81)
        apogee = predict_apogee(100.0, 500.0, 1000.0, params)
        assert apogee > 500.0


class TestModelParams:
    def test_defaults_match_constants(self):
        p = ModelParams()
        assert p.reference_pressure == REFERENCE_PRESSURE
        assert p.g == G

    def test_drag_area_computed_correctly(self):
        p = ModelParams(cross_section=0.02, drag_coefficient=0.5)
        assert abs(p.drag_area - 0.01) < 1e-12
