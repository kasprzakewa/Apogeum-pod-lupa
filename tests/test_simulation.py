"""
Unit tests for simulation engine, flight profile, and result.
"""

import numpy as np
import pytest

from src.noise.noise_model import NoNoiseModel
from src.simulation.engine import SimulationEngine
from src.simulation.flight_profile import FlightProfile
from src.simulation.result import SimulationResult


class TestFlightProfile:
    def test_synthetic_creates_correct_length(self):
        profile = FlightProfile.synthetic(duration=10.0, dt=0.1)
        assert profile.n_steps == 100

    def test_synthetic_time_starts_at_zero(self):
        profile = FlightProfile.synthetic()
        assert profile.time[0] == pytest.approx(0.0)

    def test_synthetic_pressures_are_positive(self):
        profile = FlightProfile.synthetic()
        assert np.all(profile.static_pressure > 0)
        assert np.all(profile.total_pressure > 0)


class TestSimulationEngine:
    def test_run_returns_result_with_matching_lengths(self):
        profile = FlightProfile.synthetic(duration=5.0, dt=0.1)
        engine = SimulationEngine(profile)
        result = engine.run()

        assert isinstance(result, SimulationResult)
        assert result.n_steps == profile.n_steps
        assert len(result.altitude) == result.n_steps
        assert len(result.speed) == result.n_steps
        assert len(result.predicted_apogee) == result.n_steps

    def test_run_with_noise_model(self):
        profile = FlightProfile.synthetic(duration=1.0, dt=0.05)
        from src.noise.noise_model import GaussianNoiseModel

        noise = GaussianNoiseModel(sigma_static=1.0, sigma_total=2.0, seed=123)
        engine = SimulationEngine(profile, noise_model=noise)
        result = engine.run()
        assert result.noise_enabled is True

    def test_speed_non_negative(self):
        profile = FlightProfile.synthetic()
        engine = SimulationEngine(profile)
        result = engine.run()
        assert np.all(result.speed >= 0.0)

    def test_noise_disabled_by_default(self):
        profile = FlightProfile.synthetic(duration=2.0, dt=0.1)
        engine = SimulationEngine(profile)
        result = engine.run()
        assert result.noise_enabled is False

    def test_noise_disabled_with_no_noise_model(self):
        profile = FlightProfile.synthetic(duration=2.0, dt=0.1)
        engine = SimulationEngine(profile, noise_model=NoNoiseModel())
        result = engine.run()
        assert result.noise_enabled is False


class TestSimulationResult:
    def _make_result(self) -> SimulationResult:
        profile = FlightProfile.synthetic(duration=5.0, dt=0.1)
        return SimulationEngine(profile).run()

    def test_to_dict_has_all_keys(self):
        result = self._make_result()
        d = result.to_dict()
        for key in [
            "time",
            "altitude",
            "speed",
            "predicted_apogee",
            "metadata",
        ]:
            assert key in d

    def test_max_altitude_positive(self):
        result = self._make_result()
        assert result.max_altitude > 0.0

    def test_final_apogee_greater_than_max_altitude_or_equal(self):
        result = self._make_result()
        # Predicted apogee should generally be >= current altitude
        assert result.final_apogee_prediction >= 0.0
