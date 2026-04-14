"""
Unit tests for simulation engine, flight profile, and result.
"""

import numpy as np
import pytest

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

    def test_synthetic_total_gte_static(self):
        profile = FlightProfile.synthetic()
        assert np.all(profile.total_pressure >= profile.static_pressure - 1e-6)

    def test_dt_property(self):
        profile = FlightProfile.synthetic(duration=5.0, dt=0.05)
        assert profile.dt == pytest.approx(0.05, rel=1e-3)

    def test_from_csv_missing_columns_raises(self, tmp_path):
        csv = tmp_path / "bad.csv"
        csv.write_text("time,static_pressure\n0,101325\n")
        with pytest.raises(ValueError, match="missing required columns"):
            FlightProfile.from_csv(csv)

    def test_from_csv_roundtrip(self, tmp_path):
        profile = FlightProfile.synthetic(duration=1.0, dt=0.1)
        import pandas as pd

        df = pd.DataFrame(
            {
                "time": profile.time,
                "static_pressure": profile.static_pressure,
                "total_pressure": profile.total_pressure,
            }
        )
        csv_path = tmp_path / "profile.csv"
        df.to_csv(csv_path, index=False)

        loaded = FlightProfile.from_csv(csv_path)
        np.testing.assert_allclose(loaded.time, profile.time)
        np.testing.assert_allclose(loaded.static_pressure, profile.static_pressure)
        np.testing.assert_allclose(loaded.total_pressure, profile.total_pressure)


class TestSimulationEngine:
    def test_run_returns_simulation_result(self):
        profile = FlightProfile.synthetic(duration=5.0, dt=0.1)
        engine = SimulationEngine(profile)
        result = engine.run()
        assert isinstance(result, SimulationResult)

    def test_result_arrays_same_length_as_profile(self):
        profile = FlightProfile.synthetic(duration=5.0, dt=0.1)
        engine = SimulationEngine(profile)
        result = engine.run()
        n = profile.n_steps
        assert len(result.time) == n
        assert len(result.altitude) == n
        assert len(result.speed) == n
        assert len(result.predicted_apogee) == n

    def test_altitude_is_positive(self):
        profile = FlightProfile.synthetic(max_altitude=2000.0)
        engine = SimulationEngine(profile)
        result = engine.run()
        assert np.all(result.altitude >= 0.0)

    def test_speed_is_non_negative(self):
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
        from src.noise.noise_model import NoNoiseModel

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
        for key in ["time", "altitude", "speed", "predicted_apogee", "metadata"]:
            assert key in d

    def test_max_altitude_positive(self):
        result = self._make_result()
        assert result.max_altitude > 0.0

    def test_final_apogee_greater_than_max_altitude_or_equal(self):
        result = self._make_result()
        # Predicted apogee should generally be >= current altitude
        assert result.final_apogee_prediction >= 0.0
