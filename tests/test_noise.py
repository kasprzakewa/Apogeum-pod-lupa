"""
Unit tests for the noise model module.
"""

import pytest

from src.noise.noise_model import NoiseModel, NoNoiseModel


class TestNoNoiseModel:
    def test_pass_through_static(self):
        model = NoNoiseModel()
        ps, _ = model.apply(101325.0, 102000.0)
        assert ps == pytest.approx(101325.0)

    def test_pass_through_total(self):
        model = NoNoiseModel()
        _, pt = model.apply(101325.0, 102000.0)
        assert pt == pytest.approx(102000.0)

    def test_enabled_is_false(self):
        assert NoNoiseModel().enabled is False

    def test_is_subclass_of_noise_model(self):
        assert isinstance(NoNoiseModel(), NoiseModel)

    def test_reset_does_not_raise(self):
        model = NoNoiseModel()
        model.reset(seed=42)

    def test_repr_contains_class_name(self):
        assert "NoNoiseModel" in repr(NoNoiseModel())
