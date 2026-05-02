"""Tests for POST /api/v1/monte-carlo/prediction-error."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_prediction_error_synthetic_gaussian_small_mc():
    payload = {
        "simulation": {
            "synthetic_profile": {
                "duration": 10.0,
                "dt": 0.1,
                "max_altitude": 1000.0,
                "max_speed": 100.0,
                "burnout_time": 2.0,
            },
            "noise_config": {"noise_type": "gaussian", "params": {"sigma_static": 5.0, "sigma_total": 8.0}},
        },
        "n_runs": 5,
        "base_seed": 1,
        "include_all_runs": True,
        "scatter_max_points": 5000,
    }
    r = client.post("/api/v1/monte-carlo/prediction-error", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "reference_apogee_m" in data
    assert abs(data["burnout_time_used_s"] - 2.0) <= 0.2
    assert len(data["time"]) == len(data["signed_error"]["mean"])
    assert data["mean_abs_error_full_flight_m"] >= 0.0
    assert data["mean_abs_error_post_burnout_m"] is not None
    assert len(data["scatter_time"]) == len(data["scatter_signed_error"])


def test_prediction_error_response_format_png():
    payload = {
        "simulation": {
            "synthetic_profile": {
                "duration": 8.0,
                "dt": 0.2,
                "max_altitude": 800.0,
                "max_speed": 80.0,
                "burnout_time": 1.5,
            },
            "noise_config": {"noise_type": "gaussian", "params": {"sigma_static": 3.0, "sigma_total": 5.0}},
        },
        "n_runs": 4,
        "base_seed": 0,
        "response_format": "png",
        "figure_dpi": 100,
    }
    r = client.post("/api/v1/monte-carlo/prediction-error", json=payload)
    assert r.status_code == 200, r.text
    assert r.headers.get("content-type", "").startswith("image/png")
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_prediction_error_csv_uses_clean_reference_burnout(tmp_path):
    csv_content = (
        "time,static_pressure,total_pressure\n"
        "0,101325,101325\n"
        "1,100000,101000\n"
    )
    p = tmp_path / "short.csv"
    p.write_text(csv_content)
    payload = {
        "simulation": {
            "csv_path": str(p),
            "noise_config": {"noise_type": "none", "params": {}},
        },
        "n_runs": 2,
        "base_seed": 0,
    }
    r = client.post("/api/v1/monte-carlo/prediction-error", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["burnout_time_used_s"] is not None
    assert data["burnout_time_used_s"] == 1.0
    assert data["mean_abs_error_post_burnout_m"] is not None
    assert data["rmse_post_burnout_m"] is not None


def test_prediction_error_post_burnout_metrics_use_burnout_to_apogee_window():
    payload = {
        "simulation": {
            "synthetic_profile": {
                "duration": 60.0,
                "dt": 0.05,
                "max_altitude": 1200.0,
                "max_speed": 150.0,
                "burnout_time": 2.5,
            },
            "noise_config": {"noise_type": "none", "params": {}},
        },
        "n_runs": 2,
        "base_seed": 0,
    }
    r = client.post("/api/v1/monte-carlo/prediction-error", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()

    t_b = data["burnout_time_used_s"]
    t_a = data["apogee_time_used_s"]
    assert t_b is not None
    assert t_a is not None
    assert t_b <= t_a

    # For noise_type=none all runs are identical, so MAE/RMSE can be reconstructed
    # from signed_error.mean for the same time windows used by the API.
    t = data["time"]
    e = data["signed_error"]["mean"]

    full_window = [abs(v) for tt, v in zip(t, e) if tt <= t_a]
    assert len(full_window) > 0
    expected_full_mae = sum(full_window) / len(full_window)
    full_window_sq = [v * v for tt, v in zip(t, e) if tt <= t_a]
    expected_full_rmse = (sum(full_window_sq) / len(full_window_sq)) ** 0.5

    assert abs(data["mean_abs_error_full_flight_m"] - expected_full_mae) < 1e-6
    assert abs(data["rmse_full_flight_m"] - expected_full_rmse) < 1e-6

    window = [abs(v) for tt, v in zip(t, e) if t_b <= tt <= t_a]
    assert len(window) > 0
    expected_mae = sum(window) / len(window)

    window_sq = [v * v for tt, v in zip(t, e) if t_b <= tt <= t_a]
    expected_rmse = (sum(window_sq) / len(window_sq)) ** 0.5

    assert abs(data["mean_abs_error_post_burnout_m"] - expected_mae) < 1e-6
    assert abs(data["rmse_post_burnout_m"] - expected_rmse) < 1e-6
