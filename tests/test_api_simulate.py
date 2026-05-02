"""Tests for POST /api/v1/simulate."""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_simulate_synthetic_returns_series_and_metadata():
    payload = {
        "synthetic_profile": {
            "duration": 2.0,
            "dt": 0.1,
            "max_altitude": 500.0,
            "max_speed": 50.0,
            "burnout_time": 1.0,
        },
        "noise_config": {"noise_type": "none", "params": {}},
    }
    r = client.post("/api/v1/simulate", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    n = data["metadata"]["n_steps"]
    assert len(data["altitude"]) == n
    assert len(data["predicted_apogee"]) == n
    assert data["metadata"]["noise_type"] == "NoNoiseModel"
