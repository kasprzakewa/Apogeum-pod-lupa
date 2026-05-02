# Apogeum Pod Lupą

Software-in-the-loop (SIL) simulation environment for student rocket apogee prediction with sensor noise modelling and filter evaluation via Monte Carlo analysis.

## Overview

The project simulates a 1D vertical rocket flight using static and total pressure sensor readings. At each time step the engine:

1. Applies optional sensor noise (pneumatic lag, white noise, vibration, quantisation)
2. Optionally smooths raw pressures with a **pressure-domain EMA filter** (before barometric conversion)
3. Computes **altitude** from static pressure via the barometric formula (ISA)
4. Computes **airspeed** from differential pressure via Bernoulli's equation
5. Optionally estimates altitude and velocity with a **linear Kalman filter** (state-space)
6. Predicts **apogee** from the current kinematic state with aerodynamic drag correction

All three channels (raw, EMA-filtered, KF-filtered) are available simultaneously and exposed through the HTTP API.

## Project Structure

```
Apogeum-pod-lupa/
├── pyproject.toml
├── data/
│   ├── or_smaller_converted.csv   # Converted OpenRocket flight profile (recommended)
│   └── sample_flight.csv          # Legacy synthetic reference profile
├── src/
│   ├── models/
│   │   ├── constants.py           # Physical constants (P0, rho0, g, …)
│   │   └── physics.py             # altitude, speed, apogee prediction, air density
│   ├── simulation/
│   │   ├── flight_profile.py      # FlightProfile – CSV loader and synthetic generator
│   │   ├── engine.py              # SimulationEngine – discrete-time loop
│   │   └── result.py              # SimulationResult dataclass
│   ├── noise/
│   │   └── noise_model.py         # NoiseModel ABC + all concrete models
│   ├── filters/
│   │   ├── kalman.py              # Linear Kalman filter (altitude, velocity state)
│   │   └── pressure_filter.py     # Dual-channel EMA filter on raw pressures
│   ├── visualization/
│   │   ├── plots.py               # 2×2 simulation plot
│   │   └── mc_prediction_error_plot.py  # Monte Carlo error figure (raw / KF / EMA)
│   └── api/
│       ├── main.py                # FastAPI application
│       ├── routes.py              # Endpoint handlers
│       └── schemas.py             # Pydantic request/response models
├── scripts/
│   ├── or_csv_to_flight.py        # Convert OpenRocket CSV → (time, p_static, p_total)
│   ├── tune_kalman.py             # KF parameter grid search with MC evaluation
│   └── tune_ema.py                # EMA parameter grid search with MC evaluation
├── results/                       # Output plots from tuning scripts (git-ignored)
└── tests/
    ├── test_physics.py
    ├── test_simulation.py
    ├── test_noise.py
    ├── test_api_simulate.py
    └── test_api_prediction_error.py
```

## Requirements

- Python 3.12+
- [Poetry](https://python-poetry.org/) for dependency management

## Installation

```bash
poetry install
```

## Running the API

```bash
poetry run uvicorn src.api.main:app --reload
```

Interactive documentation:
- **http://127.0.0.1:8000/docs** — Swagger UI
- **http://127.0.0.1:8000/redoc** — ReDoc

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/v1/health` | Health check |
| `POST` | `/api/v1/simulate` | Run simulation, return JSON time-series |
| `POST` | `/api/v1/plot` | Run simulation, return PNG plot |
| `POST` | `/api/v1/monte-carlo/prediction-error` | Monte Carlo prediction-error analysis — JSON or PNG |

### POST /api/v1/simulate

Runs one simulation. Returns all three output channels (raw, EMA-filtered, KF-filtered) depending on which filters are enabled.

```bash
# Synthetic profile, Binczar noise, EMA pressure filter enabled
curl -s -X POST http://127.0.0.1:8000/api/v1/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "noise_config": {"noise_type": "binczar", "params": {}},
    "pressure_filter_config": {"enabled": true, "tau_static": 0.03, "tau_dynamic": 0.02},
    "synthetic_profile": {"duration": 60.0, "dt": 0.1, "max_altitude": 3000.0,
                          "max_speed": 300.0, "burnout_time": 5.0}
  }'
```

```bash
# OpenRocket CSV profile, no noise, no filter
curl -s -X POST http://127.0.0.1:8000/api/v1/simulate \
  -H "Content-Type: application/json" \
  -d '{"csv_path": "data/or_smaller_converted.csv"}'
```

### POST /api/v1/monte-carlo/prediction-error

Runs `n_runs` noisy simulations and compares `predicted_apogee` to a clean noise-free reference at every time step. Returns per-step statistics (mean, std, P05, P95) and scalar MAE/RMSE metrics, split into full-flight and post-burnout windows.

Set `response_format` to `"png"` to get the figure directly instead of JSON.

**Example — EMA filter with Binczar noise, PNG output:**

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/monte-carlo/prediction-error \
  -H "Content-Type: application/json" \
  -d '{
    "base_seed": 42,
    "n_runs": 50,
    "response_format": "png",
    "scatter_max_points": 50000,
    "include_all_runs": true,
    "figure_title": "MC: EMA filter vs raw",
    "simulation": {
      "csv_path": "data/or_smaller_converted.csv",
      "noise_config": {"noise_type": "binczar", "params": {}},
      "pressure_filter_config": {"enabled": true, "tau_static": 0.03, "tau_dynamic": 0.02}
    }
  }' --output results/mc_ema.png
```

**Example — Kalman filter enabled alongside EMA:**

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/monte-carlo/prediction-error \
  -H "Content-Type: application/json" \
  -d '{
    "base_seed": 42,
    "n_runs": 50,
    "response_format": "png",
    "include_all_runs": true,
    "simulation": {
      "csv_path": "data/or_smaller_converted.csv",
      "noise_config": {"noise_type": "binczar", "params": {}},
      "filter_config": {"enabled": true, "sigma_a": 30, "sigma_h": 20, "sigma_v": 30},
      "pressure_filter_config": {"enabled": true, "tau_static": 0.03, "tau_dynamic": 0.02}
    }
  }' --output results/mc_kf_ema.png
```

## Running Tests

```bash
poetry run pytest
poetry run pytest -v
```

## Tuning Scripts

### KF parameter grid search

```bash
poetry run python3 scripts/tune_kalman.py \
    --csv data/or_smaller_converted.csv \
    --n-mc 20 \
    --sigma-a 5 15 45 120 \
    --sigma-h 5 15 45 120 \
    --sigma-v 2 7 20 60 \
    --output results/kf_tune
```

Produces `results/kf_tune_heatmaps.png`, `_ranking.png`, `_timeseries.png`.

### EMA parameter grid search

```bash
poetry run python3 scripts/tune_ema.py \
    --csv data/or_smaller_converted.csv \
    --n-mc 20 \
    --tau-static 0.01 0.05 0.15 0.5 \
    --tau-dynamic 0.005 0.02 0.08 0.3 \
    --output results/ema_tune
```

Produces `results/ema_tune_heatmap.png`, `_ranking.png`, `_timeseries.png`.

### OpenRocket CSV conversion

```bash
poetry run python3 scripts/or_csv_to_flight.py \
    data/openrocket_export.csv \
    --output data/or_smaller_converted.csv \
    --method isentropic
```

Converts an OpenRocket CSV (columns: time, altitude, vertical velocity, Mach, air pressure) to the `(time, static_pressure, total_pressure)` format required by the simulation engine.

## Physical Model

### Altitude (barometric formula, ISA)

```
h = 44300 · (1 − (p / p₀)^0.190263)
```

### Airspeed (Bernoulli)

```
rho(h) = rho0 · (1 − h / 44300)^4.256
v = sqrt(2 · dp / rho(h))
```

### Apogee prediction (with drag)

```
Fd = q · A · C_D

if Fd > 0:
    h_apogee = h + (m · v² · ln(1 + Fd / (m·g))) / (2·Fd)
else:
    h_apogee = h + v² / (2·g)
```

### Default model parameters

| Symbol | Value | Description |
|--------|-------|-------------|
| p₀ | 101 325 Pa | Sea-level reference pressure |
| ρ₀ | 1.225 kg/m³ | Sea-level air density |
| g | 9.81 m/s² | Gravitational acceleration |
| A | 0.018 146 m² | Rocket cross-sectional area |
| C_D | 0.6 | Drag coefficient |
| m | 50 kg | Rocket mass |

All parameters are configurable via `ModelParamsSchema` in the API request.

## Noise Models

| Class | Description |
|-------|-------------|
| `NoNoiseModel` | Pass-through, ideal sensors |
| `BinczarNoiseModel` | Realistic model: pneumatic lag, white noise, vibration, temperature drift, ADC quantisation |
| `IdealGaussianNoiseModel` | Independent Gaussian noise on static and dynamic pressure only |
| `GaussianNoiseModel` | Independent Gaussian noise on static and total pressure |

## Filters

| Class | Input domain | Notes |
|-------|-------------|-------|
| `PressureEMAFilter` | Raw sensor pressures (Pa) | Applied before barometric conversion; lag stays in Pa, not amplified by v² |
| `KalmanFilter` | Altitude + velocity (m, m/s) | Linear 2-state KF; performs poorly at large dt (>0.1 s) due to v² amplification |

Default EMA parameters (`tau_static=0.03 s`, `tau_dynamic=0.02 s`) were tuned via Monte Carlo grid search on `or_smaller_converted.csv` with `BinczarNoiseModel`, yielding ~11% MAE reduction over raw noisy predictions in the post-burnout window.

## License

Apache License 2.0 – see [LICENSE](LICENSE) for details.
