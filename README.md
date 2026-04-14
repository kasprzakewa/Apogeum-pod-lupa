# Apogeum Pod Lupą

Software-in-the-loop (SIL) simulation environment for student rocket apogee prediction.

## Overview

The project simulates a 1D vertical rocket flight using static and total pressure sensor readings. It computes:

- **Altitude** – from static pressure via the barometric formula (ISA model)
- **Airspeed** – from differential pressure via Bernoulli's equation
- **Predicted apogee** – from current kinematic state with aerodynamic drag correction

The architecture is designed for future extension with an **Extended Kalman Filter (EKF)** and **Monte Carlo** uncertainty analysis.

## Project Structure

```
Apogeum-pod-lupa/
├── pyproject.toml              # Poetry project and dependency configuration
├── poetry.lock                 # Locked dependency versions
├── data/
│   └── sample_flight.csv       # Example synthetic flight profile (600 steps, dt=0.1 s)
├── src/
│   ├── models/
│   │   ├── constants.py        # Physical constants (P₀, ρ₀, g, …)
│   │   └── physics.py          # Core functions: altitude, speed, apogee prediction
│   ├── simulation/
│   │   ├── flight_profile.py   # FlightProfile – CSV loader and synthetic generator
│   │   ├── engine.py           # SimulationEngine – discrete-time loop
│   │   └── result.py           # SimulationResult dataclass
│   ├── noise/
│   │   └── noise_model.py      # NoiseModel (abstract base) + NoNoiseModel
│   ├── filters/
│   │   └── ekf.py              # ExtendedKalmanFilter – interface placeholder
│   ├── monte_carlo/
│   │   └── runner.py           # MonteCarloRunner – interface placeholder
│   ├── visualization/
│   │   └── plots.py            # Fixed 2×2 matplotlib plot layout
│   └── api/
│       ├── main.py             # FastAPI application
│       ├── routes.py           # Endpoint handlers
│       └── schemas.py          # Pydantic request/response models
└── tests/
    ├── test_physics.py         # Unit tests for physics functions
    ├── test_simulation.py      # Unit tests for engine and flight profile
    └── test_noise.py           # Unit tests for noise models
```

## Requirements

- Python 3.12+
- [Poetry](https://python-poetry.org/) for dependency management

## Installation

```bash
# Install all dependencies and create virtual environment
poetry install

# Activate the virtual environment
poetry shell
```

## Running the API

```bash
poetry run uvicorn src.api.main:app --reload
```

Interactive documentation is available at:
- **[http://127.0.0.1:8000](http://127.0.0.1:8000)** → redirects to Swagger UI
- **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** → Swagger UI
- **[http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)** → ReDoc

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/v1/health` | Health check |
| `POST` | `/api/v1/simulate` | Run simulation, return JSON time-series |
| `POST` | `/api/v1/plot` | Run simulation, return PNG plot |

### POST /api/v1/simulate

Run a simulation and get raw time-series data.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "synthetic_profile": {
      "duration": 60.0,
      "dt": 0.1,
      "max_altitude": 3000.0,
      "max_speed": 300.0,
      "burnout_time": 5.0
    }
  }'
```

Using a CSV file instead of the synthetic profile:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/simulate \
  -H "Content-Type: application/json" \
  -d '{"csv_path": "data/sample_flight.csv"}'
```

### POST /api/v1/plot

Run a simulation and return a **PNG image** with a fixed 2×2 panel layout:

```
┌─────────────────────────────┬──────────────────────┐
│  Altitude + Predicted Apogee │        Speed         │
├─────────────────────────────┼──────────────────────┤
│      Static Pressure        │   Dynamic Pressure   │
└─────────────────────────────┴──────────────────────┘
```

```bash
# Basic plot (ideal simulation)
curl -X POST http://127.0.0.1:8000/api/v1/plot \
  -H "Content-Type: application/json" \
  -d '{"title": "Rocket SIL Simulation"}' \
  --output plot.png
```

## Running Tests

```bash
poetry run pytest

# With verbose output
poetry run pytest -v
```

## Using the Python API Directly

```python
from src.simulation import FlightProfile, SimulationEngine
from src.visualization import plot_simulation

# Create flight profile (synthetic or from CSV)
profile = FlightProfile.synthetic(duration=60.0, max_altitude=3000.0)
# profile = FlightProfile.from_csv("data/sample_flight.csv")

# Run simulation
result = SimulationEngine(profile).run()

# Plot – fixed 2×2 layout
fig = plot_simulation(result, title="Rocket SIL Simulation")
fig.savefig("simulation.png", dpi=150)
fig.show()
```

## Physical Model

### Altitude (barometric formula, ISA)

```
h = 44300 · (1 − (p / p₀)^0.190263)
```

### Airspeed (Bernoulli)

```
ρ(h) = ρ₀ · (1 − h / 44300)^4.256

v = sqrt(2 · Δp / ρ(h))
```

### Apogee prediction

```
Fd = q · A · C_D          (aerodynamic drag force)

if Fd > 0:
    h_apogee = h + (m · v² · ln(1 + Fd / (m·g))) / (2 · Fd)
else:
    h_apogee = h + v² / (2·g)
```

### Default constants

| Symbol | Value | Description |
|--------|-------|-------------|
| p₀ | 101 325 Pa | Sea-level reference pressure |
| ρ₀ | 1.225 kg/m³ | Sea-level air density |
| g | 9.81 m/s² | Gravitational acceleration |
| A | 0.018 146 m² | Rocket cross-sectional area |
| C_D | 0.6 | Drag coefficient |
| m | 50 kg | Rocket mass |

All parameters are configurable via `ModelParams`.

## Noise Models

Currently only the pass-through (no-noise) model is implemented.
Concrete noise models will be added in future iterations.

| Class | Description |
|-------|-------------|
| `NoiseModel` | Abstract base class – defines the `apply()` interface |
| `NoNoiseModel` | Pass-through, ideal sensors (default) |

## Roadmap

- [ ] Implement concrete noise models
- [ ] Implement Extended Kalman Filter (`src/filters/ekf.py`)
- [ ] Implement Monte Carlo runner (`src/monte_carlo/runner.py`)
- [ ] Add sensitivity analysis (Sobol indices)
- [ ] Load real flight telemetry from hardware sensors
- [ ] Add CI/CD pipeline with automated tests

## License

Apache License 2.0 – see [LICENSE](LICENSE) for details.
