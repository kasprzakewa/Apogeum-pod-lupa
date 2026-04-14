"""
Monte Carlo simulation runner - placeholder module.

This module defines the intended interface for running repeated simulations
with randomised noise parameters to analyse apogee prediction uncertainty.

Planned workflow:
    1. Define a distribution over noise parameters (sigma_static, sigma_total, etc.).
    2. Run N independent simulations, each with a different noise realisation.
    3. Collect final_apogee_prediction from each SimulationResult.
    4. Compute statistics: mean, std, percentiles, histogram.
    5. (Optional) Vary model parameters (mass, drag) to analyse sensitivity.

TODO:
    1. Implement MCConfig dataclass for parameter distributions.
    2. Implement MonteCarloRunner.run() with parallel execution (concurrent.futures).
    3. Implement MCResult with aggregation and statistics methods.
    4. Add sensitivity analysis (Sobol indices or one-at-a-time).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class MCConfig:
    """
    Configuration for a Monte Carlo experiment.

    TODO: Replace fixed sigma values with full distribution specs
    (scipy.stats frozen distributions) for arbitrary parameter uncertainty.

    Args:
        n_runs:       Number of Monte Carlo runs.
        sigma_static: Std. deviation of Gaussian noise on static pressure [Pa].
        sigma_total:  Std. deviation of Gaussian noise on total pressure [Pa].
        base_seed:    Base RNG seed; each run uses base_seed + run_index.
    """

    n_runs: int = 1000
    sigma_static: float = 10.0
    sigma_total: float = 15.0
    base_seed: int = 42


@dataclass
class MCResult:
    """
    Aggregated results from a Monte Carlo experiment.

    Args:
        apogee_predictions: Array of final apogee predictions from each run [m].
        config:             The MCConfig used to produce these results.
    """

    apogee_predictions: np.ndarray
    config: MCConfig

    @property
    def mean(self) -> float:
        return float(np.mean(self.apogee_predictions))

    @property
    def std(self) -> float:
        return float(np.std(self.apogee_predictions))

    @property
    def p05(self) -> float:
        """5th percentile."""
        return float(np.percentile(self.apogee_predictions, 5))

    @property
    def p95(self) -> float:
        """95th percentile."""
        return float(np.percentile(self.apogee_predictions, 95))

    def summary(self) -> dict:
        return {
            "n_runs": self.config.n_runs,
            "mean_apogee_m": self.mean,
            "std_apogee_m": self.std,
            "p05_apogee_m": self.p05,
            "p95_apogee_m": self.p95,
        }


class MonteCarloRunner:
    """
    Runs repeated simulations with randomised noise to estimate apogee uncertainty.

    Args:
        simulation_factory: Callable that accepts a seed (int) and returns a
                            configured SimulationEngine ready to run.
        config:             MCConfig specifying the number of runs and noise params.

    Example (intended usage after implementation)::

        def factory(seed):
            profile = FlightProfile.synthetic()
            noise = GaussianNoiseModel(sigma_static=10, sigma_total=15, seed=seed)
            return SimulationEngine(profile, noise_model=noise)

        runner = MonteCarloRunner(factory, MCConfig(n_runs=500))
        result = runner.run()
        print(result.summary())
    """

    def __init__(
        self,
        simulation_factory: Callable[[int], object],
        config: MCConfig | None = None,
    ) -> None:
        self.simulation_factory = simulation_factory
        self.config = config or MCConfig()

    def run(self) -> MCResult:
        """
        Execute all Monte Carlo runs sequentially and collect results.

        Returns:
            MCResult with aggregated apogee predictions.

        Raises:
            NotImplementedError: Until implemented.
        """
        raise NotImplementedError(
            "MonteCarloRunner.run() is not yet implemented. "
            "Implement sequential or parallel execution of self.simulation_factory(seed) "
            "for seed in range(self.config.base_seed, self.config.base_seed + self.config.n_runs), "
            "collect result.final_apogee_prediction from each run, "
            "and return MCResult(apogee_predictions=..., config=self.config)."
        )
