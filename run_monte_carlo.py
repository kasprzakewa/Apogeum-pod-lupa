#!/usr/bin/env python3
"""
Przykład uruchomienia symulacji Monte Carlo.
Mozna uzyt danych syntetycznych lub rzeczywistych z CSV.
"""

import numpy as np
from pathlib import Path
from functools import partial
from src.monte_carlo.runner import MonteCarloRunner, MCConfig
from src.simulation.flight_profile import FlightProfile
from src.simulation.engine import SimulationEngine
from src.noise.noise_model import BinczarNoiseModel

# KONFIGURACJA - zmien USE_CSV na True aby uzyc rzeczywistych danych
USE_CSV = True
CSV_PATH = "data/or_flight.csv"


def create_simulation_factory(flight_profile):
    """Funkcja zwracajaca factory z zaladowanym profilem lotu."""
    def simulation_factory(seed: int):
        """Funkcja fabryczna z realistycznym szumem i zaladowanym profilem."""
        noise_config = {
            "static_rms_base":  (2.0, 0.8),
            "dynamic_rms_base": (5.0, 1.5),
            "tau_lag_range":    (0.01, 0.08),
            "vib_sens_range":   (0.2, 2.0),
            "temp_drift_range": (-0.5, 0.5),
            "resolution":       1.5,
        }
        
        noise = BinczarNoiseModel(
            config=noise_config,
            accel_g=0.0,
            temp_c=20.0,
            seed=seed
        )
        return SimulationEngine(flight_profile, noise_model=noise)
    
    return simulation_factory


def main():
    """Glowna funkcja uruchamiajaca symulacje Monte Carlo."""
    print("Uruchamiam symulacje Monte Carlo...\n")
    
    # Zaladownanie profilu lotu
    if USE_CSV:
        if not Path(CSV_PATH).exists():
            print(f"BLAD: Plik {CSV_PATH} nie istnieje!")
            return
        
        try:
            profile = FlightProfile.from_csv(CSV_PATH)
            print(f"Zaladowano dane z: {CSV_PATH}")
            print(f"  - Liczba krokow: {profile.n_steps}")
            print(f"  - Czas lotu: {profile.time[-1]:.2f} s")
            print(f"  - Time step (dt): {profile.dt:.4f} s\n")
        except Exception as e:
            print(f"BLAD przy zaladowaniu CSV: {e}")
            return
    else:
        profile = FlightProfile.synthetic()
        print(f"Uzyte dane syntetyczne")
        print(f"  - Liczba krokow: {profile.n_steps}")
        print(f"  - Czas lotu: {profile.time[-1]:.2f} s")
        print(f"  - Time step (dt): {profile.dt:.4f} s\n")
    
    # Konfiguracja symulacji
    config = MCConfig(
        n_runs=1000,
        sigma_static=2.0,
        sigma_total=5.0,
        base_seed=42
    )

    # Tworzenie runner'a
    simulation_factory = create_simulation_factory(profile)
    runner = MonteCarloRunner(simulation_factory, config)

    # Uruchomienie symulacji
    result = runner.run()

    print(result)

    # Wyświetlanie wyników
    print("\n" + "="*50)
    print("📊 WYNIKI SYMULACJI MONTE CARLO")
    print("="*50)
    print(f"Liczba przebiegów: {result.config.n_runs}")
    print(f"Średnia apogeum: {result.mean:.1f} m")
    print(f"Odchylenie standardowe: {result.std:.1f} m")
    print(f"5. percentyl: {result.p05:.1f} m")
    print(f"95. percentyl: {result.p95:.1f} m")

    print("\n" + "-"*30)
    print("📈 STATYSTYKI SZCZEGÓŁOWE")
    print("-"*30)
    summary = result.summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    print("\n✅ Symulacja zakończona pomyślnie!")


if __name__ == "__main__":
    main()
