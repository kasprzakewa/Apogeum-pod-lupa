#!/usr/bin/env python3
"""
Convert an OpenRocket CSV export to the flight-profile format expected by the
simulation engine (time, static_pressure, total_pressure).

Expected OR export columns (order-independent, matched by header text):
  Time (s), Altitude (m), Vertical velocity (m/s), Mach number, Air pressure (mbar)

Two methods for computing total pressure:

  bernoulli  [default]
      q = 0.5 · ρ(altitude) · v²           (incompressible)
      p_total = p_static + q

      Uses the same ISA air-density model as the simulation engine, so the
      engine will recover the exact OR velocity when it reads the file back.

  isentropic
      p_total = p_static · (1 + (γ−1)/2 · M²)^(γ/(γ−1))   with γ = 1.4

      Physically more accurate for M > 0.3 (high-power rocketry peak speeds),
      but the simulation's Bernoulli-based speed recovery will then be slightly
      off from the true OR velocity.  Use this if you want the pitot pressure to
      be physically correct; pair with a simulation σ_v tuned for the resulting
      Mach-corrected differential.

Usage examples:
  poetry run python3 scripts/or_csv_to_flight.py data/or_flight_smaller.csv
  poetry run python3 scripts/or_csv_to_flight.py data/or_flight_smaller.csv \\
      --output data/my_flight.csv --method isentropic --dt 0.05 --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ── project constants (mirrors src/models/constants.py) ───────────────────────
_P_REF   = 101325.0   # Pa
_RHO_REF = 1.225      # kg/m³
_H_REF   = 44300.0    # m  (scale height used in physics.py)
_N_RHO   = 4.256      # density exponent
_GAMMA   = 1.4        # ratio of specific heats (air)
_MBAR_TO_PA = 100.0   # 1 mbar = 100 Pa

# ── ISA helpers ───────────────────────────────────────────────────────────────

def _air_density(altitude: np.ndarray) -> np.ndarray:
    """Same power-law model as calculate_air_density() in physics.py."""
    factor = np.clip(1.0 - altitude / _H_REF, 0.0, None)
    return _RHO_REF * factor ** _N_RHO


def _total_pressure_bernoulli(p_static: np.ndarray,
                               altitude: np.ndarray,
                               velocity: np.ndarray) -> np.ndarray:
    """
    p_total = p_static + 0.5 · ρ(h) · v²

    Sign-safe: if velocity is negative (descent), q is still positive.
    """
    rho = _air_density(altitude)
    q = 0.5 * rho * velocity ** 2
    return p_static + q


def _total_pressure_isentropic(p_static: np.ndarray,
                                mach: np.ndarray) -> np.ndarray:
    """
    Isentropic pitot total pressure (exact for subsonic compressible flow):
        p_total = p_static · (1 + (γ−1)/2 · M²)^(γ/(γ−1))

    For γ = 1.4: exponent = 3.5
    """
    exponent = _GAMMA / (_GAMMA - 1.0)         # 3.5
    return p_static * (1.0 + (_GAMMA - 1.0) / 2.0 * mach ** 2) ** exponent


# ── OR-CSV parsing ─────────────────────────────────────────────────────────────

_COL_ALIASES = {
    "time":              ["time", "time (s)"],
    "altitude":          ["altitude", "altitude (m)"],
    "vertical_velocity": ["vertical velocity", "vertical velocity (m/s)",
                          "vertical_velocity", "vertical_velocity_(m/s)"],
    "mach":              ["mach number", "mach_number", "mach number ()", "mach number (​)"],
    "air_pressure":      ["air pressure", "air pressure (mbar)", "air_pressure",
                          "air_pressure_(mbar)"],
}


def _match_header(raw_header: str) -> dict[str, int]:
    """
    Match CSV header tokens to canonical column names, case/space insensitive.

    Returns dict: canonical_name → column_index.
    Raises ValueError if required columns are missing.
    """
    tokens = [t.strip().lower() for t in raw_header.split(",")]
    mapping: dict[str, int] = {}
    for canonical, aliases in _COL_ALIASES.items():
        for i, tok in enumerate(tokens):
            if any(tok == alias or tok.startswith(alias) for alias in aliases):
                mapping[canonical] = i
                break

    required = set(_COL_ALIASES.keys())
    missing = required - set(mapping.keys())
    if missing:
        raise ValueError(
            f"Could not find columns {missing} in header: '{raw_header}'\n"
            f"  Found: {list(mapping.keys())}\n"
            f"  Expected aliases: { {k: v for k, v in _COL_ALIASES.items() if k in missing} }"
        )
    return mapping


def _read_or_csv(path: Path) -> tuple[dict[str, int], list[str]]:
    """Read OR CSV, skip comment lines (#), return (column_map, data_lines)."""
    header_line: str | None = None
    data_lines: list[str] = []

    with path.open(encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                # OR embeds the header in a comment line
                candidate = line.lstrip("# ").strip()
                # Heuristic: header line contains "time" (case-insensitive)
                if "time" in candidate.lower() and header_line is None:
                    header_line = candidate
                continue
            # First non-comment line may also be the header
            if header_line is None and "time" in line.lower():
                header_line = line
                continue
            data_lines.append(line)

    if header_line is None:
        raise ValueError("Could not detect a header row in the OR CSV file.")

    col_map = _match_header(header_line)
    return col_map, data_lines


def _parse_data(
    col_map: dict[str, int],
    data_lines: list[str],
) -> dict[str, np.ndarray]:
    """Parse data lines into numpy arrays using column mapping."""
    n_cols = max(col_map.values()) + 1
    rows: list[list[float]] = []
    skipped = 0
    for line in data_lines:
        parts = line.split(",")
        if len(parts) < n_cols:
            skipped += 1
            continue
        try:
            rows.append([float(parts[i]) for i in sorted(col_map.values())])
        except ValueError:
            skipped += 1
    if skipped:
        print(f"  ⚠  Skipped {skipped} malformed data rows.", file=sys.stderr)

    arr = np.array(rows, dtype=np.float64)
    sorted_keys = [k for k, _ in sorted(col_map.items(), key=lambda kv: kv[1])]
    return {k: arr[:, j] for j, k in enumerate(sorted_keys)}


# ── resampling ─────────────────────────────────────────────────────────────────

def _resample(
    arrays: dict[str, np.ndarray],
    dt_out: float | None,
) -> dict[str, np.ndarray]:
    """Linearly interpolate all channels to a uniform time grid (if dt_out given)."""
    if dt_out is None:
        return arrays
    t_in  = arrays["time"]
    t_out = np.arange(t_in[0], t_in[-1] + 1e-9, dt_out)
    return {k: np.interp(t_out, t_in, v) for k, v in arrays.items()}


# ── main conversion ────────────────────────────────────────────────────────────

def convert(
    input_path: Path,
    output_path: Path,
    method: str,
    dt: float | None,
    verbose: bool,
) -> None:
    if verbose:
        print(f"Reading  : {input_path}")

    col_map, data_lines = _read_or_csv(input_path)
    raw = _parse_data(col_map, data_lines)

    if verbose:
        print(f"  Rows parsed : {len(raw['time'])}")
        print(f"  Time range  : {raw['time'][0]:.3f} – {raw['time'][-1]:.3f} s")
        print(f"  Altitude    : {raw['altitude'].min():.1f} – {raw['altitude'].max():.1f} m")
        print(f"  Velocity    : {raw['vertical_velocity'].min():.1f} – "
              f"{raw['vertical_velocity'].max():.1f} m/s")
        print(f"  Mach        : {raw['mach'].min():.3f} – {raw['mach'].max():.3f}")
        print(f"  Pressure    : {raw['air_pressure'].min():.2f} – "
              f"{raw['air_pressure'].max():.2f} mbar")

    arrays = _resample(raw, dt)
    if dt is not None and verbose:
        print(f"  Resampled to dt={dt}s → {len(arrays['time'])} rows")

    t           = arrays["time"]
    altitude    = arrays["altitude"]
    velocity    = np.abs(arrays["vertical_velocity"])   # speed (non-negative for q)
    mach        = arrays["mach"]
    p_static    = arrays["air_pressure"] * _MBAR_TO_PA  # mbar → Pa

    if method == "bernoulli":
        p_total = _total_pressure_bernoulli(p_static, altitude, velocity)
        if verbose:
            q_max = np.max(p_total - p_static)
            print(f"  Method: Bernoulli  |  max q = {q_max:.1f} Pa")
    elif method == "isentropic":
        p_total = _total_pressure_isentropic(p_static, mach)
        if verbose:
            q_max = np.max(p_total - p_static)
            print(f"  Method: Isentropic  |  max q = {q_max:.1f} Pa")
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'bernoulli' or 'isentropic'.")

    # Sanity: p_total must be ≥ p_static
    bad = np.sum(p_total < p_static)
    if bad:
        print(f"  ⚠  {bad} rows with p_total < p_static (clamped to p_static).",
              file=sys.stderr)
        p_total = np.maximum(p_total, p_static)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        out.write("time,static_pressure,total_pressure\n")
        for ti, ps, pt in zip(t, p_static, p_total):
            out.write(f"{ti:.6g},{ps:.6f},{pt:.6f}\n")

    if verbose:
        print(f"Writing  : {output_path}  ({len(t)} rows)")
        print(f"  static_pressure : {p_static.min():.1f} – {p_static.max():.1f} Pa")
        print(f"  total_pressure  : {p_total.min():.1f} – {p_total.max():.1f} Pa")
        print(f"  max Δp (dynamic): {np.max(p_total - p_static):.1f} Pa")
    else:
        print(f"Converted {input_path.name} → {output_path}  ({len(t)} rows)")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert OpenRocket CSV to simulation flight-profile CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", type=Path,
                   help="OpenRocket CSV file (may contain # comment headers).")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Output CSV path. Default: same directory, '_flight.csv' suffix.")
    p.add_argument(
        "--method", "-m",
        choices=["bernoulli", "isentropic"],
        default="bernoulli",
        help=(
            "bernoulli: p_total = p_static + 0.5·ρ(h)·v²  "
            "(consistent with simulation engine, recommended). "
            "isentropic: p_total = p_static·(1+0.2·M²)^3.5  "
            "(physically exact pitot pressure, better for M>0.3)."
        ),
    )
    p.add_argument(
        "--dt", type=float, default=None,
        help=(
            "Resample to uniform time step [s] before output. "
            "E.g. --dt 0.01 gives a 100 Hz profile. "
            "If omitted, OR's native time grid is preserved."
        ),
    )
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print detailed statistics.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        print(f"ERROR: file not found: {input_path}", file=sys.stderr)
        return 1

    if args.output is None:
        stem = input_path.stem
        if stem.endswith("_or") or stem.endswith("_raw"):
            stem = stem[:-3]
        out_name = stem + "_flight.csv"
        output_path = input_path.parent / out_name
    else:
        output_path = args.output

    try:
        convert(input_path, output_path, args.method, args.dt, args.verbose)
    except (ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
