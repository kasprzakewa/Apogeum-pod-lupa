"""Physical and aerodynamic constants used throughout the simulation."""

# Atmospheric reference values
REFERENCE_PRESSURE: float = 101325.0       # Pa – standard sea-level pressure
REFERENCE_AIR_DENSITY: float = 1.225       # kg/m³ – standard sea-level air density
REFERENCE_ALTITUDE: float = 0.0            # m

# Barometric formula constants (ISA model)
SCALE_HEIGHT: float = 44300.0              # m – used in altitude and density formulas
BARO_EXPONENT: float = 0.190263096        # dimensionless – (g*M)/(R*L)
DENSITY_EXPONENT: float = 4.256           # dimensionless

# Gravitational acceleration
G: float = 9.81                            # m/s²

# Default rocket aerodynamic / mass parameters
# These are used in apogee prediction; overridable via ModelParams.
DEFAULT_CROSS_SECTION: float = 0.0181458  # m² – reference area A
DEFAULT_DRAG_COEFFICIENT: float = 0.6     # dimensionless – C_D
DEFAULT_MASS: float = 50.0                # kg – rocket mass m
