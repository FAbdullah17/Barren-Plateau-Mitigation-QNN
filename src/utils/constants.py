"""Constants used throughout the project."""

# Quantum Circuit Constants
MIN_QUBITS = 2
MAX_QUBITS = 10
MIN_LAYERS = 1
MAX_LAYERS = 20

# Training / Metrics Defaults
DEFAULT_LOG_FREQUENCY = 10
DEFAULT_DIAGNOSTIC_SAMPLES = 100

# Supported Approaches
APPROACHES = ["baseline", "layerwise", "local_cost"]

# Visualization Constants
PLOT_DPI = 300
PLOT_FIGSIZE = (12, 8)

# Color Schemes (colorblind-friendly)
COLORS = {
    'baseline': '#E69F00',      # Orange
    'layerwise': '#56B4E9',     # Sky blue
    'local_cost': '#009E73',    # Bluish green
}