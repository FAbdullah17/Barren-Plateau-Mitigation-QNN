"""Constants used throughout the project."""

# Quantum Circuit Constants
DEFAULT_N_QUBITS = 4
DEFAULT_N_LAYERS = 4
MIN_QUBITS = 2
MAX_QUBITS = 10
MIN_LAYERS = 1
MAX_LAYERS = 20

# Training Constants
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_BATCH_SIZE = 20
DEFAULT_EPOCHS = 50
DEFAULT_EPOCHS_PER_LAYER = 10
DEFAULT_FINETUNE_EPOCHS = 10

# Data Constants
MNIST_IMAGE_SIZE = (28, 28)
DEFAULT_DOWNSAMPLED_SIZE = (4, 4)
DEFAULT_DIGIT1 = 3
DEFAULT_DIGIT2 = 6
DEFAULT_TRAIN_SIZE = 1000
DEFAULT_TEST_SIZE = 200

# Label Encoding
CLASSICAL_LABELS = {0, 1}  # For MNIST
QUANTUM_LABELS = {-1, 1}    # For QNN

# Barren Plateau Detection
BARREN_PLATEAU_THRESHOLD = 1e-6
GRADIENT_VARIANCE_THRESHOLD = 1e-8

# Success Criteria
DEFAULT_SUCCESS_THRESHOLD = 90.0  # Accuracy percentage
MIN_ACCEPTABLE_ACCURACY = 60.0

# File Paths
RESULTS_DIR = "results"
CONFIGS_DIR = "configs"
LOGS_DIR = "logs"
NOTEBOOKS_DIR = "notebooks"
DOCS_DIR = "docs"

# Supported Approaches
APPROACHES = ["baseline", "layerwise", "local_cost"]

# Experiment Parameters
MULTI_DEPTH_EXPERIMENTS = [4, 6, 8]
DEFAULT_SEEDS = [42, 123, 456, 789, 101112]

# Visualization Constants
PLOT_DPI = 300
PLOT_FIGSIZE = (12, 8)
PLOT_STYLE = 'whitegrid'

# Color Schemes (colorblind-friendly)
COLORS = {
    'baseline': '#E69F00',      # Orange
    'layerwise': '#56B4E9',     # Sky blue
    'local_cost': '#009E73',    # Bluish green
}

# Metric Names
METRIC_NAMES = {
    'train_loss': 'Training Loss',
    'test_loss': 'Test Loss',
    'train_accuracy': 'Training Accuracy (%)',
    'test_accuracy': 'Test Accuracy (%)',
    'gradient_norms': 'Gradient Norm',
    'gradient_variances': 'Gradient Variance',
}

# Statistical Test Constants
SIGNIFICANCE_LEVELS = {
    'highly_significant': 0.001,  # ***
    'very_significant': 0.01,     # **
    'significant': 0.05,          # *
}

# Effect Size Thresholds (Cohen's d)
EFFECT_SIZE_THRESHOLDS = {
    'negligible': 0.2,
    'small': 0.5,
    'medium': 0.8,
}

# Hardware Recommendations
RECOMMENDED_RAM_GB = 4
RECOMMENDED_DISK_GB = 1

# Version Information
PROJECT_VERSION = "1.0.0"
MIN_PYTHON_VERSION = "3.10"
TENSORFLOW_VERSION = "2.15.0"
TFQ_VERSION = "0.7.3"
CIRQ_VERSION = "1.3.0"

# Random Seeds for Reproducibility
NUMPY_SEED = 42
TENSORFLOW_SEED = 42
PYTHON_SEED = 42
