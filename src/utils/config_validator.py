"""Configuration validation utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from .constants import (
    MIN_QUBITS, MAX_QUBITS, MIN_LAYERS, MAX_LAYERS,
    APPROACHES, CLASSICAL_LABELS
)

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded successfully")
    return config


def validate_config(config: Dict[str, Any], approach: Optional[str] = None) -> None:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        approach: Experiment approach (baseline, layerwise, local_cost)
    
    Raises:
        ConfigValidationError: If validation fails
    """
    logger.info("Validating configuration...")
    
    # Required parameters
    required_params = [
        'n_qubits', 'n_layers', 'learning_rate', 'batch_size',
        'digit1', 'digit2', 'train_size', 'test_size'
    ]
    
    for param in required_params:
        if param not in config:
            raise ConfigValidationError(f"Missing required parameter: {param}")
    
    # Validate quantum circuit parameters
    _validate_circuit_params(config)
    
    # Validate training parameters
    _validate_training_params(config)
    
    # Validate data parameters
    _validate_data_params(config)
    
    # Approach-specific validation
    if approach:
        _validate_approach_params(config, approach)
    
    logger.info("Configuration validation passed")


def _validate_circuit_params(config: Dict[str, Any]) -> None:
    """Validate quantum circuit parameters."""
    n_qubits = config.get('n_qubits')
    n_layers = config.get('n_layers')
    
    # Validate n_qubits
    if not isinstance(n_qubits, int):
        raise ConfigValidationError(f"n_qubits must be integer, got {type(n_qubits)}")
    
    if not MIN_QUBITS <= n_qubits <= MAX_QUBITS:
        raise ConfigValidationError(
            f"n_qubits must be between {MIN_QUBITS} and {MAX_QUBITS}, got {n_qubits}"
        )
    
    # Validate n_layers
    if not isinstance(n_layers, int):
        raise ConfigValidationError(f"n_layers must be integer, got {type(n_layers)}")
    
    if not MIN_LAYERS <= n_layers <= MAX_LAYERS:
        raise ConfigValidationError(
            f"n_layers must be between {MIN_LAYERS} and {MAX_LAYERS}, got {n_layers}"
        )
    
    logger.debug(f"Circuit parameters valid: {n_qubits} qubits, {n_layers} layers")


def _validate_training_params(config: Dict[str, Any]) -> None:
    """Validate training hyperparameters."""
    learning_rate = config.get('learning_rate')
    batch_size = config.get('batch_size')
    epochs = config.get('epochs', 50)
    
    # Validate learning rate
    if not isinstance(learning_rate, (int, float)):
        raise ConfigValidationError(
            f"learning_rate must be numeric, got {type(learning_rate)}"
        )
    
    if not 1e-6 < learning_rate < 1.0:
        raise ConfigValidationError(
            f"learning_rate should be in (1e-6, 1.0), got {learning_rate}"
        )
    
    # Validate batch size
    if not isinstance(batch_size, int):
        raise ConfigValidationError(f"batch_size must be integer, got {type(batch_size)}")
    
    if not 1 <= batch_size <= 1000:
        raise ConfigValidationError(
            f"batch_size must be between 1 and 1000, got {batch_size}"
        )
    
    # Validate epochs
    if not isinstance(epochs, int):
        raise ConfigValidationError(f"epochs must be integer, got {type(epochs)}")
    
    if not 1 <= epochs <= 1000:
        raise ConfigValidationError(f"epochs must be between 1 and 1000, got {epochs}")
    
    logger.debug(f"Training parameters valid: lr={learning_rate}, batch={batch_size}, epochs={epochs}")


def _validate_data_params(config: Dict[str, Any]) -> None:
    """Validate data configuration parameters."""
    digit1 = config.get('digit1')
    digit2 = config.get('digit2')
    train_size = config.get('train_size')
    test_size = config.get('test_size')
    
    # Validate digits
    if not isinstance(digit1, int) or not isinstance(digit2, int):
        raise ConfigValidationError("digit1 and digit2 must be integers")
    
    if not (0 <= digit1 <= 9) or not (0 <= digit2 <= 9):
        raise ConfigValidationError("digit1 and digit2 must be between 0 and 9")
    
    if digit1 == digit2:
        raise ConfigValidationError("digit1 and digit2 must be different")
    
    # Validate dataset sizes
    if not isinstance(train_size, int) or not isinstance(test_size, int):
        raise ConfigValidationError("train_size and test_size must be integers")
    
    if train_size < 10:
        raise ConfigValidationError(f"train_size too small: {train_size} (minimum 10)")
    
    if test_size < 5:
        raise ConfigValidationError(f"test_size too small: {test_size} (minimum 5)")
    
    if train_size > 10000:
        logger.warning(f"Large train_size ({train_size}) may be slow")
    
    # Validate image size if specified
    if 'image_size' in config:
        image_size = config['image_size']
        if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
            raise ConfigValidationError("image_size must be [height, width]")
        
        h, w = image_size
        if not (2 <= h <= 28) or not (2 <= w <= 28):
            raise ConfigValidationError(
                f"image_size dimensions must be between 2 and 28, got {image_size}"
            )
        
        # Check compatibility with n_qubits
        n_features = h * w
        n_qubits = config['n_qubits']
        if n_features < n_qubits:
            logger.warning(
                f"Features ({n_features}) < qubits ({n_qubits}). "
                f"Consider increasing image_size or reducing n_qubits."
            )
    
    logger.debug(
        f"Data parameters valid: digits {digit1} vs {digit2}, "
        f"train={train_size}, test={test_size}"
    )


def _validate_approach_params(config: Dict[str, Any], approach: str) -> None:
    """Validate approach-specific parameters."""
    if approach not in APPROACHES:
        raise ConfigValidationError(
            f"Invalid approach: {approach}. Must be one of {APPROACHES}"
        )
    
    if approach == 'layerwise':
        # Validate layerwise-specific parameters
        if 'epochs_per_layer' not in config:
            raise ConfigValidationError(
                "layerwise approach requires 'epochs_per_layer' parameter"
            )
        
        epochs_per_layer = config['epochs_per_layer']
        if not isinstance(epochs_per_layer, int) or epochs_per_layer < 1:
            raise ConfigValidationError(
                f"epochs_per_layer must be positive integer, got {epochs_per_layer}"
            )
        
        if 'finetune_epochs' in config:
            finetune_epochs = config['finetune_epochs']
            if not isinstance(finetune_epochs, int) or finetune_epochs < 0:
                raise ConfigValidationError(
                    f"finetune_epochs must be non-negative integer, got {finetune_epochs}"
                )
        
        logger.debug(f"Layerwise parameters valid: {epochs_per_layer} epochs/layer")
    
    elif approach == 'local_cost':
        # Validate local cost parameters
        if 'local_cost' in config and not isinstance(config['local_cost'], bool):
            raise ConfigValidationError("local_cost must be boolean")
        
        logger.debug("Local cost parameters valid")


def validate_multi_depth_config(config: Dict[str, Any]) -> None:
    """
    Validate multi-depth comparison configuration.
    
    Args:
        config: Configuration dictionary
    
    Raises:
        ConfigValidationError: If validation fails
    """
    logger.info("Validating multi-depth configuration...")
    
    # Validate base parameters
    validate_config(config)
    
    # Validate depths
    if 'depths' not in config:
        raise ConfigValidationError("Multi-depth config requires 'depths' parameter")
    
    depths = config['depths']
    if not isinstance(depths, list) or len(depths) == 0:
        raise ConfigValidationError("depths must be non-empty list")
    
    for depth in depths:
        if not isinstance(depth, int) or not MIN_LAYERS <= depth <= MAX_LAYERS:
            raise ConfigValidationError(
                f"Each depth must be integer in [{MIN_LAYERS}, {MAX_LAYERS}], got {depth}"
            )
    
    # Validate seeds
    if 'seeds' not in config:
        raise ConfigValidationError("Multi-depth config requires 'seeds' parameter")
    
    seeds = config['seeds']
    if not isinstance(seeds, list) or len(seeds) == 0:
        raise ConfigValidationError("seeds must be non-empty list")
    
    for seed in seeds:
        if not isinstance(seed, int):
            raise ConfigValidationError(f"Each seed must be integer, got {seed}")
    
    # Calculate total experiments
    total_experiments = len(depths) * len(seeds) * len(APPROACHES)
    if total_experiments > 100:
        logger.warning(
            f"Large experiment count: {total_experiments}. "
            f"This may take several hours."
        )
    
    logger.info(
        f"Multi-depth validation passed: {len(depths)} depths, "
        f"{len(seeds)} seeds = {total_experiments} total experiments"
    )


def create_default_config() -> Dict[str, Any]:
    """
    Create a default configuration dictionary.
    
    Returns:
        Default configuration
    """
    from .constants import (
        DEFAULT_N_QUBITS, DEFAULT_N_LAYERS,
        DEFAULT_LEARNING_RATE, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS,
        DEFAULT_DIGIT1, DEFAULT_DIGIT2,
        DEFAULT_TRAIN_SIZE, DEFAULT_TEST_SIZE,
        DEFAULT_DOWNSAMPLED_SIZE
    )
    
    return {
        'n_qubits': DEFAULT_N_QUBITS,
        'n_layers': DEFAULT_N_LAYERS,
        'learning_rate': DEFAULT_LEARNING_RATE,
        'batch_size': DEFAULT_BATCH_SIZE,
        'epochs': DEFAULT_EPOCHS,
        'digit1': DEFAULT_DIGIT1,
        'digit2': DEFAULT_DIGIT2,
        'train_size': DEFAULT_TRAIN_SIZE,
        'test_size': DEFAULT_TEST_SIZE,
        'image_size': list(DEFAULT_DOWNSAMPLED_SIZE),
        'local_cost': False,
    }


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Configuration saved to: {output_path}")
