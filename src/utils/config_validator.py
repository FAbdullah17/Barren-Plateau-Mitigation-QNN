"""Configuration validation for experiment configs.

The experiment config is a nested YAML document with fixed sections:

    experiment: {name, approach}
    model:      {n_qubits, n_layers}
    training:   {optimizer, learning_rate, batch_size, total_updates,
                 cost_function}
    data:       {dataset, digit1, digit2, train_size, test_size, image_size,
                 preprocessing, n_components, encoding}
    seeds:      {seed_triples, base_seed}
    metrics:    {track_gradients, log_frequency, diagnostic_samples}   (optional)
    analysis:   {min_variance_separation_se, qubit_counts,
                 alpha, multiple_comparison}                         (optional)
    output:     {results_dir, save_plot}

Unknown sections, unknown keys inside known sections, and any legacy keys
(``shots``, ``backend``, ``success_threshold``, ``local_cost``, ``epochs``,
``epochs_per_layer``, ``finetune_epochs``, ``random_seeds``, ...) are rejected
so stale configs fail loudly instead of being silently misinterpreted.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from .constants import (
    MIN_QUBITS, MAX_QUBITS, MIN_LAYERS, MAX_LAYERS,
    APPROACHES,
    DEFAULT_LOG_FREQUENCY, DEFAULT_DIAGNOSTIC_SAMPLES,
)

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


# Allowed top-level sections.
ALLOWED_TOP_LEVEL = frozenset({
    'experiment', 'model', 'training', 'data', 'seeds',
    'metrics', 'analysis', 'output',
})

# Sections every config must provide.
REQUIRED_SECTIONS = ['experiment', 'model', 'training', 'data', 'seeds', 'output']

# Keys allowed inside each section (unknown/dead keys are rejected).
SECTION_KEYS = {
    'experiment': {'name', 'approach', 'description'},
    'model': {'n_qubits', 'n_layers'},
    'training': {
        'optimizer', 'learning_rate', 'batch_size',
        'total_updates', 'cost_function',
    },
    'data': {
        'dataset', 'digit1', 'digit2', 'train_size', 'test_size',
        'image_size', 'preprocessing', 'n_components', 'encoding',
    },
    'seeds': {'seed_triples', 'base_seed'},
    'metrics': {'track_gradients', 'log_frequency', 'diagnostic_samples'},
    'analysis': {
        'min_variance_separation_se', 'qubit_counts',
        'alpha', 'multiple_comparison',
    },
    'output': {'results_dir', 'save_plot'},
}

# Defaults for the optional analysis block.
DEFAULT_ANALYSIS = {
    'min_variance_separation_se': 2.0,
    'qubit_counts': [4, 6, 8],
    'alpha': 0.05,
    'multiple_comparison': 'holm',
}


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        The raw configuration dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the YAML cannot be parsed.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def validate_config(
    config: Dict[str, Any], approach: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate a configuration and return it with defaults applied.

    Args:
        config: Configuration dictionary.
        approach: Expected ``experiment.approach`` (baseline, layerwise,
            local_cost). When given, the config must declare this approach.

    Returns:
        The resolved config (optional sections filled with defaults). Callers
        may ignore the return value; validation happens for the side effect of
        raising ``ConfigValidationError`` on any violation.

    Raises:
        ConfigValidationError: If the config violates the schema.
    """
    if not isinstance(config, dict):
        raise ConfigValidationError(f"config must be a mapping, got {type(config).__name__}")

    _reject_unknown_keys(config)

    missing_sections = [
        s for s in REQUIRED_SECTIONS
        if s not in config or not isinstance(config[s], dict)
    ]
    if missing_sections:
        raise ConfigValidationError(f"Missing required config section(s): {missing_sections}")

    _validate_experiment(config, approach)
    _validate_model(config)
    _validate_training(config)
    _validate_data(config)
    _validate_seeds(config)

    resolved = dict(config)

    metrics = config.get('metrics')
    resolved['metrics'] = _validate_metrics(metrics)

    analysis = config.get('analysis')
    resolved['analysis'] = _validate_analysis(analysis)

    output = config.get('output')
    if not isinstance(output, dict) or 'results_dir' not in output:
        raise ConfigValidationError("output.results_dir is required")
    results_dir = output['results_dir']
    if not isinstance(results_dir, str) or not results_dir.strip():
        raise ConfigValidationError(
            f"output.results_dir must be a non-empty string, got {results_dir!r}"
        )

    logger.info("Configuration validation passed")
    return resolved


def derive_seed_triple(base_seed: int, index: int) -> Dict[str, int]:
    """Deterministically derive a (data, init, training) seed triple.

    ``seed_triples`` independent (data, init, training) triples are derived
    from a single ``base_seed``. The same index yields the same triple for
    every approach, so runs are paired across approaches.

    Args:
        base_seed: Seed from the config ``seeds.base_seed``.
        index: Triple index in ``[0, seed_triples)``.

    Returns:
        dict with ``data_seed``, ``init_seed``, ``training_seed``.

    Raises:
        ValueError: If either input is invalid.
    """
    if not isinstance(base_seed, int):
        raise ValueError(f"base_seed must be an integer, got {base_seed!r}")
    if not isinstance(index, int) or index < 0:
        raise ValueError(f"index must be a non-negative integer, got {index!r}")
    return {
        'data_seed': base_seed + 3 * index,
        'init_seed': base_seed + 3 * index + 1,
        'training_seed': base_seed + 3 * index + 2,
    }


def _reject_unknown_keys(config: Dict[str, Any]) -> None:
    """Reject unknown sections and unknown keys inside known sections."""
    unknown_sections = set(config) - ALLOWED_TOP_LEVEL
    if unknown_sections:
        raise ConfigValidationError(
            "Unknown config sections: "
            f"{sorted(unknown_sections)}. Allowed: {sorted(ALLOWED_TOP_LEVEL)}"
        )

    for section in ALLOWED_TOP_LEVEL & set(config):
        value = config[section]
        if not isinstance(value, dict):
            raise ConfigValidationError(f"{section} must be a mapping")
        unknown_keys = set(value) - SECTION_KEYS[section]
        if unknown_keys:
            raise ConfigValidationError(
                f"Unknown/legacy keys in {section}: {sorted(unknown_keys)}. "
                f"Allowed: {sorted(SECTION_KEYS[section])}"
            )


def _validate_experiment(config: Dict[str, Any], approach: Optional[str]) -> None:
    experiment = config['experiment']
    name = experiment.get('name')
    if not isinstance(name, str) or not name.strip():
        raise ConfigValidationError("experiment.name must be a non-empty string")
    cfg_approach = experiment.get('approach')
    if cfg_approach not in APPROACHES:
        raise ConfigValidationError(
            f"experiment.approach must be one of {APPROACHES}, got {cfg_approach!r}"
        )
    if approach is not None and cfg_approach != approach:
        raise ConfigValidationError(
            f"approach mismatch: config declares {cfg_approach!r} but the "
            f"runner expects {approach!r}"
        )


def _validate_model(config: Dict[str, Any]) -> None:
    model = config['model']
    n_qubits = model.get('n_qubits')
    if not isinstance(n_qubits, int) or isinstance(n_qubits, bool):
        raise ConfigValidationError(
            f"model.n_qubits must be an integer, got {type(n_qubits).__name__}"
        )
    if not MIN_QUBITS <= n_qubits <= MAX_QUBITS:
        raise ConfigValidationError(
            f"model.n_qubits must be in [{MIN_QUBITS}, {MAX_QUBITS}], got {n_qubits}"
        )

    n_layers = model.get('n_layers')
    if not isinstance(n_layers, int) or isinstance(n_layers, bool):
        raise ConfigValidationError(
            f"model.n_layers must be an integer, got {type(n_layers).__name__}"
        )
    if not MIN_LAYERS <= n_layers <= MAX_LAYERS:
        raise ConfigValidationError(
            f"model.n_layers must be in [{MIN_LAYERS}, {MAX_LAYERS}], got {n_layers}"
        )


def _validate_training(config: Dict[str, Any]) -> None:
    training = config['training']
    optimizer = training.get('optimizer')
    if optimizer != 'adam':
        raise ConfigValidationError(f"training.optimizer must be 'adam', got {optimizer!r}")

    learning_rate = training.get('learning_rate')
    if not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool):
        raise ConfigValidationError(
            f"training.learning_rate must be numeric, got {type(learning_rate).__name__}"
        )
    if not 1e-6 < learning_rate < 1.0:
        raise ConfigValidationError(
            f"training.learning_rate must be in (1e-6, 1.0), got {learning_rate}"
        )

    batch_size = training.get('batch_size')
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise ConfigValidationError(
            f"training.batch_size must be an integer, got {type(batch_size).__name__}"
        )
    if not 1 <= batch_size <= 1000:
        raise ConfigValidationError(
            f"training.batch_size must be in [1, 1000], got {batch_size}"
        )

    total_updates = training.get('total_updates')
    if not isinstance(total_updates, int) or isinstance(total_updates, bool):
        raise ConfigValidationError(
            f"training.total_updates must be an integer, got {type(total_updates).__name__}"
        )
    if total_updates < 1:
        raise ConfigValidationError(f"training.total_updates must be >= 1, got {total_updates}")

    cost_function = training.get('cost_function')
    if cost_function not in ('global', 'local'):
        raise ConfigValidationError(
            f"training.cost_function must be 'global' or 'local', got {cost_function!r}"
        )


def _validate_data(config: Dict[str, Any]) -> None:
    data = config['data']
    n_qubits = config['model']['n_qubits']

    dataset = data.get('dataset')
    if dataset != 'mnist':
        raise ConfigValidationError(f"data.dataset must be 'mnist', got {dataset!r}")

    digit1 = data.get('digit1')
    digit2 = data.get('digit2')
    if (
        not isinstance(digit1, int) or not isinstance(digit2, int)
        or not (0 <= digit1 <= 9) or not (0 <= digit2 <= 9)
        or digit1 == digit2
    ):
        raise ConfigValidationError(
            f"data.digit1/digit2 must be distinct digits in [0, 9], got {digit1}/{digit2}"
        )

    train_size = data.get('train_size')
    if not isinstance(train_size, int) or isinstance(train_size, bool) or train_size < 10:
        raise ConfigValidationError(
            f"data.train_size must be an integer >= 10, got {train_size!r}"
        )
    test_size = data.get('test_size')
    if not isinstance(test_size, int) or isinstance(test_size, bool) or test_size < 5:
        raise ConfigValidationError(
            f"data.test_size must be an integer >= 5, got {test_size!r}"
        )

    image_size = data.get('image_size')
    if (
        not isinstance(image_size, (list, tuple)) or len(image_size) != 2
        or not all(isinstance(v, int) and not isinstance(v, bool) for v in image_size)
        or not all(2 <= v <= 28 for v in image_size)
    ):
        raise ConfigValidationError(
            f"data.image_size must be [height, width] with both in [2, 28], got {image_size!r}"
        )

    preprocessing = data.get('preprocessing')
    if preprocessing != 'pca':
        raise ConfigValidationError(
            f"data.preprocessing must be 'pca' (fixed low-dimensional pipeline), "
            f"got {preprocessing!r}"
        )

    n_components = data.get('n_components')
    if not isinstance(n_components, int) or isinstance(n_components, bool) or n_components < 1:
        raise ConfigValidationError(
            f"data.n_components must be a positive integer, got {n_components!r}"
        )
    if n_components != n_qubits:
        raise ConfigValidationError(
            f"data.n_components ({n_components}) must equal model.n_qubits ({n_qubits})"
        )

    encoding = data.get('encoding')
    if encoding != 'ry_angle':
        raise ConfigValidationError(
            f"data.encoding must be 'ry_angle', got {encoding!r}"
        )


def _validate_seeds(config: Dict[str, Any]) -> None:
    seeds = config['seeds']
    seed_triples = seeds.get('seed_triples')
    if not isinstance(seed_triples, int) or isinstance(seed_triples, bool) or seed_triples < 1:
        raise ConfigValidationError(
            f"seeds.seed_triples must be an integer >= 1, got {seed_triples!r}"
        )
    base_seed = seeds.get('base_seed')
    if not isinstance(base_seed, int):
        raise ConfigValidationError(
            f"seeds.base_seed must be an integer, got {base_seed!r}"
        )


def _validate_metrics(metrics: Any) -> Dict[str, Any]:
    if metrics is None:
        metrics = {}
    if not isinstance(metrics, dict):
        raise ConfigValidationError("metrics must be a mapping")

    track_gradients = metrics.get('track_gradients', True)
    if not isinstance(track_gradients, bool):
        raise ConfigValidationError(
            f"metrics.track_gradients must be a boolean, got {track_gradients!r}"
        )

    log_frequency = metrics.get('log_frequency', DEFAULT_LOG_FREQUENCY)
    if not isinstance(log_frequency, int) or isinstance(log_frequency, bool) or log_frequency < 1:
        raise ConfigValidationError(
            f"metrics.log_frequency must be an integer >= 1, got {log_frequency!r}"
        )

    diagnostic_samples = metrics.get('diagnostic_samples', DEFAULT_DIAGNOSTIC_SAMPLES)
    if (
        not isinstance(diagnostic_samples, int)
        or isinstance(diagnostic_samples, bool)
        or diagnostic_samples < 1
    ):
        raise ConfigValidationError(
            f"metrics.diagnostic_samples must be an integer >= 1, "
            f"got {diagnostic_samples!r}"
        )

    return {
        'track_gradients': track_gradients,
        'log_frequency': log_frequency,
        'diagnostic_samples': diagnostic_samples,
    }


def _validate_analysis(analysis: Any) -> Dict[str, Any]:
    if analysis is None:
        analysis = {}
    if not isinstance(analysis, dict):
        raise ConfigValidationError("analysis must be a mapping")
    return {**DEFAULT_ANALYSIS, **analysis}
