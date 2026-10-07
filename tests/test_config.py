"""Config-schema tests for the validator.

Covers the config schema: required sections, ``n_components == n_qubits``,
seeds, cost function, log frequency, and rejection of legacy keys
(``shots``, ``backend``, ``success_threshold``, ``local_cost``, ``epochs``,
``epochs_per_layer``, ``finetune_epochs``, ``random_seeds``, ...). Also covers
the deterministic ``derive_seed_triple`` used to pair runs across approaches.
"""

import pytest
from pathlib import Path

from src.utils.config_validator import (
    ConfigValidationError,
    derive_seed_triple,
    load_config,
    validate_config,
)


def _valid_config():
    return {
        'experiment': {'name': 'test', 'approach': 'baseline'},
        'model': {'n_qubits': 8, 'n_layers': 4},
        'training': {
            'optimizer': 'adam', 'learning_rate': 0.01, 'batch_size': 20,
            'total_updates': 2500, 'cost_function': 'global',
        },
        'data': {
            'dataset': 'mnist', 'digit1': 3, 'digit2': 6,
            'train_size': 1000, 'test_size': 200, 'image_size': [4, 4],
            'preprocessing': 'pca', 'n_components': 8, 'encoding': 'ry_angle',
        },
        'seeds': {'seed_triples': 20, 'base_seed': 42},
        'metrics': {'track_gradients': True, 'log_frequency': 10},
        'output': {'results_dir': 'results/baseline/test', 'save_plot': True},
    }


def _set(config, dotted_key, value):
    """Set a nested key like ``"training.cost_function"``, creating sections."""
    *parts, leaf = dotted_key.split('.')
    node = config
    for part in parts:
        node = node.setdefault(part, {})
    node[leaf] = value
    return config


class TestValidConfig:
    def test_valid_config_passes_and_returns_resolved(self):
        resolved = validate_config(_valid_config())
        # Optional sections filled with defaults.
        assert resolved['metrics']['log_frequency'] == 10
        assert resolved['metrics']['track_gradients'] is True
        assert resolved['analysis']['multiple_comparison'] == 'holm'
        assert resolved['analysis']['alpha'] == 0.05
        assert resolved['output']['checkpoint_frequency'] == 500

    def test_approach_must_match(self):
        cfg = _valid_config()
        with pytest.raises(ConfigValidationError, match='approach mismatch'):
            validate_config(cfg, approach='layerwise')
        validate_config(cfg, approach='baseline')  # no error

    def test_real_8layer_config_validates(self):
        cfg_path = Path(__file__).parent.parent / 'configs' / 'baseline_8layer.yaml'
        config = load_config(str(cfg_path))
        resolved = validate_config(config, approach='baseline')
        assert resolved['model']['n_qubits'] == 8
        assert resolved['model']['n_layers'] == 8
        assert resolved['seeds']['seed_triples'] == 20


class TestMissingRequired:
    def test_missing_section(self):
        cfg = _valid_config()
        del cfg['seeds']
        with pytest.raises(ConfigValidationError, match='seeds'):
            validate_config(cfg)

    def test_missing_n_components(self):
        cfg = _valid_config()
        del cfg['data']['n_components']
        with pytest.raises(ConfigValidationError, match='n_components'):
            validate_config(cfg)

    def test_missing_total_updates(self):
        cfg = _valid_config()
        del cfg['training']['total_updates']
        with pytest.raises(ConfigValidationError, match='total_updates'):
            validate_config(cfg)

    def test_missing_results_dir(self):
        cfg = _valid_config()
        del cfg['output']['results_dir']
        with pytest.raises(ConfigValidationError, match='results_dir'):
            validate_config(cfg)


class TestValueConstraints:
    def test_n_components_must_equal_n_qubits(self):
        cfg = _valid_config()
        _set(cfg, 'data.n_components', 7)
        with pytest.raises(ConfigValidationError, match='must equal'):
            validate_config(cfg)

    def test_bad_n_components_type(self):
        cfg = _valid_config()
        _set(cfg, 'data.n_components', 'six')
        with pytest.raises(ConfigValidationError, match='n_components'):
            validate_config(cfg)

    def test_unknown_cost_function(self):
        cfg = _valid_config()
        _set(cfg, 'training.cost_function', 'mean_global')
        with pytest.raises(ConfigValidationError, match='cost_function'):
            validate_config(cfg)

    def test_bad_total_updates(self):
        cfg = _valid_config()
        _set(cfg, 'training.total_updates', 0)
        with pytest.raises(ConfigValidationError, match='total_updates'):
            validate_config(cfg)

    def test_bad_log_frequency(self):
        cfg = _valid_config()
        _set(cfg, 'metrics.log_frequency', 0)
        with pytest.raises(ConfigValidationError, match='log_frequency'):
            validate_config(cfg)

    def test_bad_seed_triples(self):
        cfg = _valid_config()
        _set(cfg, 'seeds.seed_triples', 0)
        with pytest.raises(ConfigValidationError, match='seed_triples'):
            validate_config(cfg)

    def test_bad_learning_rate(self):
        cfg = _valid_config()
        _set(cfg, 'training.learning_rate', 2.0)
        with pytest.raises(ConfigValidationError, match='learning_rate'):
            validate_config(cfg)


class TestLegacyKeyRejection:
    @pytest.mark.parametrize('key,value', [
        ('training.local_cost', True),
        ('training.epochs', 50),
        ('training.epochs_per_layer', 10),
        ('training.finetune_epochs', 10),
        ('model.circuit_type', 'hardware_efficient'),
        ('metrics.barren_plateau_threshold', 1e-6),
        ('metrics.gradient_variance', True),
        ('output.save_checkpoints', True),
    ])
    def test_legacy_nested_keys_rejected(self, key, value):
        cfg = _valid_config()
        _set(cfg, key, value)
        with pytest.raises(ConfigValidationError, match='legacy'):
            validate_config(cfg)

    def test_legacy_top_level_keys_rejected(self):
        cfg = _valid_config()
        cfg['random_seeds'] = [42, 123]
        cfg['quantum'] = {'shots': 1024, 'backend': 'cirq_simulator'}
        with pytest.raises(ConfigValidationError, match='Unknown config sections'):
            validate_config(cfg)

    def test_unknown_top_level_section_rejected(self):
        cfg = _valid_config()
        cfg['typo_section'] = {}
        with pytest.raises(ConfigValidationError, match='Unknown config sections'):
            validate_config(cfg)

    def test_shots_backend_success_threshold_rejected(self):
        cfg = _valid_config()
        cfg['quantum'] = {'shots': 1024, 'backend': 'cirq_simulator'}
        cfg['metrics']['success_threshold'] = 90.0
        with pytest.raises(ConfigValidationError):
            validate_config(cfg)


class TestSeedTriple:
    def test_deterministic(self):
        assert derive_seed_triple(42, 0) == {'data_seed': 42, 'init_seed': 43,
                                             'training_seed': 44}
        assert derive_seed_triple(42, 0) == derive_seed_triple(42, 0)

    def test_triples_are_disjoint_across_indices(self):
        a = derive_seed_triple(42, 0)
        b = derive_seed_triple(42, 1)
        assert set(a.values()).isdisjoint(set(b.values()))
        assert b['data_seed'] - a['data_seed'] == 3

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            derive_seed_triple('42', 0)
        with pytest.raises(ValueError):
            derive_seed_triple(42, -1)
        with pytest.raises(ValueError):
            derive_seed_triple(42, 1.5)