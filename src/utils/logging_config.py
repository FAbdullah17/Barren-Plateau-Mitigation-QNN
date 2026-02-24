"""Logging configuration for experiment tracking and reproducibility.

Provides structured logging setup with both console and file handlers,
and an ExperimentLogger context manager for per-experiment log isolation.
Suppresses noisy third-party loggers (TensorFlow, Cirq, Matplotlib) by
default to keep experiment output clean.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_dir: str = "logs"
) -> None:
    """
    Setup logging configuration for the project.
    
    Args:
        level: Logging level for console output (default: INFO).
        log_file: Optional log file name. If not provided, a timestamped
                  default filename is generated.
        log_dir: Directory for log files (default: "logs").
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Console formatter (concise)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File formatter (detailed with source location)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    
    handlers = [console_handler]
    
    # Setup file handler
    if log_file:
        if not log_file.endswith('.log'):
            log_file = f"{log_file}.log"
        
        file_path = log_path / log_file
        file_handler = logging.FileHandler(file_path, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    else:
        # Default timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_log = log_path / f'qnn_experiment_{timestamp}.log'
        file_handler = logging.FileHandler(default_log, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=handlers,
        force=True
    )
    
    # Suppress noisy third-party loggers
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
    logging.getLogger('cirq').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__).
    
    Returns:
        Configured Logger instance.
    """
    return logging.getLogger(name)


class ExperimentLogger:
    """
    Context manager for experiment-scoped logging.
    
    Creates an isolated log file for each experiment run, capturing all
    log output with seed and timestamp metadata.
    
    Example:
        with ExperimentLogger('baseline', seed=42) as logger:
            logger.info("Starting experiment")
            # ... experiment code ...
            logger.info("Experiment complete")
    """
    
    def __init__(self, experiment_name: str, seed: Optional[int] = None):
        self.experiment_name = experiment_name
        self.seed = seed
        self.logger = None
        self.log_file = None
    
    def __enter__(self):
        # Create experiment-specific log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        seed_str = f"_seed{self.seed}" if self.seed is not None else ""
        log_filename = f"{self.experiment_name}{seed_str}_{timestamp}.log"
        
        setup_logging(log_file=log_filename)
        self.logger = get_logger(self.experiment_name)
        
        self.logger.info("=" * 60)
        self.logger.info(f"Experiment: {self.experiment_name}")
        if self.seed is not None:
            self.logger.info(f"Random Seed: {self.seed}")
        self.logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 60)
        
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.logger.error(f"Experiment failed with error: {exc_val}")
        else:
            self.logger.info("Experiment completed successfully")
        
        self.logger.info("=" * 60)


# Initialize default logging on module import
setup_logging()
