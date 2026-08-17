"""Pytest fixtures/configuration.

Sets ``TF_USE_LEGACY_KERAS=1`` before any TensorFlow import: tensorflow-
quantum is only compatible with Keras 2, and the env var must be set before
``tensorflow`` is imported anywhere in the process.
"""

import os

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
