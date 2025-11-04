"""Data module initialization."""

from .mnist_loader import load_mnist_binary, encode_data_for_qnn

__all__ = ["load_mnist_binary", "encode_data_for_qnn"]
