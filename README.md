# Barren Plateau Mitigation in Quantum Neural Networks

**Empirical Comparison of Layerwise Training and Local Cost Functions**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![TensorFlow Quantum](https://img.shields.io/badge/TFQ-0.7.3-orange.svg)](https://www.tensorflow.org/quantum)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the implementation and experiments for a systematic empirical comparison of two prominent barren plateau mitigation strategies in quantum neural networks:

1. **Layerwise Training** (Skolik et al., 2020)
2. **Local Cost Functions** (Cerezo et al., 2021)

The research addresses a critical gap in quantum machine learning: while both approaches show theoretical promise, no direct empirical comparison exists on a standardized benchmark.

## Research Question

**How do layerwise training and local cost functions compare in their effectiveness at mitigating barren plateaus in quantum neural networks, as measured by gradient flow, training dynamics, and classification performance on the MNIST benchmark?**

## Key Features

- ✅ **Three Training Approaches**: Baseline, Layerwise, and Local Cost implementations
- ✅ **Hardware-Efficient Ansatz**: 4-qubit quantum circuits with configurable depth
- ✅ **Comprehensive Metrics**: Gradient tracking, barren plateau detection, performance evaluation
- ✅ **MNIST Binary Classification**: Standard benchmark (digits 3 vs 6)
- ✅ **Multi-Depth Testing**: Compare approaches at 4, 6, and 8 layers
- ✅ **Statistical Robustness**: Multiple random seeds for significance testing
- ✅ **Rich Visualization**: Training histories, gradient trajectories, comparative plots
- ✅ **100+ Unit Tests**: Comprehensive test coverage for all modules
- ✅ **Analysis Notebooks**: 4 Jupyter notebooks for data exploration and results
- ✅ **Complete Documentation**: Methodology and results interpretation guides
