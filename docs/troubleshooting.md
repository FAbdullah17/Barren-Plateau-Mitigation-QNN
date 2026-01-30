# Troubleshooting Guide

Common issues and solutions for the Hybrid-QNN experiments.

---

## TensorFlow Warnings

### CUDA/GPU Warnings
```
Could not find cuda drivers on your machine, GPU will not be used.
```
**Solution:** This is normal for CPU-only systems. TensorFlow Quantum runs efficiently on CPU. Ignore these warnings.

To suppress verbose warnings:
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING
```

### cuDNN/cuFFT/cuBLAS Registration Errors
```
Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
```
**Solution:** Safe to ignore. Occurs when TensorFlow is built with GPU support but no GPU is available.

---

## Memory Issues

### Out of Memory Errors
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solutions:**
1. **Reduce batch size** in config file:
   ```yaml
   training:
     batch_size: 10  # Reduce from 20
   ```

2. **Reduce training samples**:
   ```yaml
   data:
     train_size: 500  # Reduce from 1000
   ```

3. **Clear memory between runs**:
   ```python
   import gc
   gc.collect()
   ```

---

## Shape Mismatch Errors

### Predictions vs Labels Shape
```
ValueError: logits and labels must have the same shape, received ((200, 1) vs (200,))
```

**Solution:** This was fixed in `baseline_trainer.py` and `layerwise_trainer.py` by adding:
```python
predictions = tf.squeeze(predictions, axis=-1)
```

If you encounter this, ensure you have the latest code from the repository.

---

## File I/O Issues

### Permission Denied
```
PermissionError: [Errno 13] Permission denied
```

**Solutions:**
1. Close any programs (Excel, text editors) that have the file open
2. Run terminal as administrator (Windows)
3. Check file isn't read-only

### Results Not Saving
**Check:**
1. Results directory exists and is writable
2. Disk has sufficient space (run `python scripts/check_system.py`)
3. Path in config is correct

---

## Quantum Circuit Issues

### TFQ Tensor Conversion Errors
```
TypeError: Cannot convert the argument to a tensor
```

**Solution:** Ensure circuits are converted properly:
```python
import tensorflow_quantum as tfq
circuit_tensor = tfq.convert_to_tensor(circuits)
```

### Circuit Building Errors
```
ValueError: Operation ... is not in circuit
```

**Solution:** Verify qubit indices match between encoding and ansatz circuits.

---

## Training Issues

### Training Not Converging
**Symptoms:**
- Loss stays flat
- Accuracy stuck at ~50%
- Gradient norms near 0

**Solutions:**
1. **Reduce learning rate**: 0.01 → 0.001
2. **Check for barren plateau**: This is expected for deep baseline circuits
3. **Verify data loading**: Run `python tests/test_data_consistency.py`

### NaN in Loss/Gradients
```
WARNING: NaN detected in loss
```

**Solutions:**
1. Reduce learning rate
2. Check input data normalization (should be 0-1)
3. Reduce circuit depth

---

## Environment Issues

### Import Errors
```
ModuleNotFoundError: No module named 'tensorflow_quantum'
```

**Solution:**
```bash
pip install tensorflow==2.15.0 tensorflow-quantum==0.7.2
```

### Version Conflicts
```
ERROR: pip's dependency resolver...
```

**Solution:** Create fresh virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

## WSL/Linux Issues

### Line Ending Problems
```
$'\r': command not found
```

**Solution:** Convert Windows line endings to Unix:
```bash
sed -i 's/\r$//' scripts/*.py experiments/*.py
```

### Path Issues
Use forward slashes and Linux-style paths in WSL:
```bash
cd /mnt/d/Programs/PF/Hybrid-QNNs
```

---

## Getting Help

If you encounter an issue not covered here:

1. **Check the error message carefully** - it often contains the solution
2. **Run validation scripts**: `python scripts/validate_results.py results/`
3. **Check system status**: `python scripts/check_system.py`
4. **Review recent changes** - did something break after a code update?

---

**Last Updated:** January 2026
