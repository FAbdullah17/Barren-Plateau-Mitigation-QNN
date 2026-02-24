"""Baseline training implementation using standard end-to-end optimization.

Trains the full parameterized quantum circuit simultaneously using gradient
descent. Serves as the control condition to demonstrate the barren plateau
problem, against which layerwise training and local cost function approaches
are compared.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from tqdm import tqdm

from src.models import QuantumNeuralNetwork
from src.evaluation.metrics import GradientTracker


class BaselineTrainer:
    """Standard end-to-end training for quantum neural networks."""
    
    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 4,
        learning_rate: float = 0.01,
        batch_size: int = 20,
        local_cost: bool = False,
        seed: Optional[int] = None
    ):
        """
        Initialize baseline trainer.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of circuit layers
            learning_rate: Learning rate for Adam optimizer
            batch_size: Batch size for training
            local_cost: Use local cost functions
            seed: Random seed for reproducibility
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.local_cost = local_cost
        self.seed = seed
        
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)
        
        # Initialize model
        self.model = QuantumNeuralNetwork(
            n_qubits=n_qubits,
            n_layers=n_layers,
            local_cost=local_cost
        )
        
        # Optimizer and loss
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        
        # Gradient tracker
        self.gradient_tracker = GradientTracker()
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'gradient_norms': [],
            'gradient_variance': []
        }
    
    def train(
        self,
        train_circuits: tf.Tensor,
        train_labels: np.ndarray,
        val_circuits: tf.Tensor,
        val_labels: np.ndarray,
        epochs: int = 50
    ) -> Dict:
        """
        Train the model.
        
        Args:
            train_circuits: Training quantum circuits
            train_labels: Training labels
            val_circuits: Validation quantum circuits
            val_labels: Validation labels
            epochs: Number of training epochs
            
        Returns:
            Dictionary containing training history and metrics
        """
        print(f"Starting baseline training for {epochs} epochs...")
        print(f"Model: {self.n_layers} layers, {self.model.get_num_parameters()} parameters")
        print(f"Local cost: {self.local_cost}")
        
        start_time = time.time()
        
        # Create dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((train_circuits, train_labels))
        train_dataset = train_dataset.shuffle(len(train_labels)).batch(self.batch_size)
        
        for epoch in range(epochs):
            epoch_loss = []
            epoch_acc = []
            epoch_gradients = []
            
            # Training loop
            for batch_circuits, batch_labels in tqdm(
                train_dataset, 
                desc=f"Epoch {epoch+1}/{epochs}",
                leave=False
            ):
                loss, acc, gradients = self._train_step(batch_circuits, batch_labels)
                epoch_loss.append(loss.numpy())
                epoch_acc.append(acc.numpy())
                epoch_gradients.extend([g.numpy() for g in gradients])
            
            # Compute metrics
            train_loss = np.mean(epoch_loss)
            train_acc = np.mean(epoch_acc)
            
            # Validation
            val_loss, val_acc = self._evaluate(val_circuits, val_labels)
            
            # Gradient statistics
            valid_gradients = [g for g in epoch_gradients if g is not None]
            if valid_gradients:
                grad_norm = np.mean([np.linalg.norm(g) for g in valid_gradients])
                grad_var = np.var([np.linalg.norm(g) for g in valid_gradients])
            else:
                grad_norm = 0.0
                grad_var = 0.0
            
            # Track gradients
            self.gradient_tracker.update(epoch_gradients)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['gradient_norms'].append(grad_norm)
            self.history['gradient_variance'].append(grad_var)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - "
                  f"Grad Norm: {grad_norm:.6f}")
        
        training_time = time.time() - start_time
        
        # Final evaluation
        test_loss, test_acc = self._evaluate(val_circuits, val_labels)
        
        results = {
            'history': self.history,
            'final_train_loss': self.history['train_loss'][-1],
            'final_train_acc': self.history['train_acc'][-1],
            'final_val_loss': self.history['val_loss'][-1],
            'final_val_acc': self.history['val_acc'][-1],
            'test_loss': test_loss,
            'test_acc': test_acc,
            'training_time': training_time,
            'gradient_stats': self.gradient_tracker.get_statistics(),
            'barren_plateau_detected': self.gradient_tracker.detect_barren_plateau()
        }
        
        print(f"\nTraining completed in {training_time:.2f}s")
        print(f"Final Test Accuracy: {test_acc:.4f}")
        print(f"Barren Plateau: {results['barren_plateau_detected']}")
        
        return results
    
    # Note: @tf.function removed for quantum circuit compatibility
    # TFQ already uses graph compilation internally
    def _train_step(self, circuits, labels):
        """Single training step."""
        with tf.GradientTape() as tape:
            predictions = self.model(circuits, training=True)
            # Squeeze predictions to match labels shape (batch,)
            predictions = tf.squeeze(predictions, axis=-1)
            loss = self.loss_fn(labels, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Compute accuracy
        predictions_binary = tf.cast(predictions > 0.5, tf.int32)
        labels_int = tf.cast(labels, tf.int32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions_binary, labels_int), tf.float32))
        
        # Return TensorFlow tensors (not numpy), let caller convert if needed
        return loss, accuracy, gradients
    
    def _evaluate(self, circuits, labels):
        """Evaluate on given data."""
        predictions = self.model(circuits, training=False)
        # Squeeze predictions to match labels shape (batch,)
        predictions = tf.squeeze(predictions, axis=-1)
        loss = self.loss_fn(labels, predictions).numpy()
        
        predictions_binary = tf.cast(predictions > 0.5, tf.int32)
        labels_int = tf.cast(labels, tf.int32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions_binary, labels_int), tf.float32)).numpy()
        
        return loss, accuracy
