"""Layerwise training implementation (Skolik et al., 2020).

Developer Assignment (Weeks 3-4):
    Primary: Asma Zubair - Layerwise approach implementation
    Testing: All team members (Weeks 5-6)
    Experiments: Asma (4L Week 7), Frahan (6L Week 8), Fahad (8L Week 9)
"""

import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np
from typing import Dict, Optional
import time
from tqdm import tqdm

from ..models import LayerwiseQNN
from ..evaluation.metrics import GradientTracker


class LayerwiseTrainer:
    """Layerwise training for quantum neural networks."""
    
    def __init__(
        self,
        n_qubits: int = 4,
        target_layers: int = 4,
        learning_rate: float = 0.01,
        batch_size: int = 20,
        epochs_per_layer: int = 10,
        finetune_epochs: int = 10,
        local_cost: bool = False,
        seed: Optional[int] = None
    ):
        """
        Initialize layerwise trainer.
        
        Args:
            n_qubits: Number of qubits
            target_layers: Target number of layers
            learning_rate: Learning rate for Adam optimizer
            batch_size: Batch size for training
            epochs_per_layer: Epochs to train each new layer
            finetune_epochs: Epochs for final fine-tuning
            local_cost: Use local cost functions
            seed: Random seed for reproducibility
        """
        self.n_qubits = n_qubits
        self.target_layers = target_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs_per_layer = epochs_per_layer
        self.finetune_epochs = finetune_epochs
        self.local_cost = local_cost
        self.seed = seed
        
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)
        
        # Initialize layerwise QNN
        self.qnn = LayerwiseQNN(
            n_qubits=n_qubits,
            target_layers=target_layers,
            local_cost=local_cost
        )
        
        # Loss function
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
            'gradient_variance': [],
            'layer_transitions': []  # Mark when layers are added
        }
    
    def train(
        self,
        train_circuits: tf.Tensor,
        train_labels: np.ndarray,
        val_circuits: tf.Tensor,
        val_labels: np.ndarray
    ) -> Dict:
        """
        Train the model layer by layer.
        
        Args:
            train_circuits: Training quantum circuits
            train_labels: Training labels
            val_circuits: Validation quantum circuits
            val_labels: Validation labels
            
        Returns:
            Dictionary containing training history and metrics
        """
        print(f"Starting layerwise training for {self.target_layers} layers...")
        print(f"Epochs per layer: {self.epochs_per_layer}")
        print(f"Fine-tuning epochs: {self.finetune_epochs}")
        print(f"Local cost: {self.local_cost}")
        
        start_time = time.time()
        
        # Train each layer incrementally
        for layer_idx in range(self.target_layers):
            print(f"\n{'='*60}")
            print(f"Adding and training layer {layer_idx + 1}/{self.target_layers}")
            print(f"{'='*60}")
            
            # Add new layer
            model = self.qnn.add_layer()
            
            # Create optimizer for this layer
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            
            # Mark layer transition in history
            self.history['layer_transitions'].append(len(self.history['train_loss']))
            
            # Train this layer
            self._train_layer(
                model,
                optimizer,
                train_circuits,
                train_labels,
                val_circuits,
                val_labels,
                epochs=self.epochs_per_layer
            )
        
        # Fine-tuning phase: train all layers together
        if self.finetune_epochs > 0:
            print(f"\n{'='*60}")
            print(f"Fine-tuning all {self.target_layers} layers")
            print(f"{'='*60}")
            
            model = self.qnn.get_current_model()
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            
            self._train_layer(
                model,
                optimizer,
                train_circuits,
                train_labels,
                val_circuits,
                val_labels,
                epochs=self.finetune_epochs
            )
        
        training_time = time.time() - start_time
        
        # Final evaluation
        model = self.qnn.get_current_model()
        test_loss, test_acc = self._evaluate(model, val_circuits, val_labels)
        
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
        
        print(f"\nLayerwise training completed in {training_time:.2f}s")
        print(f"Final Test Accuracy: {test_acc:.4f}")
        print(f"Barren Plateau: {results['barren_plateau_detected']}")
        
        return results
    
    def _train_layer(
        self,
        model,
        optimizer,
        train_circuits,
        train_labels,
        val_circuits,
        val_labels,
        epochs: int
    ):
        """Train current layer configuration."""
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
                loss, acc, gradients = self._train_step(
                    model, optimizer, batch_circuits, batch_labels
                )
                epoch_loss.append(loss)
                epoch_acc.append(acc)
                epoch_gradients.extend(gradients)
            
            # Compute metrics
            train_loss = np.mean(epoch_loss)
            train_acc = np.mean(epoch_acc)
            
            # Validation
            val_loss, val_acc = self._evaluate(model, val_circuits, val_labels)
            
            # Gradient statistics
            grad_norm = np.mean([np.linalg.norm(g) for g in epoch_gradients])
            grad_var = np.var([np.linalg.norm(g) for g in epoch_gradients])
            
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
    
    # Note: @tf.function removed for TFQ compatibility
    def _train_step(self, model, optimizer, circuits, labels):
        """Single training step."""
        with tf.GradientTape() as tape:
            predictions = model(circuits, training=True)
            # Squeeze predictions to match labels shape (batch,)
            predictions = tf.squeeze(predictions, axis=-1)
            loss = self.loss_fn(labels, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Compute accuracy
        predictions_binary = tf.cast(predictions > 0.5, tf.int32)
        labels_int = tf.cast(labels, tf.int32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions_binary, labels_int), tf.float32))
        
        return loss.numpy(), accuracy.numpy(), [g.numpy() for g in gradients]
    
    def _evaluate(self, model, circuits, labels):
        """Evaluate on given data."""
        predictions = model(circuits, training=False)
        # Squeeze predictions to match labels shape (batch,)
        predictions = tf.squeeze(predictions, axis=-1)
        loss = self.loss_fn(labels, predictions).numpy()
        
        predictions_binary = tf.cast(predictions > 0.5, tf.int32)
        labels_int = tf.cast(labels, tf.int32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions_binary, labels_int), tf.float32)).numpy()
        
        return loss, accuracy
