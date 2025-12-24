import tensorflow as tf
from pathlib import Path
from typing import Optional, Tuple


class CheckpointManager:
    """Manages TensorFlow checkpoints for training."""
    
    def __init__(self, checkpoint_dir: str, model: tf.keras.Model, optimizer):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            model: TensorFlow model to checkpoint
            optimizer: Optimizer to checkpoint
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint
        self.checkpoint = tf.train.Checkpoint(
            optimizer=optimizer,
            model=model
        )
        
        # Create manager (keeps last 3 checkpoints)
        self.manager = tf.train.CheckpointManager(
            self.checkpoint,
            self.checkpoint_dir,
            max_to_keep=3
        )
    
    def save(self) -> str:
        """Save checkpoint and return path."""
        save_path = self.manager.save()
        return save_path
    
    def restore(self) -> bool:
        """
        Restore from latest checkpoint.
        
        Returns:
            True if checkpoint was restored, False if none found
        """
        latest = self.manager.latest_checkpoint
        if latest:
            status = self.checkpoint.restore(latest)
            status.expect_partial()  # Some variables might not be in checkpoint
            return True
        return False
    
    def latest_checkpoint_path(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        return self.manager.latest_checkpoint


def setup_checkpoint(
    checkpoint_dir: str,
    model: tf.keras.Model,
    optimizer
) -> CheckpointManager:
    """Convenience function to setup checkpointing."""
    return CheckpointManager(checkpoint_dir, model, optimizer)


