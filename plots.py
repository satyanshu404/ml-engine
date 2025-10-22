import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_training_curves(self):
    """Plot training and validation loss curves"""
    if not self.trainer:
        return

    plt.figure(figsize=(10, 6))
    epochs = range(1, len(self.trainer.train_losses) + 1)
    plt.plot(epochs, self.trainer.train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, self.trainer.val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Curves - fwd_{self.fwd}')
    plt.legend()
    plt.grid(True)
    plt.savefig(self.plots_dir / f'training_curves_fwd_{self.fwd}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_predictions_scatter(self, targets: np.ndarray, predictions: np.ndarray):
    """Plot predictions vs targets scatter plot"""
    plt.figure(figsize=(8, 8))
    plt.scatter(targets, predictions, alpha=0.6, s=12)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Test Predictions vs Actual - fwd_{self.fwd}')

    min_val, max_val = min(targets.min(), predictions.min()), max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)

    plt.grid(True, alpha=0.3)
    plt.savefig(self.plots_dir / f'predictions_scatter_fwd_{self.fwd}.png', dpi=150, bbox_inches='tight')
    plt.close()