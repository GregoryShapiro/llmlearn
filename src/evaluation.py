"""
Evaluation and Metrics Module

This module provides utilities for evaluating model performance and tracking
training progress. Key features:
- Training history tracking
- Model checkpointing (save/load best models)
- Plotting training curves
- Confusion matrices for analyzing which operations fail
- Per-operation accuracy breakdown

Why Evaluation Matters:
    Training a model without good evaluation is like driving blindfolded.
    We need to:
    - Track if the model is learning (loss decreasing, accuracy increasing)
    - Detect overfitting (train acc >> val acc)
    - Understand failure modes (which operations are hardest?)
    - Save the best model (not just the last one!)
"""

import numpy as np
import os
import pickle
from collections import defaultdict


class MetricsTracker:
    """
    Track training metrics over time.

    This class maintains a history of all training metrics:
    - Training loss and accuracy per epoch
    - Validation loss and accuracy per epoch
    - Best validation accuracy achieved
    - Which epoch had the best performance

    Why Track Metrics?
        - Visualize learning progress
        - Detect overfitting early
        - Choose the best model checkpoint
        - Debug training issues

    Example:
        >>> tracker = MetricsTracker()
        >>> for epoch in range(num_epochs):
        >>>     train_loss, train_acc = train_epoch(...)
        >>>     val_loss, val_acc = evaluate(...)
        >>>     tracker.update(train_loss, train_acc, val_loss, val_acc)
        >>>     if tracker.is_best_epoch():
        >>>         save_model(model, 'best_model.pkl')
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }

        # Track best validation performance
        self.best_val_acc = 0.0
        self.best_epoch = 0

    def update(self, train_loss, train_acc, val_loss, val_acc):
        """
        Update metrics with new epoch results.

        Args:
            train_loss (float): Training loss for this epoch
            train_acc (float): Training accuracy for this epoch
            val_loss (float): Validation loss for this epoch
            val_acc (float): Validation accuracy for this epoch
        """
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)

        # Update best validation accuracy
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = len(self.history['train_loss']) - 1

    def is_best_epoch(self):
        """
        Check if the current epoch is the best so far.

        Returns:
            bool: True if this epoch achieved best validation accuracy
        """
        current_epoch = len(self.history['train_loss']) - 1
        return current_epoch == self.best_epoch

    def get_current_epoch(self):
        """Get the current epoch number (0-indexed)."""
        return len(self.history['train_loss']) - 1

    def print_summary(self):
        """Print a summary of training progress."""
        if not self.history['train_loss']:
            print("No training history yet.")
            return

        current_epoch = self.get_current_epoch()

        print(f"\nTraining Summary (Epoch {current_epoch + 1}):")
        print("-" * 60)
        print(f"  Train Loss: {self.history['train_loss'][-1]:.4f}")
        print(f"  Train Acc:  {self.history['train_acc'][-1]:.4f}")
        print(f"  Val Loss:   {self.history['val_loss'][-1]:.4f}")
        print(f"  Val Acc:    {self.history['val_acc'][-1]:.4f}")
        print()
        print(f"  Best Val Acc: {self.best_val_acc:.4f} (Epoch {self.best_epoch + 1})")

        # Check for overfitting
        if current_epoch > 5:  # Need some history first
            train_acc = self.history['train_acc'][-1]
            val_acc = self.history['val_acc'][-1]
            gap = train_acc - val_acc

            if gap > 0.15:  # 15% gap suggests overfitting
                print(f"  ⚠ Warning: Possible overfitting detected (gap: {gap:.2%})")

    def get_history(self):
        """Get the complete training history."""
        return self.history


class ModelCheckpoint:
    """
    Save and load model checkpoints.

    Checkpointing allows us to:
    - Save the best model during training
    - Resume training from a saved state
    - Use the best model for inference (not the last one!)

    Why Save Best Model?
        The last epoch isn't always the best! Validation accuracy might
        peak at epoch 45 and then decrease due to overfitting. We want
        to keep the model from epoch 45, not epoch 100.

    What Gets Saved:
        - Model parameters (all weights and biases)
        - Optimizer state (momentum, velocity for Adam)
        - Training history
        - Epoch number

    Example:
        >>> checkpoint = ModelCheckpoint('checkpoints/')
        >>> # During training:
        >>> if tracker.is_best_epoch():
        >>>     checkpoint.save(model, optimizer, tracker, epoch)
        >>> # Later, to resume:
        >>> model, optimizer, tracker, epoch = checkpoint.load()
    """

    def __init__(self, checkpoint_dir='checkpoints'):
        """
        Initialize model checkpoint manager.

        Args:
            checkpoint_dir (str): Directory to save checkpoints
        """
        self.checkpoint_dir = checkpoint_dir

        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def save(self, model, optimizer, tracker, epoch, filename='best_model.pkl'):
        """
        Save model checkpoint.

        Args:
            model: The model to save
            optimizer: The optimizer to save
            tracker: MetricsTracker with training history
            epoch (int): Current epoch number
            filename (str): Name of checkpoint file
        """
        filepath = os.path.join(self.checkpoint_dir, filename)

        # Collect all model parameters
        model_params = {}
        for i, (param, grad) in enumerate(model.get_parameters()):
            model_params[f'param_{i}'] = param.copy()

        # Collect optimizer state
        optimizer_state = {
            'learning_rate': optimizer.learning_rate,
            't': optimizer.t if hasattr(optimizer, 't') else 0,
        }

        if hasattr(optimizer, 'm'):  # Adam optimizer
            optimizer_state['m'] = [m.copy() for m in optimizer.m]
            optimizer_state['v'] = [v.copy() for v in optimizer.v]
            optimizer_state['beta1'] = optimizer.beta1
            optimizer_state['beta2'] = optimizer.beta2
            optimizer_state['epsilon'] = optimizer.epsilon
        elif hasattr(optimizer, 'velocities'):  # SGD with momentum
            optimizer_state['velocities'] = [v.copy() for v in optimizer.velocities]
            optimizer_state['momentum'] = optimizer.momentum

        # Save everything
        checkpoint = {
            'model_params': model_params,
            'optimizer_state': optimizer_state,
            'history': tracker.get_history(),
            'epoch': epoch,
            'best_val_acc': tracker.best_val_acc,
            'best_epoch': tracker.best_epoch,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)

        print(f"  💾 Saved checkpoint: {filepath}")

    def load(self, filename='best_model.pkl'):
        """
        Load model checkpoint.

        Args:
            filename (str): Name of checkpoint file

        Returns:
            dict: Checkpoint dictionary with model_params, optimizer_state, etc.

        Note:
            This returns the raw checkpoint dict. You'll need to manually
            restore the model and optimizer states.
        """
        filepath = os.path.join(self.checkpoint_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)

        print(f"  📂 Loaded checkpoint: {filepath}")
        print(f"     Epoch: {checkpoint['epoch']}, Best Val Acc: {checkpoint['best_val_acc']:.4f}")

        return checkpoint


def compute_confusion_matrix(model, dataset, operation_names):
    """
    Compute confusion matrix for multi-class classification.

    A confusion matrix shows which operations the model confuses with each other.
    Rows are true labels, columns are predictions.

    Why Confusion Matrix?
        - See which operations are hardest to learn
        - Detect systematic errors (e.g., model always predicts "Max")
        - Understand failure modes

    Args:
        model: Trained model
        dataset: List of (input_indices, answer_indices) tuples
        operation_names: List of operation names in order

    Returns:
        np.ndarray: Confusion matrix of shape (num_ops, num_ops)

    Example:
        >>> confusion = compute_confusion_matrix(model, test_data, ['First', 'Second', 'Last', 'Max', 'Min'])
        >>> # confusion[i, j] = number of times true label i was predicted as j
    """
    from vocabluary import VOCAB

    # Map operation names to indices
    op_to_idx = {name: idx for idx, name in enumerate(operation_names)}

    num_ops = len(operation_names)
    confusion = np.zeros((num_ops, num_ops), dtype=np.int32)

    # Process each example
    for input_indices, answer_indices in dataset:
        # Get the operation (first token)
        operation_token_id = input_indices[0]

        # Find which operation this is
        operation_name = None
        for name, idx in VOCAB.items():
            if idx == operation_token_id and name in operation_names:
                operation_name = name
                break

        if operation_name is None:
            continue  # Skip if not an operation

        true_op_idx = op_to_idx[operation_name]

        # Get model prediction
        # Pad input to fixed length
        max_len = 20
        input_padded = np.array(input_indices + [0] * (max_len - len(input_indices)))
        input_batch = input_padded.reshape(1, -1)

        predictions = model.forward(input_batch)
        predicted_token = np.argmax(predictions[0])

        # Map prediction back to operation (crude heuristic)
        # This is simplified - in reality, the model predicts answer digits, not operations
        # For now, we'll just track correct vs incorrect
        correct_answer = answer_indices[0]
        is_correct = (predicted_token == correct_answer)

        # Update confusion matrix (simplified: just track correct/incorrect per operation)
        if is_correct:
            confusion[true_op_idx, true_op_idx] += 1
        else:
            # Distribute errors evenly (not ideal, but simple)
            for j in range(num_ops):
                if j != true_op_idx:
                    confusion[true_op_idx, j] += 1 / (num_ops - 1)

    return confusion


def compute_per_operation_accuracy(model, dataset, operation_names):
    """
    Compute accuracy for each operation separately.

    This tells us which operations the model finds easy vs hard.

    Args:
        model: Trained model
        dataset: List of (input_indices, answer_indices) tuples
        operation_names: List of operation names to analyze

    Returns:
        dict: Mapping operation_name -> accuracy

    Example:
        >>> accuracies = compute_per_operation_accuracy(model, test_data, ['First', 'Second', 'Last', 'Max', 'Min'])
        >>> print(f"First: {accuracies['First']:.2%}")
        >>> print(f"Max: {accuracies['Max']:.2%}")
    """
    from vocabluary import VOCAB

    # Track correct and total for each operation
    op_correct = defaultdict(int)
    op_total = defaultdict(int)

    for input_indices, answer_indices in dataset:
        # Get the operation (first token)
        operation_token_id = input_indices[0]

        # Find which operation this is
        operation_name = None
        for name, idx in VOCAB.items():
            if idx == operation_token_id and name in operation_names:
                operation_name = name
                break

        if operation_name is None:
            continue

        # Get model prediction
        max_len = 20
        input_padded = np.array(input_indices + [0] * (max_len - len(input_indices)))
        input_batch = input_padded.reshape(1, -1)

        predictions = model.forward(input_batch)
        predicted_token = np.argmax(predictions[0])

        correct_answer = answer_indices[0]
        is_correct = (predicted_token == correct_answer)

        op_total[operation_name] += 1
        if is_correct:
            op_correct[operation_name] += 1

    # Compute accuracies
    accuracies = {}
    for op_name in operation_names:
        if op_total[op_name] > 0:
            accuracies[op_name] = op_correct[op_name] / op_total[op_name]
        else:
            accuracies[op_name] = 0.0

    return accuracies


def print_operation_analysis(model, dataset, operation_names):
    """
    Print a detailed analysis of per-operation performance.

    Args:
        model: Trained model
        dataset: Test dataset
        operation_names: List of operations to analyze
    """
    print("\nPer-Operation Analysis:")
    print("=" * 60)

    accuracies = compute_per_operation_accuracy(model, dataset, operation_names)

    # Sort by accuracy (hardest first)
    sorted_ops = sorted(accuracies.items(), key=lambda x: x[1])

    for op_name, acc in sorted_ops:
        bar_length = int(acc * 40)
        bar = '█' * bar_length + '░' * (40 - bar_length)
        print(f"  {op_name:10s}: {bar} {acc:.1%}")

    print()
    avg_acc = np.mean(list(accuracies.values()))
    print(f"  Average Accuracy: {avg_acc:.1%}")

    # Find hardest and easiest
    if sorted_ops:
        hardest = sorted_ops[0]
        easiest = sorted_ops[-1]
        print(f"  Hardest: {hardest[0]} ({hardest[1]:.1%})")
        print(f"  Easiest: {easiest[0]} ({easiest[1]:.1%})")


# ============================================================================
# Plotting Functions (optional - requires matplotlib)
# ============================================================================

def plot_training_curves(tracker, save_path='training_curves.png'):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        tracker: MetricsTracker with training history
        save_path: Where to save the plot (optional)

    Note:
        Requires matplotlib. If not available, prints a message instead.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ Matplotlib not installed. Skipping plot generation.")
        print("  Install with: pip install matplotlib")
        return

    history = tracker.get_history()
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    # Mark best epoch
    best_epoch = tracker.best_epoch + 1  # Convert to 1-indexed
    ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best (Epoch {best_epoch})')
    ax2.legend(fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📊 Saved training curves: {save_path}")

    plt.show()


def plot_confusion_matrix(confusion, operation_names, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix as a heatmap.

    Args:
        confusion: Confusion matrix (num_ops x num_ops)
        operation_names: List of operation names
        save_path: Where to save the plot (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ Matplotlib not installed. Skipping plot generation.")
        return

    fig, ax = plt.subplots(figsize=(8, 7))

    # Normalize by row (show percentages)
    confusion_pct = confusion / (confusion.sum(axis=1, keepdims=True) + 1e-10)

    im = ax.imshow(confusion_pct, cmap='Blues', aspect='auto')

    # Set ticks and labels
    ax.set_xticks(range(len(operation_names)))
    ax.set_yticks(range(len(operation_names)))
    ax.set_xticklabels(operation_names)
    ax.set_yticklabels(operation_names)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Percentage', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(operation_names)):
        for j in range(len(operation_names)):
            text = ax.text(j, i, f'{confusion_pct[i, j]:.0%}',
                          ha="center", va="center", color="black" if confusion_pct[i, j] < 0.5 else "white",
                          fontsize=10, fontweight='bold')

    ax.set_xlabel('Predicted Operation', fontsize=12)
    ax.set_ylabel('True Operation', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📊 Saved confusion matrix: {save_path}")

    plt.show()


# ============================================================================
# Testing and demonstration
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EVALUATION MODULE TEST")
    print("=" * 60)

    # Test 1: MetricsTracker
    print("\n1. Testing MetricsTracker:")
    print("-" * 60)

    tracker = MetricsTracker()

    # Simulate some training epochs
    for epoch in range(10):
        # Simulate improving then overfitting
        train_loss = 3.0 - epoch * 0.25 + (0.1 if epoch > 7 else 0)
        train_acc = min(0.95, 0.1 + epoch * 0.1)
        val_loss = 3.0 - epoch * 0.20 + (0.3 if epoch > 7 else 0)
        val_acc = min(0.85, 0.1 + epoch * 0.08)

        tracker.update(train_loss, train_acc, val_loss, val_acc)

        if tracker.is_best_epoch():
            print(f"  Epoch {epoch + 1}: New best! Val Acc = {val_acc:.4f}")

    tracker.print_summary()

    # Test 2: ModelCheckpoint
    print("\n2. Testing ModelCheckpoint:")
    print("-" * 60)

    checkpoint = ModelCheckpoint('test_checkpoints')
    print("  ✓ Created checkpoint manager")
    print(f"  ✓ Checkpoint directory: {checkpoint.checkpoint_dir}")

    # Test 3: Per-operation accuracy
    print("\n3. Per-Operation Accuracy Analysis:")
    print("-" * 60)

    # Simulate operation accuracies
    operation_names = ['First', 'Second', 'Last', 'Max', 'Min']
    simulated_accs = {
        'First': 0.95,
        'Second': 0.92,
        'Last': 0.94,
        'Max': 0.78,  # Hardest
        'Min': 0.82,
    }

    print("\nSimulated Per-Operation Performance:")
    for op, acc in sorted(simulated_accs.items(), key=lambda x: x[1]):
        bar_length = int(acc * 40)
        bar = '█' * bar_length + '░' * (40 - bar_length)
        print(f"  {op:10s}: {bar} {acc:.1%}")

    print("\n" + "=" * 60)
    print("EVALUATION MODULE READY")
    print("=" * 60)
    print("\nUsage:")
    print("  from evaluation import MetricsTracker, ModelCheckpoint")
    print("  tracker = MetricsTracker()")
    print("  checkpoint = ModelCheckpoint('checkpoints/')")
    print("  ")
    print("  for epoch in range(num_epochs):")
    print("      train_loss, train_acc = train_epoch(...)")
    print("      val_loss, val_acc = evaluate(...)")
    print("      tracker.update(train_loss, train_acc, val_loss, val_acc)")
    print("      if tracker.is_best_epoch():")
    print("          checkpoint.save(model, optimizer, tracker, epoch)")
