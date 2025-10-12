"""
Training Utilities Module

This module provides utility functions for training neural networks.

Training a neural network involves many repetitive steps:
- Forward pass through model
- Compute loss
- Backward pass to compute gradients
- Update parameters with optimizer
- Track metrics

This module provides clean abstractions for these operations.
"""

import numpy as np
from loss import CrossEntropyLoss, compute_accuracy


def train_step(model, batch_tokens, batch_targets, loss_fn, optimizer):
    """
    Perform one training step (forward + backward + update).

    This is the core training loop operation. It:
    1. Runs forward pass to get predictions
    2. Computes loss
    3. Runs backward pass to compute gradients
    4. Updates parameters using optimizer

    Why Separate Function?
        - Reduces code duplication
        - Makes training loop cleaner
        - Easier to debug and test
        - Consistent training logic

    Args:
        model: The model to train (must have forward, backward, zero_grad methods)
        batch_tokens (np.ndarray): Input token IDs, shape (batch_size, seq_len)
        batch_targets (np.ndarray): Target labels, shape (batch_size,)
        loss_fn: Loss function (e.g., CrossEntropyLoss)
        optimizer: Optimizer (e.g., Adam)

    Returns:
        tuple: (loss, accuracy) for this batch
    """
    # Clear gradients from previous step
    model.zero_grad()

    # Forward pass: get predictions
    predictions = model.forward(batch_tokens)

    # Compute loss
    loss = loss_fn.forward(predictions, batch_targets)

    # Compute accuracy
    accuracy = compute_accuracy(predictions, batch_targets)

    # Backward pass: compute gradients
    grad_output = loss_fn.backward()
    model.backward(grad_output)

    # Update parameters
    optimizer.step()

    return loss, accuracy


def evaluate(model, data_tokens, data_targets, loss_fn, batch_size=32):
    """
    Evaluate model on a dataset without updating parameters.

    This is used for validation and test sets. We don't compute gradients
    or update parameters - just measure performance.

    Why Separate Evaluation?
        - Validation during training (monitor overfitting)
        - Final test set evaluation
        - No gradient computation saves memory and time

    Args:
        model: The model to evaluate
        data_tokens (np.ndarray): All input tokens, shape (num_examples, seq_len)
        data_targets (np.ndarray): All target labels, shape (num_examples,)
        loss_fn: Loss function
        batch_size (int): Batch size for evaluation (doesn't affect results)

    Returns:
        tuple: (average_loss, average_accuracy) across entire dataset
    """
    num_examples = len(data_tokens)
    num_batches = (num_examples + batch_size - 1) // batch_size  # Ceiling division

    total_loss = 0.0
    total_accuracy = 0.0

    for batch_idx in range(num_batches):
        # Get batch
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_examples)

        batch_tokens = data_tokens[start_idx:end_idx]
        batch_targets = data_targets[start_idx:end_idx]

        # Forward pass only (no backward pass)
        predictions = model.forward(batch_tokens)

        # Compute metrics
        batch_loss = loss_fn.forward(predictions, batch_targets)
        batch_accuracy = compute_accuracy(predictions, batch_targets)

        # Accumulate (weighted by batch size for correct averaging)
        batch_weight = len(batch_tokens)
        total_loss += batch_loss * batch_weight
        total_accuracy += batch_accuracy * batch_weight

    # Average across all examples
    avg_loss = total_loss / num_examples
    avg_accuracy = total_accuracy / num_examples

    return avg_loss, avg_accuracy


def train_epoch(model, train_tokens, train_targets, loss_fn, optimizer, batch_size=32, verbose=True):
    """
    Train for one complete epoch (one pass through training data).

    An epoch is one complete pass through the entire training dataset.
    We typically train for many epochs (e.g., 50-100).

    What Happens in an Epoch:
        1. Shuffle training data (prevents overfitting to batch order)
        2. Split into batches
        3. For each batch: forward, backward, update
        4. Track and report metrics

    Why Shuffle?
        Shuffling prevents the model from learning patterns in the batch order.
        For example, if all "Max" examples came first, the model might
        temporarily forget about "Min" operations.

    Args:
        model: The model to train
        train_tokens (np.ndarray): Training input tokens
        train_targets (np.ndarray): Training target labels
        loss_fn: Loss function
        optimizer: Optimizer
        batch_size (int): Number of examples per batch
        verbose (bool): Whether to print progress

    Returns:
        tuple: (average_loss, average_accuracy) for this epoch
    """
    num_examples = len(train_tokens)
    num_batches = (num_examples + batch_size - 1) // batch_size

    # Shuffle training data
    # This is important for SGD - we want random batches each epoch
    shuffle_indices = np.random.permutation(num_examples)
    train_tokens = train_tokens[shuffle_indices]
    train_targets = train_targets[shuffle_indices]

    total_loss = 0.0
    total_accuracy = 0.0

    for batch_idx in range(num_batches):
        # Get batch
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_examples)

        batch_tokens = train_tokens[start_idx:end_idx]
        batch_targets = train_targets[start_idx:end_idx]

        # Perform training step
        batch_loss, batch_accuracy = train_step(
            model, batch_tokens, batch_targets, loss_fn, optimizer
        )

        # Accumulate
        batch_weight = len(batch_tokens)
        total_loss += batch_loss * batch_weight
        total_accuracy += batch_accuracy * batch_weight

        # Optional: print progress
        if verbose and (batch_idx + 1) % max(1, num_batches // 10) == 0:
            print(f"  Batch {batch_idx + 1}/{num_batches} - "
                  f"Loss: {batch_loss:.4f}, Acc: {batch_accuracy:.4f}")

    # Average across all examples
    avg_loss = total_loss / num_examples
    avg_accuracy = total_accuracy / num_examples

    return avg_loss, avg_accuracy


def train(model, train_data, val_data, loss_fn, optimizer, num_epochs=10, batch_size=32, verbose=True):
    """
    Complete training loop for multiple epochs.

    This is the high-level training function that:
    1. Trains for multiple epochs
    2. Evaluates on validation set after each epoch
    3. Tracks training history
    4. Can implement early stopping

    Training Philosophy:
        - Train on training set (update parameters)
        - Validate on validation set (monitor overfitting)
        - Final evaluation on test set (measure generalization)

    Why Validation?
        Validation accuracy tells us if the model is overfitting:
        - Training acc ↑, Val acc ↑: Good! Model is learning and generalizing
        - Training acc ↑, Val acc →: Starting to overfit, consider stopping
        - Training acc ↑, Val acc ↓: Overfitting! Stop training

    Args:
        model: The model to train
        train_data: Tuple of (train_tokens, train_targets)
        val_data: Tuple of (val_tokens, val_targets)
        loss_fn: Loss function
        optimizer: Optimizer
        num_epochs (int): Number of epochs to train
        batch_size (int): Batch size
        verbose (bool): Whether to print progress

    Returns:
        dict: Training history with keys 'train_loss', 'train_acc', 'val_loss', 'val_acc'
    """
    train_tokens, train_targets = train_data
    val_tokens, val_targets = val_data

    # Track training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    if verbose:
        print(f"\nTraining for {num_epochs} epochs...")
        print(f"Training set: {len(train_tokens)} examples")
        print(f"Validation set: {len(val_tokens)} examples")
        print(f"Batch size: {batch_size}\n")

    for epoch in range(num_epochs):
        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs}")

        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_tokens, train_targets, loss_fn, optimizer,
            batch_size=batch_size, verbose=False  # Don't print batch-level progress
        )

        # Evaluate on validation set
        val_loss, val_acc = evaluate(
            model, val_tokens, val_targets, loss_fn, batch_size=batch_size
        )

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print epoch summary
        if verbose:
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print()

    if verbose:
        print("Training complete!")
        print(f"Final Train Accuracy: {history['train_acc'][-1]:.4f}")
        print(f"Final Val Accuracy: {history['val_acc'][-1]:.4f}")

    return history
