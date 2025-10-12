"""
Loss Functions Module

This module implements loss functions for training neural networks.

What is a Loss Function?
    A loss function measures how far the model's predictions are from the true labels.
    During training, we minimize this loss by adjusting model parameters.

    The loss function is the bridge between:
    - Forward pass: model makes predictions
    - Backward pass: gradients flow to update parameters

Why Cross-Entropy Loss for Classification?
    Cross-entropy is the standard loss for classification because it:
    1. Measures the difference between two probability distributions
    2. Penalizes confident wrong predictions heavily
    3. Has good gradient properties (doesn't saturate for wrong predictions)
    4. Derived from maximum likelihood estimation

Mathematical Foundation:
    For classification, we want to maximize the likelihood of correct predictions.
    This is equivalent to minimizing negative log-likelihood (cross-entropy).

    Formula: L = -log(p_correct)
    Where p_correct is the predicted probability for the correct class.

Why Negative Log?
    - log(1) = 0: Perfect prediction has zero loss
    - log(0) = -∞: Completely wrong prediction has infinite loss
    - Negative makes it a minimization problem
    - Convex optimization landscape (easier to optimize)

This module implements:
    - CrossEntropyLoss: Standard cross-entropy loss for classification
"""

import numpy as np


class CrossEntropyLoss:
    """
    Cross-Entropy Loss for multi-class classification.

    This is the standard loss function for classification tasks. It measures
    how well the predicted probability distribution matches the true distribution.

    Mathematical Formulation:
        For a single example:
            L = -log(p[y])
        Where:
            - p is the predicted probability distribution (after softmax)
            - y is the true class label (integer)

        For a batch of N examples:
            L = -(1/N) * sum(log(p_i[y_i]) for i in range(N))

    Why This Formula?
        Cross-entropy comes from information theory. It measures the expected
        number of bits needed to encode one distribution using another distribution.

        Intuition:
        - If prediction is correct with high confidence: log(0.99) ≈ -0.01 (low loss)
        - If prediction is correct with low confidence: log(0.10) ≈ -2.3 (high loss)
        - If prediction is wrong: log(0.01) ≈ -4.6 (very high loss)

    Gradient of Cross-Entropy + Softmax:
        The beautiful property: When combined with softmax, the gradient simplifies!

        For softmax followed by cross-entropy:
            ∂L/∂logits = p - y_onehot

        Where:
            - p is the predicted probability (after softmax)
            - y_onehot is the one-hot encoded true label

        This means the gradient is just the difference between predictions and labels!
        No complex chain rule computation needed.

    Numerical Stability:
        We must handle log(0) which is undefined. We add a small epsilon (1e-15)
        to prevent taking log of exactly zero.

        More stable approach: compute log-softmax directly (not implemented here
        for simplicity, but used in production code).

    Alternative Loss Functions (not implemented):
        - Mean Squared Error: For regression, not suitable for classification
        - Hinge Loss: Used in SVMs, less common for neural networks
        - Focal Loss: For imbalanced datasets, down-weights easy examples
        - Label Smoothing: Prevents overconfident predictions

    Args:
        None - CrossEntropyLoss has no learnable parameters

    Attributes:
        predictions_cache (np.ndarray): Cached predictions for backward pass
        targets_cache (np.ndarray): Cached targets for backward pass
        loss_cache (float): Cached loss value

    Example:
        >>> loss_fn = CrossEntropyLoss()
        >>> predictions = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])  # (batch=2, classes=3)
        >>> targets = np.array([1, 0])  # Correct classes
        >>> loss = loss_fn.forward(predictions, targets)  # Scalar loss value
        >>> grad = loss_fn.backward()  # Gradient w.r.t. predictions
    """

    def __init__(self):
        """
        Initialize Cross-Entropy Loss.

        No parameters to initialize - this is a fixed function.
        """
        self.predictions_cache = None
        self.targets_cache = None
        self.loss_cache = None

    def forward(self, predictions, targets):
        """
        Forward pass: compute cross-entropy loss.

        Computation Steps:
            1. Extract predicted probabilities for correct classes
            2. Take negative log of these probabilities
            3. Average across the batch

        Why Average Instead of Sum?
            Averaging makes the loss independent of batch size.
            This allows us to use the same learning rate for different batch sizes.

        Numerical Stability:
            We clip predictions to [eps, 1-eps] to prevent log(0).
            In practice, eps=1e-15 is small enough to not affect training.

        Args:
            predictions (np.ndarray): Predicted probabilities, shape (batch_size, num_classes)
                                     Should be output of softmax (values in [0,1], sum to 1)
            targets (np.ndarray): True class labels, shape (batch_size,)
                                 Integer labels in range [0, num_classes-1]

        Returns:
            float: Average cross-entropy loss across the batch

        Example:
            >>> predictions = np.array([[0.1, 0.9], [0.8, 0.2]])
            >>> targets = np.array([1, 0])  # First example is class 1, second is class 0
            >>> loss = loss_fn.forward(predictions, targets)
            >>> # loss ≈ -(log(0.9) + log(0.8)) / 2 ≈ 0.161
        """
        # Cache for backward pass
        self.predictions_cache = predictions
        self.targets_cache = targets

        batch_size = predictions.shape[0]

        # Numerical stability: clip probabilities to prevent log(0)
        # We use a very small epsilon to avoid affecting the optimization
        eps = 1e-15
        predictions_clipped = np.clip(predictions, eps, 1 - eps)

        # Extract predicted probabilities for the correct classes
        # This uses advanced indexing: for each row i, get element at column targets[i]
        correct_class_probs = predictions_clipped[np.arange(batch_size), targets]

        # Compute negative log-likelihood (cross-entropy)
        # Shape: (batch_size,) - one loss per example
        individual_losses = -np.log(correct_class_probs)

        # Average across batch
        loss = np.mean(individual_losses)

        # Cache for tracking
        self.loss_cache = loss

        return loss

    def backward(self):
        """
        Backward pass: compute gradient of loss with respect to predictions.

        The Magic of Softmax + Cross-Entropy:
            When cross-entropy follows softmax, the combined gradient simplifies beautifully:

            ∂L/∂logits = (p - y) / N

            Where:
            - p is the predicted probability distribution (shape: batch_size × num_classes)
            - y is one-hot encoded true labels
            - N is the batch size

        Why This Form?
            The gradient is proportional to the prediction error!
            - If we predict 0.9 for a class that's actually 1.0, gradient is (0.9 - 1.0) = -0.1
            - If we predict 0.9 for a class that's actually 0.0, gradient is (0.9 - 0.0) = +0.9

            Large errors → large gradients → bigger updates
            Small errors → small gradients → smaller updates

        Derivation Intuition:
            Cross-entropy: L = -sum(y * log(p))
            Taking derivative w.r.t. logits (before softmax):
            Through chain rule: ∂L/∂logits = ∂L/∂p * ∂p/∂logits

            The Jacobian of softmax cancels beautifully with the log derivative!
            Result: Just (p - y)

        Implementation:
            We create a one-hot encoded version of targets, subtract from predictions,
            and divide by batch size (because we averaged the loss).

        Returns:
            np.ndarray: Gradient of loss w.r.t. predictions, shape (batch_size, num_classes)

        Example:
            >>> predictions = np.array([[0.2, 0.8], [0.7, 0.3]])
            >>> targets = np.array([1, 0])
            >>> loss = loss_fn.forward(predictions, targets)
            >>> grad = loss_fn.backward()
            >>> # grad ≈ [[0.2/2, -0.2/2], [-0.3/2, 0.3/2]]
            >>> # Positive where we over-predicted, negative where we under-predicted
        """
        predictions = self.predictions_cache
        targets = self.targets_cache

        batch_size, num_classes = predictions.shape

        # Create one-hot encoded targets
        # Shape: (batch_size, num_classes)
        # For each row, all zeros except 1.0 at the target class index
        targets_onehot = np.zeros_like(predictions)
        targets_onehot[np.arange(batch_size), targets] = 1.0

        # Gradient: difference between predictions and true distribution
        # This is the combined gradient of cross-entropy + softmax
        grad_predictions = predictions - targets_onehot

        # Divide by batch size (because we averaged the loss in forward pass)
        grad_predictions = grad_predictions / batch_size

        return grad_predictions

    def __call__(self, predictions, targets):
        """
        Make the loss function callable.

        This allows using: loss = loss_fn(predictions, targets)
        Instead of: loss = loss_fn.forward(predictions, targets)
        """
        return self.forward(predictions, targets)

    def __repr__(self):
        """String representation for debugging."""
        return "CrossEntropyLoss()"


# Optional: Utility function for computing accuracy
def compute_accuracy(predictions, targets):
    """
    Compute classification accuracy.

    Accuracy = (number of correct predictions) / (total predictions)

    This is a simple but important metric that's easy to interpret.
    Unlike loss, which can be hard to interpret, accuracy directly tells us
    what percentage of examples we got right.

    Args:
        predictions (np.ndarray): Predicted probabilities, shape (batch_size, num_classes)
                                 Or predicted class labels, shape (batch_size,)
        targets (np.ndarray): True class labels, shape (batch_size,)

    Returns:
        float: Accuracy as a fraction in [0, 1]

    Example:
        >>> predictions = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        >>> targets = np.array([1, 0, 0])
        >>> accuracy = compute_accuracy(predictions, targets)
        >>> # predictions: class 1, class 0, class 1
        >>> # targets:     class 1, class 0, class 0
        >>> # correct:     ✓,       ✓,       ✗
        >>> # accuracy: 2/3 ≈ 0.667
    """
    # If predictions are probabilities (2D), convert to class labels
    if predictions.ndim == 2:
        predicted_classes = np.argmax(predictions, axis=1)
    else:
        predicted_classes = predictions

    # Count correct predictions
    correct = np.sum(predicted_classes == targets)
    total = len(targets)

    accuracy = correct / total
    return accuracy
