"""
Optimizers Module

This module implements optimization algorithms for training neural networks.

What is an Optimizer?
    An optimizer updates model parameters using gradients to minimize the loss function.
    The basic update rule: θ_new = θ_old - learning_rate * gradient

Why Not Just Use Plain Gradient Descent?
    Simple gradient descent (SGD) has problems:
    1. Slow convergence in flat regions
    2. Overshooting in steep regions
    3. Same learning rate for all parameters
    4. No momentum to escape local minima

    Modern optimizers (like Adam) solve these problems using adaptive learning rates
    and momentum.

Optimization Landscape:
    Training a neural network means navigating a high-dimensional loss landscape.
    Good optimizers help us:
    - Move quickly in consistent directions (momentum)
    - Move carefully in inconsistent directions (adaptive learning rates)
    - Converge to good solutions faster

This module implements:
    - SGD: Simple gradient descent with optional momentum
    - Adam: Adaptive Moment Estimation (state-of-the-art general-purpose optimizer)
"""

import numpy as np


class SGD:
    """
    Stochastic Gradient Descent optimizer with optional momentum.

    SGD is the simplest optimizer. It updates parameters directly using gradients:
        θ = θ - learning_rate * gradient

    With Momentum:
        Momentum helps accelerate SGD in consistent directions and dampen oscillations.

        velocity = momentum * velocity - learning_rate * gradient
        θ = θ + velocity

        Think of momentum as a ball rolling down a hill - it builds up speed
        in consistent directions and doesn't stop immediately at flat regions.

    Why Momentum?
        - Accelerates learning in consistent directions
        - Dampens oscillations in inconsistent directions
        - Helps escape shallow local minima
        - Common momentum value: 0.9 (means 90% of previous velocity is kept)

    Args:
        parameters (list): List of (param, grad) tuples from model
        learning_rate (float): Step size for updates (default: 0.01)
        momentum (float): Momentum factor (default: 0.0, range: [0, 1))

    Example:
        >>> optimizer = SGD(model.get_parameters(), learning_rate=0.01, momentum=0.9)
        >>> for epoch in range(num_epochs):
        >>>     loss = compute_loss()
        >>>     gradients = compute_gradients()
        >>>     optimizer.step()
    """

    def __init__(self, parameters, learning_rate=0.01, momentum=0.0):
        """
        Initialize SGD optimizer.

        Args:
            parameters (list): List of (param, grad) tuples
            learning_rate (float): Learning rate
            momentum (float): Momentum factor (0 = no momentum)
        """
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.momentum = momentum

        # Initialize velocity for momentum
        # velocity stores the exponentially weighted moving average of gradients
        self.velocities = [np.zeros_like(param) for param, _ in parameters]

    def step(self):
        """
        Perform one optimization step (parameter update).

        Updates all parameters using their gradients and the SGD update rule.
        """
        for i, (param, grad) in enumerate(self.parameters):
            if grad is None:
                continue  # Skip parameters without gradients

            if self.momentum > 0:
                # Update velocity: mix of previous velocity and current gradient
                self.velocities[i] = self.momentum * self.velocities[i] - self.learning_rate * grad
                # Update parameter using velocity
                param += self.velocities[i]
            else:
                # Simple SGD without momentum
                param -= self.learning_rate * grad

    def zero_grad(self):
        """Reset velocities (called at start of training, not every step)."""
        self.velocities = [np.zeros_like(param) for param, _ in self.parameters]

    def __repr__(self):
        return f"SGD(lr={self.learning_rate}, momentum={self.momentum})"


class Adam:
    """
    Adam (Adaptive Moment Estimation) optimizer.

    Adam is currently the most popular optimizer for deep learning because it:
    1. Adapts learning rate for each parameter individually
    2. Uses momentum for stable convergence
    3. Works well with default hyperparameters
    4. Handles sparse gradients well

    The Adam Algorithm:
        Adam maintains two moving averages for each parameter:
        - m (first moment): exponentially decaying average of gradients (momentum)
        - v (second moment): exponentially decaying average of squared gradients (adaptive lr)

        Update steps:
        1. m = β₁ * m + (1 - β₁) * gradient          # Update momentum
        2. v = β₂ * v + (1 - β₂) * gradient²         # Update variance
        3. m_hat = m / (1 - β₁^t)                     # Bias correction for momentum
        4. v_hat = v / (1 - β₂^t)                     # Bias correction for variance
        5. θ = θ - learning_rate * m_hat / (√v_hat + ε)  # Parameter update

    Why These Components?

    First Moment (m) - Momentum:
        - Exponentially weighted average of gradients
        - Provides momentum to accelerate learning
        - Smooths out noisy gradients
        - β₁ = 0.9 means we keep 90% of previous momentum

    Second Moment (v) - Adaptive Learning Rate:
        - Exponentially weighted average of squared gradients
        - Estimates the variance of gradients
        - Parameters with large gradients get smaller effective learning rates
        - Parameters with small gradients get larger effective learning rates
        - β₂ = 0.999 means we keep 99.9% of previous variance estimate

    Bias Correction:
        Without bias correction, m and v are biased toward zero at the start.
        Division by (1 - β^t) corrects this bias, where t is the timestep.
        As t increases, (1 - β^t) approaches 1, so correction diminishes.

    The Effective Learning Rate:
        learning_rate * m / (√v + ε)

        - If gradients are consistently large: √v is large → effective lr is small
        - If gradients are consistently small: √v is small → effective lr is large
        - This adapts to the landscape automatically!

    Why Adam Works So Well:
        1. Combines benefits of momentum (m) and adaptive learning rates (v)
        2. Bias correction ensures good performance even in early training
        3. Works well across a wide range of problems with default hyperparameters
        4. Robust to noisy gradients and sparse data

    Default Hyperparameters:
        - learning_rate: 0.001 (often works well without tuning)
        - β₁: 0.9 (momentum decay rate)
        - β₂: 0.999 (variance decay rate)
        - ε: 1e-8 (numerical stability)

    These defaults work well for most problems!

    Variants (not implemented):
        - AdamW: Adam with decoupled weight decay
        - NAdam: Adam with Nesterov momentum
        - RAdam: Adam with rectified adaptive learning rate

    Args:
        parameters (list): List of (param, grad) tuples from model
        learning_rate (float): Learning rate (default: 0.001)
        beta1 (float): Decay rate for first moment (default: 0.9)
        beta2 (float): Decay rate for second moment (default: 0.999)
        epsilon (float): Small constant for numerical stability (default: 1e-8)

    Example:
        >>> optimizer = Adam(model.get_parameters(), learning_rate=0.001)
        >>> for epoch in range(num_epochs):
        >>>     for batch in data_loader:
        >>>         loss = model.forward(batch)
        >>>         model.backward(loss)
        >>>         optimizer.step()
        >>>         model.zero_grad()
    """

    def __init__(self, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize Adam optimizer.

        Why These Defaults?
            - lr=0.001: Small enough to be stable, large enough to make progress
            - β₁=0.9: Keeps 90% of momentum, provides good acceleration
            - β₂=0.999: Keeps 99.9% of variance, provides stable adaptive lr
            - ε=1e-8: Small enough to not affect optimization, prevents division by zero

        Args:
            parameters (list): List of (param, grad) tuples
            learning_rate (float): Learning rate (alpha in paper)
            beta1 (float): Decay rate for first moment estimates
            beta2 (float): Decay rate for second moment estimates
            epsilon (float): Small constant for numerical stability
        """
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Initialize first moment (momentum) - exponentially decaying average of gradients
        self.m = [np.zeros_like(param) for param, _ in parameters]

        # Initialize second moment (variance) - exponentially decaying average of squared gradients
        self.v = [np.zeros_like(param) for param, _ in parameters]

        # Timestep for bias correction
        # t starts at 0, increments each time step() is called
        self.t = 0

    def step(self):
        """
        Perform one optimization step (parameter update).

        This implements the Adam update rule with bias correction.

        The algorithm:
            1. Increment timestep
            2. For each parameter:
                a. Update biased first moment estimate (momentum)
                b. Update biased second moment estimate (variance)
                c. Compute bias-corrected first moment
                d. Compute bias-corrected second moment
                e. Update parameter using adaptive learning rate

        Why This Order?
            The order matters! We must:
            - Increment t first (for correct bias correction)
            - Update m and v before computing corrections
            - Apply corrections before parameter update
        """
        # Increment timestep
        self.t += 1

        # Loop over all parameters
        for i, (param, grad) in enumerate(self.parameters):
            if grad is None:
                continue  # Skip parameters without gradients (shouldn't happen in practice)

            # Step 1 & 2: Update biased first and second moment estimates
            # m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
            # v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # Step 3 & 4: Compute bias-corrected moment estimates
            # Without bias correction, m and v are biased toward zero initially
            # Bias correction: divide by (1 - β^t)
            # As t → ∞, (1 - β^t) → 1, so correction vanishes
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Step 5: Update parameter
            # θ_t = θ_{t-1} - α * m_hat / (√v_hat + ε)
            # The division by √v_hat adapts the learning rate per parameter
            # ε prevents division by zero
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def zero_grad(self):
        """
        Reset optimizer state.

        This is typically NOT called during training. It's here for completeness.
        In practice, you'd create a new optimizer if you want to reset state.
        """
        self.m = [np.zeros_like(param) for param, _ in self.parameters]
        self.v = [np.zeros_like(param) for param, _ in self.parameters]
        self.t = 0

    def get_learning_rate(self):
        """
        Get current learning rate.

        Note: In Adam, each parameter has its own effective learning rate
        based on its gradient history. This returns the base learning rate.

        Returns:
            float: Base learning rate
        """
        return self.learning_rate

    def set_learning_rate(self, learning_rate):
        """
        Set learning rate.

        Useful for learning rate scheduling (reducing lr during training).

        Args:
            learning_rate (float): New learning rate
        """
        self.learning_rate = learning_rate

    def __repr__(self):
        """String representation for debugging."""
        return (f"Adam(lr={self.learning_rate}, β₁={self.beta1}, "
                f"β₂={self.beta2}, ε={self.epsilon}, t={self.t})")


# Learning rate schedulers (not implemented, but useful for reference)
"""
Common learning rate schedules:

1. Step Decay:
   Reduce learning rate by factor every N epochs
   Example: lr = initial_lr * 0.1^(epoch // 30)

2. Exponential Decay:
   Reduce learning rate exponentially
   Example: lr = initial_lr * exp(-decay_rate * epoch)

3. Cosine Annealing:
   Reduce learning rate following cosine function
   Example: lr = min_lr + (max_lr - min_lr) * (1 + cos(π * epoch / max_epochs)) / 2

4. Warm-up:
   Gradually increase learning rate at the start
   Example: lr = target_lr * min(1, epoch / warmup_epochs)

These can significantly improve training! Usually implemented as:
    for epoch in range(num_epochs):
        adjust_learning_rate(optimizer, epoch)
        train_one_epoch()
"""
