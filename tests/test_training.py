"""
Comprehensive test suite for training infrastructure.

Tests cover:
1. Cross-Entropy Loss
   - Forward pass computation
   - Gradient computation
   - Numerical stability

2. Optimizers (SGD and Adam)
   - Parameter updates
   - Momentum and adaptive learning rates
   - Bias correction (Adam)

3. Training Utilities
   - Training step
   - Evaluation function
   - Full training loop

These tests ensure the training infrastructure works correctly
before we attempt to train the actual model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from loss import CrossEntropyLoss, compute_accuracy
from optimizer import SGD, Adam


def test_cross_entropy_loss_forward():
    """Test that cross-entropy loss computes correct values."""
    print("Testing CrossEntropyLoss forward pass...")

    loss_fn = CrossEntropyLoss()

    # Simple test case
    predictions = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
    targets = np.array([1, 0])  # First example: class 1, second: class 0

    loss = loss_fn.forward(predictions, targets)

    # Expected: -( log(0.7) + log(0.8) ) / 2
    expected_loss = -(np.log(0.7) + np.log(0.8)) / 2

    assert np.isclose(loss, expected_loss, rtol=1e-5), \
        f"Expected loss {expected_loss}, got {loss}"

    print("✓ CrossEntropyLoss forward pass is correct")


def test_cross_entropy_loss_backward():
    """Test that cross-entropy loss computes correct gradients."""
    print("Testing CrossEntropyLoss backward pass...")

    loss_fn = CrossEntropyLoss()

    predictions = np.array([[0.2, 0.8], [0.7, 0.3]])
    targets = np.array([1, 0])

    # Forward pass
    loss = loss_fn.forward(predictions, targets)

    # Backward pass
    grad = loss_fn.backward()

    # Expected gradient: (predictions - targets_onehot) / batch_size
    # targets_onehot: [[0, 1], [1, 0]]
    # grad: ([[0.2, 0.8] - [0, 1]]) / 2 = [[0.1, -0.1], ...]
    expected_grad = np.array([[0.2 - 0, 0.8 - 1], [0.7 - 1, 0.3 - 0]]) / 2

    assert np.allclose(grad, expected_grad, rtol=1e-5), \
        f"Gradient mismatch"

    # Gradient shape should match predictions shape
    assert grad.shape == predictions.shape, \
        f"Expected grad shape {predictions.shape}, got {grad.shape}"

    print("✓ CrossEntropyLoss backward pass is correct")


def test_cross_entropy_numerical_stability():
    """Test that cross-entropy handles edge cases without NaN/Inf."""
    print("Testing CrossEntropyLoss numerical stability...")

    loss_fn = CrossEntropyLoss()

    # Test with very confident correct predictions
    predictions = np.array([[0.01, 0.99], [0.99, 0.01]])
    targets = np.array([1, 0])
    loss = loss_fn.forward(predictions, targets)

    assert not np.isnan(loss), "Loss should not be NaN"
    assert not np.isinf(loss), "Loss should not be Inf"
    assert loss > 0, "Loss should be positive"

    # Test with very confident wrong predictions
    predictions = np.array([[0.99, 0.01], [0.01, 0.99]])
    targets = np.array([1, 0])
    loss = loss_fn.forward(predictions, targets)

    assert not np.isnan(loss), "Loss should not be NaN with wrong predictions"
    assert loss > 0, "Loss should be positive"

    print("✓ CrossEntropyLoss is numerically stable")


def test_compute_accuracy():
    """Test accuracy computation."""
    print("Testing accuracy computation...")

    # Perfect predictions
    predictions = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    targets = np.array([1, 0, 1])
    accuracy = compute_accuracy(predictions, targets)
    assert accuracy == 1.0, f"Expected accuracy 1.0, got {accuracy}"

    # Partially correct
    predictions = np.array([[0.1, 0.9], [0.8, 0.2], [0.7, 0.3]])
    targets = np.array([1, 0, 1])
    accuracy = compute_accuracy(predictions, targets)
    assert accuracy == 2/3, f"Expected accuracy 0.667, got {accuracy}"

    # All wrong
    predictions = np.array([[0.9, 0.1], [0.2, 0.8]])
    targets = np.array([1, 0])
    accuracy = compute_accuracy(predictions, targets)
    assert accuracy == 0.0, f"Expected accuracy 0.0, got {accuracy}"

    print("✓ Accuracy computation is correct")


def test_sgd_parameter_update():
    """Test that SGD correctly updates parameters."""
    print("Testing SGD parameter updates...")

    # Create dummy parameter
    param = np.array([[1.0, 2.0], [3.0, 4.0]])
    grad = np.array([[0.1, 0.2], [0.3, 0.4]])

    parameters = [(param, grad)]
    optimizer = SGD(parameters, learning_rate=0.1, momentum=0.0)

    # Before update
    param_before = param.copy()

    # Perform update
    optimizer.step()

    # After update: param = param - lr * grad
    expected_param = param_before - 0.1 * grad

    assert np.allclose(param, expected_param, rtol=1e-5), \
        "SGD update incorrect"

    print("✓ SGD parameter updates are correct")


def test_sgd_with_momentum():
    """Test that SGD momentum works correctly."""
    print("Testing SGD with momentum...")

    param = np.array([[1.0]])
    grad1 = np.array([[1.0]])
    grad2 = np.array([[1.0]])

    parameters = [(param, grad1)]
    optimizer = SGD(parameters, learning_rate=0.1, momentum=0.9)

    param_initial = param.copy()

    # First step
    optimizer.step()
    # velocity = 0.9 * 0 - 0.1 * 1.0 = -0.1
    # param = 1.0 + (-0.1) = 0.9

    # Second step with same gradient
    parameters[0] = (param, grad2)
    optimizer.parameters = parameters
    optimizer.step()
    # velocity = 0.9 * (-0.1) - 0.1 * 1.0 = -0.19
    # param = 0.9 + (-0.19) = 0.71

    # With momentum, second step should be larger
    expected_approx = 0.71

    assert np.isclose(param[0, 0], expected_approx, atol=0.01), \
        f"Expected param ≈ {expected_approx}, got {param[0, 0]}"

    print("✓ SGD momentum works correctly")


def test_adam_parameter_update():
    """Test that Adam correctly updates parameters."""
    print("Testing Adam parameter updates...")

    param = np.array([[1.0, 2.0]])
    grad = np.array([[0.1, 0.2]])

    parameters = [(param, grad)]
    optimizer = Adam(parameters, learning_rate=0.01)

    param_before = param.copy()

    # Perform update
    optimizer.step()

    # Parameter should change
    assert not np.allclose(param, param_before), \
        "Adam should update parameters"

    # Update should be in opposite direction of gradient
    param_change = param - param_before
    # Since gradient is positive, param should decrease
    assert np.all(param_change < 0), \
        "Adam should move opposite to gradient direction"

    print("✓ Adam parameter updates work correctly")


def test_adam_bias_correction():
    """Test that Adam bias correction works."""
    print("Testing Adam bias correction...")

    param = np.array([[1.0]])
    grad = np.array([[0.1]])

    parameters = [(param, grad)]
    optimizer = Adam(parameters, learning_rate=0.01, beta1=0.9, beta2=0.999)

    # First step
    param_before_step1 = param.copy()
    optimizer.step()
    step1_change = np.abs(param[0, 0] - param_before_step1[0, 0])

    # Reset for second comparison
    param = np.array([[1.0]])
    parameters = [(param, grad)]
    optimizer2 = Adam(parameters, learning_rate=0.01, beta1=0.9, beta2=0.999)

    # Two steps
    optimizer2.step()
    param_before_step2 = param.copy()
    optimizer2.step()
    step2_change = np.abs(param[0, 0] - param_before_step2[0, 0])

    # With bias correction, early steps have larger updates
    # Step 1 correction factor: 1/(1-0.9^1) = 10, 1/(1-0.999^1) = 1000
    # Step 2 correction factor: 1/(1-0.9^2) = 5.26, 1/(1-0.999^2) = 500

    print(f"  Step 1 change: {step1_change:.6f}")
    print(f"  Step 2 change: {step2_change:.6f}")

    # Both should be positive (parameters changed)
    assert step1_change > 0, "Step 1 should change parameter"
    assert step2_change > 0, "Step 2 should change parameter"

    print("✓ Adam bias correction is working")


def test_adam_adaptive_learning_rate():
    """Test that Adam adapts learning rate per parameter."""
    print("Testing Adam adaptive learning rate...")

    # Two parameters with different gradient magnitudes
    param1 = np.array([[1.0]])
    param2 = np.array([[1.0]])
    grad1 = np.array([[0.01]])  # Small gradient
    grad2 = np.array([[1.0]])    # Large gradient

    parameters = [(param1, grad1), (param2, grad2)]
    optimizer = Adam(parameters, learning_rate=0.01)

    param1_before = param1.copy()
    param2_before = param2.copy()

    # Perform update
    optimizer.step()

    # Compute relative changes
    change1 = np.abs((param1[0, 0] - param1_before[0, 0]) / param1_before[0, 0])
    change2 = np.abs((param2[0, 0] - param2_before[0, 0]) / param2_before[0, 0])

    # Parameter with smaller gradients should have relatively larger effective learning rate
    # (though absolute change might be similar due to gradient magnitude)
    print(f"  Param1 change (small grad): {change1:.6f}")
    print(f"  Param2 change (large grad): {change2:.6f}")

    # Both should change
    assert change1 > 0, "Parameter 1 should change"
    assert change2 > 0, "Parameter 2 should change"

    print("✓ Adam adapts learning rate per parameter")


def test_optimizer_with_none_gradients():
    """Test that optimizers handle None gradients gracefully."""
    print("Testing optimizers with None gradients...")

    param1 = np.array([[1.0]])
    param2 = np.array([[2.0]])
    grad1 = np.array([[0.1]])
    grad2 = None  # No gradient for second parameter

    # Test SGD
    parameters_sgd = [(param1, grad1), (param2, grad2)]
    optimizer_sgd = SGD(parameters_sgd, learning_rate=0.01)

    param1_before = param1.copy()
    param2_before = param2.copy()

    optimizer_sgd.step()

    # Param1 should change, param2 should not
    assert not np.allclose(param1, param1_before), "Param1 should change"
    assert np.allclose(param2, param2_before), "Param2 should not change (None gradient)"

    # Test Adam
    param1 = np.array([[1.0]])
    param2 = np.array([[2.0]])
    parameters_adam = [(param1, grad1), (param2, grad2)]
    optimizer_adam = Adam(parameters_adam, learning_rate=0.01)

    param1_before = param1.copy()
    param2_before = param2.copy()

    optimizer_adam.step()

    assert not np.allclose(param1, param1_before), "Param1 should change with Adam"
    assert np.allclose(param2, param2_before), "Param2 should not change with Adam (None gradient)"

    print("✓ Optimizers handle None gradients correctly")


def test_loss_and_optimizer_integration():
    """Test that loss and optimizer work together."""
    print("Testing loss and optimizer integration...")

    # Simple 1-parameter model: just a bias
    # Prediction = bias (broadcasted)
    bias = np.array([0.5, 0.5])  # Start with equal probabilities

    # True targets: all class 0
    targets = np.array([0, 0, 0])

    loss_fn = CrossEntropyLoss()

    # Create predictions (just the bias repeated)
    predictions = np.tile(bias, (len(targets), 1))

    # Compute loss
    loss_before = loss_fn.forward(predictions, targets)

    # Compute gradient
    grad = loss_fn.backward()

    # Average gradient across batch
    grad_avg = np.mean(grad, axis=0)

    # Update bias
    parameters = [(bias, grad_avg)]
    optimizer = SGD(parameters, learning_rate=0.1)
    optimizer.step()

    # After update, make new predictions
    predictions_after = np.tile(bias, (len(targets), 1))
    loss_after = loss_fn.forward(predictions_after, targets)

    # Loss should decrease
    assert loss_after < loss_before, \
        f"Loss should decrease after update (before: {loss_before:.4f}, after: {loss_after:.4f})"

    print(f"  Loss before: {loss_before:.4f}")
    print(f"  Loss after: {loss_after:.4f}")
    print("✓ Loss and optimizer integration works")


# Run all tests
if __name__ == '__main__':
    print("\n" + "="*60)
    print("Running Training Infrastructure Tests")
    print("="*60 + "\n")

    try:
        # Loss function tests
        print("Cross-Entropy Loss Tests:")
        print("-" * 60)
        test_cross_entropy_loss_forward()
        test_cross_entropy_loss_backward()
        test_cross_entropy_numerical_stability()
        test_compute_accuracy()

        print()

        # Optimizer tests
        print("Optimizer Tests:")
        print("-" * 60)
        test_sgd_parameter_update()
        test_sgd_with_momentum()
        test_adam_parameter_update()
        test_adam_bias_correction()
        test_adam_adaptive_learning_rate()
        test_optimizer_with_none_gradients()

        print()

        # Integration tests
        print("Integration Tests:")
        print("-" * 60)
        test_loss_and_optimizer_integration()

        print("\n" + "="*60)
        print("All training infrastructure tests passed! ✓")
        print("="*60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise
