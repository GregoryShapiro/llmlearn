# Chapter 3: Neural Networks from Scratch - Detailed Plan

**Estimated Reading Time:** 7-8 hours
**Prerequisites:** Chapter 2, gradient descent, chain rule
**Learning Objectives:**
- Understand perceptrons and their limitations
- Master activation functions and their gradients
- Implement forward and backward propagation
- Build multi-layer perceptrons from scratch
- Train neural networks with various optimizers
- Apply neural networks to real problems (MNIST)

---

## 3.1 The Perceptron: Building Block of Neural Networks

**Duration:** 50 minutes

### Content Outline:

1. **Historical Context** (8 min)
   - Rosenblatt's perceptron (1958)
   - Inspired by biological neurons
   - Initial excitement and subsequent disappointment
   - Minsky & Papert's critique (1969)

2. **The Perceptron Model** (15 min)
   - Inputs: `x = [x₁, x₂, ..., xₙ]`
   - Weights: `w = [w₁, w₂, ..., wₙ]`
   - Bias: `b`
   - Weighted sum: `z = Σ wᵢxᵢ + b = w^T x + b`
   - Step activation: `y = 1 if z ≥ 0, else 0`
   - Geometric interpretation: Hyperplane separator

3. **Perceptron Learning Algorithm** (12 min)
   - For each misclassified point:
     - If `y=1` but predicted 0: `w = w + x`, `b = b + 1`
     - If `y=0` but predicted 1: `w = w - x`, `b = b - 1`
   - Guarantees convergence for linearly separable data
   - No convergence for non-separable (like XOR)

4. **Limitations** (10 min)
   - Can only learn linear decision boundaries
   - XOR problem: The deal-breaker
   - Cannot represent simple functions like XOR, XNOR
   - This limitation caused first AI winter
   - Resolution: Multiple layers (but no training algorithm until 1980s)

5. **Modern Perspective** (5 min)
   - Perceptron is a single neuron
   - Modern neural networks: Thousands/millions of neurons in layers
   - Replace step function with smooth activations
   - Use gradient-based learning instead of perceptron rule

### Exercise 3.1a: Implementing a Single Perceptron
**Type:** Programming (45-55 min)

**Task:**
1. **Implement Perceptron class:**
   ```python
   class Perceptron:
       def __init__(self, n_features):
           self.w = np.zeros(n_features)
           self.b = 0

       def predict(self, X):
           # TODO: Compute weighted sum and apply step function
           pass

       def fit(self, X, y, epochs=100):
           # TODO: Implement perceptron learning rule
           # Track: number of updates per epoch
           pass
   ```

2. **Test on linearly separable data:**
   ```python
   # AND function
   X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
   y_and = np.array([0, 0, 0, 1])

   # Train perceptron
   # Should converge quickly
   ```

3. **Visualize:**
   - Plot data points
   - Plot decision boundary (line)
   - Animate learning process (optional)

4. **Test on XOR:**
   ```python
   X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
   y_xor = np.array([0, 1, 1, 0])

   # Try to train perceptron
   # Should NOT converge
   ```

**Expected Outcome:**
- Working perceptron implementation
- Empirical verification of linear separability requirement
- Understanding of historical limitations

### Exercise 3.1b: Understanding the Decision Boundary
**Type:** Visualization and analysis (30-35 min)

**Task:**
1. **Train perceptrons on different problems:**
   - AND, OR, NOT, NAND
   - All are linearly separable

2. **For each, visualize:**
   - Data points colored by class
   - Decision boundary line
   - Weight vector (perpendicular to boundary)
   - Margin from boundary to nearest points

3. **Analyze:**
   - What determines the slope of the boundary? (ratio of weights)
   - What determines the intercept? (bias)
   - Where would the boundary be for different initializations?

4. **Experiment:**
   - Add noise to data
   - What happens when data is almost but not quite linearly separable?
   - How does the perceptron behave?

**Expected Outcome:**
- Geometric intuition for linear classifiers
- Understanding relationship between weights and decision boundary
- Appreciation for noise sensitivity

### Exercise 3.1c: Manual Weight Updates
**Type:** Mathematical (25-30 min)

**Task:**
1. **Given:**
   - Training data: `[(x₁=[0,1], y₁=1), (x₂=[1,0], y₂=1), (x₃=[1,1], y₃=0)]`
   - Initial: `w=[0,0]`, `b=0`

2. **Manually apply perceptron rule for 2 epochs:**
   - For each point, compute `z = w^Tx + b`
   - Predict: `ŷ = 1 if z ≥ 0 else 0`
   - If wrong, update weights
   - Show all calculations

3. **Track:**
   - Weight values after each update
   - Number of mistakes per epoch
   - Final decision boundary equation

4. **Verify:**
   - Implement in code
   - Compare manual calculations to code output

**Expected Outcome:**
- Detailed understanding of perceptron mechanics
- Comfort with weight update calculations
- Foundation for backpropagation

---

## 3.2 Activation Functions (ReLU, Sigmoid, Softmax)

**Duration:** 55 minutes

### Content Outline:

1. **Why Activation Functions?** (10 min)
   - Without non-linearity: Stacked linear layers = single linear layer
   - Proof: `f(x) = W₂(W₁x) = (W₂W₁)x = Wx`
   - Need non-linearity to learn complex functions
   - Activation adds expressiveness

2. **Sigmoid Function** (12 min)
   - Formula: `σ(z) = 1 / (1 + e^(-z))`
   - Range: (0, 1) - good for probabilities
   - Shape: S-curve (smooth step function)
   - Derivative: `σ'(z) = σ(z)(1 - σ(z))`
   - Problems:
     - Vanishing gradients (saturates at 0 and 1)
     - Not zero-centered (causes zig-zagging)
   - When to use: Binary classification output layer

3. **Tanh Function** (8 min)
   - Formula: `tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))`
   - Range: (-1, 1) - zero-centered
   - Derivative: `tanh'(z) = 1 - tanh²(z)`
   - Better than sigmoid for hidden layers
   - Still has vanishing gradient problem

4. **ReLU (Rectified Linear Unit)** (12 min)
   - Formula: `ReLU(z) = max(0, z)`
   - Derivative: `1 if z > 0, else 0` (undefined at 0, use 0 or 1)
   - Advantages:
     - No vanishing gradient for positive values
     - Sparse activation (many zeros)
     - Computationally cheap
     - Empirically works very well
   - Problems:
     - "Dying ReLU": Neurons can get stuck at 0
   - Variants: Leaky ReLU, ELU, GELU
   - **Current standard for hidden layers**

5. **Softmax Function** (13 min)
   - Formula: `softmax(z)ᵢ = e^(zᵢ) / Σⱼ e^(zⱼ)`
   - Converts scores to probability distribution
   - Range: (0, 1) for each element, sum to 1
   - Used for multi-class classification output
   - Temperature parameter for sharpness control
   - Numerically stable implementation (subtract max)

### Exercise 3.2a: Plotting Activation Functions
**Type:** Programming and visualization (30-35 min)

**Task:**
1. **Implement all activation functions:**
   ```python
   def sigmoid(z):
       return 1 / (1 + np.exp(-z))

   def tanh(z):
       return np.tanh(z)

   def relu(z):
       return np.maximum(0, z)

   def softmax(z):
       # TODO: Implement with numerical stability
       pass
   ```

2. **Plot each function:**
   - Input range: -10 to 10
   - Show shape, range, saturation regions
   - All on same figure for comparison

3. **Plot derivatives:**
   - Compute numerical derivatives
   - Show where gradients vanish
   - Compare gradient magnitudes

4. **Softmax special case:**
   - Test on vector: `z = [1, 2, 3]`
   - Verify outputs sum to 1
   - Show effect of temperature scaling

**Expected Outcome:**
- Visual understanding of activation functions
- Recognition of vanishing gradient regions
- Intuition for when to use each

### Exercise 3.2b: Computing Activation Derivatives
**Type:** Mathematical derivation (35-45 min)

**Task:**
1. **Derive sigmoid derivative:**
   - Start with `σ(z) = 1 / (1 + e^(-z))`
   - Use quotient rule or chain rule
   - Show that `σ'(z) = σ(z)(1 - σ(z))`
   - Why is this convenient computationally?

2. **Derive tanh derivative:**
   - Start with `tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))`
   - Show that `tanh'(z) = 1 - tanh²(z)`

3. **Derive ReLU derivative:**
   - Piecewise function
   - What about at z=0? (Discuss subdifferential)

4. **Derive softmax derivative (challenging):**
   - For single output: `∂softmax(z)ᵢ / ∂zⱼ`
   - Result is Jacobian matrix, not simple element-wise
   - `∂sᵢ/∂zⱼ = sᵢ(δᵢⱼ - sⱼ)` where δ is Kronecker delta
   - When combined with cross-entropy, simplifies to `ŷ - y`

5. **Verify numerically:**
   - Use finite differences for each
   - Compare to analytical derivatives

**Expected Outcome:**
- Derivation skills for neural network components
- Understanding of backpropagation building blocks
- Appreciation for mathematical simplifications

### Exercise 3.2c: Comparing Activation Function Behaviors
**Type:** Experimental (30-40 min)

**Task:**
1. **Build simple 2-layer network with different activations:**
   ```python
   # Architecture: Input(2) -> Hidden(4) -> Output(1)
   # Try: sigmoid, tanh, ReLU in hidden layer
   ```

2. **Train on non-linear dataset:**
   - Use sklearn's `make_moons` or `make_circles`
   - Keep everything else constant

3. **Compare:**
   - Training speed (epochs to convergence)
   - Final accuracy
   - Gradient magnitudes during training
   - Number of "dead" neurons (always output 0)

4. **Visualize:**
   - Decision boundaries for each
   - Loss curves
   - Gradient distributions at different layers

**Expected Outcome:**
- Empirical validation of activation function properties
- Understanding of why ReLU dominates modern architectures
- Experience with dying ReLU problem

---

## 3.3 Forward Propagation

**Duration:** 60 minutes

### Content Outline:

1. **Network Architecture Notation** (10 min)
   - Layers: Input (layer 0), Hidden (layers 1 to L-1), Output (layer L)
   - Layer `l` has `nₗ` neurons
   - Weights: `W^[l]` has shape `(n_l, n_{l-1})`
   - Biases: `b^[l]` has shape `(n_l, 1)`
   - Activations: `a^[l]` (output of layer l)

2. **Forward Pass for a Single Layer** (15 min)
   - Linear transformation: `z^[l] = W^[l] a^{[l-1]} + b^[l]`
   - Non-linear activation: `a^[l] = g(z^[l])`
   - Vectorized form for batches: `Z^[l] = W^[l] A^{[l-1]} + b^[l]`
   - Broadcasting bias across batch

3. **Complete Forward Propagation** (15 min)
   - Initialize: `a^[0] = X` (input)
   - For each layer l = 1 to L:
     - Compute `z^[l] = W^[l] a^{[l-1]} + b^[l]`
     - Compute `a^[l] = g^[l](z^[l])`
   - Return `a^[L]` (final prediction)

4. **Caching for Backpropagation** (10 min)
   - Why: Need intermediate values for gradient computation
   - What to cache:
     - `z^[l]` (pre-activation)
     - `a^[l]` (post-activation)
     - `W^[l], b^[l]` (parameters)
   - Data structure: Dictionary or list of tuples

5. **Shapes and Dimensions** (10 min)
   - Critical for debugging
   - Example 2-layer network:
     - Input: (n_features, batch_size)
     - W^[1]: (n_hidden, n_features)
     - b^[1]: (n_hidden, 1)
     - z^[1]: (n_hidden, batch_size)
     - a^[1]: (n_hidden, batch_size)
     - W^[2]: (n_output, n_hidden)
     - a^[2]: (n_output, batch_size)

### Exercise 3.3a: Hand-Computing Forward Pass
**Type:** Mathematical (35-40 min)

**Task:**
1. **Given tiny network:**
   - Architecture: 2 → 3 → 1
   - Weights and biases:
     ```
     W^[1] = [[0.1, 0.2],
              [0.3, 0.4],
              [0.5, 0.6]]
     b^[1] = [[0.1], [0.2], [0.3]]

     W^[2] = [[0.7, 0.8, 0.9]]
     b^[2] = [[0.4]]
     ```
   - Input: `x = [1, 2]`
   - Activations: ReLU for hidden, sigmoid for output

2. **Compute by hand:**
   - `z^[1] = W^[1]x + b^[1]` (show all calculations)
   - `a^[1] = ReLU(z^[1])`
   - `z^[2] = W^[2]a^[1] + b^[2]`
   - `a^[2] = sigmoid(z^[2])` (final output)

3. **Track shapes:**
   - Write shape of each quantity
   - Verify matrix multiplication compatibility

4. **Verify with code:**
   - Implement the network
   - Check your hand calculations

**Expected Outcome:**
- Detailed understanding of forward propagation mechanics
- Comfort with matrix dimensions
- Debugging skills for shape mismatches

### Exercise 3.3b: Implementing Forward Propagation in NumPy
**Type:** Programming (60-75 min)

**Task:**
1. **Implement neural network class:**
   ```python
   class NeuralNetwork:
       def __init__(self, layer_sizes):
           """
           layer_sizes: List of layer sizes [input, hidden1, ..., output]
           """
           self.layer_sizes = layer_sizes
           self.parameters = {}

           # TODO: Initialize weights and biases
           # Use small random values for weights (e.g., * 0.01)
           # Use zeros for biases

       def forward(self, X):
           """
           X: Input data (n_features, batch_size)
           Returns: Output activations, cache of intermediate values
           """
           cache = {}
           A = X
           cache['A0'] = A

           # TODO: Implement forward pass through all layers
           # For each layer:
           #   - Compute Z = W @ A + b
           #   - Compute A = activation(Z)
           #   - Cache Z and A

           return A, cache
   ```

2. **Test on simple data:**
   - Create network: [2, 4, 1]
   - Random input: `X = np.random.randn(2, 10)`
   - Run forward pass
   - Check output shape: Should be (1, 10)

3. **Implement different activation combinations:**
   - ReLU for hidden, sigmoid for output (binary classification)
   - ReLU for hidden, softmax for output (multi-class)
   - Tanh for hidden, linear for output (regression)

4. **Verify with shapes:**
   - Print shape of every intermediate activation
   - Ensure everything matches expected dimensions

**Expected Outcome:**
- Working forward propagation implementation
- Flexible network architecture
- Ready for backpropagation

### Exercise 3.3c: Debugging Shape Mismatches
**Type:** Debugging practice (25-30 min)

**Task:**
1. **Given buggy forward propagation code with shape errors:**
   ```python
   # Deliberately buggy code provided
   # Students must find and fix errors
   ```

2. **Common errors to diagnose:**
   - Transposed weight matrices
   - Bias shape incompatible with broadcasting
   - Batch dimension in wrong axis
   - Inconsistent (n_samples, n_features) vs (n_features, n_samples) convention

3. **Debugging techniques:**
   - Print shapes after each operation
   - Use assertions to check expected shapes
   - Visualize small matrices to verify computations

4. **Create debugging helper function:**
   ```python
   def check_shapes(network, input_shape):
       """
       Verify all intermediate shapes are correct
       """
       # TODO: Implement
   ```

**Expected Outcome:**
- Debugging skills for neural networks
- Understanding of common pitfalls
- Defensive programming habits (assertions)

---

## 3.4 Backpropagation and Gradient Descent

**Duration:** 90 minutes

### Content Outline:

1. **The Backpropagation Algorithm** (20 min)
   - Goal: Compute ∂L/∂W^[l] and ∂L/∂b^[l] for all layers
   - Key insight: Chain rule + dynamic programming
   - Work backwards from output to input
   - Reuse computations from forward pass (hence "cache")

2. **Output Layer Gradients** (15 min)
   - Start with loss: L(y, ŷ)
   - Compute: `dL/dz^[L]` (depends on loss and activation)
   - Example: Sigmoid + BCE
     - `dz^[L] = a^[L] - y` (beautiful simplification!)
   - Example: Softmax + Cross-entropy
     - `dz^[L] = ŷ - y` (same form!)

3. **Hidden Layer Gradients** (20 min)
   - Recursive formula:
     - `dz^[l] = (W^[l+1])^T dz^[l+1] ⊙ g'(z^[l])`
     - ⊙ denotes element-wise multiplication
   - Intuition: Gradient flows backward through weights, modulated by activation derivatives
   - This is the "back" in backpropagation

4. **Parameter Gradients** (15 min)
   - Once we have `dz^[l]`:
     - `dW^[l] = (1/m) dz^[l] (a^[l-1])^T`
     - `db^[l] = (1/m) Σ dz^[l]` (sum over batch)
   - m is batch size
   - These gradients used for parameter updates

5. **Complete Algorithm** (10 min)
   ```
   Forward:
   - Compute all activations, cache z and a

   Backward:
   - dz^[L] = loss_gradient(a^[L], y)
   - For l = L down to 1:
       - dW^[l] = (1/m) dz^[l] @ (a^[l-1]).T
       - db^[l] = (1/m) sum(dz^[l], axis=1, keepdims=True)
       - if l > 1:
           dz^[l-1] = W^[l].T @ dz^[l] * activation_derivative(z^[l-1])
   ```

6. **Why It Works: Chain Rule** (10 min)
   - Formal derivation using chain rule
   - Computational graph perspective
   - Efficiency: O(n) backward vs O(n²) naive approach

### Exercise 3.4a: Deriving Backpropagation by Hand
**Type:** Mathematical derivation (60-75 min)

**Task:**
1. **Simple 2-layer network:**
   - Architecture: x → h (ReLU) → y (sigmoid)
   - Loss: Binary cross-entropy

2. **Derive all gradients from first principles:**
   - `L = -[y log(ŷ) + (1-y)log(1-ŷ)]`
   - `ŷ = sigmoid(z₂)` where `z₂ = W₂h + b₂`
   - `h = ReLU(z₁)` where `z₁ = W₁x + b₁`

3. **Compute using chain rule:**
   - `∂L/∂z₂ = ?` (show all steps)
   - `∂L/∂W₂ = ?`
   - `∂L/∂b₂ = ?`
   - `∂L/∂h = ?`
   - `∂L/∂z₁ = ?`
   - `∂L/∂W₁ = ?`
   - `∂L/∂b₁ = ?`

4. **Verify the recursive pattern:**
   - Does your derivation match the backprop algorithm?

5. **Special cases:**
   - What happens when ReLU outputs 0?
   - What happens when sigmoid saturates?

**Expected Outcome:**
- Deep understanding of backpropagation mathematics
- Comfort with chain rule in neural networks
- Foundation for implementing backprop

### Exercise 3.4b: Implementing Backward Pass
**Type:** Programming (75-90 min)

**Task:**
1. **Add backward method to NeuralNetwork class:**
   ```python
   def backward(self, dA, cache):
       """
       dA: Gradient of loss w.r.t. final activation
       cache: Cached values from forward pass
       Returns: Dictionary of parameter gradients
       """
       gradients = {}
       L = len(self.layer_sizes) - 1
       m = dA.shape[1]  # batch size

       # TODO: Implement backpropagation
       # Start from output layer, work backwards
       # For each layer:
       #   - Compute dZ
       #   - Compute dW, db
       #   - Compute dA for previous layer

       return gradients
   ```

2. **Implement activation derivatives:**
   ```python
   def sigmoid_derivative(Z):
       # TODO

   def relu_derivative(Z):
       # TODO

   def tanh_derivative(Z):
       # TODO
   ```

3. **Test on toy problem:**
   - Create small network
   - Run forward pass
   - Manually compute loss gradient
   - Run backward pass
   - Verify gradient shapes

4. **Integrate with loss function:**
   ```python
   def compute_loss(Y_pred, Y_true):
       # Binary cross-entropy
       # Return loss value and initial gradient dA
       pass
   ```

**Expected Outcome:**
- Working backpropagation implementation
- Understanding of gradient flow
- Ready to train networks

### Exercise 3.4c: Numerical Gradient Checking
**Type:** Programming and debugging (40-50 min)

**Task:**
1. **Implement gradient checking:**
   ```python
   def gradient_check(network, X, Y, epsilon=1e-7):
       """
       Verify analytical gradients match numerical gradients
       """
       # 1. Compute analytical gradients using backprop
       A, cache = network.forward(X)
       loss, dA = compute_loss(A, Y)
       analytical_grads = network.backward(dA, cache)

       # 2. Compute numerical gradients
       numerical_grads = {}
       for param_name in network.parameters:
           # For each parameter:
           #   - Perturb by +epsilon, compute loss
           #   - Perturb by -epsilon, compute loss
           #   - Numerical gradient = (loss+ - loss-) / (2*epsilon)
           pass

       # 3. Compare
       for param_name in analytical_grads:
           diff = np.linalg.norm(analytical_grads[param_name] - numerical_grads[param_name])
           relative_error = diff / (np.linalg.norm(analytical_grads[param_name]) + np.linalg.norm(numerical_grads[param_name]))
           print(f"{param_name}: relative error = {relative_error}")
           # Should be < 1e-7 for correct implementation
   ```

2. **Test your implementation:**
   - Use small network (faster)
   - Random input/output
   - Run gradient check
   - If errors > 1e-5, there's likely a bug

3. **Common bugs to check:**
   - Forgot to divide by batch size
   - Transposed matrices
   - Wrong activation derivative
   - Cache not properly passed

4. **When to use:**
   - Always when implementing new layer types
   - When debugging mysterious training issues
   - Not during training (too slow)

**Expected Outcome:**
- Verified correct backpropagation
- Critical debugging skill for all neural network work
- Confidence in implementation correctness

---

## 3.5 Multi-Layer Perceptrons (MLPs)

**Duration:** 45 minutes

### Content Outline:

1. **Architecture Flexibility** (10 min)
   - Can have any number of hidden layers
   - Each layer can have any number of neurons
   - Deeper networks can learn more complex functions
   - But: Harder to train, more prone to overfitting

2. **Universal Approximation Theorem** (12 min)
   - Single hidden layer with enough neurons can approximate any continuous function
   - Theoretical result, not practical guidance
   - In practice: Multiple smaller layers often better
   - Depth vs width trade-offs

3. **XOR Solved** (10 min)
   - Minimal network: 2-2-1 (input-hidden-output)
   - Visualization: First layer creates two linear boundaries
   - Second layer combines them (AND of two half-spaces)
   - Proof that multi-layer solves what single-layer cannot

4. **Design Choices** (13 min)
   - **Number of layers:** Start with 2-3, increase if needed
   - **Neurons per layer:** Often decreasing (e.g., 64 → 32 → 16)
   - **Activation functions:** ReLU for hidden, task-specific for output
   - **Architecture search:** Still an art, not a science
   - Common patterns:
     - Encoder (decreasing sizes): Image → features
     - Decoder (increasing sizes): Features → image
     - Autoencoder (decrease then increase): Compression

### Exercise 3.5a: Building a 2-Layer Neural Network
**Type:** Programming (50-60 min)

**Task:**
1. **Complete neural network implementation:**
   - Use your forward and backward from previous exercises
   - Add training loop:
   ```python
   def train(self, X, Y, learning_rate=0.01, epochs=1000):
       losses = []
       for epoch in range(epochs):
           # Forward pass
           A, cache = self.forward(X)

           # Compute loss
           loss, dA = compute_loss(A, Y)
           losses.append(loss)

           # Backward pass
           gradients = self.backward(dA, cache)

           # Update parameters
           self.update_parameters(gradients, learning_rate)

           if epoch % 100 == 0:
               print(f"Epoch {epoch}, Loss: {loss:.4f}")

       return losses
   ```

2. **Test on XOR:**
   - Network: [2, 4, 1]
   - Train until loss < 0.1
   - Visualize decision boundary
   - Should clearly separate XOR

3. **Compare architectures:**
   - Try [2, 2, 1], [2, 4, 1], [2, 8, 1], [2, 4, 4, 1]
   - Which trains fastest?
   - Which generalizes best?

**Expected Outcome:**
- End-to-end neural network training
- Empirical verification of XOR solution
- Understanding of architecture impacts

### Exercise 3.5b: Experimenting with Hidden Layer Sizes
**Type:** Experimental (35-45 min)

**Task:**
1. **Create non-linear dataset:**
   ```python
   from sklearn.datasets import make_moons
   X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
   ```

2. **Train networks with different hidden sizes:**
   - [2, 2, 1], [2, 5, 1], [2, 10, 1], [2, 20, 1], [2, 50, 1]
   - Same learning rate and epochs

3. **Compare:**
   - Training loss curves
   - Test accuracy (use train/test split)
   - Decision boundary complexity
   - Training time

4. **Overfitting detection:**
   - Split data: 80% train, 20% test
   - Plot both train and test accuracy
   - Identify when test accuracy stops improving (overfitting)

5. **Create plots:**
   - Grid of decision boundaries for each architecture
   - Loss curves on same plot
   - Bar chart of final accuracies

**Expected Outcome:**
- Understanding of capacity and overfitting
- Practical model selection skills
- Recognition that bigger is not always better

### Exercise 3.5c: Non-Linear Decision Boundaries
**Type:** Visualization and analysis (30-35 min)

**Task:**
1. **Create challenging datasets:**
   - Concentric circles
   - Spiral
   - Checkerboard pattern

2. **For each dataset, find minimal architecture that achieves 95% accuracy:**
   - Start small, increase until good performance
   - Record: architecture, training time, final accuracy

3. **Visualize decision boundaries:**
   - Heatmap of predicted probabilities
   - Overlay with actual data points
   - Show complexity of learned boundary

4. **Analyze:**
   - Which dataset requires deepest network?
   - Which requires most neurons?
   - Can you predict difficulty from looking at data?

**Expected Outcome:**
- Intuition for problem difficulty
- Understanding of when deeper/wider networks needed
- Visual appreciation for learned decision boundaries

---

## 3.6 Training Dynamics and Optimization

**Duration:** 75 minutes

### Content Outline:

1. **Stochastic Gradient Descent (SGD)** (15 min)
   - Batch gradient descent: Use all data
   - Stochastic GD: Use one sample at a time
   - Mini-batch GD: Use small batch (e.g., 32, 64)
   - Trade-offs:
     - Batch: Accurate gradient, slow, memory intensive
     - Stochastic: Fast, noisy, can escape local minima
     - Mini-batch: Best of both worlds (current standard)
   - Implementation: Iterate over batches

2. **Momentum** (12 min)
   - Problem: SGD oscillates in narrow valleys
   - Solution: Remember previous update direction
   - Update rule:
     - `v = β*v + (1-β)*gradient`
     - `parameter -= learning_rate * v`
   - β typically 0.9 (90% old direction, 10% new gradient)
   - Accelerates convergence

3. **Adam Optimizer** (18 min)
   - Adaptive Moment Estimation
   - Combines momentum + adaptive learning rates
   - Maintains:
     - `m` (first moment, like momentum)
     - `v` (second moment, squared gradients)
   - Update rules:
     - `m = β₁*m + (1-β₁)*gradient`
     - `v = β₂*v + (1-β₂)*gradient²`
     - `m_hat = m / (1 - β₁^t)` (bias correction)
     - `v_hat = v / (1 - β₂^t)`
     - `parameter -= α * m_hat / (√v_hat + ε)`
   - Default: β₁=0.9, β₂=0.999, ε=1e-8
   - **Current go-to optimizer**

4. **Learning Rate Schedules** (12 min)
   - **Constant:** Same throughout (can fail to converge)
   - **Step decay:** Reduce by factor every N epochs
   - **Exponential decay:** `lr = lr₀ * e^(-kt)`
   - **1/t decay:** `lr = lr₀ / (1 + kt)`
   - **Warm restart:** Periodic resets
   - Modern approach: Learning rate finder, then schedule

5. **Hyperparameter Tuning** (10 min)
   - Learning rate: Most important (try 0.001, 0.01, 0.1)
   - Batch size: Powers of 2 (16, 32, 64, 128)
   - Number of epochs: Until validation loss stops improving
   - Architecture: Systematic search or random search
   - Regularization strength (Chapter 3.7)

6. **Initialization Strategies** (8 min)
   - **Zero initialization:** Breaks symmetry, all neurons learn same thing (BAD)
   - **Small random:** `W ~ N(0, 0.01)` - works but gradients can vanish
   - **Xavier/Glorot:** `W ~ N(0, 1/√n_in)` - for tanh
   - **He initialization:** `W ~ N(0, 2/√n_in)` - for ReLU (current standard)
   - Why they work: Preserve gradient magnitudes

### Exercise 3.6a: Implementing SGD from Scratch
**Type:** Programming (40-50 min)

**Task:**
1. **Implement mini-batch data iterator:**
   ```python
   def create_mini_batches(X, Y, batch_size):
       """
       Yield mini-batches of data
       """
       # TODO: Shuffle data
       # TODO: Split into batches
       # TODO: Handle last batch (might be smaller)
       pass
   ```

2. **Modify training loop for mini-batch:**
   ```python
   def train_sgd(self, X, Y, batch_size=32, epochs=100, lr=0.01):
       losses = []
       for epoch in range(epochs):
           epoch_loss = 0
           num_batches = 0

           for X_batch, Y_batch in create_mini_batches(X, Y, batch_size):
               # TODO: Forward pass on batch
               # TODO: Compute loss
               # TODO: Backward pass
               # TODO: Update parameters
               epoch_loss += batch_loss
               num_batches += 1

           avg_loss = epoch_loss / num_batches
           losses.append(avg_loss)

       return losses
   ```

3. **Compare batch sizes:**
   - Try: 1 (pure SGD), 32, 64, full batch
   - Plot loss curves
   - Measure training time
   - Compare convergence speed

4. **Analysis:**
   - Why does mini-batch converge faster than full batch?
   - Why is it more stable than single-sample SGD?
   - What's the trade-off with batch size?

**Expected Outcome:**
- Working mini-batch SGD implementation
- Understanding of batch size effects
- Practical experience with modern training

### Exercise 3.6b: Implementing Adam Optimizer
**Type:** Programming (60-70 min)

**Task:**
1. **Implement Adam optimizer:**
   ```python
   class AdamOptimizer:
       def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
           self.lr = learning_rate
           self.beta1 = beta1
           self.beta2 = beta2
           self.epsilon = epsilon
           self.m = {}  # First moment
           self.v = {}  # Second moment
           self.t = 0   # Time step

       def update(self, parameters, gradients):
           """
           Update parameters using Adam
           """
           self.t += 1

           for param_name in parameters:
               # Initialize m and v for new parameters
               if param_name not in self.m:
                   self.m[param_name] = np.zeros_like(parameters[param_name])
                   self.v[param_name] = np.zeros_like(parameters[param_name])

               # TODO: Update first moment (m)
               # TODO: Update second moment (v)
               # TODO: Bias correction
               # TODO: Parameter update

               pass
   ```

2. **Compare optimizers:**
   - SGD (no momentum)
   - SGD with momentum
   - Adam
   - On same dataset

3. **Visualize:**
   - Loss curves for each optimizer
   - Parameter trajectories (for 2D case)
   - Convergence speed

4. **Test on difficult landscape:**
   - Create loss function with:
     - Narrow valley
     - Saddle points
     - Different scales in different directions

**Expected Outcome:**
- Working Adam implementation
- Understanding of adaptive learning rates
- Appreciation for why Adam is default choice

### Exercise 3.6c: Hyperparameter Tuning Experiment
**Type:** Experimental (45-55 min)

**Task:**
1. **Systematic search:**
   - Learning rates: [0.0001, 0.001, 0.01, 0.1]
   - Batch sizes: [16, 32, 64, 128]
   - Hidden sizes: [16, 32, 64]

2. **For each combination:**
   - Train for fixed number of epochs (e.g., 50)
   - Record final test accuracy
   - Record training time

3. **Create visualizations:**
   - Heatmap: learning rate vs batch size (color = accuracy)
   - Line plot: hidden size vs accuracy
   - Bar chart: training time for each configuration

4. **Find best configuration:**
   - Highest test accuracy
   - Good balance of accuracy and speed

5. **Random search (bonus):**
   - Instead of grid, sample random combinations
   - Compare efficiency to grid search

**Expected Outcome:**
- Practical hyperparameter tuning experience
- Understanding of hyperparameter interactions
- Methodology for model selection

### Exercise 3.6d: Learning Rate Scheduling
**Type:** Programming (35-45 min)

**Task:**
1. **Implement learning rate schedules:**
   ```python
   def step_decay(initial_lr, epoch, drop=0.5, epochs_drop=10):
       return initial_lr * (drop ** (epoch // epochs_drop))

   def exponential_decay(initial_lr, epoch, decay_rate=0.95):
       return initial_lr * (decay_rate ** epoch)
   ```

2. **Modify training loop to use schedule:**
   ```python
   def train_with_schedule(self, X, Y, schedule='step', **kwargs):
       # TODO: Implement training with changing learning rate
       pass
   ```

3. **Compare schedules:**
   - Constant learning rate
   - Step decay
   - Exponential decay
   - Plot: learning rate over time

4. **Visualize effect:**
   - Loss curves for each schedule
   - Final accuracy
   - Which converges best?

**Expected Outcome:**
- Understanding of learning rate importance
- Ability to implement schedules
- Intuition for when to use each type

---

## 3.7 Hands-on: Building a Neural Network with NumPy

**Duration:** 90 minutes (extended exercise)

### Content Outline:

1. **MNIST Dataset** (10 min)
   - 60,000 training images, 10,000 test images
   - Handwritten digits 0-9
   - 28x28 grayscale images
   - Classic benchmark for neural networks
   - Loading with sklearn or manual download

2. **Data Preprocessing** (10 min)
   - Flatten 28x28 → 784 dimensional vector
   - Normalize pixels: [0, 255] → [0, 1]
   - One-hot encode labels: 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
   - Train/validation split

3. **Network Architecture** (5 min)
   - Input: 784 (28*28)
   - Hidden 1: 128 neurons, ReLU
   - Hidden 2: 64 neurons, ReLU
   - Output: 10 neurons, softmax
   - Total parameters: ~100K

4. **Training Strategy** (5 min)
   - Optimizer: Adam
   - Batch size: 64
   - Learning rate: 0.001
   - Epochs: 20-30
   - Monitor validation accuracy

5. **Expected Results** (5 min)
   - Should achieve >95% test accuracy
   - Training time: ~5-10 minutes on CPU
   - Improvement over traditional ML: Logistic ~92%, NN ~97%

### Exercise 3.7a: Complete MNIST Digit Classifier
**Type:** Capstone project (120-150 min)

**Task:**
1. **Load and preprocess MNIST:**
   ```python
   from sklearn.datasets import fetch_openml
   X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

   # TODO: Normalize X
   # TODO: One-hot encode y
   # TODO: Train/test split
   ```

2. **Build complete pipeline:**
   ```python
   # Initialize network
   nn = NeuralNetwork([784, 128, 64, 10])

   # Train with Adam
   optimizer = AdamOptimizer(learning_rate=0.001)

   # Training loop with validation
   for epoch in range(30):
       # Train on mini-batches
       # Evaluate on validation set
       # Print progress
       pass
   ```

3. **Implement missing pieces:**
   - Softmax output layer
   - Cross-entropy loss
   - Accuracy metric
   - Early stopping (stop if validation accuracy doesn't improve)

4. **Evaluate:**
   - Final test accuracy
   - Confusion matrix
   - Visualize misclassified examples

5. **Analysis:**
   - Which digits are confused most?
   - Look at error cases: Are they understandable?
   - How does performance compare to random guessing (10%)?

**Expected Outcome:**
- End-to-end neural network project
- Real-world dataset experience
- 95%+ accuracy on MNIST
- Complete understanding of training pipeline

### Exercise 3.7b: Analyzing Training Curves
**Type:** Analysis and visualization (40-50 min)

**Task:**
1. **Track metrics during training:**
   - Training loss per epoch
   - Validation loss per epoch
   - Training accuracy per epoch
   - Validation accuracy per epoch

2. **Create comprehensive plots:**
   - Plot 1: Training vs validation loss
   - Plot 2: Training vs validation accuracy
   - Plot 3: Learning rate schedule (if applicable)

3. **Identify phenomena:**
   - Overfitting: When does it start?
   - Underfitting: Is initial model too simple?
   - Convergence: When does improvement stop?

4. **Experiment:**
   - Try different architectures
   - Try different regularization (if implemented)
   - Compare training curves

5. **Create diagnostic checklist:**
   - Model training too slowly? → Increase learning rate
   - Loss oscillating? → Decrease learning rate
   - Overfitting? → Add regularization, more data, or simpler model
   - Underfitting? → Bigger model, train longer

**Expected Outcome:**
- Ability to diagnose training issues
- Understanding of loss curve patterns
- Practical debugging skills

### Exercise 3.7c: Preventing Overfitting
**Type:** Experimental (50-60 min)

**Task:**
1. **Create overfitting scenario:**
   - Use only 5000 MNIST training samples
   - Train large network: [784, 256, 128, 10]
   - Train for many epochs

2. **Implement regularization techniques:**
   - **L2 regularization:**
     ```python
     loss = cross_entropy_loss + lambda * sum(W^2)
     # Add lambda * W to gradients
     ```
   - **Dropout (conceptual, simplified implementation):**
     ```python
     # During training: randomly zero out activations
     # During test: scale activations
     ```
   - **Early stopping:**
     ```python
     # Stop when validation loss stops improving
     ```

3. **Compare approaches:**
   - No regularization
   - L2 with different λ values
   - Dropout with different probabilities
   - Early stopping

4. **Visualize:**
   - Train/val accuracy gap for each method
   - Effect of regularization strength
   - Which method works best?

5. **Best practices:**
   - Always use train/val/test split
   - Monitor validation, not just training
   - Use regularization for deep networks
   - Early stopping as safety net

**Expected Outcome:**
- Understanding of overfitting
- Practical regularization techniques
- Model selection methodology

---

## Math Appendix 3A: Backpropagation Derivations

*(See content outline in main chapter outline - detailed mathematical treatment)*

---

## Math Appendix 3B: Optimization Theory

*(See content outline in main chapter outline - detailed mathematical treatment)*

---

## Math Appendix 3C: Initialization Strategies

*(See content outline in main chapter outline - detailed mathematical treatment)*

---

## Chapter 3 Summary

**Key Takeaways:**
1. Perceptrons are linear classifiers with fundamental limitations
2. Activation functions provide non-linearity; ReLU is current standard
3. Forward propagation: Sequential computation through layers
4. Backpropagation: Efficient gradient computation via chain rule
5. Adam optimizer is default choice for training
6. Proper initialization (He/Xavier) critical for deep networks
7. Real neural networks achieve 97%+ on MNIST

**Prerequisites for Chapter 4:**
- Solid understanding of forward/backward propagation
- Comfort with matrix operations and shapes
- Experience training neural networks
- Understanding of attention as "weighted average" concept

**Total Exercises:** 15 main exercises (3a-c for each section)
**Total Time:** 7-8 hours reading + 12-15 hours exercises = **19-23 hours**

---

## Consistency Check (Internal)

**Terminology:**
- "Forward pass" = "forward propagation" ✓
- "Backward pass" = "backpropagation" ✓
- "Parameters" (weights, biases) vs "hyperparameters" (learning rate, architecture) ✓

**Prerequisites:**
- Assumes gradient descent from Chapter 2 ✓
- Builds on chain rule from Math Appendix 2B ✓
- All exercises assume NumPy proficiency ✓

**Flow to Chapter 4:**
- MLPs process entire sequence as single vector (limitation for variable-length)
- Transformers address this with attention mechanism
- Chapter 3 provides foundation for understanding transformer layers
- Residual connections and layer norm introduced conceptually (detailed in Chapter 4)
