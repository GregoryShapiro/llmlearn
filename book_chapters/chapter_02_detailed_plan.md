# Chapter 2: Machine Learning Fundamentals - Detailed Plan

**Estimated Reading Time:** 5-6 hours
**Prerequisites:** Chapter 1, basic Python, high school math
**Learning Objectives:**
- Understand the three main types of machine learning
- Implement linear and logistic regression from scratch
- Master gradient descent optimization
- Learn model evaluation techniques
- Understand when to use traditional ML vs neural networks

---

## 2.1 Types of Machine Learning

**Duration:** 45 minutes

### Content Outline:

1. **Supervised Learning** (15 min)
   - Definition: Learning from labeled examples (input → output pairs)
   - Two categories:
     - **Regression:** Predicting continuous values (house prices, temperature)
     - **Classification:** Predicting discrete categories (spam/not spam, dog/cat)
   - Training process: Model learns mapping from features to labels
   - Examples: Email filtering, medical diagnosis, stock prediction

2. **Unsupervised Learning** (12 min)
   - Definition: Finding patterns in unlabeled data
   - Common tasks:
     - **Clustering:** Grouping similar items (customer segmentation)
     - **Dimensionality Reduction:** Compressing data (PCA, t-SNE)
     - **Anomaly Detection:** Finding outliers (fraud detection)
   - No "correct answers" to learn from
   - Examples: Market segmentation, recommendation systems

3. **Reinforcement Learning** (10 min)
   - Definition: Learning through trial and error with rewards
   - Key concepts:
     - **Agent:** The learner/decision maker
     - **Environment:** What the agent interacts with
     - **Actions:** What the agent can do
     - **Rewards:** Feedback signal (positive or negative)
   - Examples: Game playing (AlphaGo), robotics, autonomous driving

4. **Comparison and When to Use Each** (8 min)
   - Decision tree: Do you have labels? Continuous or discrete output? Sequential decisions?
   - Hybrid approaches (semi-supervised, self-supervised)
   - Real-world applications often combine multiple types

### Exercise 2.1: Classifying Real-World Problems by ML Type
**Type:** Conceptual classification (20-25 min)

**Task:**
1. Classify each problem as Supervised (Regression/Classification), Unsupervised, or Reinforcement Learning:
   - Predicting house prices from size, location, bedrooms
   - Grouping news articles by topic (no predefined topics)
   - Teaching a robot to walk
   - Detecting credit card fraud
   - Translating English to French
   - Organizing photos by faces (no names provided)
   - Recommending movies based on viewing history
   - Predicting customer churn
   - Playing chess
   - Compressing images while preserving quality

2. For 3 of the above, describe:
   - What is the input data?
   - What is the output or goal?
   - How would you measure success?

3. Design your own ML problem:
   - Choose a real-world problem you want to solve
   - Specify the ML type
   - Define inputs, outputs, and success metrics

**Expected Outcome:**
- Ability to identify ML problem types
- Understanding of problem formulation
- Recognition that some problems fit multiple categories

---

## 2.2 Regression Techniques

**Duration:** 90 minutes

### Content Outline:

1. **Linear Regression: The Simplest ML Model** (25 min)
   - Problem: Predict y from x using a straight line
   - Model: `y = wx + b`
     - `w`: weight (slope)
     - `b`: bias (intercept)
   - Loss function: Mean Squared Error (MSE)
     - `MSE = (1/n) Σ(y_pred - y_true)²`
   - Goal: Find w and b that minimize MSE

2. **Closed-Form Solution (Normal Equation)** (10 min)
   - For simple linear regression: `w = Cov(x,y) / Var(x)`
   - For multivariate: `w = (X^T X)^(-1) X^T y`
   - Advantages: Exact solution, no iteration
   - Disadvantages: Doesn't scale, doesn't work for all models

3. **Gradient Descent: Iterative Optimization** (30 min)
   - Key idea: Adjust parameters in direction of steepest descent
   - Algorithm:
     1. Initialize w, b randomly
     2. Compute predictions: `y_pred = wx + b`
     3. Compute loss: `MSE = mean((y_pred - y_true)²)`
     4. Compute gradients:
        - `dw = (2/n) Σ x(y_pred - y_true)`
        - `db = (2/n) Σ (y_pred - y_true)`
     5. Update parameters:
        - `w = w - learning_rate * dw`
        - `b = b - learning_rate * db`
     6. Repeat steps 2-5 until convergence

4. **Learning Rate: Critical Hyperparameter** (10 min)
   - Too small: Slow convergence
   - Too large: Oscillation or divergence
   - Visualization of different learning rates
   - Adaptive learning rates (preview)

5. **Polynomial Regression** (15 min)
   - Non-linear relationships with linear models
   - Feature transformation: `x → [x, x², x³, ...]`
   - Still uses linear regression on transformed features
   - Risk: Overfitting with high-degree polynomials

### Exercise 2.2a: Implementing Linear Regression from Scratch
**Type:** Programming (60-75 min)

**Task:**
1. **Generate synthetic data:**
   ```python
   import numpy as np
   np.random.seed(42)
   X = np.random.rand(100, 1) * 10  # 100 samples, 1 feature
   y = 3 * X + 7 + np.random.randn(100, 1) * 2  # y = 3x + 7 + noise
   ```

2. **Implement linear regression class:**
   ```python
   class LinearRegression:
       def __init__(self, learning_rate=0.01):
           self.lr = learning_rate
           self.w = None
           self.b = None

       def fit(self, X, y, epochs=1000):
           # TODO: Initialize w and b
           # TODO: Implement gradient descent loop
           # TODO: Track loss per epoch
           pass

       def predict(self, X):
           # TODO: Return wx + b
           pass
   ```

3. **Train and evaluate:**
   - Fit on the data
   - Plot: original data, fitted line, loss curve
   - Print final w and b (should be close to 3 and 7)

4. **Experiment:**
   - Try different learning rates: 0.001, 0.01, 0.1
   - Try different epoch counts: 100, 1000, 10000
   - What happens with very high learning rate (e.g., 1.0)?

**Expected Outcome:**
- Working linear regression implementation
- Understanding of gradient descent mechanics
- Intuition for learning rate effects

### Exercise 2.2b: Visualizing Gradient Descent Convergence
**Type:** Visualization and analysis (30-40 min)

**Task:**
1. **Modify your LinearRegression class to track:**
   - Loss at each epoch
   - Parameter values (w, b) at each epoch

2. **Create visualizations:**
   - Plot 1: Loss vs epoch (should decrease)
   - Plot 2: Parameter trajectory in (w, b) space
   - Plot 3: Contour plot of loss surface with gradient descent path

3. **Analyze:**
   - How many epochs until convergence?
   - Does the path take the most direct route?
   - What is the loss landscape shape? (Convex? Multiple minima?)

**Expected Outcome:**
- Visual understanding of optimization
- Recognition of convex optimization for linear regression
- Ability to diagnose convergence issues

### Exercise 2.2c: Computing Gradients by Hand
**Type:** Mathematical derivation (40-50 min)

**Task:**
1. **Given:**
   - Model: `y_pred = wx + b`
   - Loss: `L = (1/n) Σ(y_pred - y_true)²`

2. **Derive step-by-step:**
   - ∂L/∂w (gradient with respect to weight)
   - ∂L/∂b (gradient with respect to bias)

3. **Verify numerically:**
   - Use finite differences: `(L(w+h) - L(w-h)) / 2h` for small h
   - Compare to your analytical gradient
   - They should match within ~1e-5

4. **Extend to multivariate:**
   - Model: `y_pred = w₁x₁ + w₂x₂ + b`
   - Derive ∂L/∂w₁, ∂L/∂w₂, ∂L/∂b

**Expected Outcome:**
- Comfort with calculus for ML
- Gradient checking technique (critical for debugging)
- Foundation for backpropagation in Chapter 3

---

## 2.3 Classification Methods

**Duration:** 75 minutes

### Content Outline:

1. **Classification vs Regression** (8 min)
   - Discrete outputs instead of continuous
   - Binary classification: 2 classes (spam/not spam)
   - Multi-class: >2 classes (digit recognition: 0-9)
   - Need different loss function and output activation

2. **Logistic Regression for Binary Classification** (25 min)
   - Linear model + Sigmoid activation
   - Model: `z = wx + b`, then `y_pred = sigmoid(z) = 1/(1 + e^(-z))`
   - Sigmoid squashes output to (0, 1) → probability
   - Decision boundary: Predict class 1 if y_pred > 0.5
   - Loss: Binary Cross-Entropy
     - `BCE = -[y*log(y_pred) + (1-y)*log(1-y_pred)]`
   - Why not MSE? Cross-entropy better for probabilities

3. **Decision Trees** (15 min)
   - Hierarchical if-then-else rules
   - Splitting criteria: Information Gain, Gini Impurity
   - Advantages: Interpretable, handles non-linear boundaries
   - Disadvantages: Prone to overfitting, unstable
   - Not differentiable → can't use gradient descent

4. **Support Vector Machines (SVM)** (12 min)
   - Find the maximum-margin separator
   - Kernel trick for non-linear boundaries
   - Effective in high dimensions
   - Less popular in deep learning era but still useful

5. **Comparison of Methods** (15 min)
   - Linear models: Fast, interpretable, limited expressiveness
   - Tree-based: Non-linear, interpretable, ensemble methods (Random Forest)
   - SVMs: Powerful for small-medium datasets
   - When each method shines

### Exercise 2.3a: Binary Classification with Logistic Regression
**Type:** Programming (50-60 min)

**Task:**
1. **Generate binary classification data:**
   ```python
   from sklearn.datasets import make_classification
   X, y = make_classification(n_samples=200, n_features=2,
                               n_redundant=0, n_informative=2,
                               n_clusters_per_class=1, random_state=42)
   ```

2. **Implement Logistic Regression:**
   ```python
   class LogisticRegression:
       def __init__(self, learning_rate=0.01):
           self.lr = learning_rate
           self.w = None
           self.b = None

       def sigmoid(self, z):
           # TODO: Implement sigmoid function
           pass

       def fit(self, X, y, epochs=1000):
           # TODO: Initialize parameters
           # TODO: Gradient descent with BCE loss
           pass

       def predict_proba(self, X):
           # TODO: Return probabilities
           pass

       def predict(self, X):
           # TODO: Return class labels (0 or 1)
           pass
   ```

3. **Train and visualize:**
   - Fit the model
   - Plot decision boundary
   - Plot training data colored by true class
   - Compute accuracy

4. **Derive gradients:**
   - For BCE loss with sigmoid, derive:
     - ∂L/∂w
     - ∂L/∂b
   - Hint: Chain rule, and the derivative simplifies nicely!

**Expected Outcome:**
- Working logistic regression from scratch
- Understanding of sigmoid and probability outputs
- Visual understanding of decision boundaries

### Exercise 2.3b: Comparing Classification Algorithms
**Type:** Experimental comparison (40-50 min)

**Task:**
1. **Use scikit-learn for quick comparisons:**
   ```python
   from sklearn.linear_model import LogisticRegression
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.svm import SVC
   from sklearn.datasets import make_moons, make_circles
   ```

2. **Create 3 different datasets:**
   - Linearly separable
   - Non-linear (moons)
   - Circular boundary (circles)

3. **For each dataset, train and compare:**
   - Logistic Regression
   - Decision Tree
   - SVM (linear and RBF kernel)

4. **Visualize:**
   - Plot decision boundaries for all methods
   - Compute accuracy for each
   - Create comparison table

5. **Analysis:**
   - Which method works best for which dataset?
   - What are the trade-offs?
   - When would you choose each method?

**Expected Outcome:**
- Practical experience with multiple classifiers
- Understanding that no single method dominates
- Intuition for method selection based on data characteristics

---

## 2.4 Model Evaluation and Validation

**Duration:** 60 minutes

### Content Outline:

1. **Beyond Accuracy: Comprehensive Metrics** (20 min)
   - **Confusion Matrix:** TP, FP, TN, FN
   - **Precision:** TP / (TP + FP) - "Of predicted positives, how many are correct?"
   - **Recall (Sensitivity):** TP / (TP + FN) - "Of actual positives, how many did we find?"
   - **F1 Score:** Harmonic mean of precision and recall
   - **When to use which:** Imbalanced classes, different costs of errors

2. **The Train/Test Split** (10 min)
   - Why: Evaluate generalization, not memorization
   - Typical split: 80/20 or 70/30
   - Random vs stratified splitting

3. **Cross-Validation** (15 min)
   - K-fold CV: Split data into K parts, train K times
   - Each fold used once as validation
   - Average performance across folds
   - Reduces variance in performance estimates
   - Trade-off: More computation

4. **Overfitting and Underfitting** (10 min)
   - **Underfitting:** Model too simple, high bias
   - **Overfitting:** Model too complex, high variance
   - The bias-variance trade-off
   - Detecting: Training accuracy >> Test accuracy = overfitting
   - Solutions: More data, regularization, simpler models

5. **Regularization Preview** (5 min)
   - L1 (Lasso): Encourages sparsity
   - L2 (Ridge): Penalizes large weights
   - Full treatment in Chapter 3

### Exercise 2.4a: Computing Precision, Recall, and F1 Score
**Type:** Mathematical and programming (30-35 min)

**Task:**
1. **Given a confusion matrix:**
   ```
   Predicted:  Positive | Negative
   Actual:
   Positive       45     |    5
   Negative       10     |   40
   ```

2. **Calculate by hand:**
   - Accuracy
   - Precision
   - Recall
   - F1 Score

3. **Implement functions:**
   ```python
   def precision(y_true, y_pred):
       # TODO
       pass

   def recall(y_true, y_pred):
       # TODO
       pass

   def f1_score(y_true, y_pred):
       # TODO
       pass
   ```

4. **Test on real data:**
   - Use your logistic regression from Exercise 2.3a
   - Compute all metrics
   - Verify with sklearn.metrics

5. **Scenario analysis:**
   - Medical diagnosis: Which metric matters more, precision or recall? Why?
   - Spam detection: Which metric matters more?
   - Legal document review: Which metric matters more?

**Expected Outcome:**
- Ability to compute evaluation metrics
- Understanding of metric trade-offs
- Context-aware metric selection

### Exercise 2.4b: Cross-Validation Implementation
**Type:** Programming (35-45 min)

**Task:**
1. **Implement K-fold cross-validation:**
   ```python
   def k_fold_cv(model, X, y, k=5):
       """
       Perform k-fold cross-validation
       Returns: list of scores for each fold
       """
       # TODO: Split data into k folds
       # TODO: For each fold:
       #   - Train on k-1 folds
       #   - Validate on remaining fold
       #   - Record score
       # TODO: Return list of scores
       pass
   ```

2. **Test your implementation:**
   - Use your LogisticRegression class
   - Run 5-fold CV on the classification data
   - Compare with sklearn's cross_val_score

3. **Experiment:**
   - Try k=3, 5, 10
   - Compare: Mean score, standard deviation
   - What happens with k=n (leave-one-out CV)?

4. **Visualize:**
   - Box plot of scores across folds
   - Shows variance in performance estimates

**Expected Outcome:**
- Understanding of cross-validation mechanics
- Ability to implement evaluation schemes
- Appreciation for performance variance

---

## 2.5 The Path to Neural Networks

**Duration:** 30 minutes

### Content Outline:

1. **Limitations of Linear Models** (10 min)
   - Can only learn linear decision boundaries
   - XOR problem: Classic example of non-linear separability
   - Real-world data is rarely linearly separable
   - Feature engineering can help but is manual and domain-specific

2. **The XOR Problem Revisited** (8 min)
   - Truth table: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0
   - Plot the 4 points: No straight line can separate them
   - Single-layer perceptron cannot solve XOR
   - This limitation caused the first AI winter

3. **Need for Non-Linearity** (7 min)
   - Solution: Multiple layers + non-linear activations
   - Preview of multi-layer perceptrons (Chapter 3)
   - Each layer learns increasingly complex features
   - Composition of simple functions → complex functions

4. **When Traditional ML Still Wins** (5 min)
   - Small datasets (< 1000 samples): Linear models often better
   - Interpretability requirements: Trees, linear models
   - Fast inference needed: Simpler models
   - Limited compute: Training neural networks is expensive
   - Well-understood features: Feature engineering + linear model

### Exercise 2.5: Identifying Limitations of Traditional ML
**Type:** Experimental and analytical (30-40 min)

**Task:**
1. **The XOR Problem:**
   ```python
   # XOR data
   X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
   y = np.array([0, 1, 1, 0])
   ```
   - Try to fit logistic regression
   - Plot the decision boundary
   - What's the best accuracy you can achieve? (Should be ~50%)

2. **Feature Engineering Solution:**
   - Add a new feature: `x3 = x1 * x2` (interaction term)
   - Now try logistic regression with 3 features: [x1, x2, x1*x2]
   - Can it solve XOR now? Plot the decision boundary in 3D

3. **Other Non-Linear Problems:**
   - Create spiral dataset: `sklearn.datasets.make_spiral`
   - Try linear classifiers
   - Try polynomial features up to degree 3
   - What degree is needed for good accuracy?

4. **Reflection:**
   - Why is manual feature engineering not scalable?
   - What would you need for a 100-dimensional input space?
   - How do neural networks solve this? (Preview)

**Expected Outcome:**
- Deep understanding of linear model limitations
- Appreciation for the XOR problem's historical significance
- Motivation for neural networks in Chapter 3

---

## Math Appendix 2A: Linear Algebra Fundamentals

**Duration:** 60-90 minutes (reference material)

### Content Outline:

1. **Vectors** (15 min)
   - Definition and notation
   - Geometric interpretation
   - Vector operations: addition, scalar multiplication
   - Dot product: `a · b = Σ aᵢbᵢ`
   - Magnitude and unit vectors

2. **Matrices** (20 min)
   - Definition and notation
   - Matrix-vector multiplication
   - Matrix-matrix multiplication
   - Identity matrix
   - Transpose: `(AB)ᵀ = BᵀAᵀ`

3. **Matrix Operations for ML** (15 min)
   - Broadcasting in NumPy
   - Element-wise vs matrix multiplication
   - Common patterns in ML:
     - `y = Xw` (linear transformation)
     - `X^T X` (Gram matrix)
     - Outer products

4. **Matrix Inverse** (10 min)
   - Definition: `AA⁻¹ = I`
   - When it exists (non-singular matrices)
   - Computing inverse (not practical for large matrices)
   - Solving linear systems

5. **Eigenvalues and Eigenvectors** (15 min)
   - Definition: `Av = λv`
   - Geometric interpretation
   - Applications in ML: PCA, power iteration
   - Computing eigenvalues (not detailed here)

6. **Practical NumPy Reference** (15 min)
   ```python
   # Vector operations
   a = np.array([1, 2, 3])
   b = np.array([4, 5, 6])
   np.dot(a, b)  # Dot product

   # Matrix operations
   A = np.array([[1, 2], [3, 4]])
   B = np.array([[5, 6], [7, 8]])
   A @ B  # Matrix multiplication
   A.T   # Transpose
   np.linalg.inv(A)  # Inverse
   np.linalg.eig(A)  # Eigenvalues
   ```

---

## Math Appendix 2B: Calculus for Machine Learning

**Duration:** 90-120 minutes (reference material)

### Content Outline:

1. **Derivatives** (20 min)
   - Definition: Rate of change
   - Geometric interpretation: Slope of tangent line
   - Power rule: `d/dx (xⁿ) = nxⁿ⁻¹`
   - Common functions:
     - `d/dx (eˣ) = eˣ`
     - `d/dx (ln x) = 1/x`
     - `d/dx (sin x) = cos x`

2. **Partial Derivatives** (20 min)
   - Functions of multiple variables: `f(x, y)`
   - Partial derivative: Derivative w.r.t. one variable, holding others constant
   - Notation: `∂f/∂x`, `∂f/∂y`
   - Example: `f(x,y) = x²y + y³`
     - `∂f/∂x = 2xy`
     - `∂f/∂y = x² + 3y²`

3. **Chain Rule** (25 min)
   - Single variable: `d/dx f(g(x)) = f'(g(x)) · g'(x)`
   - Multiple variables: Accumulate gradients along paths
   - Critical for backpropagation!
   - Examples:
     - `f(x) = (x² + 1)³`
     - `f(x) = e^(x²)`
     - `f(x, y) = sin(x² + y²)`

4. **Gradient Vectors** (20 min)
   - Gradient: Vector of all partial derivatives
   - `∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]`
   - Points in direction of steepest ascent
   - Gradient descent: Move opposite to gradient

5. **Matrix Calculus** (25 min)
   - Derivatives of matrix expressions
   - Key identities:
     - `∂/∂x (wx) = w`
     - `∂/∂x (x^T A x) = (A + A^T)x`
     - `∂/∂W (Wx) = x^T`
   - Used extensively in neural network gradients

6. **Jacobian and Hessian** (15 min)
   - Jacobian: Matrix of all first-order partial derivatives
   - Hessian: Matrix of all second-order partial derivatives
   - Applications: Optimization, error analysis

---

## Math Appendix 2C: Probability and Statistics

**Duration:** 75-90 minutes (reference material)

### Content Outline:

1. **Probability Basics** (15 min)
   - Sample space, events
   - Probability axioms
   - Conditional probability: `P(A|B) = P(A∩B) / P(B)`
   - Independence: `P(A∩B) = P(A)P(B)`

2. **Random Variables and Distributions** (20 min)
   - Discrete vs continuous
   - Probability mass function (PMF)
   - Probability density function (PDF)
   - Cumulative distribution function (CDF)
   - Common distributions:
     - Bernoulli (binary)
     - Binomial
     - Normal (Gaussian)
     - Uniform

3. **Expectation and Variance** (15 min)
   - Expected value: `E[X] = Σ x P(X=x)`
   - Variance: `Var(X) = E[(X - μ)²]`
   - Standard deviation: `σ = √Var(X)`
   - Properties:
     - `E[aX + b] = aE[X] + b`
     - `Var(aX + b) = a²Var(X)`

4. **Maximum Likelihood Estimation** (15 min)
   - Idea: Find parameters that maximize probability of observed data
   - Log-likelihood (easier to work with)
   - MLE for normal distribution
   - Connection to loss functions in ML

5. **Bayes' Theorem** (10 min)
   - `P(A|B) = P(B|A)P(A) / P(B)`
   - Prior, likelihood, posterior
   - Applications: Naive Bayes classifier
   - Bayesian vs frequentist perspectives

---

## Chapter 2 Summary

**Key Takeaways:**
1. Three types of ML: Supervised, Unsupervised, Reinforcement
2. Linear/logistic regression are foundational models
3. Gradient descent is the workhorse optimization algorithm
4. Proper evaluation requires train/test split and cross-validation
5. Linear models have fundamental limitations → need for neural networks

**Prerequisites for Chapter 3:**
- Comfort with gradient descent
- Understanding of forward pass (prediction) and backward pass (gradients)
- Basic NumPy implementation skills
- Completed Exercise 2.2a (linear regression from scratch)

**Total Exercises:** 8 main exercises + 3 math appendices
**Total Time:** 5-6 hours reading + 6-8 hours exercises = **11-14 hours**

---

## Consistency Check (Internal)

**Terminology:**
- "Training" and "learning" used interchangeably ✓
- "Parameters" (w, b) vs "hyperparameters" (learning rate) clearly distinguished ✓
- "Loss" and "cost function" used interchangeably ✓

**Prerequisites:**
- Assumes Chapter 1 completion ✓
- Math appendices support all concepts ✓
- Exercises build on each other sequentially ✓

**Flow to Chapter 3:**
- Exercise 2.5 explicitly motivates neural networks ✓
- Gradient computation skills transfer directly ✓
- XOR problem sets up multi-layer perceptrons ✓
