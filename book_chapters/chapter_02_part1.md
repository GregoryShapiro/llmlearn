# Chapter 2: Machine Learning Fundamentals (Part 1)

> "Machine learning is the science of getting computers to learn without being explicitly programmed." — Andrew Ng

Welcome to Chapter 2! This is where your journey into practical machine learning begins. By the end of this chapter, you'll have implemented core machine learning algorithms from scratch, understood the optimization techniques that power all of AI, and built the foundation for neural networks.

**What you'll learn in this part:**
- The three paradigms of machine learning (supervised, unsupervised, reinforcement)
- Linear regression: Your first learning algorithm
- Gradient descent: The optimization engine of machine learning
- How to implement learning from data using only NumPy

Let's dive in.

---

## 2.1 Types of Machine Learning

Before we implement any algorithms, we need to understand the landscape of machine learning. Not all learning problems are the same, and different types of problems require different approaches. Machine learning is broadly divided into three paradigms, each suited to different scenarios.

### Supervised Learning: Learning from Examples

**Supervised learning** is the most common type of machine learning. In supervised learning, we have:
- **Input data** (features, attributes, X)
- **Output data** (labels, targets, y)
- **Goal:** Learn a function that maps inputs to outputs

The term "supervised" comes from the idea that we're providing the algorithm with the "correct answers" during training, like a teacher supervising a student's learning.

**Two Categories of Supervised Learning:**

**1. Regression: Predicting Continuous Values**

When the output is a continuous number, we have a regression problem.

Examples:
- **Predicting house prices** from square footage, location, number of bedrooms
  - Input: [2000 sqft, 3 bedrooms, suburban]
  - Output: $350,000

- **Estimating temperature** from atmospheric conditions
  - Input: [pressure, humidity, wind speed]
  - Output: 72.5°F

- **Stock price prediction** from historical data and market indicators
  - Input: [previous prices, trading volume, market sentiment]
  - Output: $142.73

The key characteristic: The output can take any value within a range. There are infinitely many possible outputs.

**2. Classification: Predicting Discrete Categories**

When the output is a category or class, we have a classification problem.

Examples:
- **Email spam detection**
  - Input: Email text, sender, metadata
  - Output: "spam" or "not spam" (binary classification)

- **Medical diagnosis**
  - Input: Symptoms, test results, patient history
  - Output: Disease category (multi-class classification)

- **Image recognition**
  - Input: Image pixels
  - Output: "cat", "dog", "bird", etc.

The key characteristic: The output is one of a fixed set of categories. There are finitely many possible outputs.

**The Supervised Learning Process:**

```
1. Collect labeled training data: (x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)
2. Choose a model (hypothesis function): ŷ = f(x; θ)
3. Define a loss function: L(ŷ, y)
4. Find parameters θ that minimize loss on training data
5. Use trained model to predict on new, unseen data
```

**Why "Learning"?**

The model isn't explicitly programmed with rules. Instead, it learns patterns from examples. Given enough examples of houses and their prices, the model discovers that:
- Larger houses cost more
- Location matters enormously
- Modern kitchens increase value
- Proximity to good schools is valuable

These rules aren't coded by a programmer—they emerge from data.

### Unsupervised Learning: Finding Patterns Without Labels

**Unsupervised learning** has a fundamentally different goal. We have:
- **Input data** (X)
- **No output labels** (no y)
- **Goal:** Discover hidden structure or patterns in the data

Without labeled examples, the algorithm must find interesting structure on its own.

**Common Unsupervised Learning Tasks:**

**1. Clustering: Grouping Similar Items**

Partition data into groups (clusters) where items in the same group are similar.

Examples:
- **Customer segmentation** in marketing
  - Input: Purchase history, demographics, browsing behavior
  - Output: Customer segments (e.g., "bargain hunters", "luxury buyers", "window shoppers")
  - Use: Targeted marketing campaigns

- **Document organization**
  - Input: Collection of news articles
  - Output: Topics (sports, politics, technology, etc.)
  - Use: Automatic categorization

- **Genomics**
  - Input: Gene expression data
  - Output: Groups of related genes
  - Use: Understanding genetic pathways

**2. Dimensionality Reduction: Compressing Data**

Reduce the number of features while preserving important information.

Examples:
- **Principal Component Analysis (PCA)**
  - Input: High-dimensional data (e.g., 1000 features)
  - Output: Low-dimensional representation (e.g., 2-3 features)
  - Use: Visualization, noise reduction, faster computation

- **t-SNE** for visualization
  - Input: High-dimensional embeddings
  - Output: 2D scatter plot
  - Use: Visualizing how data clusters in high dimensions

**3. Anomaly Detection: Finding Outliers**

Identify data points that don't fit the normal pattern.

Examples:
- **Credit card fraud detection**
  - Learn normal spending patterns
  - Flag unusual transactions

- **Manufacturing quality control**
  - Learn normal product specifications
  - Detect defective items

- **Network intrusion detection**
  - Learn normal network traffic
  - Identify potential attacks

**Why Unsupervised Learning?**

Labels are expensive. Labeling thousands of images as "cat" or "dog" requires human effort. Labeling medical images requires expert doctors. Unsupervised learning can:
- Discover patterns humans haven't noticed
- Work with the vast amounts of unlabeled data available
- Reduce data to its essential features
- Serve as preprocessing for supervised learning

### Reinforcement Learning: Learning Through Trial and Error

**Reinforcement learning** (RL) is fundamentally different from both supervised and unsupervised learning. Instead of learning from a fixed dataset, an RL agent learns by interacting with an environment.

**Key Concepts:**

**Agent:** The learner/decision maker
**Environment:** The world the agent interacts with
**State:** Current situation of the agent
**Action:** What the agent can do
**Reward:** Feedback from the environment (positive or negative)
**Policy:** The agent's strategy for choosing actions

**The Reinforcement Learning Loop:**

```
1. Agent observes current state
2. Agent chooses action based on policy
3. Environment transitions to new state
4. Environment provides reward
5. Agent updates policy to maximize future rewards
6. Repeat
```

**Examples:**

**Game Playing:**
- **AlphaGo** (Go)
  - State: Board position
  - Actions: Possible moves
  - Reward: +1 for winning, -1 for losing, 0 during play
  - Learned: Strategies that beat world champions

- **Atari games** (DeepMind)
  - State: Screen pixels
  - Actions: Controller inputs (up, down, left, right, fire)
  - Reward: Game score
  - Learned: Superhuman performance on many games

**Robotics:**
- **Robotic manipulation**
  - State: Joint angles, object positions
  - Actions: Motor commands
  - Reward: Successfully grasping object
  - Learned: How to grasp various objects

- **Autonomous driving**
  - State: Sensor data (cameras, LIDAR)
  - Actions: Steering, acceleration, braking
  - Reward: Safe arrival at destination
  - Learned: Driving policy

**Resource Management:**
- **Data center cooling** (Google)
  - State: Temperature sensors, server load
  - Actions: Cooling system adjustments
  - Reward: Energy saved while maintaining temperature
  - Learned: Optimal cooling strategy (40% energy reduction)

**Key Difference from Supervised Learning:**

In supervised learning, we tell the agent the correct answer for each input. In RL, we only provide feedback on outcomes. The agent must figure out which sequence of actions led to good outcomes.

Example: Teaching a robot to walk
- **Supervised:** Show examples of "correct" joint angles at each timestep (hard to specify!)
- **Reinforcement:** Give reward for forward movement, penalty for falling (much easier!)

**Challenges:**

- **Exploration vs. Exploitation:** Should agent try new actions (explore) or stick with known good actions (exploit)?
- **Credit Assignment:** Which action in a long sequence deserves credit for the reward?
- **Sample Efficiency:** RL often requires millions of trials to learn (expensive in real world)

### Comparing the Three Paradigms

| Aspect | Supervised | Unsupervised | Reinforcement |
|--------|-----------|--------------|---------------|
| **Input** | Features + Labels | Features only | State + Actions + Rewards |
| **Goal** | Predict labels | Find patterns | Maximize cumulative reward |
| **Feedback** | Correct answers | None | Rewards/penalties |
| **Examples** | Spam detection, medical diagnosis | Clustering, PCA | Game playing, robotics |
| **When to use** | Have labeled data | Want to discover structure | Sequential decision making |

**Hybrid Approaches:**

Real-world systems often combine paradigms:
- **Semi-supervised learning:** Small amount of labeled data + large amount of unlabeled data
- **Self-supervised learning:** Create labels automatically from data structure (e.g., predict next word from previous words)
- **Imitation learning:** Supervised learning from expert demonstrations, then RL fine-tuning

### Which Paradigm for Which Problem?

**Use Supervised Learning when:**
- You have labeled training examples
- You want to predict specific outputs
- The relationship between input and output is learnable from examples
- Examples: Image classification, price prediction, language translation

**Use Unsupervised Learning when:**
- You have no labels (or labels are too expensive)
- You want to discover hidden patterns
- You need to reduce data complexity
- Examples: Customer segmentation, anomaly detection, data visualization

**Use Reinforcement Learning when:**
- Problem involves sequential decision making
- Feedback is in the form of rewards, not correct answers
- Agent must learn through interaction
- Examples: Game playing, robotics, resource allocation

**This Chapter's Focus:**

We'll focus on **supervised learning**, specifically:
- Regression (predicting continuous values)
- Classification (predicting categories)

These form the foundation for neural networks, which are predominantly supervised learning methods. The techniques you'll learn—gradient descent, loss functions, backpropagation—all come from supervised learning.

---

## Exercise 2.1: Classifying Real-World Problems by ML Type

**Time: 20-25 minutes**

**Objective:** Develop intuition for identifying which type of machine learning fits different problems.

**Part 1: Problem Classification (15 min)**

For each problem below, determine:
1. **Type:** Supervised (Regression/Classification), Unsupervised, or Reinforcement Learning
2. **Input:** What data is available?
3. **Output/Goal:** What are we trying to predict or achieve?
4. **Why this type?** Brief justification

**Problems to classify:**

**A. Predicting house prices from size, location, and age**
- Type: _____________
- Input: _____________
- Output: _____________
- Why: _____________

**B. Grouping news articles by topic (no predefined topics)**
- Type: _____________
- Input: _____________
- Output: _____________
- Why: _____________

**C. Teaching a robot to walk**
- Type: _____________
- Input: _____________
- Output: _____________
- Why: _____________

**D. Detecting fraudulent credit card transactions**
- Type: _____________
- Input: _____________
- Output: _____________
- Why: _____________

**E. Translating English sentences to French**
- Type: _____________
- Input: _____________
- Output: _____________
- Why: _____________

**F. Organizing photos by faces (no names provided)**
- Type: _____________
- Input: _____________
- Output: _____________
- Why: _____________

**G. Recommending movies based on viewing history**
- Type: _____________ (Trick question: Could be multiple types!)
- Input: _____________
- Output: _____________
- Why: _____________

**H. Predicting whether a customer will churn (leave the service)**
- Type: _____________
- Input: _____________
- Output: _____________
- Why: _____________

**I. Playing chess optimally**
- Type: _____________
- Input: _____________
- Output: _____________
- Why: _____________

**J. Compressing images while preserving quality**
- Type: _____________
- Input: _____________
- Output: _____________
- Why: _____________

**Part 2: Deep Dive (10 min)**

Choose three problems from above. For each, answer:

1. **What would make this problem easier or harder?**
   - More/less data?
   - Different features?
   - Simpler/more complex patterns?

2. **How would you measure success?**
   - What metric would you use?
   - What performance level would be "good enough"?

3. **What could go wrong?**
   - Potential failure modes?
   - Ethical considerations?
   - What if predictions are wrong?

**Example Response (for Spam Detection):**

**Type:** Supervised Learning - Binary Classification
**Input:** Email text, sender information, subject line, metadata
**Output:** "spam" or "not spam"
**Why:** We have labeled examples (users mark emails as spam), and we want to predict a category for new emails.

**What makes it easier/harder:**
- Easier: More labeled examples, clear spam signals (viagra, lottery)
- Harder: Spammers adapt, legitimate emails from unknown senders, context-dependent (work email vs. personal)

**Success metric:**
- Precision: Of emails marked spam, how many are actually spam? (minimize false positives)
- Recall: Of actual spam, how much did we catch? (minimize false negatives)
- Balance: False positives frustrate users more than false negatives

**What could go wrong:**
- False positives: Important emails missed (job offer, medical results)
- False negatives: Inbox cluttered with spam
- Bias: Legitimate emails from certain demographics wrongly flagged
- Adversarial: Spammers adapt to evade filter

**Part 3: Design Your Own Problem (5 min)**

Think of a real-world problem you'd like to solve with machine learning.

1. **Problem description:** What do you want to achieve?
2. **ML type:** Which paradigm is most appropriate?
3. **Data requirements:** What data would you need?
4. **Success criteria:** How would you know it's working?
5. **Challenges:** What would be difficult?

**What You'll Learn:**

By completing this exercise, you'll:
- Recognize the appropriate ML paradigm for different problems
- Understand the relationship between problem structure and learning type
- Appreciate the importance of framing problems correctly
- See that some problems can be solved multiple ways

This skill—problem framing—is often the difference between ML success and failure.

---

## 2.2 Regression Techniques

Now we get to implement our first machine learning algorithm: **linear regression**. This is where theory meets practice. By the end of this section, you'll have built a learning algorithm from scratch using only NumPy.

### The Simplest Machine Learning Model: Linear Regression

Linear regression is the "Hello, World!" of machine learning. It's simple enough to understand completely, yet powerful enough to be useful in real applications. More importantly, it introduces all the key concepts you'll use in neural networks:
- Model parameters (weights and biases)
- Loss functions
- Gradient descent
- Training loops

**The Problem:**

Given examples of inputs x and outputs y, learn a function that predicts y from x.

**The Assumption:**

The relationship is linear (or approximately linear):

```
y = wx + b
```

Where:
- `x`: Input (feature, independent variable)
- `y`: Output (target, dependent variable)
- `w`: Weight (slope) - what we learn
- `b`: Bias (intercept) - what we learn

**Geometric Interpretation:**

We're fitting a line through data points. The line is defined by slope `w` and intercept `b`.

```
    y
    ^
    |     *
  8 |       *     Line: y = 2x + 1
    |   *
  6 |     *
    | *
  4 | *
    |*
  2 |
    |________________> x
    0  1  2  3  4
```

**Example: Predicting House Prices**

Suppose we have data:

| Size (sqft) | Price ($1000s) |
|-------------|----------------|
| 1000 | 200 |
| 1500 | 250 |
| 2000 | 300 |
| 2500 | 350 |
| 3000 | 400 |

We want to learn: `Price = w × Size + b`

Looking at the data, we can see the pattern:
- For every additional 1000 sqft, price increases by about $50k
- A 0 sqft house would theoretically cost $150k (unrealistic, but that's the intercept)

So approximately: `Price = 0.1 × Size + 150`

But how do we find these numbers automatically from data?

### The Loss Function: Measuring Error

We need a way to measure how good our current parameters (w, b) are. This is the **loss function** (also called cost function or objective function).

**Mean Squared Error (MSE):**

The most common loss for regression:

```
L(w, b) = (1/n) Σᵢ (ŷᵢ - yᵢ)²

Where:
- n: number of training examples
- ŷᵢ: prediction for example i (ŷᵢ = wxᵢ + b)
- yᵢ: actual value for example i
- (ŷᵢ - yᵢ): error for example i
```

**Why squared error?**

1. **Penalizes large errors more:** Error of 10 contributes 100, but error of 1 contributes only 1
2. **Always positive:** We don't want positive and negative errors to cancel out
3. **Mathematically convenient:** Derivatives are clean
4. **Statistical justification:** Under certain assumptions, MSE is the maximum likelihood estimator

**Example Calculation:**

Suppose we have three data points and current parameters w=1, b=0:

| x | y (actual) | ŷ (predicted) | Error | Squared Error |
|---|------------|---------------|-------|---------------|
| 1 | 3 | 1 | 2 | 4 |
| 2 | 5 | 2 | 3 | 9 |
| 3 | 7 | 3 | 4 | 16 |

```
MSE = (4 + 9 + 16) / 3 = 29/3 ≈ 9.67
```

This is high! We need better parameters. If instead w=2, b=1:

| x | y (actual) | ŷ (predicted) | Error | Squared Error |
|---|------------|---------------|-------|---------------|
| 1 | 3 | 3 | 0 | 0 |
| 2 | 5 | 5 | 0 | 0 |
| 3 | 7 | 7 | 0 | 0 |

```
MSE = (0 + 0 + 0) / 3 = 0
```

Perfect! These are the optimal parameters for this data.

**The Learning Problem:**

```
Find w*, b* = argmin_{w,b} L(w, b)
```

In words: Find the parameters that minimize the loss function.

### Gradient Descent: The Optimization Workhorse

How do we find the optimal parameters? For linear regression with MSE, there's a closed-form solution (the "Normal Equation"), but it doesn't scale to large datasets or more complex models.

Instead, we use **gradient descent**, the fundamental optimization algorithm in machine learning.

**The Intuition:**

Imagine you're on a mountain in thick fog, trying to reach the valley (minimum altitude). You can't see where the valley is, but you can feel the slope under your feet. Strategy:
1. Check which direction is steepest downhill
2. Take a step in that direction
3. Repeat until you reach the bottom

Gradient descent does exactly this in parameter space.

**The Algorithm:**

```
1. Initialize w, b randomly (or to zero)
2. Repeat until convergence:
   a. Compute gradient of loss w.r.t. w and b
   b. Update: w = w - α × ∂L/∂w
   c. Update: b = b - α × ∂L/∂b
```

Where:
- `α`: Learning rate (step size)
- `∂L/∂w`: Partial derivative of loss with respect to w (gradient)
- `∂L/∂b`: Partial derivative of loss with respect to b

**The Gradients:**

For MSE loss with linear model:

```
∂L/∂w = (1/n) Σᵢ 2(ŷᵢ - yᵢ) × xᵢ
       = (2/n) Σᵢ (wxᵢ + b - yᵢ) × xᵢ

∂L/∂b = (1/n) Σᵢ 2(ŷᵢ - yᵢ)
       = (2/n) Σᵢ (wxᵢ + b - yᵢ)
```

(The factor of 2 is often dropped or absorbed into the learning rate)

**Visual Example:**

```
Loss
  ^
  |
  |     *
  |    / \
  | * /   \ *
  |  /     \
  | /       \
  |/_________\______> w
    w₀  w₁ w₂  w*

Starting at w₀:
- Gradient is negative (slope goes down to the right)
- Update: w₁ = w₀ - α × (negative) = w₀ + positive
- We move right toward the minimum

At w₂:
- Gradient is positive (slope goes up to the right)
- Update: w₃ = w₂ - α × (positive) = w₂ - positive
- We move left toward the minimum

Eventually converge to w*
```

**The Learning Rate: A Critical Hyperparameter**

The learning rate α determines step size:

**Too Small (α = 0.001):**
- Safe: Won't overshoot minimum
- Problem: Very slow convergence, thousands of iterations
- Might get stuck in plateaus

**Too Large (α = 1.0):**
- Fast: Takes big steps
- Problem: Might overshoot and oscillate
- Could diverge (loss increases instead of decreases!)

**Just Right (α = 0.1):**
- Converges in reasonable time
- Stable updates
- Reaches near-optimal solution

**Example Convergence:**

```
Epoch | w | b | Loss
------|---|---|------
0     | 0.0 | 0.0 | 100.00
1     | 0.5 | 0.3 | 85.23
2     | 0.9 | 0.5 | 72.45
...
50    | 1.98 | 0.97 | 2.13
100   | 2.00 | 1.00 | 0.01
```

We see loss steadily decreasing, converging to near-zero.

### Step-by-Step Example

Let's work through a complete example by hand.

**Data:**

| x | y |
|---|---|
| 1 | 3 |
| 2 | 5 |
| 3 | 7 |

True relationship: y = 2x + 1 (but our algorithm doesn't know this)

**Initialization:**
```
w = 0, b = 0, α = 0.1
```

**Iteration 1:**

Predictions:
```
ŷ₁ = 0×1 + 0 = 0
ŷ₂ = 0×2 + 0 = 0
ŷ₃ = 0×3 + 0 = 0
```

Loss:
```
MSE = [(0-3)² + (0-5)² + (0-7)²] / 3
    = [9 + 25 + 49] / 3
    = 83/3 ≈ 27.67
```

Gradients:
```
∂L/∂w = (2/3) × [(0-3)×1 + (0-5)×2 + (0-7)×3]
      = (2/3) × [-3 - 10 - 21]
      = (2/3) × (-34)
      = -22.67

∂L/∂b = (2/3) × [(0-3) + (0-5) + (0-7)]
      = (2/3) × (-15)
      = -10
```

Updates:
```
w = 0 - 0.1 × (-22.67) = 2.267
b = 0 - 0.1 × (-10) = 1.0
```

**Iteration 2:**

Predictions:
```
ŷ₁ = 2.267×1 + 1.0 = 3.267
ŷ₂ = 2.267×2 + 1.0 = 5.534
ŷ₃ = 2.267×3 + 1.0 = 7.801
```

Loss:
```
MSE = [(3.267-3)² + (5.534-5)² + (7.801-7)²] / 3
    = [0.071 + 0.285 + 0.641] / 3
    = 0.997/3 ≈ 0.33
```

Much better! Loss dropped from 27.67 to 0.33.

Gradients (I'll spare you the arithmetic):
```
∂L/∂w ≈ 1.78
∂L/∂b ≈ 0.67
```

Updates:
```
w = 2.267 - 0.1 × 1.78 = 2.089
b = 1.0 - 0.1 × 0.67 = 0.933
```

**After many iterations:**
```
w → 2.0
b → 1.0
Loss → 0.0
```

We've recovered the true parameters!

### Polynomial Regression: Beyond Lines

What if the relationship isn't linear? We can use **feature engineering** to fit non-linear relationships with linear models.

**Idea:** Transform input features

Instead of:
```
y = w₁x + b
```

Use:
```
y = w₁x + w₂x² + w₃x³ + b
```

This is still linear in the parameters (w₁, w₂, w₃), so we can use the same linear regression algorithm!

**Example: Quadratic Fit**

Data that follows y = x²:

| x | x² | y |
|---|-----|---|
| 1 | 1 | 1 |
| 2 | 4 | 4 |
| 3 | 9 | 9 |

We create new features [x, x²] and fit:
```
y = w₁x + w₂x²
```

Gradient descent will find w₁ ≈ 0, w₂ ≈ 1.

**The Risk: Overfitting**

With high-degree polynomials, we can fit training data perfectly but generalize poorly:

```
# 9th degree polynomial through 10 points
y = w₁x + w₂x² + ... + w₉x⁹ + b
```

This might pass through all training points (zero training error) but wiggle wildly between them, performing poorly on new data.

**Regularization** (covered in Chapter 3) helps prevent overfitting by penalizing large weights.

---

## Exercise 2.2a: Implementing Linear Regression from Scratch

**Time: 60-75 minutes**

**Objective:** Build a complete linear regression implementation using only NumPy.

**Part 1: Generate Synthetic Data (5 min)**

```python
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate data: y = 3x + 7 + noise
n_samples = 100
X = np.random.rand(n_samples, 1) * 10  # Random x values from 0 to 10
y = 3 * X + 7 + np.random.randn(n_samples, 1) * 2  # Add noise

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Training Data')
plt.grid(True)
plt.show()

print(f"Data shape: X {X.shape}, y {y.shape}")
print(f"X range: [{X.min():.2f}, {X.max():.2f}]")
print(f"y range: [{y.min():.2f}, {y.max():.2f}]")
```

**Part 2: Implement Linear Regression Class (30-40 min)**

```python
class LinearRegression:
    def __init__(self, learning_rate=0.01):
        """
        Initialize linear regression model.

        Parameters:
        learning_rate: float, step size for gradient descent
        """
        self.lr = learning_rate
        self.w = None  # Weight (slope)
        self.b = None  # Bias (intercept)
        self.losses = []  # Track loss per epoch

    def fit(self, X, y, epochs=1000, verbose=True):
        """
        Train the model using gradient descent.

        Parameters:
        X: np.array, shape (n_samples, 1), input features
        y: np.array, shape (n_samples, 1), target values
        epochs: int, number of training iterations
        verbose: bool, whether to print progress

        Returns:
        self: trained model
        """
        n_samples = X.shape[0]

        # Initialize parameters
        self.w = np.zeros((1, 1))
        self.b = np.zeros((1, 1))

        # Training loop
        for epoch in range(epochs):
            # TODO: Forward pass - compute predictions
            # y_pred = ...

            # TODO: Compute loss (MSE)
            # loss = ...

            # TODO: Compute gradients
            # dw = ...
            # db = ...

            # TODO: Update parameters
            # self.w = ...
            # self.b = ...

            # Track progress
            self.losses.append(loss)

            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:4d}: Loss = {loss:.4f}, w = {self.w[0,0]:.4f}, b = {self.b[0,0]:.4f}")

        return self

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X: np.array, shape (n_samples, 1), input features

        Returns:
        predictions: np.array, shape (n_samples, 1)
        """
        # TODO: Implement prediction
        # return ...
        pass

    def score(self, X, y):
        """
        Compute R² score (coefficient of determination).

        Parameters:
        X: np.array, shape (n_samples, 1), input features
        y: np.array, shape (n_samples, 1), true values

        Returns:
        r2: float, R² score (1.0 is perfect, 0.0 is baseline)
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
        ss_tot = np.sum((y - y.mean()) ** 2)  # Total sum of squares
        r2 = 1 - (ss_res / ss_tot)
        return r2
```

**Part 3: Train and Evaluate (15-20 min)**

```python
# Create and train model
model = LinearRegression(learning_rate=0.01)
model.fit(X, y, epochs=1000)

# Make predictions
y_pred = model.predict(X)

# Evaluate
r2 = model.score(X, y)
print(f"\nFinal Results:")
print(f"Learned parameters: w = {model.w[0,0]:.4f}, b = {model.b[0,0]:.4f}")
print(f"True parameters: w = 3.0, b = 7.0")
print(f"R² score: {r2:.4f}")

# Visualize results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Fitted line
ax1.scatter(X, y, alpha=0.5, label='Data')
ax1.plot(X, y_pred, 'r-', linewidth=2, label='Fitted line')
ax1.set_xlabel('X')
ax1.set_ylabel('y')
ax1.set_title('Linear Regression Fit')
ax1.legend()
ax1.grid(True)

# Plot 2: Loss curve
ax2.plot(model.losses)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss (MSE)')
ax2.set_title('Training Loss Over Time')
ax2.set_yscale('log')  # Log scale to see convergence better
ax2.grid(True)

plt.tight_layout()
plt.show()
```

**Part 4: Experiment with Learning Rates (10-15 min)**

```python
# Try different learning rates
learning_rates = [0.001, 0.01, 0.1, 0.5]

plt.figure(figsize=(12, 8))
for lr in learning_rates:
    model = LinearRegression(learning_rate=lr)
    model.fit(X, y, epochs=200, verbose=False)

    plt.plot(model.losses, label=f'LR = {lr}')

plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Effect of Learning Rate on Convergence')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()

# What happens with very high learning rate?
print("\nTrying learning rate = 1.0 (too high):")
model_diverge = LinearRegression(learning_rate=1.0)
model_diverge.fit(X, y, epochs=100, verbose=True)
```

**Expected Outcomes:**

1. **Successful training:**
   - Loss decreases smoothly
   - Final w ≈ 3.0, b ≈ 7.0 (close to true values)
   - R² ≈ 0.97-0.99 (high correlation)

2. **Learning rate effects:**
   - LR too small (0.001): Slow convergence, may not reach minimum in 200 epochs
   - LR optimal (0.01-0.1): Fast, smooth convergence
   - LR too large (0.5-1.0): Oscillation or divergence

3. **Visual confirmation:**
   - Fitted line passes through data cloud
   - Loss curve shows exponential decay (looks linear on log scale)

**Common Issues and Solutions:**

**Issue: "Loss is NaN"**
- Cause: Learning rate too high, causing overflow
- Solution: Reduce learning rate by 10x

**Issue: "Loss not decreasing"**
- Cause: Learning rate too small, or bug in gradient calculation
- Solution: Check gradient formulas, try larger learning rate

**Issue: "Parameters not close to [3, 7]"**
- Cause: Not enough epochs, or data has high noise
- Solution: Train longer, check data generation

**What You've Accomplished:**

- ✓ Implemented gradient descent from scratch
- ✓ Trained a model to learn from data
- ✓ Visualized learning process
- ✓ Understood learning rate effects
- ✓ Built intuition for optimization

This is a major milestone! You've implemented the core of machine learning: learning parameters from data through optimization.

---

**End of Part 1**

In the next part, we'll continue with:
- Exercise 2.2b: Visualizing Gradient Descent Convergence
- Exercise 2.2c: Computing Gradients by Hand
- Section 2.3: Classification Methods
- Section 2.4: Model Evaluation and Validation

**Total time for Part 1:** Approximately 2-2.5 hours
**Words:** ~5,200 words
