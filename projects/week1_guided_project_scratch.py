"""
Week 1 - Guided Project: Implementing a Basic MLP to Classify XOR (NumPy Version)
--------------------------------------------------------------------------------
This script implements a Multi-Layer Perceptron (MLP) **from scratch** using NumPy
to solve the XOR classification problem.

Concepts Covered:
✅ Forward Propagation (Manual Computation)
✅ Activation Functions (Sigmoid, ReLU)
✅ Backpropagation (Gradient Computation)
✅ Binary Cross-Entropy Loss
✅ Stochastic Gradient Descent (SGD)
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------
# 📌 Step 1: Define XOR Dataset
# ------------------------------------------------------
"""
XOR Dataset:
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [[0], [1], [1], [0]]

We need a hidden layer to introduce non-linearity and solve XOR.
"""

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
y = np.array([[0], [1], [1], [0]])  # Labels

np.random.seed(42)  # Ensure reproducibility

# ------------------------------------------------------
# 📌 Step 2: Initialize Weights & Biases
# ------------------------------------------------------
"""
We use **random weight initialization**:
    W1: (2x2) matrix for input → hidden layer
    b1: (1x2) bias for hidden layer
    W2: (2x1) matrix for hidden → output layer
    b2: (1x1) bias for output layer

Weights are initialized using Xavier (Glorot) initialization:
    W ~ N(0, 1 / fan_in)
"""
input_size = 2
hidden_size = 2
output_size = 1

W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1 / input_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1 / hidden_size)
b2 = np.zeros((1, output_size))

# ------------------------------------------------------
# 📌 Step 3: Define Activation & Loss Functions
# ------------------------------------------------------
def sigmoid(x):
    """ Sigmoid Activation Function """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """ Derivative of Sigmoid (for Backpropagation) """
    return sigmoid(x) * (1 - sigmoid(x))

def binary_cross_entropy(y_true, y_pred):
    """ Binary Cross-Entropy Loss Function """
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# ------------------------------------------------------
# 📌 Step 4: Training Loop (Forward + Backpropagation)
# ------------------------------------------------------
"""
1. Forward Pass:
    h = sigmoid(W1 * X + b1)
    ŷ = sigmoid(W2 * h + b2)

2. Compute Loss:
    L = - Σ [ y * log(ŷ) + (1 - y) * log(1 - ŷ) ]

3. Backpropagation:
    Compute gradients using chain rule.

4. Gradient Descent:
    W_new = W - η * dL/dW
"""

epochs = 10000
learning_rate = 0.1
losses = []

for epoch in range(epochs):
    # Forward Pass
    Z1 = np.dot(X, W1) + b1  # Input → Hidden layer
    H = sigmoid(Z1)  # Activation in hidden layer
    Z2 = np.dot(H, W2) + b2  # Hidden → Output layer
    y_pred = sigmoid(Z2)  # Activation in output layer

    # Compute Loss
    loss = binary_cross_entropy(y, y_pred)
    losses.append(loss)

    # Backpropagation
    dL_dy = (y_pred - y)  # Error at output layer
    dL_dW2 = np.dot(H.T, dL_dy * sigmoid_derivative(Z2))  # Gradient for W2
    dL_db2 = np.sum(dL_dy * sigmoid_derivative(Z2), axis=0, keepdims=True)  # Gradient for b2

    dL_dH = np.dot(dL_dy * sigmoid_derivative(Z2), W2.T)  # Error at hidden layer
    dL_dW1 = np.dot(X.T, dL_dH * sigmoid_derivative(Z1))  # Gradient for W1
    dL_db1 = np.sum(dL_dH * sigmoid_derivative(Z1), axis=0, keepdims=True)  # Gradient for b1

    # Gradient Descent Updates
    W2 -= learning_rate * dL_dW2
    b2 -= learning_rate * dL_db2
    W1 -= learning_rate * dL_dW1
    b1 -= learning_rate * dL_db1

    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# ------------------------------------------------------
# 📌 Step 5: Visualize the Training Loss Curve
# ------------------------------------------------------
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()

# ------------------------------------------------------
# 📌 Step 6: Test the Model
# ------------------------------------------------------
"""
After training, we expect the model to classify XOR correctly.
Expected Output:
    [[0], [1], [1], [0]]
"""

# Final Forward Pass for Predictions
Z1 = np.dot(X, W1) + b1
H = sigmoid(Z1)
Z2 = np.dot(H, W2) + b2
y_pred_test = sigmoid(Z2)
y_pred_test = np.round(y_pred_test)  # Convert probabilities to 0 or 1

print("Predictions:")
print(y_pred_test)

# ------------------------------------------------------
# 📌 Experimentation & Extra Tasks
# ------------------------------------------------------
"""
Try modifying:
✅ Learning Rate (e.g., 0.01, 1.0)
✅ Use ReLU instead of Sigmoid (ReLU activation & derivative required)
✅ Increase Hidden Layer Neurons (e.g., 4 instead of 2)
✅ Experiment with different weight initializations
"""
