"""
Week 1 - Guided Project: Implementing a Basic MLP to Classify XOR
---------------------------------------------------------------
This script implements a simple Multi-Layer Perceptron (MLP) using PyTorch to solve the XOR classification problem.

Concepts Covered:
✅ Feedforward Networks
✅ Activation Functions (Sigmoid, ReLU)
✅ Backpropagation & Gradient Descent
✅ Binary Cross-Entropy Loss
✅ Stochastic Gradient Descent (SGD)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------
# 📌 Step 1: Define XOR Dataset
# ------------------------------------------------------
"""
XOR Dataset:
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [[0], [1], [1], [0]]

The XOR function is not linearly separable, so a single-layer perceptron cannot solve it.
We need a Multi-Layer Perceptron (MLP) with a hidden layer.
"""
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)  # Inputs
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)  # Labels

torch.manual_seed(42)  # Ensure reproducibility

# ------------------------------------------------------
# 📌 Step 2: Define the MLP Model
# ------------------------------------------------------
"""
Mathematical Representation:

Hidden Layer:
    h = σ(W1 * X + b1)
Output Layer:
    ŷ = σ(W2 * h + b2)

Where:
- W1, W2 = weight matrices
- b1, b2 = biases
- σ (sigma) = Activation function (Sigmoid)

We use two hidden neurons to introduce non-linearity.
"""

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(2, 2)  # 2-input → 2-hidden neurons
        self.output = nn.Linear(2, 1)  # 2-hidden → 1-output neuron

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))  # Sigmoid activation in hidden layer
        x = torch.sigmoid(self.output(x))  # Sigmoid activation in output layer
        return x

# Initialize model
model = MLP()

# ------------------------------------------------------
# 📌 Step 3: Define Loss Function & Optimizer
# ------------------------------------------------------
"""
Loss Function: Binary Cross-Entropy (BCE)
    L = - Σ [ y * log(ŷ) + (1 - y) * log(1 - ŷ) ]

Optimizer: Stochastic Gradient Descent (SGD)
    W_new = W - η * dL/dW

Where:
- L = Loss
- η = Learning Rate (step size)
- dL/dW = Gradient of Loss w.r.t Weights
"""
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.1)  # Learning rate = 0.1

# ------------------------------------------------------
# 📌 Step 4: Train the Model
# ------------------------------------------------------
"""
Training Loop:
1. Perform forward pass: Compute predictions
2. Compute loss using BCE
3. Perform backpropagation: Compute gradients using chain rule
4. Update weights using SGD
5. Repeat for multiple epochs
"""
epochs = 10000
losses = []

for epoch in range(epochs):
    # Forward pass
    y_pred = model(X)
    
    # Compute loss
    loss = criterion(y_pred, y)
    
    # Backpropagation
    optimizer.zero_grad()  # Reset gradients
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights
    
    # Store loss for visualization
    losses.append(loss.item())

    # Print every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ------------------------------------------------------
# 📌 Step 5: Visualize the Training Process
# ------------------------------------------------------
"""
The loss function should decrease over epochs, indicating learning.
"""
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()

# ------------------------------------------------------
# 📌 Step 6: Test the Model
# ------------------------------------------------------
"""
After training, we expect the model to correctly classify XOR:
Expected Output:
    [[0], [1], [1], [0]]
"""
y_pred_test = model(X).detach().numpy()
y_pred_test = np.round(y_pred_test)

print("Predictions:")
print(y_pred_test)

# ------------------------------------------------------
# 📌 Experimentation & Extra Tasks
# ------------------------------------------------------
"""
Try modifying:
✅ Learning Rate (e.g., 0.01, 1.0)
✅ Use ReLU instead of Sigmoid
✅ Increase Hidden Layer Neurons (e.g., 4 instead of 2)
✅ Change Optimizer to Adam (optim.Adam)
"""
