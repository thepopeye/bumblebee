# **Week 1: Introduction to Deep Learning**

---

## **Abstract**  

This week introduces the fundamental concepts of **deep learning**, focusing on **feedforward neural networks (FNNs)**, **backpropagation**, and **activation functions**. Participants will explore how neural networks function as universal function approximators and understand why multi-layer perceptrons (MLPs) are essential for solving non-linearly separable problems like XOR.

Key topics include the **mathematical foundation** of FNNs, weight initialization, and how **gradient descent** updates model parameters through **backpropagation**. The role of activation functions such as **Sigmoid, ReLU, and Tanh** will be studied, emphasizing their impact on model training, convergence speed, and performance.

The hands-on component consists of implementing an **MLP from scratch** and training it to classify XOR. The **challenge project** extends this by applying an MLP to the **MNIST digit classification task**, encouraging experimentation with network architectures and hyperparameters.

By the end of the week, participants will have a **solid foundation in neural networks**, understand the importance of **hidden layers**, and gain hands-on experience implementing backpropagation. This knowledge sets the stage for deeper exploration into **optimization and regularization in Week 2**.

---

# **Mathematical Foundations of Deep Learning**

## **1. Feedforward Neural Networks**  

### **Definition**  
A **feedforward neural network (FNN)** is a function mapping input **\( x \)** to output **\( y \)** through **hidden layers** with weights and biases.  

\[
h = \sigma(Wx + b)
\]

where:  
- **\( W \)** = weight matrix  
- **\( b \)** = bias  
- **\( \sigma \)** = activation function  

### **Multi-Layer Perceptron (MLP)**  
An MLP with a single hidden layer follows:

1. **Hidden Layer Transformation**  
   \[
   h = \sigma(W^{(1)}x + b^{(1)})
   \]
2. **Output Layer Transformation**  
   \[
   y = \sigma(W^{(2)}h + b^{(2)})
   \]

📌 **Reference:**  
- **Goodfellow:** Chapter 6 (Deep Feedforward Networks)  
- **D2L:** Chapter 3 (Multilayer Perceptrons)  

---

## **2. Activation Functions**  

### **Why Are They Needed?**  
- Without non-linearity, an MLP would collapse into a linear function.
- Activation functions **enable non-linearity**, making deep learning effective.

### **Common Activation Functions**  

#### **Sigmoid**
\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]
- **Pros:** Smooth, differentiable.  
- **Cons:** Saturates, causes vanishing gradients.

#### **ReLU (Rectified Linear Unit)**
\[
\text{ReLU}(x) = \max(0, x)
\]
- **Pros:** Efficient, avoids vanishing gradient.  
- **Cons:** Can cause “dead neurons” (output always 0).

#### **Tanh**
\[
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
\]
- **Pros:** Centered at zero (unlike sigmoid).  
- **Cons:** Still suffers from saturation.

📌 **Reference:**  
- **Goodfellow:** Chapter 6.3 (Hidden Units)  
- **D2L:** Chapter 3.1 (Activation Functions)  

---

## **3. Gradient Descent & Backpropagation**  

### **Gradient Descent**  
Used to minimize the loss function **\( L \)** by updating weights **\( W \)**:

\[
W_{\text{new}} = W - \eta \frac{\partial L}{\partial W}
\]

where **\( \eta \)** = learning rate.  

### **Backpropagation**  
Backpropagation **computes gradients efficiently** using the chain rule:

1. **Output Layer Gradient**
   \[
   \frac{\partial L}{\partial W^{(2)}} = \delta^{(2)} h^T
   \]

2. **Hidden Layer Gradient**
   \[
   \delta^{(1)} = (\delta^{(2)} W^{(2)}) \circ \sigma'(h)
   \]

where **\( \delta^{(l)} \)** represents the gradient at layer \( l \).

📌 **Reference:**  
- **Goodfellow:** Chapter 6.5 (Backpropagation)  
- **D2L:** Chapter 3.3 (Training a Neural Network)  

---

## **4. Loss Functions**  

### **Mean Squared Error (MSE)**
Used for regression tasks:

\[
L = \frac{1}{N} \sum (y_{\text{true}} - y_{\text{pred}})^2
\]

### **Cross-Entropy Loss**
Used for classification:

\[
L = - \sum y \log \hat{y}
\]

📌 **Reference:**  
- **Goodfellow:** Chapter 6.2 (Loss Functions)  
- **D2L:** Chapter 3.4 (Loss Functions)  

---

## **5. Weight Initialization**
Proper initialization prevents issues like vanishing/exploding gradients.

### **Xavier (Glorot) Initialization**  
\[
W \sim \mathcal{N}\left(0, \frac{1}{\text{fan-in}}\right)
\]

### **He Initialization (for ReLU)**
\[
W \sim \mathcal{N}\left(0, \frac{2}{\text{fan-in}}\right)
\]

📌 **Reference:**  
- **Goodfellow:** Chapter 8.4 (Parameter Initialization)  
- **D2L:** Chapter 3.2 (Parameter Initialization)  

---

## **6. Summary of Key Equations**
| Concept | Equation |
|---------|----------|
| Feedforward Layer | \( h = \sigma(Wx + b) \) |
| Backpropagation | \( W_{\text{new}} = W - \eta \frac{\partial L}{\partial W} \) |
| Sigmoid Activation | \( \sigma(x) = \frac{1}{1 + e^{-x}} \) |
| ReLU Activation | \( \max(0, x) \) |
| MSE Loss | \( L = \frac{1}{N} \sum (y - \hat{y})^2 \) |
| Cross-Entropy Loss | \( L = - \sum y \log \hat{y} \) |

---

📌 **Recommended Reading:**  
- **Goodfellow:** Chapters 6 & 8  
- **D2L:** Chapter 3  
