# bumblebee
# **Building Foundations for LLMs**

This **9-week study plan** covers deep learning fundamentals, sequence modeling, and Transformers, culminating in the implementation of a **small LLM** trained on Wikipedia data.

---

## **Phase 1: Foundations (Weeks 1–3)**
**Goal**: Understand deep learning basics, optimization, and regularization to prepare for sequence modeling.

### **Week 1: Introduction to Deep Learning**
- **Topics**:
  - Feedforward networks  
  - Gradient descent and backpropagation  
  - Activation functions (Sigmoid, ReLU, Tanh)  
- **Primary Resources**:
  - Goodfellow: Chapter 6 (Deep Feedforward Networks)  
  - D2L: Chapter 3 (Multilayer Perceptrons)  
- **Deliverables**:
  1. **Quiz**: Basics of feedforward networks and backpropagation  
  2. **Guided Project**: Implement a basic MLP to classify XOR  
  3. **Challenge**: Train an MLP for MNIST digit classification  

---

### **Week 2: Optimization Techniques**
- **Topics**:
  - Stochastic gradient descent (SGD)  
  - Advanced optimizers (Adam, RMSProp)  
  - Weight initialization and its impact  
- **Primary Resources**:
  - Goodfellow: Chapter 8 (Optimization)  
  - D2L: Chapter 6 (Optimization Algorithms)  
- **Deliverables**:
  1. **Quiz**: Differences between optimization algorithms  
  2. **Guided Project**: Visualize gradient descent with a simple loss surface  
  3. **Challenge**: Compare SGD, Adam, and RMSProp on CIFAR-10 classification  

---

### **Week 3: Regularization and Generalization**
- **Topics**:
  - Overfitting and underfitting  
  - Regularization techniques (dropout, weight decay)  
  - Bias-variance tradeoff  
- **Primary Resources**:
  - Goodfellow: Chapter 7 (Regularization)  
  - D2L: Chapter 4 (Underfitting and Overfitting)  
- **Deliverables**:
  1. **Quiz**: Identify overfitting and regularization techniques  
  2. **Guided Project**: Apply dropout to prevent overfitting in a neural network  
  3. **Challenge**: Implement L1/L2 regularization for a regression task  

---

## **Phase 2: Sequence Modeling and Transformers (Weeks 4–7)**
**Goal**: Understand sequence data processing and the foundations of Transformers.

### **Week 4: Sequence Modeling Basics**
- **Topics**:
  - Recurrent Neural Networks (RNNs)  
  - Vanishing gradients in RNNs  
  - Sequence-to-sequence models  
- **Primary Resources**:
  - Goodfellow: Chapter 10 (Sequence Modeling)  
  - D2L: Chapter 8 (Sequence Modeling with RNNs)  
- **Deliverables**:
  1. **Quiz**: Core concepts of RNNs and their challenges  
  2. **Guided Project**: Implement a basic RNN for text prediction  
  3. **Challenge**: Train a sequence-to-sequence model for character-level translation  

---

### **Week 5: Advanced RNNs (LSTMs and GRUs)**
- **Topics**:
  - Long Short-Term Memory (LSTM)  
  - Gated Recurrent Units (GRU)  
  - Practical applications in text data  
- **Primary Resources**:
  - Goodfellow: Chapter 10 (Advanced RNNs)  
  - D2L: Chapter 9 (Modern RNNs)  
- **Deliverables**:
  1. **Quiz**: Differences between LSTMs and GRUs  
  2. **Guided Project**: Build an LSTM for sentence sentiment analysis  
  3. **Challenge**: Train a GRU for time-series forecasting  

---

### **Week 6: Attention Mechanisms**
- **Topics**:
  - Self-attention and scaled dot-product attention  
  - Key-query-value (KQV) mechanism  
  - Positional encodings  
- **Primary Resources**:
  - D2L: Chapter 10 (Attention Mechanisms)  
  - ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al.)  
- **Deliverables**:
  1. **Quiz**: Components of attention mechanisms  
  2. **Guided Project**: Implement scaled dot-product attention  
  3. **Challenge**: Build a basic attention-based encoder-decoder for translation  

---

### **Week 7: Transformers**
- **Topics**:
  - Transformer architecture  
  - Multi-head self-attention  
  - Feedforward layers in Transformers  
- **Primary Resources**:
  - D2L: Chapter 11 (Transformers)  
  - ["Attention is All You Need"](https://arxiv.org/abs/1706.03762)  
- **Deliverables**:
  1. **Quiz**: Key components of Transformer architecture  
  2. **Guided Project**: Implement a simplified Transformer block  
  3. **Challenge**: Build a Transformer for text classification  

---

## **Phase 3: LLMs (Weeks 8–9)**
**Goal**: Explore pretraining, fine-tuning, and build a small LLM.

### **Week 8: Pretraining and Fine-Tuning**
- **Topics**:
  - Masked language modeling (MLM) and causal LM  
  - Pretraining objectives in BERT and GPT  
  - Transfer learning  
- **Primary Resources**:
  - D2L: Chapter 12 (Pretraining and Fine-Tuning)  
  - ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"](https://arxiv.org/abs/1810.04805)  
  - ["Language Models are Few-Shot Learners (GPT-3)"](https://arxiv.org/abs/2005.14165)  
- **Deliverables**:
  1. **Quiz**: Pretraining vs. fine-tuning  
  2. **Guided Project**: Fine-tune a pretrained BERT model for text classification  
  3. **Challenge**: Implement MLM and train on a subset of Wikipedia  

---

### **Week 9: Scaling and Building an LLM**
- **Topics**:
  - Scaling models (parameters, data, compute)  
  - Building an LLM for text generation and retrieval  
- **Primary Resources**:
  - D2L and Hugging Face tutorials  
  - ["Scaling Laws for Neural Language Models"](https://arxiv.org/abs/2001.08361)  
- **Deliverables**:
  1. **Quiz**: Key considerations for scaling LLMs  
  2. **Guided Project**: Fine-tune GPT-2 for conversational tasks  
  3. **Challenge**: Build an end-to-end pipeline for a small LLM using Wikipedia data  

---

### **Next Steps**
- Join the **Bumblebee Discussions** (Zoom calls, GitHub issues, forums).  
- Complete **weekly projects and challenges** to reinforce learning.  
- Contribute to the **final LLM implementation** by Week 9.  
