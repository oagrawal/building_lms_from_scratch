# From-Scratch Neural Networks: A Deep Learning Portfolio

This repository showcases a portfolio of neural network implementations built from scratch, demonstrating a deep understanding of fundamental machine learning concepts and advanced architectures. The projects follow Andrej Karpathy's "Neural Networks: Zero to Hero" course, but go beyond simple replication by focusing on the underlying mathematics and engineering principles.

> Each of the following jupyter notebooks has an associated blog post I wrote at: **[Om's Blog](https://omagrawal.tech/blog.html)**

## 🎯 Key Skills Demonstrated

- **Deep Learning Fundamentals**: Solid understanding of core concepts like backpropagation, gradient descent, activation functions, loss functions, and regularization techniques.
- **Neural Network Architectures**: Experience in implementing and training various architectures including Multilayer Perceptrons (MLPs) and Convolutional Neural Networks (CNNs).
- **From-Scratch Implementation**: Proven ability to build complex systems from first principles, demonstrating a deep understanding of the underlying mechanics (e.g., automatic differentiation).
- **PyTorch**: Proficient in using PyTorch for building, training, and debugging deep learning models.
- **ML Engineering Practices**: Experience with hyperparameter tuning, model evaluation, and diagnosing training issues (e.g., vanishing/exploding gradients, internal covariate shift).
- **Mathematical Foundations**: Strong grasp of the calculus, linear algebra, and probability theory that form the bedrock of machine learning.

## 🛠️ Core Projects & Implementations

### **1. Micrograd: A from-scratch automatic differentiation engine**

- **File**: `micrograd.ipynb`
- **Description**: Implemented a lightweight automatic differentiation engine in Python. This project involved creating a `Value` object to build and represent mathematical expressions as a Directed Acyclic Graph (DAG). Backpropagation is performed by traversing the graph and applying the chain rule recursively to compute gradients for every parameter.
- **Technical Skills**: Automatic Differentiation (Autograd), Backpropagation, Chain Rule, Directed Acyclic Graphs (DAGs), Gradient-based optimization, Python.

### **2. Bigram Character-level Language Model**

- **File**: `makemore/Lecture_2.ipynb`
- **Description**: Developed a simple language model using PyTorch to predict the next character in a sequence based on the previous one. This involved creating a lookup table of bigram probabilities from a large text corpus, and then sampling from the probability distribution to generate new text. The model is trained to minimize the negative log-likelihood loss.
- **Technical Skills**: Language Modeling, N-grams, PyTorch, Tensors, Negative Log-Likelihood Loss, Model Training, Text Generation, Sampling.

### **3. Multilayer Perceptron (MLP) for Character-level Language Modeling**

- **File**: `makemore/Lecture 3.ipynb`
- **Description**: Built a deep neural network to predict the next character based on a sequence of preceding characters. This project covers the fundamentals of deep learning, including the implementation of an MLP with embedding layers, non-linear activation functions (e.g., Tanh), and a softmax output layer. It also involved practical aspects of training like hyperparameter tuning (learning rate, embedding size, context length), and proper model evaluation using train/dev/test splits to prevent overfitting.
- **Technical Skills**: Multilayer Perceptron (MLP), Deep Learning, Embedding Layers, Activation Functions, Softmax, Cross-Entropy Loss, Hyperparameter Tuning, Overfitting, Model Evaluation.

### **4. Batch Normalization Implementation and Analysis**

- **Files**: `makemore/Lecture_4.ipynb` | `makemore/Lecture_4.html`
- **Description**: Implemented the Batch Normalization layer from scratch and integrated it into the MLP. This project involved a deep dive into the internal covariate shift problem and how BatchNorm helps to mitigate it. I analyzed the effects of BatchNorm on activation statistics and gradient flow, and used diagnostic tools to visualize the health of the network during training.
- **Technical Skills**: Batch Normalization, Internal Covariate Shift, Gradient Flow Analysis, Activation Statistics, Deep Network Training Stability, PyTorch Hooks.

### **5. Manual Backpropagation for an MLP**

- **File**: `makemore/Lecture_5.ipynb`
- **Description**: Manually derived and implemented the backpropagation algorithm for the entire MLP, including the cross-entropy loss, linear layers, Tanh activation, and Batch Normalization layers. This exercise solidified my understanding of how gradients are calculated and propagated through a complex compute graph at the tensor level, without relying on PyTorch's autograd.
- **Technical Skills**: Backpropagation, Gradient Descent, Chain Rule, Tensor Calculus, PyTorch Internals.

### **6. WaveNet-style Hierarchical Convolutional Network**

- **File**: `makemore/Lecture_6.ipynb`
- **Description**: Implemented a deep convolutional neural network for sequence modeling, inspired by the WaveNet architecture. This project involved building a hierarchical structure of convolutional layers to capture long-range dependencies in the input sequence. While the original WaveNet uses causal dilated convolutions for efficiency, this implementation focuses on the hierarchical aspect of the architecture.
- **Technical Skills**: Convolutional Neural Networks (CNNs), Sequence Modeling, Hierarchical Architectures, Receptive Field, PyTorch `torch.nn` module.

## 📖 Blog Series

Each implementation is accompanied by detailed blog posts that dive deeper into the mathematical concepts, implementation challenges, and insights gained. These posts serve as comprehensive guides for understanding both the theory and practice of neural network development.

- **Micrograd**: [Autograd Deep Dive](https://omagrawal.tech/blog/AutogradDeepDive.html)
- **Bigram LM**: [Bigram Language Model](https://omagrawal.tech/blog/BigramLM.html)
- **MLP**: [MLP Language Model](https://omagrawal.tech/blog/MLPLanguageModel.html)
- **BatchNorm**: [Batch Normalization](https://omagrawal.tech/blog/BatchNorm.html)
- **Manual Backprop**: [Manual Backpropagation](https://omagrawal.tech/blog/ManualBackProp.html)
- **WaveNet**: [WaveNet Implementation](https://omagrawal.tech/blog/WaveNet.html)

## 📋 Prerequisites

- **Programming**: Solid Python knowledge
- **Mathematics**: High school calculus (derivatives, chain rule)
- **Optional**: Basic linear algebra and statistics

## 🚀 Getting Started

1. Start with `micrograd.ipynb` to understand automatic differentiation
2. Progress through the `makemore/` lectures in numerical order
3. Read the accompanying blog posts for deeper insights
4. Experiment with the code and try your own modifications
