---
title:  "Understanding Backpropagation in Neural Networks"
date:   2025-08-19 20:00:00 +0800
categories: [supervised learning, deep learning]
tags: [gradient descent]
math: true
---

## Introduction

Neural networks are at the heart of modern machine learning. They power image recognition, natural language processing, recommender systems, and large language models. But have you ever wondered: **how does a neural network actually learn?**

The answer lies in an algorithm called **Backpropagation** (short for "backward propagation of errors"). Backpropagation is the engine that allows neural networks to adjust their internal parameters (weights and biases) so that predictions become more accurate over time.

In this post, we'll break down backpropagation from intuition to mathematics, walk through its algorithm, and highlight practical issues and applications.

## Neural Network Basics

Before diving into backpropagation, let's quickly review how a neural network works.
- **Neuron model**: Each neuron computes $z = W \cdot x + b, a = g(z)$
    - $x$ is the input
    - $W$ is the weight vector
    - $b$ is the bias term
    - $g$ is the activation function
- **Forward propagation**: Input flows layer by layer through the network, producing the final output $\hat{y}$.
- **Loss function**: Measures the difference between the predicted output $\hat{y}$ and the true target $y$. For example, mean squared error (MSE): $L = \frac{1}{2}(y - \hat{y})^2$

This sets the stage for backpropagation: we need to know how much each weight contributed to the error, so we can update it.

## Intuition of Backpropagation

Think of learning as a feedback loop:
- A **student** solves a problem (forward propagation).
- The **teacher** checks the answer and points out errors (loss calculation).
- The **student** traces back by step to see where the mistakes came from (backpropagation).
- Finally, the student **adjusts their strategy** (gradient descent update).

In short: **forward to predict, backward to learn.**

## Mathematical Foundation

### The Chain Rule

Backpropagation is essentially an application of the chain rule from calculus:

$$\frac{dL}{dx} = \frac{dL}{dy} \cdot \frac{dy}{dx}$$

This allows us to "pass back" the error derivatives through the network layer by layer.

### Single-Layer Networks

![](/assets/img/posts/single_layer_network.png){: width="500" .normal}

Suppose:

$$z = w \cdot x + b, \quad a = g(z), \quad L = \text{Loss}(a, y)$$

Then derivatives of loss with respect to the parameters:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$

$$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial b}$$

### Multi-Layer Networks

![](/assets/img/posts/two_layers_neural_network.png){: width="800" .normal}

Here we used two layers neural network as an example to illustrate how it works for multi-layer networks.

For layer $1$, we have:

$$z[1] = W[1] \cdot x + b[1], \quad a[1] = g(z[1])$$

layer $2$, we have:

$$z[2] = W[2] \cdot a[1] + b[2], \quad a[2] = g(z[2])$$

and loss function:

$$L = \text{Loss}(a[2], y)$$

Then derivatives of the loss with respect to the parameters:

$$\frac{\partial L}{\partial W[2]} = \frac{\partial L}{\partial a[2]} \cdot \frac{\partial a[2]}{\partial z[2]} \cdot \frac{\partial z[2]}{\partial W[2]}$$

$$\frac{\partial L}{\partial b[2]} = \frac{\partial L}{\partial a[2]} \cdot \frac{\partial a[2]}{\partial z[2]} \cdot \frac{\partial z[2]}{\partial b[2]}$$

$$\frac{\partial L}{\partial W[1]} = \frac{\partial L}{\partial a[2]} \cdot \frac{\partial a[2]}{\partial z[2]} \cdot \frac{\partial z[2]}{\partial a[1]} \cdot \frac{\partial a[1]}{\partial z[1]} \cdot \frac{\partial z[1]}{\partial W[1]}$$

$$\frac{\partial L}{\partial b[1]} = \frac{\partial L}{\partial a[2]} \cdot \frac{\partial a[2]}{\partial z[2]} \cdot \frac{\partial z[2]}{\partial a[1]} \cdot \frac{\partial a[1]}{\partial z[1]} \cdot \frac{\partial z[1]}{\partial b[1]}$$

## Backpropagation Algorithm

The backpropagation algorithm consists of three major steps:

1. **Forward pass**
    - Compute activations for each layer
    - Compute the final loss
2. **Backward pass**
    - Start from the output layer and compute gradients using the chain rule
    - Propagate these gradients backward through the network layer by layer
3. **Parameter update**
    - Update each weight and bias using gradient descent:
    - $$W = W - \eta \cdot \frac{\partial L}{\partial W}$$
    - $$b = b - \eta \cdot \frac{\partial L}{\partial b}$$
    - where $\eta$ is the learning rate

Pseudocode for Backpropagation:

```python
# Forward pass
a = forward(x)
loss = compute_loss(a, y)

# Backward pass
grads = backward(loss)

# Parameter update
for each parameter in params:
    parameter -= learning_rate * grads[parameter]
```

## Common Challenges

Backpropagation works beautifully, but comes with pitfalls:
- **Vanishing gradients**: Sigmoid/tanh activations shrink gradients in deep networks.
- **Exploding gradients**: Large weights can cause unstable updates.
- **Learning rate issues**: Too high → oscillations; too low → painfully slow convergence.
- **Activation choice**: ReLU helps mitigate vanishing gradients.

## Conclusion

Backpropagation is the **mathematical backbone of deep learning**. By applying the chain rule to propagate signals backward, it enables neural networks to adjust their parameters and learn from data.