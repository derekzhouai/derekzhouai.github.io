---
title:  "Backpropagation Step-by-Step"
date:   2025-08-25 20:00:00 +0800
categories: [supervised learning, deep learning]
tags: [gradient descent]
math: true
---

Let's use a two-layer neural network for a regression problem as an example to illustrate backpropagation step by step.

## Two-Layer Network

![](/assets/img/posts/20250825_network_layer.png){: width="400" .normal}

- **Input**: X $\in\mathbb{R}^{m\times 2}$ (mini-batch of m examples, 2 features each)
- **Hidden layer**: 3 neurons, sigmoid activation
- **Output layer**: 1 neuron, linear activation
- **Loss**: Mean Squared Error (MSE)

## Forward Pass

![](/assets/img/posts/20250825_computation_graph.png)

**Parameters**
- $W_1\in\mathbb{R}^{2\times 3},\quad b_1\in\mathbb{R}^{1\times 3}$
- $W_2\in\mathbb{R}^{3\times 1},\quad b_2\in\mathbb{R}^{1\times 1}$

**Activations**
- Hidden: $\sigma(z)=\dfrac{1}{1+e^{-z}}$ (applied element-wise)
- Output: linear (identity)

**Forward pass**

$$
\begin{aligned}
Z_1 &= XW_1 + b_1 \quad &(m,2)(2,3)\to(m,3) \\
A_1 &= \sigma(Z_1) \quad &(m,3) \\
Z_2 &= A_1W_2 + b_2 \quad &(m,3)(3,1)\to(m,1) \\
A_2 &= Z_2 \quad &(m,1)
\end{aligned}
$$

**Loss (MSE)**

$
\mathcal{L} \;=\; \frac{1}{2m}\,\|A_2 - y\|\_F^2
\;=\; \frac{1}{2m}\sum_{i=1}^m (A_2^{(i,1)}-y^{(i,1)})^2
$

## Backward Pass

### Output Layer

$
\mathcal{L}=\frac{1}{2m}\|A_2-y\|_F^2, \quad\text{with } A_2=Z_2.
$

$
Z_2=A_1W_2+b_2
$

**Output error**

$
\delta_2 = \frac{\partial L}{\partial A_2} = \frac{\partial L}{\partial Z_2} = \frac{1}{m}(A_2 - y) \quad (m,1)
$

**Output gradient**

$
\nabla_{W_2} = \frac{\partial L}{\partial W_2} = A_1^\top\delta_2 \quad (3,1)
$

$
\nabla_{b_2} = \frac{\partial L}{\partial b_2} = \text{rowsum}(\delta_2) \quad (1,1)
$

### Hidden Layer

$
A_1 = \sigma(Z_1) \Rightarrow \frac{\partial A_1}{\partial Z_1} \;=\; \sigma(Z_1)\odot \big(1-\sigma(Z_1)\big) \;=\; A_1\odot (1-A_1).
$

$Z_1=XW_1+b_1$

**Hidden error**

$
\delta_1 = \left(\frac{\partial \mathcal{L}}{\partial A_1}\right)\odot \left(\frac{\partial A_1}{\partial Z_1}\right)
\;=\; \big(\delta_2 W_2^\top\big)\odot \big(A_1\,(1-A_1)\big) \quad (m,3)
$

**Hidden gradient**

$
\nabla_{W_1} = X^\top\delta_1 \quad (2,3)
$

$
\nabla_{b_1} = \text{rowsum}(\delta_1) \quad (1,3)
$

## Summary

- **Forward:**

$Z_1=XW_1+b_1,\ A_1=\sigma(Z_1),\ Z_2=A_1W_2+b_2,\ A_2=Z_2$

- **Errors:**

$\delta_2=\tfrac{1}{m}(A_2-y)$

$\delta_1=(\delta_2 W_2^\top)\odot(A_1(1-A_1))$

- **Gradients:**

$\nabla_{W_2}=A_1^\top\delta_2,\ \nabla_{b_2}=\text{rowsum}(\delta_2)$

$\nabla_{W_1}=X^\top\delta_1,\ \nabla_{b_1}=\text{rowsum}(\delta_1)$

This is the complete, fully vectorized backpropagation for a two-layer MLP with a sigmoid hidden layer and linear output under MSE.