---
title:  "Explaining Backpropagation with Example"
date:   2025-08-25 20:00:00 +0800
categories: [supervised learning, deep learning]
tags: [gradient descent, backpropagation, MLP]
math: true
---

Backpropagation is the key algorithm that makes training deep neural networks possible.

In this post, I'll walk step by step through **a simple two-layer neural network** (one hidden layer with sigmoid activation and one linear output layer) to illustrate how the forward and backward passes work.

## Two-Layer Network

![](/assets/img/posts/20250825_network_layer.png){: width="400" .normal}

- **Input**: X $\in\mathbb{R}^{m\times 2}$ (mini-batch of m examples, 2 features each)
- **Hidden layer**: 3 neurons, sigmoid activation
- **Output layer**: 1 neuron, linear activation
- **Loss**: Mean Squared Error (MSE)

---

## Forward Pass

![](/assets/img/posts/20250825_computation_graph.png)

**Parameters:**
- $W_1 \in \mathbb{R}^{2 \times 3},\quad b_1 \in \mathbb{R}^{1  \times 3}$
- $W_2  \in \mathbb{R}^{3 \times 1},\quad b_2 \in \mathbb{R}^{1 \times 1}$

**Activations:**
- Hidden: $\sigma(z)=\dfrac{1}{1+e^{-z}}$ (applied element-wise)
- Output: linear (identity)

**Equations:**

$$
\begin{aligned}
Z_1 &= XW_1 + b_1 \quad &(m,2)(2,3)\to(m,3) \\
A_1 &= \sigma(Z_1) \quad &(m,3) \\
Z_2 &= A_1W_2 + b_2 \quad &(m,3)(3,1)\to(m,1) \\
A_2 &= Z_2 \quad &(m,1)
\end{aligned}
$$

**Loss (MSE):**

$
L = \frac{1}{2m} \parallel A_2 - y\parallel_F^2 = \frac{1}{2m}\sum_{i=1}^m (A_2^{(i,1)}-y^{(i,1)})^2 \in \mathbb{R}
$

---

## Backward Pass

We now compute the gradients layer by layer.

### Output Layer

$
L=\frac{1}{2m} \parallel A_2 - y \parallel_F^2, \quad\text{with } A_2 = Z_2.
$

$
Z_2 = A_1W_2 + b_2
$

**Output error term:**

$
\delta_2 = \frac{\partial L}{\partial Z_2} = \frac{\partial L}{\partial A_2} = \frac{1}{m}(A_2 - y) \in \mathbb{R}^{m \times 1}
$

**Output gradients:**

$
\nabla_{W_2} = \frac{\partial L}{\partial W_2} = A_1^\top\delta_2 \in \mathbb{R}^{3 \times 1}
$

$
\nabla_{b_2} = \frac{\partial L}{\partial b_2} = \text{rowsum}(\delta_2) \in \mathbb{R}^{1 \times 1}
$

---

### Hidden Layer

$
A_1 = \sigma(Z_1) \Rightarrow \frac{\partial A_1}{\partial Z_1} \;=\; \sigma(Z_1)\odot \big(1-\sigma(Z_1)\big) \;=\; A_1\odot (1-A_1).
$

$Z_1=XW_1+b_1$

**Hidden error term:**

$
\delta_1 = \frac{\partial L}{\partial Z_1} = \left(\frac{\partial L}{\partial A_1}\right)\odot \left(\frac{\partial A_1}{\partial Z_1}\right)
\;=\; \big(\delta_2 W_2^\top\big)\odot \big(A_1 \odot (1-A_1)\big) \in \mathbb{R}^{m \times 3}
$

**Hidden gradient:**

$
\nabla_{W_1} = X^\top\delta_1 \in \mathbb{R}^{2 \times 3}
$

$
\nabla_{b_1} = \text{rowsum}(\delta_1) \in \mathbb{R}^{1 \times 3}
$

---

## Summary

- **Forward:**

$Z_1=XW_1+b_1,\ A_1=\sigma(Z_1),\ Z_2=A_1W_2+b_2,\ A_2=Z_2$

- **Errors:**

$\delta_2=\tfrac{1}{m}(A_2-y)$

$\delta_1=(\delta_2 W_2^\top) \odot (A_1 \odot (1-A_1))$

- **Gradients:**

$\nabla_{W_2}=A_1^\top\delta_2,\ \nabla_{b_2}=\text{rowsum}(\delta_2)$

$\nabla_{W_1}=X^\top\delta_1,\ \nabla_{b_1}=\text{rowsum}(\delta_1)$

---

This is the complete **vectorized backpropagation** for a two-layer MLP with a sigmoid hidden layer and linear output under MSE loss. It captures the essence of backpropagation: **propagate errors backward and compute gradients.**

I also implemented this example in Python. If you are interested in the code, please refer to [here](https://github.com/derekzhouai/derekzhou-ai-blog-code/blob/main/explaining_backpropagation.ipynb).