---
title:  "Linear Regression Implementation from Scratch (Pytorch & Numpy)"
date:   2025-08-10 11:00:00 +0800
categories: [machine learning, deep learning]
tags: [linear regression, pytorch]
---

In this post, we will implement Linear Regression from scratch using Pytorch and Numpy. This will help us understand the underlying mechanics of the algorithm and how it can be applied to real-world datasets.

## 1. Importing Libraries

```python
import numpy as np
import random
import torch
```

## 2. Generating Synthetic Data

Let's create a synthetic dataset based on known parameters (w, b) to simulate a linear relationship between features and target variable.

```python
def synthetic_data(w, b, num_samples):
    """
    Generates synthetic data based on linear equation with noise.
    y = Xw + b + noise
    Args:
        w: Coefficients (weights), torch tensor with shape [n_features]
        b: Intercept (bias), a scalar value
        num_samples: Number of samples to generate
    """
    X = torch.normal(0, 1, (num_samples, len(w)))  # Random features
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)  # Adding noise
    return X, y.reshape(-1, 1)  # Reshape y to be a column vector
```

```python
true_w = torch.tensor([3.0, -1.2])  # True weights
true_b = 2.0  # True bias
num_samples = 1000  # Number of samples
features, labels = synthetic_data(true_w, true_b, num_samples)
```

## 3. Create data iterator

Let's create a function to generate mini-batches of data for training. We will use this to feed data into our model during training.

```python
def data_iter(batch_size, features, labels):
    """
    Generates mini-batches of data.
    Args:
        batch_size: Size of each mini-batch
        features: Input features, torch tensor with shape [num_samples, n_features]
        labels: Target labels, torch tensor with shape [num_samples, 1]
    Yields:
        A tuple of (features, labels) for each mini-batch
    """
    num_samples = len(features)
    indices = list(range(num_samples))
    random.shuffle(indices)  # Shuffle indices for randomness
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:min(i + batch_size, num_samples)]
        yield features[batch_indices], labels[batch_indices]
```

## 4. Initialize model parameters

Let's initialize the model parameters (weights and bias) that we will optimize during training.

```python
w = torch.normal(0, 0.01, size=(features.shape[1], 1), requires_grad=True)  # Weights
b = torch.zeros(1, requires_grad=True)  # Bias
```

## 5. Define the model

```python
def linear_regression(X, w, b):
    """
    Computes the linear regression model output.
    Args:
        X: Input features, torch tensor with shape [num_samples, n_features]
        w: Weights, torch tensor with shape [n_features, 1]
        b: Bias, a scalar value
    Returns:
        Output of the linear regression model, torch tensor with shape [num_samples, 1]
    """
    return torch.matmul(X, w) + b
```

## 6. Define the loss function

Create a loss function to measure the difference between predicted and true values. We'll use Mean Squared Error (MSE) as our loss function.

```python
def squared_loss(y_hat, y):
    """
    Computes the mean squared loss between predicted and true values.
    Args:
        y_hat: Predicted values, torch tensor with shape [num_samples, 1]
        y: True values, torch tensor with shape [num_samples, 1]
    Returns:
        Mean squared loss, a scalar tensor
    """
    num_samples = y_hat.shape[0]
    return ((y_hat - y) ** 2).mean() / 2
```

## 7. Define the optimization algorithm

We create a function for Stochastic Gradient Descent (SGD) to update the model parameters.

```python
def sgd(w, b, lr):
    """
    Performs Stochastic Gradient Descent (SGD) to update model parameters.
    Args:
        w: Weights, torch tensor with shape [n_features, 1]
        b: Bias, a scalar value
        lr: Learning rate, a scalar value
    """
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()  # Reset gradients
        b.grad.zero_()  # Reset gradients
```

## 8. Training the model

Now we can train our linear regression model using the synthetic data we generated. We'll iterate through the data in mini-batches, compute predictions, calculate loss, and update the parameters using SGD.

```python   
lr = 0.03  # Learning rate
num_epochs = 3  # Number of epochs
batch_size = 10

for epoch in range(num_epochs):
    for X_batch, y_batch in data_iter(batch_size, features, labels):
        # Forward pass
        y_hat = linear_regression(X_batch, w, b)
        loss = squared_loss(y_hat, y_batch)

        # Backward pass
        loss.backward()
        sgd(w, b, lr)

    print(f'Epoch {epoch + 1}, Loss: {loss.item():.5f}')

# Epoch 1, Loss: 0.05720
# Epoch 2, Loss: 0.00017
# Epoch 3, Loss: 0.00001
```

## Summary

In this implementation, we have built a linear regression model from scratch using PyTorch. We generated synthetic data, defined the model architecture, loss function, and optimization algorithm, and trained the model using mini-batch gradient descent. The training process involved iterating over the data, computing predictions, calculating loss, and updating the model parameters. After training for a few epochs, we observed a significant reduction in loss, indicating that the model was learning to fit the data.
