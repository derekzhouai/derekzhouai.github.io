---
title:  "Linear Regression Implementation from Scratch (Pytorch)"
date:   2025-08-10 11:00:00 +0800
categories: [supervised learning, deep learning]
tags: [linear regression, pytorch]
math: true
---

In this post, we will implement **Linear Regression** from scratch using Pytorch. This will help us understand the underlying mechanics of this algorithm and how it can be applied to a synthetic dataset.

## Implementation

### 1. Importing libraries

```python
import torch
import random
```

### 2. Initializing parameters

Let's create a function to initialize the model parameters (weights and bias) that we will optimize during training.

```python
def init_params(num_features: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize the parameters for linear regression.
    Args:
        num_features (int): The number of features (input dimensions).
    Returns:
        tuple[torch.Tensor, torch.Tensor]: The initialized Weight tensor w (num_features, 1) and bias tensor b (1,).
    """
    w = torch.normal(0, 0.01, size=(num_features, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return w, b
```

### 3. Defining the model

Let's define the linear regression model function based on the equation $(y = wx + b)$.

```python
def linear_regression(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Perform linear regression prediction.
    Args:
        X (torch.Tensor): Input features of shape (num_samples, num_features).
        w (torch.Tensor): Weight tensor of shape (num_features, 1).
        b (torch.Tensor): Bias tensor of shape (1,).
    Returns:
        torch.Tensor: Predicted output of shape (num_samples, 1).
    """
    return torch.matmul(X, w) + b
```

### 4. Defining the loss function

Create a loss function to measure the difference between predicted and true values. We'll use **Mean Squared Error (MSE)** as our loss function.


```python
def squared_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the squared mean loss between predicted and true values.
    Args:
        y_hat (torch.Tensor): Predicted output of shape (num_samples, 1).
        y (torch.Tensor): True output of shape (num_samples, 1).
    Returns:
        torch.Tensor: Squared loss value.
    """
    return ((y_hat - y) ** 2).mean() / 2
```

### 5. Defining the optimizer

We create a function to update (optimize) parameters using gradient descent.

```python
def updater(w: torch.Tensor, b: torch.Tensor, lr: float) -> None:
    """
    Update model parameters using gradient descent.
    Args:
        w (torch.Tensor): Weight tensor of shape (num_features, 1).
        b (torch.Tensor): Bias tensor of shape (1,).
        lr (float): Learning rate.
    """
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_() # Reset gradients
        b.grad.zero_() # Reset gradients
```

### 6. Creating mini-batch stochastic gradient descent (SGD) iterator

We will use stochastic gradient descent (SGD) for optimization. So we need to create a data iterator that yields mini-batches of data.

```python
def data_iter(batch_size: int, features: torch.Tensor, labels: torch.Tensor):
    """
    Generate mini-batches of data.
    Args:
        batch_size (int): Number of samples per batch.
        features (torch.Tensor): Input features.
        labels (torch.Tensor): Corresponding labels.
    Yields:
        A mini-batch of features (batch_size, num_features) and labels (batch_size, 1).
    """
    num_samples = features.shape[0]
    indices = list(range(num_samples))
    random.shuffle(indices)  # Shuffle indices
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:min(i + batch_size, num_samples)]
        yield features[batch_indices], labels[batch_indices]
```

### 7. Creating the training function

Since we already defined the model, loss function, and optimizer, we can create the training function to train the model.

```python
def train(
    features: torch.Tensor, labels: torch.Tensor, # Training data
    w: torch.Tensor, b: torch.Tensor,             # Parameters
    num_epochs: int, batch_size: int, lr: float   # Hyperparameters
) -> None:
    """
    Train the linear regression model.
    Args:
        features (torch.Tensor): Input features with shape (num_samples, num_features).
        labels (torch.Tensor): Corresponding labels with shape (num_samples, 1).
        w (torch.Tensor): Weights with shape (num_features, 1).
        b (torch.Tensor): Bias with shape (1,).
        num_epochs (int): Number of training epochs.
        batch_size (int): Number of samples per batch.
        lr (float): Learning rate.
    """
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            # Forward pass
            y_hat = linear_regression(X, w, b) # Compute predictions
            l = squared_loss(y_hat, y) # Calculate the loss

            # Backward pass
            l.backward() # Compute gradients
            sgd(w, b, lr) # Update parameters

        print(f'Epoch {epoch + 1}, Loss: {l.item():.5f}')
```

## Test Implementation


### 1. Generating synthetic data

Let's create a synthetic dataset based on known parameters $(w, b)$ to simulate a linear relationship between features and target variable.

```python
def synthetic_data(w, b, num_samples) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates synthetic data based on linear equation with noise.
    y = Xw + b + noise
    Args:
        w (torch.Tensor): Weight tensor of shape (num_features, 1).
        b (torch.Tensor): Bias tensor of shape (1,).
    Returns:
        tuple: A tuple containing the feature tensor X (num_samples, num_features) and the label tensor y (num_samples, 1).
    """
    num_features = w.shape[0]
    X = torch.normal(0, 1, (num_samples, num_features))
    y = torch.matmul(X, w) + b 
    y += torch.normal(0, 0.01, y.shape)  # Adding noise
    return X, y
```

```python
true_w = torch.tensor([2.8, -1.7, 3.6]).reshape(-1, 1)  # True weights: (num_features, 1)
true_b = torch.tensor([3.0])
num_samples = 1000  # Number of samples
features, labels = synthetic_data(true_w, true_b, num_samples)
```

### 2. Training the model

Now we can train our linear regression model using the synthetic data we generated. We'll iterate through the data in mini-batches, compute predictions, calculate loss, and update the parameters using SGD.

```python
num_epochs = 3   # Number of epochs
batch_size = 10  # Number of samples per batch
lr = 0.03        # Learning rate
w, b = init_params(features.shape[1])

train(features, labels, w, b, num_epochs, batch_size, lr)

# Epoch 1, Loss: 0.02504
# Epoch 2, Loss: 0.00011
# Epoch 3, Loss: 0.00004
```

> As the parameters (w, b) were initialized randomly, the loss may vary between different runs.
{: .prompt-info }

## Summary

In this implementation, we have built a linear regression model from scratch using PyTorch. We defined the model, loss function, and optimization algorithm, and trained the model with synthetic data using mini-batch stochastic gradient descent. The training process involved iterating over the data, computing predictions, calculating loss, and updating the model parameters. After training for a few epochs, we observed a significant reduction in loss, indicating that the model was learning to fit the data.

## References

- [Linear Neural Networks for Regression](https://d2l.ai/chapter_linear-regression/index.html)