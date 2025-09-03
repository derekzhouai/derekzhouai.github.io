---
title:  "Linear Regression Concise Implementation (Pytorch)"
date:   2025-08-12 16:00:00 +0800
categories: [Linear Models]
tags: [linear regression, pytorch]
math: true
---

In the [previous post](/posts/linear-regression-implementation-scratch/), we implement linear regression from scratch by defining the model, loss function, and optimizer by ourself. Actually, we can leverage Pytorch's built-in functionalities to achieve the same goal more concisely.

## Implementation

### Importing the libraries

```python
import torch
from torch import nn
from torch.utils import data
```

### Generating synthetic data

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

```python
batch_size = 10
dataset = data.TensorDataset(features, labels)
data_iter = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
```

### Training the model

```python
# 1. Defining the model
num_features = features.shape[1]
net = nn.Linear(num_features, 1)

# 2. Initializing the parameters
net.weight.data.normal_(0, 0.01)
net.bias.data.fill_(0)

# 3. Defining the loss function
loss = nn.MSELoss()

# 4. Defining the optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=0.03)

# 5. Training the model
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        # Forward pass
        y_hat = net(X)
        l = loss(y_hat, y)

        # Backward pass
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {l.item():.5f}')

# Epoch 1, Loss: 0.00060
# Epoch 2, Loss: 0.00012
# Epoch 3, Loss: 0.00007
```

## Summary

After using Pytorch built-in functionalities, we were able to implement linear regression more concisely and efficiently. The high-level APIs provided by Pytorch allowed us to focus on the model architecture and training loop without worrying about the low-level details of tensor operations and gradient calculations. However, understanding these low-level details is still important for debugging and optimizing models.

## References

- [Linear Neural Networks for Regression](https://d2l.ai/chapter_linear-regression/index.html)