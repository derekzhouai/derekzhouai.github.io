---
title:  "Softmax Regression Concise Implementation (Pytorch)"
date:   2025-08-14 11:00:00 +0800
categories: [deep learning]
tags: [softmax regression, pytorch]
math: true
---

In the [previous post](/posts/softmax-regression-implementation-scratch/), we implement softmax regression from scratch by defining the model, loss function, and optimizer by ourself. Actually, we can leverage Pytorch's built-in functionalities to achieve the same goal more concisely.

## Implementation

### Importing the libraries


```python
import torch
import torchvision
from torchvision import transforms
from torch.utils import data
from torch import nn
```

### Generating data

Let's create a data loading function for the Fashion MNIST dataset.

```python
def load_data_fashion_mnist(batch_size: int) -> tuple[data.DataLoader, data.DataLoader]:
    """
    Load the Fashion MNIST dataset.
    Args:
        batch_size (int): The number of samples per batch.
    Returns:
        tuple[data.DataLoader, data.DataLoader]: The training and test data loaders.
    """
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=trans, download=True)
    train_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_iter = data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    return train_iter, test_iter
```

```python
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
```

### Defining the evaluation function

We create `accuracy` function to compute the total number of correct predictions and `evaluate_accuracy` function to evaluate the accuracy of the model on the given data.

```python
def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute the total number of correct predictions.
    Args:
        y_hat: Predicted probabilities of shape (num_samples, num_classes)
        y: True labels of shape (num_samples,)
    Returns:
        Scalar representing the number of correct predictions.
    """
    y_pred = y_hat.argmax(dim=1) # (num_samples,)
    return float((y_pred == y).sum())

def evaluate_accuracy(net: nn.Module, data_iter: data.DataLoader) -> float:
    """
    Evaluate the accuracy of the model on the given data.
    Args:
        data_iter (data.DataLoader): DataLoader for the dataset.
        net (nn.Module): The neural network model.
    Returns:
        float: Accuracy of the model on the dataset.
    """
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        y_hat = net(X)
        acc_sum += accuracy(y_hat, y)
        n += len(y)
    return acc_sum / n
```

### Training the model

```python
num_inputs, num_outputs = 28 * 28, 10

# 1. Defining the model
net = nn.Sequential(nn.Flatten(), nn.Linear(num_inputs, num_outputs))

# 2. Initializing the parameters
net[1].weight.data.normal_(mean=0.0, std=0.01)
net[1].bias.data.fill_(0)

# 3. Defining the loss
loss = nn.CrossEntropyLoss()

# 4. Defining the optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# 5. Training the model
num_epochs = 10
for epoch in range(num_epochs):
    for X, y in train_iter:
        # Forward pass
        y_hat = net(X)
        l = loss(y_hat, y)

        # Backward pass
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

    # Evaluate on test set
    test_acc = evaluate_accuracy(net, test_iter)

    print(f'Epoch {epoch + 1}, Loss: {l.item():.4f}, Test Accuracy: {test_acc:.4f}')

# Epoch 1, Loss: 0.5089, Test Accuracy: 0.7835
# Epoch 2, Loss: 0.7589, Test Accuracy: 0.8085
# Epoch 3, Loss: 0.4804, Test Accuracy: 0.8127
# Epoch 4, Loss: 0.3893, Test Accuracy: 0.8257
# Epoch 5, Loss: 0.6210, Test Accuracy: 0.8242
# Epoch 6, Loss: 0.6150, Test Accuracy: 0.8143
# Epoch 7, Loss: 0.4659, Test Accuracy: 0.8256
# Epoch 8, Loss: 0.3286, Test Accuracy: 0.8290
# Epoch 9, Loss: 0.4537, Test Accuracy: 0.8340
# Epoch 10, Loss: 0.5178, Test Accuracy: 0.8316
```

## Summary

After using Pytorch built-in functionalities, we were able to implement softmax regression more concisely and efficiently. The high-level APIs provided by Pytorch allowed us to focus on the model architecture and training loop without worrying about the low-level details of tensor operations and gradient calculations. However, understanding these low-level details is still important for debugging and optimizing models.

## References

- [Linear Neural Networks for Classification](https://d2l.ai/chapter_linear-classification/index.html)