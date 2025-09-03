---
title:  "Softmax Regression Implementation from Scratch (Pytorch)"
date:   2025-08-13 18:00:00 +0800
categories: [Linear Models]
tags: [softmax regression, pytorch]
math: true
---

In this post, we will implement **Softmax Regression** from scratch using Pytorch. This will help us understand the underlying mechanics of this algorithm and how it can be applied to multi-class classification problems.

## Implementation

### 1. Importing libraries

```python
import torch
import torchvision
from torchvision import transforms
from torch.utils import data
```

### 2. Initializing parameters

Let's create a function to initialize the model parameters (weights and bias) that we will optimize during training.

```python
def init_params(num_inputs: int, num_outputs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize the parameters for the softmax regression model.
    Args:
        num_inputs (int): Number of input features.
        num_outputs (int): Number of output classes.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: Initialized weight W (num_inputs, num_outputs) and bias tensor b (num_outputs).
    """
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)
    return W, b
```

### 3. Defining the model

Let's define the softmax regression model function based on the equation $y = softmax(Wx + b)$.

$$
\begin{align*}
\text{softmax}(\mathbf{X})_{ij} &= \frac{e^{X_{ij}}}{\sum_{k} e^{X_{ik}}}
\end{align*}
$$

```python
def softmax(X: torch.Tensor) -> torch.Tensor:
    """
    Compute the softmax for each row of the input tensor.
    Args:
        X: Input tensor of shape (num_samples, num_classes)
    Returns:
        Output tensor of the same shape as X, containing the softmax probabilities.
    """
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition
```

```python
def softmax_regression(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute the softmax regression output.
    Args:
        X: Input tensor of shape (num_samples, num_inputs)
        W: Weight tensor of shape (num_inputs, num_outputs)
        b: Bias tensor of shape (num_outputs)
    Returns:
        Output tensor of shape (num_samples, num_outputs) containing the softmax probabilities.
    """
    num_inputs = W.shape[0]
    logits = torch.matmul(X.reshape(-1, num_inputs), W) + b
    return softmax(logits)
```

### 4. Defining the loss function

Create a loss function to measure the difference between predicted and true values. We'll use **Cross-Entropy Loss** as our loss function.

$$
L(y, \hat{y}) = -\sum_{i} y_i \log(\hat{y}_i)
$$

```python
def cross_entropy_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the cross-entropy loss.
    Args:
        y_hat: Predicted probabilities of shape (num_samples, num_classes)
        y: True labels of shape (num_samples,)
    Returns:
        Scalar tensor representing the cross-entropy loss.
    """
    return -torch.mean(torch.log(y_hat[range(len(y_hat)), y]))
```

### 5. Defining the optimizer

We create a function to update (optimize) parameters using gradient descent.

```python
def updater(W: torch.Tensor, b: torch.Tensor, lr: float) -> None:
    """
    Update model parameters using gradient descent.
    Args:
        W (torch.Tensor): Weight tensor of shape (num_features, num_classes).
        b (torch.Tensor): Bias tensor of shape (num_classes,).
        lr (float): Learning rate.
    """
    with torch.no_grad():
        W -= lr * W.grad
        b -= lr * b.grad
        W.grad.zero_() # Reset gradients
        b.grad.zero_() # Reset gradients
```

### 6. Creating mini-batch stochastic gradient descent (SGD) iterator

We will use stochastic gradient descent (SGD) for optimization. So we need to create a data iterator that yields mini-batches of data. Here we use Fashion MNIST dataset as an example.

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

### 7. Evaluating the accuracy

We create `accuracy` function to compute the total number of correct predictions and `evaluate_accuracy` function to evaluate the accuracy of the model with specific parameters on the given data.

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
```

```python
def evaluate_accuracy(data_iter: data.DataLoader, W: torch.Tensor, b: torch.Tensor) -> float:
    """
    Evaluate the accuracy of the model with specific parameters on the given data.
    Args:
        data_iter (data.DataLoader): DataLoader for the dataset.
        W (torch.Tensor): Weight tensor.
        b (torch.Tensor): Bias tensor.
    Returns:
        float: Accuracy of the model on the dataset.
    """
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        y_hat = softmax_regression(X, W, b)
        acc_sum += accuracy(y_hat, y)
        n += len(y)
    return acc_sum / n
```

### 8. Creating the training function

Since we already defined the model, loss function, and optimizer, we can create the training function to train the model.

```python
def train(
    train_iter: data.dataloader.DataLoader,        # Train data loader
    test_iter: data.dataloader.DataLoader,         # Test data loader
    W: torch.Tensor, b: torch.Tensor,              # Parameters
    num_epochs: int, batch_size: int, lr: float    # Hyperparameters
) -> None:
    """
    Train the softmax regression model.
    Args:
        train_iter: DataLoader for the training dataset.
        test_iter: DataLoader for the test dataset.
        W: Weight tensor with shape (num_inputs, num_outputs)
        b: Bias tensor with shape (num_outputs,)
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
    """
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # Forward pass
            y_hat = softmax_regression(X, W, b)
            l = cross_entropy_loss(y_hat, y)

            # Backward pass
            l.backward()
            updater(W, b, lr)

        # Evaluate on test set
        eval_acc = evaluate_accuracy(test_iter, W, b)

        print(f'Epoch {epoch + 1}, Loss: {l.item():.4f}, Test Accuracy: {eval_acc:.4f}')
```

## Test Implementation

### Training the model

Now we can train our softmax regression model using the Fashion MNIST dataset. We'll iterate through the data in mini-batches, compute predictions, calculate loss, and update the parameters using SGD.

```python
num_inputs, num_outputs = 28 * 28, 10       # Number of input features and output classes

num_epochs = 10                             # Number of epochs
batch_size = 256                            # Number of samples per batch
lr = 0.1                                    # Learning rate
W, b = init_params(num_inputs, num_outputs) # Parameters

# Load the data
train_iter, test_iter = load_data_fashion_mnist(batch_size)

# Train the model
train(train_iter, test_iter, W, b, num_epochs, batch_size, lr)

# Epoch 1, Loss: 0.5702, Test Accuracy: 0.7934
# Epoch 2, Loss: 0.5597, Test Accuracy: 0.7986
# Epoch 3, Loss: 0.4802, Test Accuracy: 0.8180
# Epoch 4, Loss: 0.4342, Test Accuracy: 0.8152
# Epoch 5, Loss: 0.4613, Test Accuracy: 0.8299
# Epoch 6, Loss: 0.5082, Test Accuracy: 0.8279
# Epoch 7, Loss: 0.5834, Test Accuracy: 0.8253
# Epoch 8, Loss: 0.4803, Test Accuracy: 0.8306
# Epoch 9, Loss: 0.4344, Test Accuracy: 0.8318
# Epoch 10, Loss: 0.4029, Test Accuracy: 0.8310
```

> As the parameters (w, b) were initialized randomly, the loss and accuracy may vary between different runs.
{: .prompt-info }

## Summary

In this implementation, we have built a softmax regression model from scratch using PyTorch. We defined the model, loss function, and optimization algorithm, and trained the model with the Fashion MNIST dataset using mini-batch stochastic gradient descent. The training process involved iterating over the data, computing predictions, calculating loss, and updating the model parameters. After training for a few epochs, we observed a significant reduction in loss, indicating that the model was learning to fit the data.

## References

- [Linear Neural Networks for Classification](https://d2l.ai/chapter_linear-classification/index.html)