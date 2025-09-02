---
title:  "Building a Simple MLP for MNIST Classification"
date:   2025-08-27 10:00:00 +0800
categories: [Neural Networks on MNIST]
tags: [mlp, mnist]
math: true
---

## 1. Introduction

The **MNIST dataset** of handwritten digits is one of the most famous datasets in machine learning. Each sample is a grayscale image of size 28×28, labeled from 0 to 9. Although MNIST is simple compared to modern image datasets, it remains a great benchmark for testing neural network architectures.

In this post, we’ll start with a **Multi-Layer Perceptron (MLP)**, one of the simplest types of neural networks. This will serve as our baseline model for classifying digits. In future posts, we will explore more powerful architectures, such as Convolutional Neural Networks (CNNs), and compare their performance.

## 2. What is an MLP?

A Multi-Layer Perceptron (MLP) is a feedforward neural network consisting of:
- **Input layer**: receives the raw features.
- **Hidden layers**: one or more layers of fully connected neurons with activation functions (e.g., ReLU, sigmoid).
- **Output layer**: produces predictions.

For MNIST, the input size is **784** (28×28 pixels). An MLP treats each pixel as an independent input, ignoring spatial relationships—but it’s still powerful enough to reach ~97–98% accuracy.

## 3. Dataset Preparation

We’ll use **PyTorch** and torchvision to load MNIST.

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.ToTensor()

train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
```

```python
print(f"Number of training samples: {len(train_data)}")
print(f"Number of test samples: {len(test_data)}")

# Number of training samples: 60000
# Number of test samples: 10000
```

## 4. Model Architecture

Our MLP will use:
- One hidden layer with **256 neurons**.
- **ReLU activation** for non-linearity.
- An output layer with **10 units** (one for each digit: 0-9).

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, hidden=256, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),               # 28x28 -> 784
            nn.Linear(28 * 28, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.net(x)
```

**Number of parameters**:
- Hidden layer: 28 * 28 * 256 + 256 = 200,960
- Output layer: 256 * 10 + 10 = 2,570
- Total: 28 * 28 * 256 + 256 + 256 * 10 + 10 = 203,530

> **Note**: This model already has ~200k parameters, much larger than the dataset size, which makes regularization important to prevent overfitting.

## 5. Training Setup

We'll use **CrossEntropyLoss** for classification and the **Adam optimizer** for efficient training.

```python
import torch
import torch.optim as optim

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

model = MLP()
model.to(device)

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10
```

Hyperparameters:
- Batch size: 128
- Learning rate: 0.001
- Epochs: 10

## 6. Training the Model

Here's a simple training loop:

```python
def train(model, loader, optimizer, loss):
    model.train()
    total_loss = 0.0
    total_correct, total_num = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        y_hat = model(X)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()

        total_loss += l.item() * X.size(0)
        preds = y_hat.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_num += X.size(0)
    return total_loss / total_num, total_correct / total_num

def evaluate(model, loader, loss):
    model.eval()
    total_loss = 0.0
    total_correct, total_num = 0, 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            
            y_hat = model(X)
            l = loss(y_hat, y)

            total_loss += l.item() * X.size(0)
            preds = y_hat.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_num += X.size(0)
    return total_loss / total_num, total_correct / total_num

for epoch in range(epochs):
    train_loss, train_acc = train(model, train_loader, optimizer, loss)
    test_loss, test_acc = evaluate(model, test_loader, loss)
    print(f"Epoch {epoch+1}/{epochs}: "
          f"Train (loss: {train_loss:.4f}, acc: {train_acc:.4f}) | "
          f"Test (loss: {test_loss:.4f}, acc: {test_acc:.4f})")
```

```python
Epoch 1/10: Train (loss: 0.3568, acc: 0.9061) | Test (loss: 0.1840, acc: 0.9476)
Epoch 2/10: Train (loss: 0.1535, acc: 0.9561) | Test (loss: 0.1315, acc: 0.9631)
Epoch 3/10: Train (loss: 0.1052, acc: 0.9698) | Test (loss: 0.1026, acc: 0.9695)
Epoch 4/10: Train (loss: 0.0777, acc: 0.9773) | Test (loss: 0.0831, acc: 0.9752)
Epoch 5/10: Train (loss: 0.0615, acc: 0.9818) | Test (loss: 0.0752, acc: 0.9768)
Epoch 6/10: Train (loss: 0.0486, acc: 0.9855) | Test (loss: 0.0756, acc: 0.9758)
Epoch 7/10: Train (loss: 0.0391, acc: 0.9884) | Test (loss: 0.0766, acc: 0.9762)
Epoch 8/10: Train (loss: 0.0313, acc: 0.9918) | Test (loss: 0.0671, acc: 0.9793)
Epoch 9/10: Train (loss: 0.0261, acc: 0.9925) | Test (loss: 0.0660, acc: 0.9786)
Epoch 10/10: Train (loss: 0.0202, acc: 0.9947) | Test (loss: 0.0630, acc: 0.9806)
```

## 7. Evaluation

After training, the MLP should achieve **97-98% accuracy** on the MNIST test set. 

We can visualize predictions like this:

```python
import matplotlib.pyplot as plt

X, y = next(iter(test_loader))
X, y = X[:8], y[:8]
pred = model(X.to(device)).argmax(1).cpu()

plt.figure(figsize=(10,2))
for i in range(8):
    plt.subplot(1,8,i+1)
    plt.imshow(X[i].squeeze(), cmap="gray")
    plt.title(f"P:{pred[i].item()}\nT:{y[i].item()}")
    plt.axis("off")
plt.show()
```

![](./assets/img/posts/20250902_evaluation.png)

## 8. Key Takeways

- A simple MLP already performs very well on MNIST (~97-98% accuracy).
- However, it **does not capture spatial structure**, treating pixels independently.
- The parameter count is relatively high for such a simple task, which could lead to overfitting.

## 9. Conclusion & Next Steps

We built a simple MLP for MNIST digit classification. While it performs strongly, it has limitations that motivate the use of **Convolutional Neural Networks (CNNs)**, which are designed to capture spatial features in images.

👉 In the next post, we’ll implement **LeNet-5**, one of the earliest CNN architectures, and compare it to our MLP baseline.

[GitHub Code](https://github.com/derekzhouai/derekzhou-ai-blog-code/blob/main/mnist_mlp.ipynb)