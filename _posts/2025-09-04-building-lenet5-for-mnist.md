---
title:  "LeNet-5: The Origin of CNNs"
date:   2025-09-04 09:00:00 +0800
categories: [Classic CNNs]
tags: [deepinglearning, cnn, lenet5]
math: true
---

## 1. Introduction

**LeNet-5** was proposed in 1998 by Yann LeCun and colleagues for handwritten digit recognition. 
Although computing resources were limited at the time, LeNet-5 was revolutionary: it showed that **convolutional neural networks (CNNs) could outperform traditional machine learning methods** on image recognition tasks.

This network was originally designed to recognize handwritten digits (MNIST), and its ideas laid the groundwork for modern deep learning.

## 2. Background: What is LeNet-5?

![](./assets/img/posts/20250904_lenet5_architecture.png)

**Key ideas in LeNet-5**:
- **Convolutions**: learn local patterns and features (e.g., edges, textures) that are useful across the image.
- **Pooling (subsampling)**: reduce dimensionality while maintaining important features.
- **Fully connected layers**: combine extracted features for classification.

![](/assets/img/posts/20250904_lenet5_architecture_2.png)

**Architecture**:
- Input: 32x32 grayscale image → 1x32x32
- Layer #1:
    - Conv: 6 filters of size 5x5, stride 1 → 6x28x28
    - Activation: ReLU → 6x28x28
    - Pool: 2x2 average pooling, stride 2 → 6x14x14
- Layer #2:
    - Conv: 16 filters of size 5x5, stride 1 → 16x10x10
    - Activation: ReLU → 16x10x10
    - Pool: 2x2 average pooling, stride 2 → 16x5x5
- Layer #3:
    - Linear: 120 units → 120
    - Activation: ReLU → 120
- Layer #4:
    - Linear: 84 units → 84
    - Activation: ReLU → 84
- Layer #5(Output): 
    - Linear: 10 units (digit classes 0-9) → 10

> The original LeNet-5 used sigmoid/tanh activations, but here we will use ReLU for better performance.
{: .prompt-info }

Since MNIST images are 28x28, we'll **pad them to 32x32** before feeding them into the network.

## 3. Dataset Preparation

We'll use PyTorch and torchvision to load and preprocess the MNIST dataset.

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Pad(2),  # 28x28 → 32x32
    transforms.ToTensor()
])

train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
```

```python
print(f"Number of training samples: {len(train_data)}")
print(f"Number of test samples: {len(test_data)}")

# Output:
# Number of training samples: 60000
# Number of test samples: 10000
```

## 4. Model Architecture

We'll build a PyTorch version of LeNet-5.

```python
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            # Layer #1
            nn.Conv2d(1, 6, kernel_size=5, stride=1),      # 1x32x32 -> 6x28x28
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),         # 6x28x28 -> 6x14x14

            # Layer #2
            nn.Conv2d(6, 16, kernel_size=5, stride=1),     # 6x14x14 -> 16x10x10
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),         # 16x10x10 -> 16x5x5

            nn.Flatten(),                                  # 16x5x5 -> 400

            # Layer #3
            nn.Linear(400, 120),                           # 400 -> 120
            nn.ReLU(),

            # Layer #4
            nn.Linear(120, 84),                            # 120 -> 84
            nn.ReLU(),

            # Layer #5
            nn.Linear(84, num_classes)                     # 84 -> 10
        )

    def forward(self, x):
        return self.net(x)
```

**Number of parameters**: 
- Convolutional layers: 156 + 2,416 = **2,572**
  - Conv#1: (1 * 5 * 5 * 6) + 6 = 156
  - Conv#2: (6 * 5 * 5 * 16) + 16 = 2,416
- Fully connected layers: 48,120 + 10,164 + 850 = **59,134**
  - FC#1: (16 * 5 * 5 * 120) + 120 = 48,120
  - FC#2: (120 * 84) + 84 = 10,164
  - Output: (84 * 10) + 10 = 850
- Total: 2,572 + 59,134 = **61,706**

## 5. Training the Model

```python
import torch

def evaluate(model, loader, loss, device):
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

def train(model, train_loader, test_loader, num_epochs, lr, device):
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_correct, total_num = 0.0, 0, 0

        for X, y in train_loader:
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

        train_loss = total_loss / total_num
        train_acc = total_correct / total_num

        test_loss, test_acc = evaluate(model, test_loader, loss, device)
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
```

```python
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
model = LeNet5()
num_epochs = 10
lr = 0.5

train(model, train_loader, test_loader, num_epochs, lr, device)
```

```python
Output:
Epoch 1/10: Train Loss: 0.8300, Train Acc: 0.7093 | Test Loss: 0.1049, Test Acc: 0.9671
Epoch 2/10: Train Loss: 0.0924, Train Acc: 0.9725 | Test Loss: 0.0623, Test Acc: 0.9802
Epoch 3/10: Train Loss: 0.0580, Train Acc: 0.9825 | Test Loss: 0.0557, Test Acc: 0.9826
Epoch 4/10: Train Loss: 0.0432, Train Acc: 0.9869 | Test Loss: 0.0711, Test Acc: 0.9778
Epoch 5/10: Train Loss: 0.0361, Train Acc: 0.9886 | Test Loss: 0.0312, Test Acc: 0.9896
Epoch 6/10: Train Loss: 0.0297, Train Acc: 0.9906 | Test Loss: 0.0415, Test Acc: 0.9886
Epoch 7/10: Train Loss: 0.0253, Train Acc: 0.9920 | Test Loss: 0.0473, Test Acc: 0.9858
Epoch 8/10: Train Loss: 0.0235, Train Acc: 0.9923 | Test Loss: 0.0356, Test Acc: 0.9890
Epoch 9/10: Train Loss: 0.0202, Train Acc: 0.9936 | Test Loss: 0.0614, Test Acc: 0.9818
Epoch 10/10: Train Loss: 0.0182, Train Acc: 0.9941 | Test Loss: 0.0418, Test Acc: 0.9866
```

## 6. Testing the Model

LeNet-5 typically achieves **~99% accuracy** on MNIST, significantly better than a simple MLP.

We can visualize predictions:

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

![](./assets/img/posts/20250904_evaluation.png)

## 7. Key Takeaways
- LeNet-5 outperforms simple MLP on MNIST, reaching ~99% accuracy.
- It uses far fewer parameters thanks to convolutional weight sharing.
- CNNs are translation invariant and better at recognizing spatial patterns.
- LeNet-5 was the starting point for many modern CNNs like AlexNet, VGG, Inception, and ResNet.

## 8. Conclusion & Next Steps

We built and trained a **LeNet-5 model** for MNIST classification. It achieves ~99% accuracy, demonstrating the power of convolutional networks for image tasks.

In the next post, we'll extend CNNs further by experimenting with more advanced architectures like **VGG** and **ResNet**.

Stay tuned!

[GitHub Code](https://github.com/derekzhouai/derekzhou-ai-blog-code/blob/main/mnist_lenet5.ipynb)