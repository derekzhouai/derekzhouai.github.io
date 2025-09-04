---
title:  "Building a LeNet-5 for MNIST Classification"
date:   2025-09-04 09:00:00 +0800
categories: [Neural Networks on MNIST]
tags: [lenet5, mnist]
math: true
---

## 1. Introduction

In the [previous post](/posts/building-mlp-for-mnist/) of this series, we built a simple **Multi-Layer Perceptron (MLP)** to classify digits in the MNIST dataset. Despite treating each pixel as an independent feature, the MLP achieved around 97-98% accuracy - an impressive baseline.

However, MLPs have limitations: they ignore the **spatial structure** of images. A pixel in the top-left corner is treated no differently than one in the bottom-right corner. For images, this means that the relationships between neighboring pixels are lost.

This is where **Convolutional Neural Networks (CNNs)** come into play. In this post, we'll implement **LeNet-5**, one of the earliest CNN architectures, and show how it improves upon MLPs for digit recognition.

## 2. Background: What is LeNet-5?

![](./assets/img/posts/20250904_lenet5_architecture.png)

LeNet-5 was proposed in 1998 by **Yann LeCun** and colleagues for handwritten digit recognition.

**Key ideas in LeNet-5**:
- **Convolutions**: learn local patterns and features (e.g., edges, textures) that are useful across the image.
- **Pooling (subsampling)**: reduce dimensionality while maintaining important features.
- **Fully connected layers**: combine extracted features for classification.

**Original architecture**:
- Input: 32x32 grayscale image
- Conv1: 6 filters of size 5x5, stride 1 → 28x28x6
- Pool1: 2x2 average pooling, stride 2 → 14x14x6
- Conv2: 16 filters of size 5x5, stride 1 → 10x10x16
- Pool2: 2x2 average pooling, stride 2 → 5x5x16
- Flatten: 5x5x16 → 400
- FC1: 120 units
- FC2: 84 units
- Output: 10 units (digit classes)

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
            nn.Conv2d(1, 6, kernel_size=5, stride=1),             # 32x32x1 -> 28x28x6
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),                # 28x28x6 -> 14x14x6

            nn.Conv2d(6, 16, kernel_size=5, stride=1),            # 14x14x6 -> 10x10x16
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),                # 10x10x16 -> 5x5x16

            nn.Flatten(),                                         # 5x5x16 -> 400

            nn.Linear(400, 120),                                  # 400 -> 120
            nn.ReLU(),

            nn.Linear(120, 84),                                   # 120 -> 84
            nn.ReLU(),

            nn.Linear(84, num_classes)                            # 84 -> 10
        )

    def forward(self, x):
        return self.net(x)
```

**Number of parameters**: 
- Convolutional layers: 
  - Conv1: (5 * 5 * 1 * 6) + 6 = 156
  - Conv2: (5 * 5 * 6 * 16) + 16 = 2,416
- Fully connected layers:
  - FC1: (400 * 120) + 120 = 48,120
  - FC2: (120 * 84) + 84 = 10,164
  - Output: (84 * 10) + 10 = 850
- **Total**: 156 + 2,416 + 48,120 + 10,164 + 850 = 61,706

> This model has **~60k parameters**, far fewer than the MLP's ~200k, but it performs better because convolutional layers reuse filters across spatial locations.

## 5. Training Setup

We'll use the same loss function (CrossEntropyLoss). To compare optimizers, we'll try **SGD with momentum** this time.

```python
import torch
import torch.optim as optim

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

model = LeNet5()
model.to(device)

loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
epochs = 10
```

**Hyperparameters**:
- Batch size: 128 (train), 256 (test)
- Learning rate: 0.02
- Momentum: 0.9
- Epochs: 10

## 6. Training the Model

We can reuse the training/evaluation loops from the MLP post:

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
Epoch 1/10: Train (loss: 0.6650, acc: 0.7823) | Test (loss: 0.1295, acc: 0.9609)
Epoch 2/10: Train (loss: 0.1157, acc: 0.9645) | Test (loss: 0.0731, acc: 0.9765)
Epoch 3/10: Train (loss: 0.0775, acc: 0.9761) | Test (loss: 0.0676, acc: 0.9781)
Epoch 4/10: Train (loss: 0.0597, acc: 0.9814) | Test (loss: 0.0481, acc: 0.9845)
Epoch 5/10: Train (loss: 0.0495, acc: 0.9848) | Test (loss: 0.0447, acc: 0.9861)
Epoch 6/10: Train (loss: 0.0415, acc: 0.9866) | Test (loss: 0.0424, acc: 0.9873)
Epoch 7/10: Train (loss: 0.0363, acc: 0.9884) | Test (loss: 0.0421, acc: 0.9866)
Epoch 8/10: Train (loss: 0.0299, acc: 0.9899) | Test (loss: 0.0397, acc: 0.9872)
Epoch 9/10: Train (loss: 0.0274, acc: 0.9913) | Test (loss: 0.0399, acc: 0.9874)
Epoch 10/10: Train (loss: 0.0248, acc: 0.9919) | Test (loss: 0.0324, acc: 0.9894)
```

## 7. Evaluation

LeNet-5 typically achieves **~99% accuracy** on MNIST, outperforming the MLP model.

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

## 8. Key Takeaways
- **LeNet-5 outperforms MLP** on MNIST, reaching ~99% accuracy.
- It uses far fewer parameters thanks to convolutional weight sharing.
- CNNs are translation invariant and better at recognizing spatial patterns.
- LeNet-5 was the starting point for many modern CNNs like AlexNet, VGG, ResNet, and Inception.

## 9. Conclusion & Next Steps

We built and trained a **LeNet-5 model** for MNIST classification, improving upon the MLP baseline. This demonstrates the power of convolutional networks for image tasks.

In the next post, we'll extend CNNs further by experimenting with **data augmentation, deeper CNN architectures, and regularization techniques**.

Stay tuned!

[GitHub Code](https://github.com/derekzhouai/derekzhou-ai-blog-code/blob/main/mnist_lenet5.ipynb)