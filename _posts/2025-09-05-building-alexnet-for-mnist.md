---
title:  "Building an AlexNet for MNIST Classification"
date:   2025-09-05 10:00:00 +0800
categories: [Neural Networks on MNIST]
tags: [alexnet, mnist]
math: true
---

## 1. Introduction

So far in this series, we have explored two models on the MNIST dataset:
- **Multi-Layer Perceptron (MLP)**: A simple feedforward neural network treating each pixel independently (~97-98% accuracy).
- **LeNet-5**: One of earliest CNNs (~99% accuracy).

Both worked well, but what happens if we apply a **larger, deeper CNN** that once changed the field of computer vision? In this post, we'll adapt **AlexNet** - the model that won the 2012 ImageNet competition - and see how it performs on MNIST.

## 2. Background: What is AlexNet?

**AlexNet**, designed by Alex Krizhevsky and colleagues, was the breakthrough that brought deep learning to the forefront of computer vision. Key innovations included:

- **Deep architecture** with 5 convolutional layers and 3 fully connected layers.
- **ReLU activation** instead of sigmoid/tanh, speeding up training.
- **Dropout** for regularization.
- **Max pooling** for downsampling.
- Training on **GPUs** - a critical factor in making the model feasible.

Originally, AlexNet was trained on **224x224 RGB images** from ImageNet, which is much larger and more complex than MNIST (28x28 grayscale). We'll adapt the architecture accordingly.

## 3. Adapting AlexNet to MNIST

To make AlexNet work on MNIST, we need to make a few adjustments:
1. **Input size**: MNIST is 28x28, but AlexNet expects 224x224. We'll resize MNIST images to 224x224.
2. **Input channels**: MNIST is grayscale (1 channel), not RGB (3 channels). We'll modify the first convolutional layer to accept 1 channel.
3. **Output classes**: MNIST has 10 classes (digits 0-9), so the final layer will have 10 outputs instead of 1000.

This lets us train a slightly smaller but structurally similar model of AlexNet on MNIST.

## 4. Dataset Preparation

We’ll use torchvision.transforms.Resize to upscale MNIST images.

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize(224),    # Upscale 28x28 → 224x224
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

## 5. Model Architecture

![](/assets/img/posts/20250905_alexnet_architecture.png)

Here’s an AlexNet-inspired model adapted for MNIST:

```python
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),         #1x224x224 → 96x54x54
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),                         #96x54x54 → 96x26x26

            nn.Conv2d(96, 256, kernel_size=5, padding=2),                  #96x26x26 → 256x26x26
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),                         #256x26x26 → 256x12x12

            nn.Conv2d(256, 384, kernel_size=3, padding=1),                 #256x12x12 → 384x12x12
            nn.ReLU(),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),                 #384x12x12 → 384x12x12
            nn.ReLU(),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),                 #384x12x12 → 256x12x12
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),                         #256x12x12 → 256x5x5

            nn.Flatten(),
            nn.Linear(256 * 5 * 5, 4096),  # Adjusted for input size
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        return self.net(x)
```

**Number of parameters**:
- Convolutional layers: 11,712 + 614,656 + 885,120 + 1,327,488 + 884,992 = 3,723,968
    - Conv#1: (11 * 11 * 1 * 96) + 96 = 11,712
    - Conv#2: (5 * 5 * 96 * 256) + 256 = 614,656
    - Conv#3: (3 * 3 * 256 * 384) + 384 = 885,120
    - Conv#4: (3 * 3 * 384 * 384) + 384 = 1,327,488
    - Conv#5: (3 * 3 * 384 * 256) + 256 = 884,992
- Fully connected layers: 27,140,096 + 16,781,312 + 40,970 = 43,962,378
    - FC#1: (256 * 5 * 5 * 4096) + 4096 = 27,140,096
    - FC#2: (4096 * 4096) + 4096 = 16,781,312
    - Output: (4096 * 10) + 10 = 40,970
- Total: 3,723,968 + 43,962,378 = **47,686,346**

> This model has **~47M** parameters, significantly larger than LeNet-5's ~60K, showcasing the depth and complexity of AlexNet.

## 6. Training Setup

We’ll reuse the same loss and optimizer as before.

```python
import torch
import torch.optim as optim

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

model = AlexNet()
model.to(device)

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5
```

**Hyperparameters**:
- Batch size: 128
- Learning rate: 0.001
- Epochs: 5

## 7. Training the Model

We can reuse the same training/evaluation functions from previous posts:
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
# Output:   
Epoch 1/5: Train (loss: 0.3830, acc: 0.8698) | Test (loss: 0.0659, acc: 0.9790)
Epoch 2/5: Train (loss: 0.0652, acc: 0.9805) | Test (loss: 0.0444, acc: 0.9881)
Epoch 3/5: Train (loss: 0.0509, acc: 0.9853) | Test (loss: 0.0477, acc: 0.9856)
Epoch 4/5: Train (loss: 0.0428, acc: 0.9872) | Test (loss: 0.0404, acc: 0.9890)
Epoch 5/5: Train (loss: 0.0394, acc: 0.9885) | Test (loss: 0.0255, acc: 0.9923)
```

## 8. Evaluation

AlexNet on MNIST will usually achieve **~99% accuracy** - similar to LeNet-5.

However:
- Training is **far slower**.
- Model has **tens of millions of parameters**, making it impractical for such a simple task.
- Accuracy doesn't improve much over LeNet-5 because MNIST is too simple.

## 9. Key Takeaways

- AlexNet was revolutionary for **large-scale datasets** like ImageNet.
- On MNIST, it's **too large** and provides little gain over LeNet-5.
- This highlights the importance of **matching model complexity to dataset size**.
- Still, building AlexNet for MNIST is a valuable exercise to understand deep CNNs.

## 10. Conclusion & Next Steps

In this post, we adapted **AlexNet** for MNIST classification. While it performs well (~99%), it showed that bigger isn't always better, especially on simple datasets.

In next posts, we'll explore **modern CNN architectures** such as **VGG** or **ResNet**, and see how they compare on MNIST.

[GitHub Code](https://github.com/derekzhouai/derekzhou-ai-blog-code/blob/main/mnist_alexnet.ipynb)