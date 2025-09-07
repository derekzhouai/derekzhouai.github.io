---
title:  "Building a VGG for MNIST Classification"
date:   2025-09-06 20:00:00 +0800
categories: [Neural Networks on MNIST]
tags: [vgg, mnist]
math: true
---

## 1. Introduction

In this series, we have explored different neural network architectures on the **MINIST dataset**:
- **MLP**: A simple feedforward neural network treating each pixel independently (~97-98% accuracy).
- **LeNet-5**: One of earliest CNNs (~99% accuracy), proposed by Yann LeCun in 1998.
- **AlexNet**: A deeper, revolutionized CNN in 2012, but overkill for MNIST (~99%).

This time, we'll look at **VGG** - a model from 2014 that demonstrated how **depth + small convolutional filters** can build powerful models.

Our goal is to adapt a VGG-style architecture for MNIST and see how it performs.

## 2. Background: What is VGG?

![](/assets/img/posts/20250906_vgg_block.png){: .normal }

**VGG** was proposed by Simonyan and Zisserman in 2014, achieving state-of-the-art results on ImageNet.

**Key ideas**:
- Use **3x3 convolutions** stacked in depth instead of larger filters (e.g., 5x5, 7x7).
- Uniform architecture: **VGG Block**
    - **VGG Block** is a block of (3x3 Conv) → ... → (3x3 Conv) → (2x2 MaxPool).
    - **3x3 Convs have a padding of 1 pixel** to preserve spatial dimensions.
    - **2x2 MaxPool with stride 2** halves the spatial dimensions.
- Very deep models (VGG-16, VGG-19) with 100M+ parameters.

Although it's heavy by today's standards, VGG showed that **deeper networks with simple design principles** could outperform shallower, more complex architectures.

## 3. Adapting VGG to MNIST

**Challenges**:
- MNIST images are **28x28 grayscale**.
- VGG was designed for **224x224 RGB** images.

**Adaptations**:
1. **Keep input channel = 1**.
2. **Reduce number of filters** in each block and **number of blocks** to avoid overfitting and excessive computation.

## 4. Data Preparation

Same preprocessing as before (no need to upscale to 224×224 since I will use MiniVGG).

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
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

### VGG-16

![](/assets/img/posts/20250906_vgg16_architecture_1.png){: .normal }

![](/assets/img/posts/20250906_vgg16_architecture_2.webp){: .normal }
_The architecture of VGG16. Source: Researchgate.net_

```python
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(),     # 64x224x224
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),    # 64x224x224
            nn.MaxPool2d(kernel_size=2, stride=2),                     # 64x112x112

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),   # 128x112x112
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),  # 128x112x112
            nn.MaxPool2d(kernel_size=2, stride=2),                     # 128x56x56

            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),  # 256x56x56
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),  # 256x56x56
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),  # 256x56x56
            nn.MaxPool2d(kernel_size=2, stride=2),                     # 256x28x28

            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),  # 512x28x28
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),  # 512x28x28
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),  # 512x28x28
            nn.MaxPool2d(kernel_size=2, stride=2),                     # 512x14x14

            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),  # 512x14x14
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),  # 512x14x14
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),  # 512x14x14
            nn.MaxPool2d(kernel_size=2, stride=2),                     # 512x7x7

            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        return self.net(x)
```

**Number of parameters (VGG-16)**: 14,714,688 + 119,586,826 = **134,301,514**
- Conv layers: 38,720 + 221,440 + 1,475,328 + 5,899,776 + 7,079,424 = **14,714,688**
    - Conv#1: 1,792 + 36,928 = 38,720
        - Conv#1-1: 3x3x3x64 + 64 = 1,792
        - Conv#1-2: 3x3x64x64 + 64 = 36,928
    - Conv#2: 73,856 + 147,584 = 221,440
        - Conv#2-1: 3x3x64x128 + 128 = 73,856
        - Conv#2-2: 3x3x128x128 + 128 = 147,584
    - Conv#3: 295,168 + 590,080 + 590,080 = 1,475,328
        - Conv#3-1: 3x3x128x256 + 256 = 295,168
        - Conv#3-2: 3x3x256x256 + 256 = 590,080
        - Conv#3-3: 3x3x256x256 + 256 = 590,080
    - Conv#4: 1,180,160 + 2,359,808 + 2,359,808 = 5,899,776
        - Conv#4-1: 3x3x256x512 + 512 = 1,180,160
        - Conv#4-2: 3x3x512x512 + 512 = 2,359,808
        - Conv#4-3: 3x3x512x512 + 512 = 2,359,808
    - Conv#5: 2,359,808 + 2,359,808 + 2,359,808 = 7,079,424
        - Conv#5-1: 3x3x512x512 + 512 = 2,359,808
        - Conv#5-2: 3x3x512x512 + 512 = 2,359,808
        - Conv#5-3: 3x3x512x512 + 512 = 2,359,808
- FC layers: 102,764,544 + 16,781,312 + 40,970 = **119,586,826**
    - FC#1: 512 * 7 * 7 * 4096 + 4096 = 102,764,544
    - FC#2: 4096 * 4096 + 4096 = 16,781,312
    - FC#3: 4096 * 10 + 10 = 40,970

### MiniVGG

![](/assets/img/posts/20250906_minivgg_architecture.png){: width="600" .normal }

To fit simple dataset MNIST, we will use a smaller version of VGG (MiniVGG) with fewer layers:

```python
import torch.nn as nn

class MiniVGG(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),     # 32x28x28
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),    # 32x28x28
            nn.MaxPool2d(kernel_size=2, stride=2),                     # 32x14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),    # 64x14x14
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),    # 64x14x14
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),    # 64x14x14
            nn.MaxPool2d(kernel_size=2, stride=2),                     # 64x7x7

            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        return self.net(x)
```

**Number of parameters (MiniVGG)**: 101,920 + 29,671,434 = **29,773,354**
- Conv layers: 320 + 9,248 + 18,496 + 36,928 + 36,928 = 101,920
    - Conv#1-1: 3x3x1x32 + 32 = 320
    - Conv#1-2: 3x3x32x32 + 32 = 9,248
    - Conv#2-1: 3x3x32x64 + 64 = 18,496
    - Conv#2-2: 3x3x64x64 + 64 = 36,928
    - Conv#2-3: 3x3x64x64 + 64 = 36,928
- FC layers: 12,849,152 + 40,970 = 29,671,434
    - FC#1: 64 * 7 * 7 * 4096 + 4096 = 12,849,152
    - FC#2: 4096 * 4096 + 4096 = 16,781,312
    - FC#3: 4096 * 10 + 10 = 40,970

## 6. Training Setup

We’ll use the same setup as in earlier posts.

```python
import torch
import torch.optim as optim

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

model = MiniVGG()
model.to(device)

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5
```

**Hyperparameters**:
- **Batch Size**: 128 (train), 256 (test)
- **Learning Rate**: 0.001
- **Epochs**: 5

## 7. Training the Model

We can reuse the same training loop:

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
Epoch 1/5: Train (loss: 0.2207, acc: 0.9274) | Test (loss: 0.0555, acc: 0.9818)
Epoch 2/5: Train (loss: 0.0566, acc: 0.9835) | Test (loss: 0.0495, acc: 0.9846)
Epoch 3/5: Train (loss: 0.0418, acc: 0.9878) | Test (loss: 0.0338, acc: 0.9913)
Epoch 4/5: Train (loss: 0.0314, acc: 0.9906) | Test (loss: 0.0358, acc: 0.9883)
Epoch 5/5: Train (loss: 0.0288, acc: 0.9917) | Test (loss: 0.0290, acc: 0.9914)
```

## 8. Evaluation

**Results**:
- Test accuracy: ~99+%, a bit higher than LeNet-5 and AlexNet.
- Since we only used two VGG blocks, training is faster and less memory-intensive.

## 9. Key Takeaways

- **VGG confirms the trend**: CNNs are highly effective for image tasks.
- Stacking small 3x3 convolutions works well - the main design principle of VGG.
- On MNIST, VGG doesn't outperform LeNet significantly, but it's an excellent example of **systematic deep CNN design**.

## 10. Conclusion & Next Steps

We've now implemented VGG-style networks for MNIST. While performance is similar with LetNet-5 and AlexNet, this exercise demonstrates the value of **deeper architectures with consistent design**.

In the next post, we’ll explore **ResNet**, which introduced skip connections to solve the vanishing gradient problem and enable training of very deep networks.

[GitHub Code](https://github.com/derekzhouai/derekzhou-ai-blog-code/blob/main/mnist_vgg.ipynb)