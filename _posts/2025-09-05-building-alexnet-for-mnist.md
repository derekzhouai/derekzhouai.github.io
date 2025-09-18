---
title:  "AlexNet: The Breakthrough CNN"
date:   2025-09-05 10:00:00 +0800
categories: [Classic CNNs]
tags: [deepinglearning, cnn, alexnet]
math: true
---

## 1. Introduction

**AlexNet** was introduced in 2012 by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, and it **changed the history of deep learning**.  

Winning the **ImageNet Large Scale Visual Recognition Challenge (ILSVRC 2012)** with a top-5 error rate of 15.3% (compared to the second-best 26.2%), AlexNet showed that deep convolutional neural networks could dominate computer vision.

This breakthrough revived interest in neural networks, fueled by three key factors:

- Availability of **large datasets** (ImageNet: 1.2M labeled images).  
- Affordable **GPU computing** (NVIDIA GTX 580).  
- Better **training techniques** (ReLU, Dropout, Data Augmentation).  

## 2. AlexNet Model Architecture

![](/assets/img/posts/20250905_alexnet_architecture_2.png)

AlexNet is deeper and larger than LeNet-5. It has **5 convolutional layers** and **3 fully connected layers**, totaling about **60 million parameters**.

**Architecture**:
- Input: 3x224x224 RGB image.  
- Conv Layer #1: 
    - Conv: 96 filters, 11x11, stride 4 → 96x54x54
    - Activation: ReLU → 96x54x54
    - Pool: MaxPool 3x3, stride 2 → 96x26x26
- Conv Layer #2: 
    - Conv: 256 filters, 5x5, padding 2 → 256x26x26
    - Activation: ReLU → 256x26x26
    - Pool: MaxPool 3x3, stride 2 → 256x12x12
- Conv Layer #3: 
    - Conv: 384 filters, 3x3 → 384x12x12
    - Activation: ReLU → 384x12x12
- Conv Layer #4: 
    - Conv: 384 filters, 3x3 → 384x12x12
    - Activation: ReLU → 384x12x12
- Conv Layer #5: 
    - Conv: 256 filters, 3x3 → 256x12x12
    - Activation: ReLU → 256x12x12
    - Pool: MaxPool 3x3, stride 2 → 256x5x5
- Flatten: 256x5x5 → 6400
- FC Layer #1: 
    - Linear: 4096 units → 4096
    - Activation: ReLU → 4096
    - Dropout (0.5) → 4096
- FC Layer #2: 
    - Linear: 4096 units → 4096
    - Activation: ReLU → 4096
    - Dropout (0.5) → 4096
- FC Layer #3 (Output): 
    - Linear: 1000 units → 1000
    
**Key innovations in AlexNet**:
- **ReLU** activation (faster convergence vs sigmoid/tanh).  
- **Dropout** for regularization in fully connected layers.  
- **Data augmentation** (cropping, flipping, color jitter).  
- **GPU training** for scalability.  


Originally, AlexNet was trained on **224x224 RGB images** from ImageNet, which is much larger and more complex than MNIST (28x28 grayscale). We'll adapt the architecture accordingly.

## 3. AlexNet Model Implementation

AlexNet is originally designed for **ImageNet (1000 classes with RGB images)**. For demonstration, we’ll adapt AlexNet for **FashionMNIST (grayscale, 10 classes)**. Since FashionMNIST images are 28x28 grayscale, we’ll resize them to 224x224 and change the input channel to 1.

```python
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            # Layer #1
            nn.Conv2d(1, 96, kernel_size=11, stride=4),    # 1x224x224 → 96x54x54
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),         # 96x54x54 → 96x26x26

            # Layer #2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),  # 96x26x26 → 256x26x26
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),         # 256x26x26 → 256x12x12

            # Layer #3
            nn.Conv2d(256, 384, kernel_size=3, padding=1), # 256x12x12 → 384x12x12
            nn.ReLU(),

            # Layer #4
            nn.Conv2d(384, 384, kernel_size=3, padding=1), # 384x12x12 → 384x12x12
            nn.ReLU(),

            # Layer #5
            nn.Conv2d(384, 256, kernel_size=3, padding=1), # 384x12x12 → 256x12x12
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),         # 256x12x12 → 256x5x5

            nn.Flatten(),                                  # 256x5x5 → 6400

            # Layer #6
            nn.Linear(256 * 5 * 5, 4096),                  # 6400 → 4096
            nn.ReLU(),
            nn.Dropout(),

            # Layer #7
            nn.Linear(4096, 4096),                         # 4096 → 4096
            nn.ReLU(),
            nn.Dropout(),

            # Layer #8 (Output)
            nn.Linear(4096, num_classes)                   # 4096 → 10
        )

    def forward(self, x):
        return self.net(x)
```

**Number of parameters**:
- Convolutional layers: 11,712 + 614,656 + 885,120 + 1,327,488 + 884,992 = 3,723,968
    - Conv#1: (1 * 11 * 11 * 96) + 96 = 11,712
    - Conv#2: (96 * 5 * 5 * 256) + 256 = 614,656
    - Conv#3: (256 * 3 * 3 * 384) + 384 = 885,120
    - Conv#4: (384 * 3 * 3 * 384) + 384 = 1,327,488
    - Conv#5: (384 * 3 * 3 * 256) + 256 = 884,992
- Fully connected layers: 27,140,096 + 16,781,312 + 40,970 = 43,962,378
    - FC#1: (256 * 5 * 5 * 4096) + 4096 = 27,140,096
    - FC#2: (4096 * 4096) + 4096 = 16,781,312
    - Output: (4096 * 10) + 10 = 40,970
- Total: 3,723,968 + 43,962,378 = **47,686,346**

## 4. AlexNet Model Training

### Preparing the Data

To make AlexNet work on FashionMNIST, we need to make a few adjustments:
1. **Input size**: FashionMNIST is 28x28, but AlexNet expects 224x224. We'll resize FashionMNIST images to 224x224.
2. **Input channels**: FashionMNIST is grayscale (1 channel), not RGB (3 channels). We'll modify the first convolutional layer to accept 1 channel.
3. **Output classes**: FashionMNIST has 10 classes (clothing items), so the final layer will have 10 outputs instead of 1000.

This lets us train a slightly smaller but structurally similar model of AlexNet on FashionMNIST.

We’ll use `torchvision.transforms.Resize` to upscale FashionMNIST images.

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_data_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.Resize(224),    # Upscale 28x28 → 224x224
        transforms.ToTensor()
    ])

    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
```

```python
train_loader, test_loader = get_data_loaders(batch_size=128)
print(f"Number of training samples: {len(train_loader.dataset)}")
print(f"Number of test samples: {len(test_loader.dataset)}")

# Output:
# Number of training samples: 60000
# Number of test samples: 10000
```

```python
X, y = next(iter(train_loader))
print(f"X.shape: {X.shape}")
print(f"y.shape: {y.shape}")

# Output:
# X.shape: torch.Size([128, 1, 224, 224])
# y.shape: torch.Size([128])
```

### Training Loop

We reuse the same **training/evaluation** functions from previous posts.

```python
import torch

def evaluate(model, loader, loss, device):
    model.eval()
    total_loss, total_correct, total_num = 0.0, 0, 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            
            y_hat = model(X)
            l = loss(y_hat, y)

            total_loss += l.item() * X.size(0)
            total_correct += (y_hat.argmax(dim=1) == y).sum().item()
            total_num += X.size(0)
            
    return total_loss / total_num, total_correct / total_num

def train(model, num_epochs, batch_size, lr, device):
    model.to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    train_loader, test_loader = get_data_loaders(batch_size)

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
            total_correct += (y_hat.argmax(dim=1) == y).sum().item()
            total_num += X.size(0)

        train_loss = total_loss / total_num
        train_acc = total_correct / total_num

        test_loss, test_acc = evaluate(model, test_loader, loss, device)
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train => Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Test => Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
```

```python
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
model = AlexNet()
num_epochs = 10
batch_size = 128
lr = 0.1

train(model, num_epochs, batch_size, lr, device)
```

```python
# Output:   
Epoch 1/10: Train => Loss: 1.3983, Acc: 0.4714 | Test => Loss: 0.5943, Acc: 0.7637
Epoch 2/10: Train => Loss: 0.4979, Acc: 0.8139 | Test => Loss: 0.4608, Acc: 0.8285
Epoch 3/10: Train => Loss: 0.3696, Acc: 0.8618 | Test => Loss: 0.3330, Acc: 0.8776
Epoch 4/10: Train => Loss: 0.3177, Acc: 0.8798 | Test => Loss: 0.3193, Acc: 0.8809
Epoch 5/10: Train => Loss: 0.2885, Acc: 0.8924 | Test => Loss: 0.3089, Acc: 0.8864
Epoch 6/10: Train => Loss: 0.2625, Acc: 0.9010 | Test => Loss: 0.2878, Acc: 0.8928
Epoch 7/10: Train => Loss: 0.2430, Acc: 0.9089 | Test => Loss: 0.2703, Acc: 0.8999
Epoch 8/10: Train => Loss: 0.2260, Acc: 0.9150 | Test => Loss: 0.2697, Acc: 0.9045
Epoch 9/10: Train => Loss: 0.2093, Acc: 0.9214 | Test => Loss: 0.2414, Acc: 0.9135
Epoch 10/10: Train => Loss: 0.1968, Acc: 0.9262 | Test => Loss: 0.2596, Acc: 0.9047
```

> The training must be run on a machine with a GPU. On CPU, it will be very slow.
{: .prompt-warning }

## 5. AlexNet Model Testing

The final test accuracy is around **90.47%** after 10 epochs, which is quite good for FashionMNIST.

Let's visualize predictions:

```python
import torch
import matplotlib.pyplot as plt

# FashionMNIST class names
FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def show_fashionmnist_preds(model, test_loader, device, n=8):
    """
    Show n FashionMNIST test images with predicted and true labels.
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        X, y = next(iter(test_loader))
        X, y = X[:n], y[:n]
        pred = model(X.to(device)).argmax(1).cpu()

    plt.figure(figsize=(2*n, 2.6))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        img = X[i].squeeze().cpu()
        plt.imshow(img, cmap="gray")
        p_idx, t_idx = pred[i].item(), y[i].item()
        plt.title(f"P:{FASHION_CLASSES[p_idx]}\nT:{FASHION_CLASSES[t_idx]}", fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

show_fashionmnist_preds(model, test_loader, device, n=8)
```

![](/assets/img/posts/20250905_output.png)

## 6. Key Takeaways

- AlexNet proved **deep CNNs scale** to large datasets (ImageNet).
- **ReLU** enabled faster training compared to sigmoid/tanh.
- **Dropout** + **Data Augmentation** reduced overfitting in large networks.
- **GPUs** were essential — AlexNet would have taken weeks on CPUs.

## 7. Conclusion & Next Steps

We built and trained **AlexNet** on FashionMNIST. Despite being over a decade old, AlexNet still performs strongly and illustrates the **turning point of modern deep learning**.

In the next post, we’ll explore **VGG**, which simplified CNN design with stacked **3×3 convolutions** and pushed depth even further.

Stay tuned for the next article in the **Classic CNNs** series!

[GitHub Code](https://github.com/derekzhouai/derekzhou-ai-blog-code/blob/main/mnist_alexnet.ipynb)