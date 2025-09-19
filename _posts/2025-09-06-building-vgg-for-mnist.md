---
title:  "VGG: Deep but Simple"
date:   2025-09-06 20:00:00 +0800
categories: [Classic CNNs]
tags: [deeplearning, cnn, vgg]
math: true
---

## 1. Introduction

**VGG** network was introduced in 2014 by Karen Simonyan and Andrew Zisserman in their paper *“Very Deep Convolutional Networks for Large-Scale Image Recognition”*.  

It demonstrated that **depth matters** — simply stacking small $3 \times 3$ convolutions could significantly improve performance.

VGG achieved **2nd place** in the ImageNet ILSVRC 2014 competition, but its influence has been long-lasting: the clean, uniform design became a backbone for many later models (e.g., Faster R-CNN, SSD).

## 2. VGG Model Architecture

![](/assets/img/posts/20250906_vgg11_architecture_1.png)
_VGG11 Architecture_

VGG's design principle was **simplicity**:
- Use only **$3 \times 3$ convolutions**, stride 1, padding 1.  
- Use **$2 \times 2$ max pooling** to halve spatial resolution.  
- Increase feature channels gradually: 64 → 128 → 256 → 512.  
- End with **3 fully connected layers** (two 4096-unit + output). 

**VGG-11 Architecture**
- Input: 3x224x224 RGB image.
- VGG Block Layer #1:
    - Conv: 64 filters, 3x3, padding 1 → 64x224x224
    - Activation: ReLU
    - Pool: MaxPool 2x2, stride 2 → 64x112x112
- VGG Block Layer #2:
    - Conv: 128 filters, 3x3, padding 1 → 128x112x112
    - Activation: ReLU
    - Pool: MaxPool 2x2, stride 2 → 128x56x56
- VGG Block Layer #3:
    - Conv: 256 filters, 3x3, padding 1 → 256x56x56
    - Activation: ReLU
    - Conv: 256 filters, 3x3, padding 1 → 256x56x56
    - Activation: ReLU
    - Pool: MaxPool 2x2, stride 2 → 256x28x28
- VGG Block Layer #4:
    - Conv: 512 filters, 3x3, padding 1 → 512x28x28
    - Activation: ReLU
    - Conv: 512 filters, 3x3, padding 1 → 512x28x28
    - Activation: ReLU
    - Pool: MaxPool 2x2, stride 2 → 512x14x14
- VGG Block Layer #5:
    - Conv: 512 filters, 3x3, padding 1 → 512x14x14
    - Activation: ReLU
    - Conv: 512 filters, 3x3, padding 1 → 512x14x14
    - Activation: ReLU
    - Pool: MaxPool 2x2, stride 2 → 512x7x7
- Flatten: 512x7x7 → 25088
- FC Layer #1:
    - Linear: 4096 units → 4096
    - Activation: ReLU
    - Dropout: 0.5
- FC Layer #2:
    - Linear: 4096 units → 4096
    - Activation: ReLU
    - Dropout: 0.5
- FC Layer #3 (Output): 
    - Linear: 1000 units → 1000

**Key Innovations**

- **Small filters, deep stacks**: Replacing large $7 \times 7$ kernels with multiple $3 \times 3$ kernels improves representation power while reducing parameters.  
- **Uniform design**: Only one type of convolution (3×3), making the network easier to generalize.  
- **Scalability**: Depth can be scaled up systematically (11 → 16 → 19).  

**VGG Variants**:  
- VGG-11 (shallow, fewer conv layers)  
- VGG-16 (16 conv + FC layers, the most famous)  
- VGG-19 (deeper)  

## 3. VGG Model Implementation

VGG is originally designed for **ImageNet (1000 classes with RGB images)**. For demonstration, we’ll adapt VGG for **FashionMNIST (grayscale, 10 classes)**. Since FashionMNIST images are 28x28 grayscale, we’ll resize them to 224x224 and change the input channel to 1.

```python
import torch.nn as nn

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super().__init__()
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
```

```python
class VGG11(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            # Layer #1: 1 (convolutional layer + ReLU) + MaxPool
            VGGBlock(1, 64, 1),      # 1x224x224 → 64x112x112

            # Layer #2: 1 (convolutional layer + ReLU) + MaxPool
            VGGBlock(64, 128, 1),    # 64x112x112 → 128x56x56

            # Layer #3: 2 (convolutional layers + ReLU) + MaxPool
            VGGBlock(128, 256, 2),   # 128x56x56 → 256x28x28

            # Layer #4: 2 (convolutional layers + ReLU) + MaxPool
            VGGBlock(256, 512, 2),   # 256x28x28 → 512x14x14

            # Layer #5: 2 (convolutional layers + ReLU) + MaxPool
            VGGBlock(512, 512, 2),   # 512x14x14 → 512x7x7

            nn.Flatten(),            # 512x7x7 → 25088

            # Layer #6:
            nn.Linear(25088, 4096),  # 25088 → 4096
            nn.ReLU(),
            nn.Dropout(0.5),

            # Layer #7:
            nn.Linear(4096, 4096),  # 4096 → 4096
            nn.ReLU(),
            nn.Dropout(0.5),

            # Layer #8:
            nn.Linear(4096, num_classes), # 4096 → 10
        )

    def forward(self, x):
        return self.net(x)
```

**Number of parameters**:
- Convolutional layers: 640 + 73,856 + 885,248 + 3,539,968 + 4,719,616 = 9,219,328
    - VGG Block #1: (1 * 3 * 3 * 64) + 64 = 640
    - VGG Block #2: (64 * 3 * 3 * 128) + 128 = 73,856
    - VGG Block #3: (128 * 3 * 3 * 256) + 256 + (256 * 3 * 3 * 256) + 256 = 885,248
    - VGG Block #4: (256 * 3 * 3 * 512) + 512 + (512 * 3 * 3 * 512) + 512 = 3,539,968
    - VGG Block #5: (512 * 3 * 3 * 512) + 512 + (512 * 3 * 3 * 512) + 512 = 4,719,616
- Fully connected layers: 102,764,544 + 16,781,312 + 40,970 = 119,586,826
    - FC#1: (512 * 7 * 7 * 4096) + 4096 = 102,764,544
    - FC#2: (4096 * 4096) + 4096 = 16,781,312
    - Output: (4096 * 10) + 10 = 40,970
- Total: 9,219,328 + 119,586,826 = **128,806,154**

## 4. VGG Model Training

### Preparing the Data

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader
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
model = VGG11()
num_epochs = 10
batch_size = 128
lr = 0.1

train(model, num_epochs, batch_size, lr, device)
```

## 5. VGG Model Testing

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

## 6. Key Takeaways

- VGG proved that **depth + simplicity** can yield powerful results.
- Replacing large filters with **stacked 3×3 convolutions** was a game-changer.
- Even today, **VGG features are used as backbones** in many vision tasks.

## 7. Conclusion & Next Steps

We built and trained **VGG-11** on FashionMNIST, showing its simple yet deep architecture still performs well on modern datasets.

Next in the Classic CNNs series, we’ll explore GoogLeNet, which introduced the innovative Inception modules to efficiently capture multi-scale features.

Stay tuned!


[GitHub Code](https://github.com/derekzhouai/derekzhou-ai-blog-code/blob/main/cnn_vgg.ipynb)