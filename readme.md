# Recognition using Deep Networks

A comprehensive deep learning project implementing CNNs for handwritten digit recognition, featuring transfer learning, filter visualization, and systematic hyperparameter optimization.

![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-green)
![MNIST](https://img.shields.io/badge/Dataset-MNIST-yellow)
![License](https://img.shields.io/badge/license-MIT-orange)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Usage](#usage)
- [Network Architecture](#network-architecture)
- [Task Breakdown](#task-breakdown)
- [Transfer Learning](#transfer-learning)
- [Experimental Results](#experimental-results)
- [Filter Analysis](#filter-analysis)
- [Performance Metrics](#performance-metrics)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Future Enhancements](#future-enhancements)

## 🎯 Overview

This project provides a comprehensive exploration of deep learning through the development and analysis of Convolutional Neural Networks (CNNs). Starting with MNIST digit recognition, the project progresses through network analysis, transfer learning for Greek letter classification, and systematic experimentation with various architectural choices and hyperparameters.

**Key Achievements:**
- Built custom CNN achieving >98% accuracy on MNIST
- Implemented transfer learning for Greek letter recognition
- Analyzed and visualized convolutional filters
- Conducted systematic hyperparameter optimization
- Best model: 91.19% accuracy on FashionMNIST

## ✨ Features

### Core Functionality

#### 1. **Custom CNN Architecture**
- 2 Convolutional layers (10 and 20 filters)
- Max pooling for spatial down-sampling
- Dropout regularization (5-50%)
- Fully connected layers with ReLU activation
- Log-Softmax output for classification

#### 2. **Training Pipeline**
- Adam optimizer for efficient convergence
- Negative log-likelihood loss
- Batch processing (batch size: 64)
- Training/test loss monitoring
- Model checkpointing

#### 3. **Network Analysis**
- First layer filter visualization
- Filter effect demonstration
- Feature map analysis
- Weight inspection and interpretation

#### 4. **Transfer Learning**
- Pre-trained model adaptation
- Layer freezing strategies
- Fine-tuning for new classes
- Greek letter classification (α, β, γ)

#### 5. **Custom Image Testing**
- Handwritten digit recognition
- Image preprocessing pipeline
- Real-world generalization testing
- Confidence score visualization

#### 6. **Hyperparameter Experiments**
- Grid search optimization
- Performance vs. efficiency analysis
- Systematic architecture variations
- Comprehensive result comparison

## 💻 Usage

### Training the Network

#### Basic Training
```bash
# Train on MNIST
python train.py --epochs 5 --batch-size 64 --lr 0.001

# With custom parameters
python train.py --epochs 10 --batch-size 128 --dropout 0.25 --filters1 10 --filters2 20
```

#### Command-line Arguments
```bash
--epochs          Number of training epochs (default: 5)
--batch-size      Training batch size (default: 64)
--lr              Learning rate (default: 0.001)
--dropout         Dropout rate (default: 0.25)
--filters1        First conv layer filters (default: 10)
--filters2        Second conv layer filters (default: 20)
--dense-size      Dense layer size (default: 50)
--save-model      Save trained model
```

### Evaluation

```bash
# Evaluate on test set
python evaluate.py --model-path saved_models/mnist_model.pth

# Test on custom images
python test_custom.py --image-dir custom_digits/ --model-path saved_models/mnist_model.pth
```

### Transfer Learning

```bash
# Train on Greek letters
python transfer_learning.py --pretrained saved_models/mnist_model.pth --epochs 30

# Fine-tune with different learning rate
python transfer_learning.py --pretrained saved_models/mnist_model.pth --lr 0.0001 --epochs 50
```

### Filter Visualization

```bash
# Visualize first layer filters
python visualize_filters.py --model-path saved_models/mnist_model.pth

# Show filter effects on sample image
python show_filter_effects.py --model-path saved_models/mnist_model.pth --image-idx 0
```

### Hyperparameter Experiments

```bash
# Run full grid search
python experiments.py --dataset fashionmnist --filters 10 20 40 --dropout 0.1 0.25 0.5 --dense 50 100
```

## 🏗️ Network Architecture

### CNN Architecture Diagram

```
Input (28×28×1)
      ↓
Conv2d (10 filters, 5×5) → ReLU → MaxPool2d (2×2)
      ↓
Conv2d (20 filters, 5×5) → Dropout → ReLU → MaxPool2d (2×2)
      ↓
Flatten
      ↓
Linear (320 → 50) → ReLU
      ↓
Linear (50 → 10) → Log-Softmax
      ↓
Output (10 classes)
```

### Detailed Layer Specifications

**Layer 1: Convolutional**
```python
nn.Conv2d(1, 10, kernel_size=5)
# Input: 1×28×28
# Output: 10×24×24
# Parameters: 10*(5*5*1 + 1) = 260
```

**MaxPool + Activation**
```python
nn.MaxPool2d(2)
F.relu()
# Output: 10×12×12
```

**Layer 2: Convolutional**
```python
nn.Conv2d(10, 20, kernel_size=5)
# Input: 10×12×12
# Output: 20×8×8
# Parameters: 20*(5*5*10 + 1) = 5,020
```

**Dropout + MaxPool + Activation**
```python
nn.Dropout2d(p=0.25)
nn.MaxPool2d(2)
F.relu()
# Output: 20×4×4
```

**Fully Connected Layers**
```python
nn.Linear(320, 50)
# Parameters: 320*50 + 50 = 16,050

nn.Linear(50, 10)
# Parameters: 50*10 + 10 = 510

# Total Parameters: ~21,840
```

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, filters1=10, filters2=20, dropout=0.25, dense_size=50):
        super(Net, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, filters1, kernel_size=5)
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=5)
        
        # Dropout layer
        self.conv2_drop = nn.Dropout2d(p=dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(filters2 * 4 * 4, dense_size)
        self.fc2 = nn.Linear(dense_size, 10)
    
    def forward(self, x):
        # First conv block
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        
        # Second conv block
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        
        # Flatten
        x = x.view(-1, self.num_flat_features(x))
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
```

## 📊 Task Breakdown

### Task 1: Building and Training the Network

#### A. Data Acquisition
```python
from torchvision import datasets, transforms

# Load MNIST dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1000, shuffle=False)
```

#### B. Training Loop
```python
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]'
                  f'\tLoss: {loss.item():.6f}')

# Training configuration
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, 6):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
```

#### C. Evaluation
```python
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return test_loss, accuracy
```

#### D. Results Visualization
```python
import matplotlib.pyplot as plt

def plot_training_curves(train_losses, test_losses):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, 'r.', label='Test Loss', markersize=10)
    plt.xlabel('Number of training examples seen')
    plt.ylabel('Negative log likelihood loss')
    plt.legend()
    plt.title('Training Progress')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
```

#### E. Custom Image Testing
```python
import cv2
from PIL import Image

def preprocess_custom_image(image_path):
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize to 28x28
    img = cv2.resize(img, (28, 28))
    
    # Normalize (MNIST format: white digits on black background)
    img = img.astype('float32') / 255.0
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    
    return img_tensor

def predict_custom_digits(model, image_dir):
    model.eval()
    
    for img_file in os.listdir(image_dir):
        if img_file.endswith('.png') or img_file.endswith('.jpg'):
            img_path = os.path.join(image_dir, img_file)
            img_tensor = preprocess_custom_image(img_path)
            
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.exp(output[0])
                prediction = output.argmax(dim=1).item()
                
            print(f'{img_file}: Predicted={prediction}, '
                  f'Confidence={probabilities[prediction]:.2%}')
```

### Task 2: Analyzing the Network

#### A. Filter Visualization
```python
def visualize_first_layer_filters(model):
    # Extract first layer weights
    filters = model.conv1.weight.data.cpu().numpy()
    
    # filters shape: [10, 1, 5, 5]
    n_filters = filters.shape[0]
    
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    axes = axes.ravel()
    
    for i in range(n_filters):
        axes[i].imshow(filters[i, 0], cmap='gray')
        axes[i].set_title(f'Filter {i}')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_filters, 12):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('conv1_filters.png')
    plt.show()

# Run visualization
visualize_first_layer_filters(model)
```

**Filter Interpretation:**
- **Edge Detectors**: Filters emphasizing vertical/horizontal transitions (Sobel-like)
- **Corner Detectors**: Filters highlighting corner features
- **Texture Extractors**: Filters capturing diagonal patterns
- **Contrast Enhancers**: Filters responding to brightness changes

#### B. Filter Effect Demonstration
```python
def show_filter_effects(model, image, save_path='filter_effects.png'):
    model.eval()
    
    # Get first layer filters
    filters = model.conv1.weight.data.cpu().numpy()
    
    # Convert image to numpy
    img_np = image.squeeze().numpy()
    
    fig, axes = plt.subplots(5, 4, figsize=(12, 15))
    
    for i in range(10):
        row = i // 2
        col = (i % 2) * 2
        
        # Show filter
        axes[row, col].imshow(filters[i, 0], cmap='gray')
        axes[row, col].set_title(f'Filter {i}')
        axes[row, col].axis('off')
        
        # Apply filter using OpenCV
        with torch.no_grad():
            filtered = cv2.filter2D(img_np, -1, filters[i, 0])
        
        # Show result
        axes[row, col + 1].imshow(filtered, cmap='gray')
        axes[row, col + 1].set_title(f'Output {i}')
        axes[row, col + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
```

**Observations:**
- Vertical edge filters respond strongly to digits with vertical strokes (1, 4, 7)
- Horizontal edge filters activate on digits like 2, 3, 7
- Diagonal filters capture slanted features
- Combined responses create unique feature maps for each digit

### Task 3: Transfer Learning on Greek Letters

#### Implementation
```python
class GreekTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def __call__(self, x):
        return self.transform(x)

def transfer_learning_greek(pretrained_model_path, greek_data_path):
    # Load pre-trained model
    model = Net()
    model.load_state_dict(torch.load(pretrained_model_path))
    
    # Freeze all layers except the last
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace final layer for 3 classes (alpha, beta, gamma)
    model.fc2 = nn.Linear(50, 3)
    
    # Only train the new layer
    optimizer = torch.optim.Adam(model.fc2.parameters(), lr=0.001)
    
    # Load Greek letter dataset
    greek_train = datasets.ImageFolder(
        root=f'{greek_data_path}/train',
        transform=GreekTransform()
    )
    
    greek_loader = DataLoader(greek_train, batch_size=5, shuffle=True)
    
    # Train
    for epoch in range(30):
        train(model, device, greek_loader, optimizer, epoch)
    
    return model
```

**Transfer Learning Advantages:**
- Faster convergence (30 epochs vs 100+ from scratch)
- Better performance with limited data
- Leverages learned low-level features
- Reduces training time significantly

**Results:**
- Training accuracy: 75-80% (limited dataset)
- Shows successful feature transfer
- Greek letters benefit from digit-learned edge detection

## 🔬 Experimental Results

### Task 4: Hyperparameter Optimization on FashionMNIST

#### Experimental Design

**Variables:**
1. **Number of Filters** (Conv Layer 1): 10, 20, 40
2. **Dropout Rate**: 0.1, 0.25, 0.5
3. **Dense Layer Size**: 50, 100

**Total Configurations**: 3 × 3 × 2 = 18 experiments

#### Results Summary

| Filters | Dropout | Dense | Accuracy | Time (s) |
|---------|---------|-------|----------|----------|
| 10 | 0.10 | 50 | 88.55% | 53.14 |
| 10 | 0.10 | 100 | 88.95% | 52.92 |
| 10 | 0.25 | 50 | 88.62% | 52.45 |
| 10 | 0.25 | 100 | 89.05% | 52.64 |
| 10 | 0.50 | 50 | 88.22% | 52.47 |
| 10 | 0.50 | 100 | 88.76% | 52.28 |
| 20 | 0.10 | 50 | 89.41% | 57.11 |
| 20 | 0.10 | 100 | 89.98% | 56.57 |
| 20 | 0.25 | 50 | 90.11% | 56.83 |
| 20 | 0.25 | 100 | **90.53%** | 55.94 |
| 20 | 0.50 | 50 | 89.87% | 55.39 |
| 20 | 0.50 | 100 | 90.01% | 55.37 |
| 40 | 0.10 | 50 | 90.61% | 69.18 |
| 40 | 0.10 | 100 | 90.62% | 67.02 |
| 40 | 0.25 | 50 | 90.86% | 67.23 |
| 40 | 0.25 | 100 | 90.99% | 67.67 |
| 40 | 0.50 | 50 | 90.41% | 66.92 |
| 40 | 0.50 | 100 | **91.19%** | 66.55 |

#### Key Findings

**1. Impact of Filter Count:**
- **10 filters**: 88.22% - 89.05% accuracy
- **20 filters**: 89.41% - 90.53% accuracy (+1.5%)
- **40 filters**: 90.41% - 91.19% accuracy (+2.1%)
- **Conclusion**: More filters → better accuracy, but diminishing returns

**2. Optimal Dropout Rate:**
- For 10 filters: 0.25 works best
- For 20 filters: 0.25 works best
- For 40 filters: 0.5 works best (prevents overfitting in larger models)
- **Conclusion**: Larger models need higher dropout

**3. Dense Layer Size:**
- Marginal improvement (0.2-0.5%) with 100 vs 50 units
- Cost: Minimal additional training time
- **Conclusion**: Worth using 100 units for slight accuracy boost

**4. Training Time:**
- Linear relationship with filter count
- 10 filters: ~52s, 20 filters: ~56s, 40 filters: ~67s
- **Conclusion**: 26% more time for 2.7% more accuracy (good trade-off)

#### Best Configuration
```python
# Optimal hyperparameters for FashionMNIST
best_config = {
    'filters1': 40,
    'filters2': 80,  # Double the first layer
    'dropout': 0.5,
    'dense_size': 100,
    'learning_rate': 0.001,
    'batch_size': 64
}

# Achieves 91.19% test accuracy
```

## 📈 Performance Metrics

### MNIST Results

| Metric | Value |
|--------|-------|
| Training Accuracy | 99.5% |
| Test Accuracy | 98.8% |
| Training Loss (final) | 0.015 |
| Test Loss (final) | 0.041 |
| Training Time | ~5 min (5 epochs) |
| Parameters | 21,840 |

### FashionMNIST Results (Best Model)

| Metric | Value |
|--------|-------|
| Test Accuracy | 91.19% |
| Training Time | 66.55s/epoch |
| Total Parameters | 43,420 |
| Model Size | 170 KB |

### Per-Class Accuracy (MNIST)

| Digit | Accuracy | Common Errors |
|-------|----------|---------------|
| 0 | 99.5% | Confused with 6 |
| 1 | 99.7% | Rare errors |
| 2 | 98.9% | Confused with 7 |
| 3 | 98.7% | Confused with 5, 8 |
| 4 | 99.1% | Confused with 9 |
| 5 | 98.4% | Confused with 3, 6 |
| 6 | 99.0% | Confused with 0, 5 |
| 7 | 98.6% | Confused with 1, 2 |
| 8 | 98.3% | Confused with 3, 5 |
| 9 | 98.8% | Confused with 4, 7 |


## 🎯 Future Enhancements

- [ ] Implement batch normalization
- [ ] Try different optimizers (SGD with momentum, AdamW)
- [ ] Data augmentation (rotation, translation, scaling)
- [ ] Deeper architectures (ResNet, DenseNet)
- [ ] Learning rate scheduling
- [ ] Early stopping with validation set
- [ ] Ensemble methods
- [ ] Model quantization for deployment
- [ ] ONNX export for cross-platform inference
- [ ] Grad-CAM for interpretability
- [ ] Adversarial robustness testing
- [ ] Mobile deployment (TensorFlow Lite)

### Resources
- PyTorch Documentation: https://pytorch.org/docs/
- CS231n Stanford: http://cs231n.stanford.edu/
- Deep Learning Book: https://www.deeplearningbook.org/

