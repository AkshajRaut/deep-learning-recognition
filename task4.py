# Akshaj Raut
# CS 5330 Computer Vision
# Spring 2025
# Project 5 - Recognition using Deep Networks
# Experimentation with network architecture on Fashion MNIST

import sys
import time
import argparse
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ParameterizedCNN(nn.Module):
    """
    A parameterized convolutional neural network.
    
    Architecture:
      - Conv1: 1 input channel, 'filters1' output channels, kernel size 5
      - Conv2: 'filters1' input channels, 'filters1 * 2' output channels, kernel size 5
      - Dropout after Conv2 with given dropout rate
      - MaxPool layers (kernel size 2) after each conv layer
      - Fully connected layer with 'dense_units' nodes
      - Final fully connected layer to 10 output classes with log_softmax activation
    """
    def __init__(self, filters1=10, dropout_rate=0.25, dense_units=50):
        super(ParameterizedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, filters1, kernel_size=5)
        self.conv2 = nn.Conv2d(filters1, filters1 * 2, kernel_size=5)
        self.dropout = nn.Dropout(dropout_rate)
        self.pool = nn.MaxPool2d(2)
        # After conv and pooling:
        # Image size: 28 -> conv1: (28-5+1)=24 -> pool: 12
        # conv2: (12-5+1)=8 -> pool: 4. So the flattened dimension = filters1*2 * 4 * 4.
        self.fc1 = nn.Linear(filters1 * 2 * 4 * 4, dense_units)
        self.fc2 = nn.Linear(dense_units, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.dropout(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

def train_model(model, device, train_loader, optimizer, criterion, epochs):
    model.train()
    total_epoch_loss = 0
    for epoch in range(epochs):
        epoch_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        total_epoch_loss += avg_loss
    return total_epoch_loss / epochs

def evaluate_model(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            test_loss += criterion(output, labels).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
    avg_loss = test_loss / len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    return avg_loss, accuracy

def main(argv):
    parser = argparse.ArgumentParser(description="Experiment with CNN architecture on Fashion MNIST")
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train each model')
    parser.add_argument('--batch-size', type=int, default=64, help='Training batch size')
    args = parser.parse_args(argv[1:])
    
    # Use the original device selection argument as provided
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Running on device: {device}")
    
    # Prepare Fashion MNIST data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Define parameter grid
    filters1_options = [10, 20, 40]
    dropout_options = [0.1, 0.25, 0.5]
    dense_units_options = [50, 100]
    
    results = []
    combo_idx = 0
    
    for filters1, dropout_rate, dense_units in itertools.product(filters1_options, dropout_options, dense_units_options):
        combo_idx += 1
        print(f"Experiment {combo_idx}: filters1={filters1}, dropout_rate={dropout_rate}, dense_units={dense_units}")
        
        model = ParameterizedCNN(filters1=filters1, dropout_rate=dropout_rate, dense_units=dense_units).to(device)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.NLLLoss()
        
        start_time = time.time()
        train_loss = train_model(model, device, train_loader, optimizer, criterion, args.epochs)
        elapsed_time = time.time() - start_time
        
        test_loss, test_accuracy = evaluate_model(model, device, test_loader, criterion)
        
        results.append({
            'filters1': filters1,
            'dropout_rate': dropout_rate,
            'dense_units': dense_units,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'training_time': elapsed_time
        })
        print(f" -> Test Accuracy: {test_accuracy*100:.2f}%, Training Time: {elapsed_time:.2f} sec\n")
    
    # Summarize results
    print("Final Experiment Results:")
    print("Idx\tFilt1\tDropout\tDense\tTrainLoss\tTestLoss\tTestAcc\tTime(sec)")
    for i, res in enumerate(results, start=1):
        print(f"{i}\t{res['filters1']}\t{res['dropout_rate']}\t{res['dense_units']}\t"
              f"{res['train_loss']:.4f}\t\t{res['test_loss']:.4f}\t\t{res['test_accuracy']*100:.2f}%\t{res['training_time']:.2f}")

if __name__ == "__main__":
    main(sys.argv)