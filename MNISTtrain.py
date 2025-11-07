# Akshaj Raut
# CS 5330 Computer Vision
# Spring 2025
# Project 5 - Recognition using Deep Networks 

import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class MyNetwork(nn.Module):
    """A simple CNN for MNIST classification."""
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.dropout = nn.Dropout(0.25)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(20 * 4 * 4, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # Forward pass through convolutional and pooling layers, then fully connected layers
        x = torch.relu(self.pool(self.conv1(x)))
        x = torch.relu(self.pool(self.dropout(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return torch.log_softmax(self.fc2(x), dim=1)

def main(argv):
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--out', default='mnist_model.pth')
    args = parser.parse_args()

    # Determine device (using accelerator if available)
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Running on device: {device}")
    transform = transforms.ToTensor()

    # Prepare MNIST dataset for training and testing
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, args.batch_size, shuffle=False)

    # Build the model, optimizer, and loss function
    model = MyNetwork().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.NLLLoss()

    train_loss_vals = []
    train_x_vals = []
    test_loss_vals = []
    test_x_vals = []
    total_examples_seen = 0 

    # Training loop: train the model and evaluate on the test set after each epoch
    for epoch in range(1, args.epochs + 1):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_examples_seen += len(imgs)
            train_loss_vals.append(loss.item())
            train_x_vals.append(total_examples_seen)

        # Evaluate on the test set
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                output = model(imgs)
                test_loss += criterion(output, labels).item()
        test_loss /= len(test_loader)
        test_loss_vals.append(test_loss)
        test_x_vals.append(total_examples_seen)

        print(f"Epoch {epoch}: Last-batch train_loss={loss.item():.4f}, test_loss={test_loss:.4f}")

    # Plot and save the training and test loss graph
    plt.figure()
    plt.plot(train_x_vals, train_loss_vals, 'c-', label='Train Loss')
    plt.plot(test_x_vals, test_loss_vals, 'ro', label='Test Loss')
    plt.xlabel('Number of training examples seen')
    plt.ylabel('Negative log likelihood loss')
    plt.legend()
    plt.savefig('train_test_loss.png')
    plt.close()

    # Save the trained model
    torch.save(model.state_dict(), args.out)
    print(f"Saved model to {args.out}")

if __name__ == "__main__":
    main(sys.argv)
