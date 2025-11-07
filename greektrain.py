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
from MNISTtrain import MyNetwork

def build_greek_model(model_path, device):
    model = MyNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    for param in model.parameters():
        param.requires_grad = False

    # Replace last layer and move it to device
    model.fc2 = nn.Linear(model.fc2.in_features, 3).to(device) 
    # model.to(device) 
    return model

def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss/len(loader), correct/len(loader.dataset)

def plot_metric(values, ylabel, fname):
    plt.figure()
    plt.plot(range(1, len(values)+1), values, marker='o')
    plt.xlabel('Epoch'); plt.ylabel(ylabel)
    plt.savefig(fname); plt.close()

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()

    # Use the original device selection argument as provided
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Running on device: {device}")
    model = build_greek_model(args.model_path, device)
    print(model)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: transforms.functional.rgb_to_grayscale(x)),
        transforms.Lambda(lambda x: transforms.functional.affine(x, 0, (0,0), 36/128, 0)),
        transforms.CenterCrop((28,28)),
        transforms.Lambda(lambda x: transforms.functional.invert(x)),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=5, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc2.parameters(), lr=1e-3)

    losses, accs = [], []
    for epoch in range(1, args.epochs+1):
        loss, acc = train(model, loader, criterion, optimizer, device)
        losses.append(loss); accs.append(acc)
        print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.2%}")

    plot_metric(losses, 'Loss', 'greek_training_loss.png')
    plot_metric(accs, 'Accuracy', 'greek_training_acc.png')
    torch.save(model.state_dict(), 'greek_model.pth')
    print("Saved Greek model to greek_model.pth")

if __name__ == '__main__':
    main(sys.argv)
