# Akshaj Raut
# CS 5330 Computer Vision
# Spring 2025
# Project 5 - Recognition using Deep Networks 

import sys
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from MNISTtrain import MyNetwork

def evaluate_first_ten(model, device, loader):
    """
    Evaluate the first 10 test images:
    - Print 10 network output values (formatted to 2 decimal places)
    - Print the index of the max output and the true label for each image
    - Plot the first 9 images in a 3x3 grid with the predicted label above each image
    """
    model.eval()
    with torch.no_grad():
        # Retrieve one batch of test images
        images, labels = next(iter(loader))
        images, labels = images.to(device), labels.to(device)
        # Process only the first 10 images
        outputs = model(images[:10]).cpu().exp()  # Using exp() to convert log probabilities to probabilities

    for i in range(10):
        out_values = [f"{v:.2f}" for v in outputs[i]]
        predicted_index = outputs[i].argmax().item()
        print(f"Image {i}: {out_values}  Pred={predicted_index}  True={labels[i].item()}")

    # Plot the first 9 images in a 3x3 grid with predictions
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i].cpu().squeeze(), cmap='gray')
        ax.set_title(f"Pred: {outputs[i].argmax().item()}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('plot-predictions.png')
    plt.close()

def main(argv):
    # Use the original device selection argument as provided
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Running on device: {device}")

    # Load the trained model and set it to evaluation mode
    model = MyNetwork().to(device)
    model.load_state_dict(torch.load('mnist_model.pth', map_location=device))

    # Create the MNIST test dataset loader
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Evaluate the first 10 examples from the test set
    evaluate_first_ten(model, device, test_loader)

if __name__ == "__main__":
    main(sys.argv)