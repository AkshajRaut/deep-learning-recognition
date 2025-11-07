# Akshaj Raut
# CS 5330 Computer Vision
# Spring 2025
# Project 5 - Recognition using Deep Networks

import sys
import argparse
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
from MNISTtrain import MyNetwork

def show_filters(weights):
    with torch.no_grad():
        fig, axes = plt.subplots(3, 4, figsize=(8, 6))
        for i, ax in enumerate(axes.flatten()):
            if i < weights.shape[0]:
                filt = weights[i, 0].detach().cpu().numpy()
                ax.imshow(filt, cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis('off')
        plt.tight_layout()
        plt.savefig('conv1_filters.png')
        plt.close()

def apply_filters(image, filters):
    with torch.no_grad():
        img_np = image.squeeze().cpu().numpy().astype(np.float32)
        results = []
        for i in range(filters.shape[0]):
            kernel = filters[i, 0].detach().cpu().numpy().astype(np.float32)
            filtered = cv2.filter2D(img_np, -1, kernel)
            results.append(filtered)
        return results

def show_filter_results(results):
    with torch.no_grad():
        fig, axes = plt.subplots(3, 4, figsize=(8, 6))
        for i, ax in enumerate(axes.flatten()):
            if i < len(results):
                ax.imshow(results[i], cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis('off')
        plt.tight_layout()
        plt.savefig('conv1_filterResults.png')
        plt.close()

def show_combined(filters, results):
    with torch.no_grad():
        num_filters = len(results)
        fig, axes = plt.subplots(5, 4, figsize=(8, 10))
        axes = axes.flatten()
        
        for i in range(num_filters):
            left_index = 2 * i
            right_index = 2 * i + 1
            
            # Display filter
            filt = filters[i, 0].detach().cpu().numpy()
            axes[left_index].imshow(filt, cmap='gray')
            axes[left_index].set_title(f"Filter {i}")
            axes[left_index].set_xticks([])
            axes[left_index].set_yticks([])
            
            # Display filtered image
            axes[right_index].imshow(results[i], cmap='gray')
            axes[right_index].set_title(f"Output {i}")
            axes[right_index].set_xticks([])
            axes[right_index].set_yticks([])
        
        for idx in range(len(axes) - 2 * num_filters):
            axes[-(idx + 1)].axis('off')

        plt.tight_layout()
        plt.savefig('conv1_combined.png')
        plt.close()

def load_mnist_image(img_index):
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    image, _ = train_ds[img_index]
    return image

def load_custom_image(file_path):
    # Open, convert to grayscale, resize to 28x28, and transform to tensor
    img = Image.open(file_path).convert("L")
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    return transform(img)

def main(argv):
    parser = argparse.ArgumentParser(description="Examine MNIST CNN first layer filters with OpenCV")
    parser.add_argument('--model-path', required=True, help="Path to the trained model file")
    parser.add_argument('--img-index', type=int, default=0, help="Index of MNIST image to use")
    parser.add_argument('--custom-img', type=str, help="Path to a custom image file (overrides MNIST image)")
    args = parser.parse_args()

    # Use the original device selection argument as provided
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Running on device: {device}")
    model = MyNetwork().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()  # Ensure evaluation mode (e.g., for dropout behavior)
    print(model)

    with torch.no_grad():
        conv1_weights = model.conv1.weight
        print(f"conv1 weight tensor shape: {conv1_weights.shape}")
        print(f"First filter weights:\n{conv1_weights[0,0]}")

    # Visualize the conv1 filters in a 3x4 grid
    show_filters(conv1_weights)

    # Load image: either a custom image (if provided) or an MNIST image by index
    if args.custom_img:
        print(f"Loading custom image from {args.custom_img}")
        image = load_custom_image(args.custom_img)
    else:
        print(f"Loading MNIST image at index {args.img_index}")
        image = load_mnist_image(args.img_index)
    
    results = apply_filters(image, conv1_weights)

    # Visualize the filtered images in a 3x4 grid
    show_filter_results(results)

    # Display a combined view: each filter next to its filtered output in a grid
    show_combined(conv1_weights, results)

if __name__ == '__main__':
    main(sys.argv)