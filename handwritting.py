# Akshaj Raut
# CS 5330 Computer Vision
# Spring 2025
# Project 5 - Recognition using Deep Networks

import sys
import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from MNISTtrain import MyNetwork

def preprocess_image(image_path):
    """
    Load an image, convert it to grayscale, resize it to 28x28, and normalize it 
    using MNIST statistics. Assumes images are already in the MNIST format (white digits on black background).
    """
    img = Image.open(image_path).convert("L")
    pipeline = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    img_tensor = pipeline(img)
    # No inversion is done because the images are already white on black.
    normalizer = transforms.Normalize((0.1307,), (0.3081,))
    return normalizer(img_tensor).unsqueeze(0)

def test_handwritten_digits(model, device, directory):
    """
    Loads images from a directory, predicts using the trained model, and prints the output probabilities,
    predicted label, and true label. Also plots the first 9 images with their predicted labels.
    """
    file_list = sorted(os.listdir(directory))
    tensor_collection = []
    true_labels = []
    
    for file_name in file_list:
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Assume file names are like "3.png", where '3' is the true label.
            label = int(os.path.splitext(file_name)[0])
            img_tensor = preprocess_image(os.path.join(directory, file_name)).to(device)
            tensor_collection.append(img_tensor)
            true_labels.append(label)
    
    model.eval()
    with torch.no_grad():
        predictions = torch.cat([model(tensor) for tensor in tensor_collection]).exp()

    # Print predictions for each image.
    for i, pred in enumerate(predictions):
        prob_str = " ".join(f"{p:.2f}" for p in pred)
        predicted_label = pred.argmax().item()
        print(f"Image {i}: [{prob_str}]  Predicted={predicted_label}  True={true_labels[i]}")

    # Plot the first 9 images in a 3x3 grid with predicted labels.
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for idx, ax in enumerate(axes.flatten()):
        if idx < len(tensor_collection):
            image_disp = tensor_collection[idx].cpu().squeeze()
            ax.imshow(image_disp, cmap="gray")
            ax.set_title(f"Pred: {predictions[idx].argmax().item()}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("plot-handwritten-predictions.png")
    plt.close()

def main(argv):
    parser = argparse.ArgumentParser(description="Evaluate MNIST model on handwritten digit images")
    parser.add_argument("--model-path", required=True, help="Path to the trained model file")
    parser.add_argument("--image-dir", required=True, help="Directory containing handwritten digit images")
    args = parser.parse_args(argv[1:])

    # Use the original device selection argument as provided
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Running on device: {device}")
    model = MyNetwork().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    test_handwritten_digits(model, device, args.image_dir)

if __name__ == "__main__":
    main(sys.argv)