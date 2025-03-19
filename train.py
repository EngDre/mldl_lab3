# Import necessary libraries
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import requests
from zipfile import ZipFile
from io import BytesIO
import numpy as np

# Define the path to the dataset
dataset_path = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'  # Replace with the path to your dataset

# Send a GET request to the URL
response = requests.get(dataset_path)
# Check if the request was successful
if response.status_code == 200:
    # Open the downloaded bytes and extract them
    with ZipFile(BytesIO(response.content)) as zip_file:
        zip_file.extractall('/dataset')
    print('Download and extraction complete!')

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()  # Convert PIL Image to PyTorch Tensor
])

# Load the dataset
tiny_imagenet_dataset_train = ImageFolder(root='/dataset/tiny-imagenet-200/train', transform=transform)
tiny_imagenet_dataset_test = ImageFolder(root='/dataset/tiny-imagenet-200/test', transform=transform)

# Create a DataLoader
dataloader_train = DataLoader(tiny_imagenet_dataset_train, batch_size=64, shuffle=True)
dataloader_test = DataLoader(tiny_imagenet_dataset_test, batch_size=64, shuffle=True)

# Determine the number of classes and samples
num_classes = len(tiny_imagenet_dataset_train.classes)
num_samples = len(tiny_imagenet_dataset_train)

print(f'Number of classes: {num_classes}')
print(f'Number of samples: {num_samples}')

# Function to denormalize image for visualization
def denormalize(image):
    image = image.to('cpu').numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image

# Visualize one example for each class for 10 classes
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
classes_sampled = []
found_classes = 0

for i, (inputs, classes) in enumerate(dataloader_train):
  for j in range(inputs.shape[0]):
        if tiny_imagenet_dataset_train.classes[classes[j]] not in classes_sampled:
            # This block will be executed if the condition in the 'if' statement is True
            classes_sampled.append(tiny_imagenet_dataset_train.classes[classes[j]])
            row = found_classes // 5
            col = found_classes % 5
            # The error was here, trying to access axes[2, 0] which is out of bounds
            # Since axes has shape (2, 5), row can only be 0 or 1.
            # The fix is to break the loop once we've filled all 10 subplots (2 rows * 5 cols)
            if row >= 2:  # Check if row index is out of bounds before accessing axes
                break
            axes[row, col].imshow(denormalize(inputs[j]))
            axes[row, col].set_title(tiny_imagenet_dataset_train.classes[classes[j]])
            axes[row, col].axis('off')
            found_classes += 1
        if found_classes == 10:
            break  # Break the outer loop as well once 10 classes are found
  if found_classes == 10:
      break # Break the outer loop as well once 10 classes are found
plt.show()
