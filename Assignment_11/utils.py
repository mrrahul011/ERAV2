
"""Module to define utility functions for the project."""
import os

import torch


def get_device():
    """
    Function to get the device to be used for training and testing.
    """

    # Check if cuda is available
    cuda = torch.cuda.is_available()

    # Based on check enable cuda if present, if not available
    if cuda:
        final_choice = "cuda"
    else:
        final_choice = "cpu"

    # pylint: disable=E1101
    return final_choice, torch.device(final_choice)


def get_num_workers(model_run_location):
    """Given a run mode, return the number of workers to be used for data loading."""

    # calculate the number of workers
    num_workers = (os.cpu_count() - 1) if os.cpu_count() > 3 else 2

    # If run_mode is local, use only 2 workers
    num_workers = num_workers if model_run_location == "colab" else 0

    return num_workers


def get_correct_prediction_count(prediction, label):
    """
    Function to get the count of correct predictions.
    """
    return prediction.argmax(dim=1).eq(label).sum().item()


# Function to save the model
# https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/
def save_model(epoch, model, optimizer, scheduler, batch_size, criterion, file_name):
    """
    Function to save the trained model along with other information to disk.
    """
    # print(f"Saving model from epoch {epoch}...")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "batch_size": batch_size,
            "loss": criterion,
        },
        file_name,
    )


# Given a list of train_losses, train_accuracies, test_losses,
# test_accuracies, loop through epoch and print the metrics
def pretty_print_metrics(num_epochs, results):
    """
    Function to print the metrics in a pretty format.
    """
    # Extract train_losses, train_acc, test_losses, test_acc from results
    train_losses = results["train_loss"]
    train_acc = results["train_acc"]
    test_losses = results["test_loss"]
    test_acc = results["test_acc"]

    for i in range(num_epochs):
        print(
            f"Epoch: {i+1:02d}, Train Loss: {train_losses[i]:.4f}, "
            f"Test Loss: {test_losses[i]:.4f}, Train Accuracy: {train_acc[i]:.4f}, "
            f"Test Accuracy: {test_acc[i]:.4f}"
        )




# Needed for image transformations
import albumentations as A

# # Needed for padding issues in albumentations
# import cv2
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

# Use precomputed values for mean and standard deviation of the dataset
CIFAR_MEAN = (0.4915, 0.4823, 0.4468)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CUTOUT_SIZE = 16

# Create class labels and convert to tuple
CIFAR_CLASSES = tuple(
    c.capitalize()
    for c in [
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
)


class CIFAR10Transforms(Dataset):
    """Apply albumentations augmentations to CIFAR10 dataset"""

    # Given a dataset and transformations,
    # apply the transformations and return the dataset
    def __init__(self, dataset, transforms):
        self.transforms = transforms
        self.dataset = dataset

    def __getitem__(self, idx):
        # Get the image and label from the dataset
        image, label = self.dataset[idx]

        # Apply transformations on the image
        image = self.transforms(image=np.array(image))["image"]

        return image, label

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return (
            f"CIFAR10Transforms(dataset={self.dataset}, transforms={self.transforms})"
        )

    def __str__(self):
        return (
            f"CIFAR10Transforms(dataset={self.dataset}, transforms={self.transforms})"
        )


def apply_cifar_image_transformations(mean, std, cutout_size):
    """
    Function to apply the required transformations to the MNIST dataset.
    """
    # Apply the required transformations to the MNIST dataset
    train_transforms = A.Compose(
        [
            # normalize the images with mean and standard deviation from the whole dataset
            # https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Normalize
            # # transforms.Normalize(cifar_mean, cifar_std),
            A.Normalize(mean=list(mean), std=list(std)),
            # RandomCrop 32, 32 (after padding of 4)
            # https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.PadIfNeeded
            # MinHeight and MinWidth are set to 36 to ensure that the image is padded to 36x36 after padding
            # border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            # cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            # Default: cv2.BORDER_REFLECT_101
            #A.PadIfNeeded(min_height=36, min_width=36),
            # https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.RandomCrop
            #A.RandomCrop(32, 32),
            # CutOut(8, 8)
            # # https://albumentations.ai/docs/api_reference/augmentations/dropout/cutout/#albumentations.augmentations.dropout.cutout.Cutout
            # # Because we normalized the images with mean and standard deviation from the whole dataset, the fill_value is set to the mean of the dataset
            A.Cutout(
                 num_holes=1, max_h_size=cutout_size, max_w_size=cutout_size, p=1.0
             ),
            # https://albumentations.ai/docs/api_reference/augmentations/dropout/coarse_dropout/#coarsedropout-augmentation-augmentationsdropoutcoarse_dropout
          
            #A.CoarseDropout(
            #    max_holes=1,
            #    max_height=cutout_size,
            #   max_width=cutout_size,
            #    min_holes=1,
            #    min_height=cutout_size,
            #    min_width=cutout_size,
            #    p=1.0,
            #),
        
            # Convert the images to tensors
            # # transforms.ToTensor(),
            ToTensorV2(),
        ]
        )

    # Test data transformations
    test_transforms = A.Compose(
        # Convert the images to tensors
        # normalize the images with mean and standard deviation from the whole dataset
        [
            A.Normalize(mean=list(mean), std=list(std)),
            # Convert the images to tensors
            ToTensorV2(),
        ]
    )

    return train_transforms, test_transforms

def calculate_mean_std(dataset):
    """Function to calculate the mean and standard deviation of CIFAR dataset"""
    data = dataset.data.astype(np.float32) / 255.0
    mean = np.mean(data, axis=(0, 1, 2))
    std = np.std(data, axis=(0, 1, 2))
    return mean, std


def split_cifar_data(data_path, train_transforms, test_transforms):
    """
    Function to download the MNIST data
    """
    # print("Downloading CIFAR10 dataset\n")
    # Download CIFAR dataset
    train_data = datasets.CIFAR10(data_path, train=True, download=True)
    test_data = datasets.CIFAR10(data_path, train=False, download=True)

    # Calculate and print the mean and standard deviation of the dataset
    #mean, std = calculate_mean_std(train_data)
    # print(f"\nMean: {mean}")
    # print(f"Std: {std}")

    # Apply transforms on the dataset
    # Use the above class to apply transforms on the dataset using albumentations
    train_data = CIFAR10Transforms(train_data, train_transforms)
    test_data = CIFAR10Transforms(test_data, test_transforms)

    # print("Transformations applied on the dataset")

    return train_data, test_data



import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def convert_back_image(image):
    """Using mean and std deviation convert image back to normal"""
    cifar10_mean = (0.4914, 0.4822, 0.4471)
    cifar10_std = (0.2469, 0.2433, 0.2615)
    image = image.numpy().astype(dtype=np.float32)

    for i in range(image.shape[0]):
        image[i] = (image[i] * cifar10_std[i]) + cifar10_mean[i]

    # To stop throwing a warning that image pixels exceeds bounds
    image = image.clip(0, 1)

    return np.transpose(image, (1, 2, 0))


def plot_sample_training_images(batch_data, batch_label, class_label, num_images=30):
    """Function to plot sample images from the training data."""
    images, labels = batch_data, batch_label

    # Calculate the number of images to plot
    num_images = min(num_images, len(images))
    # calculate the number of rows and columns to plot
    num_cols = 5
    num_rows = int(np.ceil(num_images / num_cols))

    # Initialize a subplot with the required number of rows and columns
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))

    # Iterate through the images and plot them in the grid along with class labels

    for img_index in range(1, num_images + 1):
        plt.subplot(num_rows, num_cols, img_index)
        plt.tight_layout()
        plt.axis("off")
        plt.imshow(convert_back_image(images[img_index - 1]))
        plt.title(class_label[labels[img_index - 1].item()])
        plt.xticks([])
        plt.yticks([])

    return fig, axs


def plot_train_test_metrics(results):
    """
    Function to plot the training and test metrics.
    """
    # Extract train_losses, train_acc, test_losses, test_acc from results
    train_losses = results["train_loss"]
    train_acc = results["train_acc"]
    test_losses = results["test_loss"]
    test_acc = results["test_acc"]

    # Plot the graphs in a 1x2 grid showing the training and test metrics
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Loss plot
    axs[0].plot(train_losses, label="Train")
    axs[0].plot(test_losses, label="Test")
    axs[0].set_title("Loss")
    axs[0].legend(loc="upper right")

    # Accuracy plot
    axs[1].plot(train_acc, label="Train")
    axs[1].plot(test_acc, label="Test")
    axs[1].set_title("Accuracy")
    axs[1].legend(loc="upper right")

    return fig, axs


def plot_misclassified_images(data, class_label, num_images=10):
    """Plot the misclassified images from the test dataset."""
    # Calculate the number of images to plot
    num_images = min(num_images, len(data["ground_truths"]))
    # calculate the number of rows and columns to plot
    num_cols = 5
    num_rows = int(np.ceil(num_images / num_cols))

    # Initialize a subplot with the required number of rows and columns
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))

    # Iterate through the images and plot them in the grid along with class labels

    for img_index in range(1, num_images + 1):
        # Get the ground truth and predicted labels for the image
        label = data["ground_truths"][img_index - 1].cpu().item()
        pred = data["predicted_vals"][img_index - 1].cpu().item()
        # Get the image
        image = data["images"][img_index - 1].cpu()
        # Plot the image
        plt.subplot(num_rows, num_cols, img_index)
        plt.tight_layout()
        plt.axis("off")
        plt.imshow(convert_back_image(image))
        plt.title(f"""ACT: {class_label[label]} \nPRED: {class_label[pred]}""")
        plt.xticks([])
        plt.yticks([])

    return fig, axs


# Function to plot gradcam for misclassified images using pytorch_grad_cam
def plot_gradcam_images(
    model,
    data,
    class_label,
    target_layers,
    device,
    targets=None,
    num_images=10,
    image_weight=0.25,
):
    """Show gradcam for misclassified images"""

    # Flag to enable cuda
    use_cuda = device == "cuda"

    # Calculate the number of images to plot
    num_images = min(num_images, len(data["ground_truths"]))
    # calculate the number of rows and columns to plot
    num_cols = 5
    num_rows = int(np.ceil(num_images / num_cols))

    # Initialize a subplot with the required number of rows and columns
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))

    # Initialize the GradCAM object
    # https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/grad_cam.py
    # https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/base_cam.py
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)

    # Iterate through the images and plot them in the grid along with class labels
    for img_index in range(1, num_images + 1):
        # Extract elements from the data dictionary
        # Get the ground truth and predicted labels for the image
        label = data["ground_truths"][img_index - 1].cpu().item()
        pred = data["predicted_vals"][img_index - 1].cpu().item()
        # Get the image
        image = data["images"][img_index - 1].cpu()

        # Get the GradCAM output
        # https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/utils/model_targets.py
        grad_cam_output = cam(
            input_tensor=image.unsqueeze(0),
            targets=targets
            # aug_smooth=True,
            # eigen_smooth=True,
        )
        grad_cam_output = grad_cam_output[0, :]

        # Overlay gradcam on top of numpy image
        overlayed_image = show_cam_on_image(
            convert_back_image(image),
            grad_cam_output,
            use_rgb=True,
            image_weight=image_weight,
        )

        # Plot the image
        plt.subplot(num_rows, num_cols, img_index)
        plt.tight_layout()
        plt.axis("off")
        plt.imshow(overlayed_image)
        plt.title(f"""ACT: {class_label[label]} \nPRED: {class_label[pred]}""")
        plt.xticks([])
        plt.yticks([])
    return fig, axs


import torchinfo

def model_summary(model, input_size, device,):
    """Define a function to print the model summary."""

    # https://github.com/TylerYep/torchinfo
    torchinfo.summary(
        model,
        input_size=input_size,
        batch_dim=0,
        col_names=(
            "input_size",
            "kernel_size",
            "output_size",
            "num_params",
            "trainable",
        ),
        device=device,
        verbose=1,
        col_width=16,
    )