
# Needed for image transformations
import albumentations as A

# # Needed for padding issues in albumentations
# import cv2
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from utils import apply_cifar_image_transformations, split_cifar_data
from model.resnet import ResNet18 as Net
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


def calculate_mean_std(dataset):
    """Function to calculate the mean and standard deviation of CIFAR dataset"""
    data = dataset.data.astype(np.float32) / 255.0
    mean = np.mean(data, axis=(0, 1, 2))
    std = np.std(data, axis=(0, 1, 2))
    return mean, std


def get_cifar_dataloaders(data_path, batch_size, num_workers, seed):
    """Get the final train and test data loaders"""

    ## Data Transformations
    train_transforms, test_transforms = apply_cifar_image_transformations(
        mean=CIFAR_MEAN, std=CIFAR_STD, cutout_size=CUTOUT_SIZE
    )

    # print(f"Train and test data path: {data_path}")

    # Train and Test data
    # print("Splitting the dataset into train and test\n")
    train_data, test_data = split_cifar_data(
        data_path, train_transforms, test_transforms
    )

    # To be passed to dataloader
    def _init_fn(worker_id):
        np.random.seed(int(seed))

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_init_fn,
    )

    # print(f"Dataloader arguments: {dataloader_args}\n")
    # print("Creating train and test dataloaders\n")
    # train dataloader
    train_loader = DataLoader(train_data, **dataloader_args)

    # test dataloader
    test_loader = DataLoader(test_data, **dataloader_args)

    return train_loader, test_loader




"""Module to define the train and test functions."""

# from functools import partial

import torch
from utils import get_correct_prediction_count, save_model
from torch_lr_finder import LRFinder
from tqdm import tqdm

############# Train and Test Functions #############


def train_model(model, device, train_loader, optimizer, criterion):
    """
    Function to train the model on the train dataset.
    """

    # Initialize the model to train mode
    model.train()

    # Initialize progress bar
    pbar = tqdm(train_loader)

    # Reset the loss and correct predictions for the epoch
    train_loss = 0
    correct = 0
    processed = 0

    # Iterate over the train loader
    for batch_idx, (data, target) in enumerate(pbar):
        # Move data and labels to device
        data, target = data.to(device), target.to(device)
        # Clear the gradients for the optimizer to avoid accumulation
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss for the batch
        loss = criterion(pred, target)
        # Update the loss
        train_loss += loss.item()

        # Backpropagation to calculate the gradients
        loss.backward()
        # Update the weights
        optimizer.step()

        # Get the count of correct predictions
        correct += get_correct_prediction_count(pred, target)
        processed += len(data)

        # Update the progress bar
        # msg = f"Progress:\tBatch = {batch_idx} "
        msg = f"Train: Loss={loss.item():0.4f}, Batch={batch_idx}, Accuracy={100*correct/processed:0.2f}"
        pbar.set_description(desc=msg)

    # Close the progress bar
    pbar.close()

    # Return the final loss and accuracy for the epoch
    current_train_accuracy = 100 * correct / processed
    current_train_loss = train_loss / len(train_loader)

    # print(f"Training:\tAverage Loss: {current_train_loss:.5f}\tAccuracy: {current_train_accuracy:.2f}%")

    return current_train_accuracy, current_train_loss


def test_model(
    model,
    device,
    test_loader,
    criterion,
    misclassified_image_data,
    save_incorrect_predictions=False,
):
    """
    Function to test the model on the test dataset.
    """

    # Initialize the model to evaluation mode
    model.eval()

    # Reset the loss and correct predictions for the epoch
    test_loss = 0
    correct = 0

    # Disable gradient calculation while testing
    with torch.no_grad():
        for data, target in test_loader:
            # Move data and labels to device
            data, target = data.to(device), target.to(device)

            # Predict using model
            output = model(data)
            # Calculate loss for the batch
            test_loss += criterion(output, target).item()

            # Get the index of the max log-probability
            pred = output.argmax(dim=1)
            # Check if the prediction is correct
            correct_mask = pred.eq(target)
            # Save the incorrect predictions
            incorrect_indices = ~correct_mask

            # Do this only for last epoch, if not you will run out of memory
            if save_incorrect_predictions:
                # Store images incorrectly predicted, generated predictions and the actual value
                misclassified_image_data["images"].extend(data[incorrect_indices])
                misclassified_image_data["ground_truths"].extend(target[incorrect_indices])
                misclassified_image_data["predicted_vals"].extend(pred[incorrect_indices])

            # Get the count of correct predictions
            correct += get_correct_prediction_count(output, target)

    # Calculate the final loss
    test_loss /= len(test_loader.dataset)

    # Return the final loss and accuracy for the epoch
    current_test_accuracy = 100.0 * correct / len(test_loader.dataset)
    current_test_loss = test_loss

    # Print the final test loss and accuracy
    # print(
    #     f"Testing:\tAverage Loss: {current_test_loss:.5f}\tAccuracy: {current_test_accuracy:.2f}%",
    # )
    print(
        f"Test set: Average loss: {current_test_loss:.4f}, ",
        f"Accuracy: {current_test_accuracy:.2f}%",
    )

    # Return the final loss and accuracy for the epoch
    return current_test_accuracy, current_test_loss


def train_and_test_model(
    batch_size,
    num_epochs,
    model,
    device,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    scheduler,
    misclassified_image_data,
):
    """Trains and tests the model by iterating through epochs"""

    print(f"\n\nBatch size: {batch_size}, Total epochs: {num_epochs}\n\n")

    # Hold the results for every epoch
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # Run the model for NUM_EPOCHS
    for epoch in range(1, num_epochs + 1):
        # Print the current epoch
        print(f"Epoch {epoch}")

        # Train the model
        epoch_train_accuracy, epoch_train_loss = train_model(model, device, train_loader, optimizer, criterion)

        # Should we save the incorrect predictions for this epoch?
        # Do this only for the last epoch, if not you will run out of memory
        if epoch == num_epochs:
            save_incorrect_predictions = True
        else:
            save_incorrect_predictions = False

        # Test the model
        epoch_test_accuracy, epoch_test_loss = test_model(
            model,
            device,
            test_loader,
            criterion,
            misclassified_image_data,
            save_incorrect_predictions,
        )

        # Append the train and test accuracies and losses
        results["train_loss"].append(epoch_train_loss)
        results["train_acc"].append(epoch_train_accuracy)
        results["test_loss"].append(epoch_test_loss)
        results["test_acc"].append(epoch_test_accuracy)

        # Check if the accuracy is the best accuracy till now
        # Save the model if you get the best test accuracy
        if max(results["test_acc"]) == epoch_test_accuracy:
            # print("Saving the model as best test accuracy till now is achieved!")
            save_model(
                epoch,
                model,
                optimizer,
                scheduler,
                batch_size,
                criterion,
                file_name="model_best_epoch.pth",
            )

        # # Passing the latest test loss in list to scheduler to adjust learning rate
        # scheduler.step(test_losses[-1])
        scheduler.step()
        # # # Line break before next epoch
        print("\n")

    return results


def find_optimal_lr(model, optimizer, criterion, train_loader):
    """Use LR Finder to find the best starting learning rate"""

    # https://github.com/davidtvs/pytorch-lr-finder
    # https://github.com/davidtvs/pytorch-lr-finder#notes
    # https://github.com/davidtvs/pytorch-lr-finder/blob/master/torch_lr_finder/lr_finder.py

    # Create LR finder object
    lr_finder = LRFinder(model, optimizer, criterion)
    lr_finder.range_test(train_loader=train_loader, end_lr=10, num_iter=100)
    # https://github.com/davidtvs/pytorch-lr-finder/issues/88
    _, suggested_lr = lr_finder.plot(suggest_lr=True)
    lr_finder.reset()
    # plot.figure.savefig("LRFinder - Suggested Max LR.png")

    return suggested_lr