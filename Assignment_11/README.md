# README

## Overview

This repository contains code for training and testing ResNet models (ResNet18 and ResNet34) using the CIFAR-10 dataset. The main files in the repository include:

- [resnet.py](https://github.com/mrrahul011/ERAV2/blob/main/Assignment_11/model/resnet.py)
- [main.py](https://github.com/mrrahul011/ERAV2/blob/main/Assignment_11/main.py)
- [utils.py](https://github.com/mrrahul011/ERAV2/blob/main/Assignment_11/utils.py)

The code is structured to facilitate the training and testing process, as well as provide utility functions for data loading and visualization.

## File Descriptions

### [resnet.py](https://github.com/mrrahul011/ERAV2/blob/main/Assignment_11/model/resnet.py)

This file contains the definitions of the ResNet18 and ResNet34 models.

- **ResNet18**: Implements the ResNet18 architecture.
- **ResNet34**: Implements the ResNet34 architecture.

### [main.py](https://github.com/mrrahul011/ERAV2/blob/main/Assignment_11/main.py)

This file contains the main functions for training and testing the model using CIFAR-10 data.

- **get_cifar_dataloaders()**: Returns the train and test dataloaders.
- **train_model()**: Trains the model given the optimizer and criterion.
- **test_model()**: Tests the model given the criterion.
- **train_and_test_model()**: Trains and tests the model.
- **find_optimal_lr()**: Uses the learning rate finder to determine the optimal starting learning rate.

### [utils.py](https://github.com/mrrahul011/ERAV2/blob/main/Assignment_11/utils.py)

This file contains utility functions to support training and testing.

- **get_device()**: Detects and returns the correct device (GPU or CPU).
- **get_correct_prediction_count()**: Returns the cumulative correct prediction count for a set of predictions and labels.
- **save_model()**: Saves the model, epoch, optimizer, scheduler, loss, and batch size.
- **split_cifar_data()**: Downloads and splits the CIFAR-10 data into test and train sets.
- **apply_cifar_image_transformations()**: Creates train and test image transformations compatible with Albumentations.
- **convert_back_image()**: Converts a normalized image back to its original form using mean and standard deviation for each channel.
- **plot_sample_training_images()**: Plots sample training images along with the labels.
- **plot_train_test_metrics()**: Plots training and testing metrics.
- **plot_misclassified_images()**: Plots incorrectly classified images with ground truth and predicted classes.
- **plot_gradcam_images()**: Shows Grad-CAM for misclassified images.

## How to Run

To run the code, download the [Assignment11.ipynb](https://github.com/mrrahul011/ERAV2/blob/main/Assignment_11/Assignment11.ipynb) notebook. Then, run the notebook and all necessary modules will be automatically imported. Follow the instructions in the notebook for training and testing the model.

