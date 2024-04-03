# Assignment 10

## Introduction
This repository contains the implementation of Assignment 10. It includes the LRFinder implementation, training logs, and network implementation for the CIFAR-10 classification task.

## Files
- Assignment10.ipynb: Main file containing the LRFinder implementation and training logs.
- data_loader.py: Python file for downloading CIFAR-10 data and applying transformations.
- network.py: Python file containing the network architecture implementation.
- train_test.py: Python file containing the train and test loops.

## Suggested LR
The suggested learning rate (LR) for training is 2.31E-03.

## LRMIN and LRMAX
- LRMIN: 2.31E-04
- LRMAX: 2.31E-03

## Network Architecture
Input
|
|--> [Conv2d(3, 64, kernel_size=3, stride=1, padding=1)]
|--> [BatchNorm2d(64)]
|--> [ReLU(inplace=True)]
|   
|--> [Conv2d(64, 128, kernel_size=3, stride=1, padding=1)]
|--> [MaxPool2d(kernel_size=2, stride=2)]
|--> [BatchNorm2d(128)]
|--> [ReLU(inplace=True)]
|   
|--> Residual Block
|       |--> [Conv2d(128, 128, kernel_size=3, stride=1, padding=1)]
|       |--> [BatchNorm2d(128)]
|       |--> [ReLU(inplace=True)]
|       |--> [Conv2d(128, 128, kernel_size=3, stride=1, padding=1)]
|       |--> [BatchNorm2d(128)]
|   
|--> [Conv2d(128, 256, kernel_size=3, stride=1, padding=1)]
|--> [MaxPool2d(kernel_size=2, stride=2)]
|--> [BatchNorm2d(256)]
|--> [ReLU(inplace=True)]
|   
|--> [Conv2d(256, 512, kernel_size=3, stride=1, padding=1)]
|--> [MaxPool2d(kernel_size=2, stride=2)]
|--> [BatchNorm2d(512)]
|--> [ReLU(inplace=True)]
|   
|--> Residual Block
|       |--> [Conv2d(512, 512, kernel_size=3, stride=1, padding=1)]
|       |--> [BatchNorm2d(512)]
|       |--> [ReLU(inplace=True)]
|       |--> [Conv2d(512, 512, kernel_size=3, stride=1, padding=1)]
|       |--> [BatchNorm2d(512)]
|   
|--> [MaxPool2d(kernel_size=4, stride=4)]
|--> [Flatten()]
|--> [Linear(512, 10)]
|
Output


## Train and Test
- train_test.py contains the train and test loops.
- Accuracy at the 24th epoch is 90.48.

