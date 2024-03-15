
## Assignment_7.ipynb

This notebook contains the implementation of different models along with their analysis and optimization steps.

## MODEL_1.py

### Target:
1. Get the setup ready
2. Set the transforms
3. Set the data loader
4. Set basic working code ready
5. Set basic training and testing loop code ready
6. Plot the graph

### Results
1. Parameters: 6.3 Million
2. Best Training Accuracy: 99.6
3. Best Test Accuracy: 99.34

### Analysis
1. Very large number of parameters
2. Model cannot be further trained to achieve accuracy above 99.4, since gap is very less between train and test accuracy.
3. Model is working, didn't observe overfitting, but too large model
4. Need to change the model

## MODEL_2.py

### Target:
1. Reduce the number of parameters
2. Fix the base model
3. Avoid overfitting

### Results
1. Parameters: 1.23 Million
2. Best Training Accuracy: 99.28
3. Best Test Accuracy: 99.24

### Analysis
1. Model is still large
2. No overfitting
3. Reduce parameters in the next stage

## MODEL_3.py

### Target:
1. Reduce the number of parameters

### Results
1. Parameters: 11,380
2. Best Training Accuracy: 99.36
3. Best Test Accuracy: 98.76

### Analysis
1. Model is still large
2. No overfitting
3. Add batch normalization and regularization in the next stage.
4. Add global average pooling

## MODEL_4.py

### Target:
1. Add batch normalization, regularization, and GAP

### Results
1. Parameters: 6,636
2. Best Training Accuracy: 99.38
3. Best Test Accuracy: 99.10

### Analysis
1. Model parameter under 8k
2. No overfitting
3. Could reach 99.4 if further trained

