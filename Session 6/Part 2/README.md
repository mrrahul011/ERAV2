# Part 2

### How many layers
The neural network consists of 7 layers.

## Layer Descriptions

### Layer 1
- **Convolutional layer**: `self.conv1_1`
- **ReLU activation**: `self.relu1_1`
- **Batch normalization**: `self.bn1_1`
- **Convolutional layer**: `self.conv1_2`
- **ReLU activation**: `self.relu1_2`
- **Batch normalization**: `self.bn1_2`
- **Convolutional layer**: `self.conv1_3`
- **ReLU activation**: `self.relu1_3`
- **Batch normalization**: `self.bn1_3`
- **Max pooling**: `self.maxpool1`
- **Dropout**: `self.dropout1`

### Layer 2
- **Convolutional layer**: `self.conv2_1`
- **ReLU activation**: `self.relu2_1`
- **Batch normalization**: `self.bn2_1`
- **Convolutional layer**: `self.conv2_2`
- **ReLU activation**: `self.relu2_2`
- **Batch normalization**: `self.bn2_2`
- **Convolutional layer**: `self.conv2_3`
- **ReLU activation**: `self.relu2_3`
- **Batch normalization**: `self.bn2_3`
- **Max pooling**: `self.maxpool2`
- **Dropout**: `self.dropout2`

### Layer 3
- **Convolutional layer**: `self.conv3_1`
- **ReLU activation**: `self.relu3_1`
- **Batch normalization**: `self.bn3_1`
- **Convolutional layer**: `self.conv3_2`
- **ReLU activation**: `self.relu3_2`
- **Convolutional layer**: `self.conv3_3`
- **ReLU activation**: `self.relu3_3`
- **Convolutional layer**: `self.conv3_4`
- **Dropout**: Used at the end

### Layer 4
- **Average pooling**: `F.avg_pool2d`

### Layer 5
- **Fully connected layer**: `self.fc`

### Layer 6
- **Reshape**: `x.view(-1, 10)`

### Layer 7
- **Log softmax activation**: `F.log_softmax`

## Additional Details
- **Receptive Field**: 43
- **SoftMax**: Log_softmax at output
- **Kernels**: 3x3 kernel
- **Batch Normalization**: After every convolution operation, but not used in Layer 3, towards the end
- **Position of MaxPooling**: Max pooling layer at the Layer 1 and Layer 2. In Layer 3, 3x3 convolution without padding.
- **DropOut**: Used in Layer 1 and Layer 2 but not in Layer 3

