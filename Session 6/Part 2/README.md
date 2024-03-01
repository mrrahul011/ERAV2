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

## Explanation

### Layer 1
This layer starts with three successive convolutional operations (`self.conv1_1`, `self.conv1_2`, `self.conv1_3`), each followed by a rectified linear unit (ReLU) activation function and batch normalization. These operations extract various features from the input images. The max pooling operation (`self.maxpool1`) reduces the spatial dimensions of the features while retaining the most important information. Dropout (`self.dropout1`) is applied to prevent overfitting by randomly setting a fraction of input units to zero during training.

### Layer 2
Similar to Layer 1, this layer consists of three convolutional operations (`self.conv2_1`, `self.conv2_2`, `self.conv2_3`) followed by ReLU activation, batch normalization, max pooling, and dropout. These operations further extract and refine features from the previously processed data.

### Layer 3
Layer 3 begins with two convolutional operations (`self.conv3_1`, `self.conv3_2`) followed by ReLU activation and batch normalization. However, in this layer, there is no batch normalization after the second convolution. Instead, a third convolutional operation (`self.conv3_3`) is applied directly after the second one. The final convolutional operation (`self.conv3_4`) does not have batch normalization and is followed by dropout.

### Layer 4
An average pooling operation (`F.avg_pool2d`) is applied to the output of Layer 3. Average pooling reduces the spatial dimensions of the features to a single value, effectively summarizing the information across the entire feature map.

### Layer 5
A fully connected layer (`self.fc`) is used to connect the output of the previous layer to the output classes. This layer performs a linear transformation on the input data, mapping it to the output space.

### Layer 6
A reshape operation (`x.view(-1, 10)`) reshapes the output of the fully connected layer into the desired output shape, which typically corresponds to the number of classes in the classification task.

### Layer 7
The final layer applies a log softmax activation (`F.log_softmax`) to the output of the previous layer. This activation function computes the logarithm of the softmax function, which normalizes the output into a probability distribution over the output classes.

## Additional Details

- **Receptive Field**: The receptive field of the network is 43, indicating the spatial extent of the input that influences a particular neuron in the output.
- **SoftMax**: Log softmax is applied at the output layer to produce probabilities for each class.
- **Kernels**: Convolutional layers use 3x3 kernels to convolve over the input data.
- **Batch Normalization**: Batch normalization is applied after every convolution operation except in Layer 3, where it is omitted towards the end.
- **Position of MaxPooling**: Max pooling layers are placed after each group of convolutional operations in Layer 1 and Layer 2. In Layer 3, a 3x3 convolution without padding is used instead of max pooling.
- **DropOut**: Dropout is utilized in Layer 1 and Layer 2 but not in Layer 3 to prevent overfitting during training.


