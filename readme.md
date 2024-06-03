## Data Preparation

### Splitting the Data

The dataset is split into training, validation, and testing sets to ensure the model trains effectively and generalizes well on unseen data. I use a script to divide the data into a 70% training set, 15% validation set, and 15% testing set. This process shuffles the data and ensures that each subset contains a representative distribution of each font class, which is crucial for maintaining the integrity of my model's evaluation.

### Data Transformation and DataLoader Setup

To prepare the images for training and validation, I apply a series of transformations to standardize and augment mydataset. This ensures that my model learns to generalize well and not just memorize the specific details of the training images. The transformations include:

- Resizing all images to 128x128 pixels.
- Randomly flipping images horizontally to introduce variability.
- Randomly rotating images by up to 20 degrees to simulate different orientations.
- Converting images to tensors for model processing.
- Normalizing images with a predefined mean and standard deviation, which aligns with common practice for image data.

For the DataLoader setup:

- **Training DataLoader**: Shuffles the data to prevent the model from learning any order-dependent patterns in the data, which helps to improve the generalization capabilities of the model.
- **Validation DataLoader**: Does not shuffle the data, as the order does not affect validation performance but ensures consistent results across different runs.

## Model Architecture

My model, `FontClassifierCNN`, utilizes a convolutional neural network (CNN) designed to accurately classify different fonts from images. Below is a description of the model's architecture:

### Convolutional Layers

- **Conv1**: First convolutional layer with 16 filters of size 5x5 and padding set to 2. This is followed by ReLU activation and a 2x2 max pooling operation.
- **Conv2**: Second convolutional layer with 32 filters of size 5x5 and padding set to 2, followed by ReLU activation and a 2x2 max pooling.
- **Conv3**: Third convolutional layer with 64 filters of size 3x3 and padding set to 1, followed by ReLU activation and a 2x2 max pooling.
- **Conv4**: Fourth convolutional layer with 128 filters of size 3x3 and padding set to 1, followed by ReLU activation and a 2x2 max pooling.
- **Conv5**: Fifth convolutional layer with 256 filters of size 3x3 and padding set to 1, followed by ReLU activation and a 2x2 max pooling.

### Fully Connected Layers

- **FC1**: First fully connected layer with 4096 input features, reduced to 1024 features, coupled with ReLU activation.
- **FC2**: Second fully connected layer reduces the feature size from 1024 to 128, using ReLU activation.
- **FC3**: The final fully connected layer maps 128 features to 10 output classes, corresponding to the 10 different fonts we aim to classify.

### Forward Pass

During the forward pass, the image data is processed through sequential convolutional layers, which are designed to extract and amplify features relevant to font recognition. After passing through multiple pooling and ReLU activation layers to reduce dimensionality and introduce non-linearity, the data is flattened and passed through fully connected layers. The output from the final layer provides the classification scores for each font.

This architecture is tailored to capture the intricate details necessary for distinguishing between various font styles, ensuring high accuracy and robustness in font classification.


## Model Training

The training process involves several techniques to optimize performance and ensure stable convergence:

- **Model Architecture**: I use a custom convolutional neural network (CNN) designed for font classification, termed `FontClassifierCNN`.
- **Loss Function**: I employ CrossEntropyLoss, which is well-suited for multi-class classification tasks, ensuring that my model accurately learns to distinguish between different font types.
- **Optimizer**: The Adam optimizer is used with an initial learning rate of 0.001. Adam is chosen for its effectiveness in handling sparse gradients on noisy problems.

### Training Enhancements

- **Learning Rate Scheduler**: A decaying learning rate is applied to address significant fluctuations in the loss towards the final training steps, helping the model converge smoothly towards the end of training.
- **Gradient Clipping**: To prevent the exploding gradients problem, I clip gradients to a maximum norm of 1.0, which helps in faster and more stable convergence.

### Training Process

The model is trained over 100 epochs with the following steps per epoch:
1. **Training Phase**:
   - Each batch of the training data is forwarded through the model to compute the loss.
   - Backpropagation is performed, and model weights are updated.
   - Training loss is logged to monitor progress.

2. **Validation Phase**:
   - The model is set to evaluation mode (`.eval()`), which disables dropout and batch normalization layers for consistent performance metrics.
   - Validation data is used to compute loss and accuracy, providing insight into how well the model generalizes to unseen data.

### Performance Monitoring

- Loss and accuracy are recorded for each epoch, both to trace training progress and to intervene if needed (e.g., early stopping if validation performance degrades).

## Model Evaluation

To assess the performance of my trained model, `FontClassifierCNN`, on unseen data, we conduct an evaluation using the test dataset. This dataset is not used during the training process and provides an objective measure of how well my model generalizes.

### Test Dataset and DataLoader

The test dataset is prepared with the same image transformations as the training and validation datasets. It is loaded using a `DataLoader` that processes the images in batches of 32 without shuffling, as the order does not impact the performance evaluation.

### Evaluation Process

The model loaded with weights from the best-performing model during validation (`cnn_grad_clipped.pth`) is set to evaluation mode. This disables layers like dropout and batch normalization that behave differently during training versus testing.

We use a loop to process each batch from the test DataLoader, where:
- **Images** are forwarded through the model to generate predictions.
- **Predictions** are compared to the actual labels to determine correctness.

Gradient computations are disabled (`torch.no_grad()` context) to reduce memory usage and speed up the computations, as gradients are not needed for model evaluation.

### Accuracy Calculation

Accuracy is calculated by comparing the predicted labels against the true labels across all test samples:
- **Correct Predictions**: Number of times the model predictions match the labels.
- **Total Predictions**: Total number of images processed.

The formula used is:

$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}} \times 100$

### Result

After running the evaluation, the accuracy of the model on the test data is 95.24%

## Submission
The submission model is provided at the path

```
model/cnn_grad_clipped.pth
```



