# Font Classifier Project Documentation


## How to Run the Project for Submission

The model for submission is saved at 
```
model/Resnet_best.pth
```
To load the model, run:
```
import torch 
from augmentation import augs
from torchvision import models
transform = augs()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 10)
model_path = 'model/Resnet_best.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
test_dataset = datasets.ImageFolder(root='your_data_path', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

To run the final results from scratch, if you have not split the data into train,validation, and test dataset., run 
```
split_data.py
```
Then run the Resnet.ipynb to train the pretrained ResNet-50 Model and make predictions on sample_eval_data. 


## Overview
This document outlines the development process, architectural decisions, and evaluation of the Font Classifier project, aimed at classifying different fonts from images using deep learning techniques.

## Data Preparation

The data preparation for the Font Classifier project consists of two main steps: generating synthetic data and splitting the data into appropriate sets for training, validation, and testing. Below, I describe these steps in detail, utilizing custom Python scripts.

### Generating Synthetic Data (`generate_fonts.py`)


To create a diverse and extensive dataset, the `generate_fonts.py` script generates synthetic images of text using various fonts. This process ensures a broad representation of styles, improving the model's generalization capabilities.

#### Key Features of the Script:
- **Random Text**: Generates random strings of varying lengths, including letters, digits, and punctuation, with a preference for more spaces to mimic natural text spacing.
- **Font Handling**: Utilizes TrueType fonts (.ttf files) located in a designated `fonts` folder, generating a specified number of images for each font.
- **Dynamic Text Sizing**: Adjusts font size dynamically to ensure the text fits within the image dimensions, starting from 50% of the image height and decreasing as necessary.
- **Image Customization**: Adds deep black shadows along the edges of the images to introduce variations and complexities that the model might encounter in real-world scenarios, this is inspired from the sample_eval_data given. 


#### Output:
The script saves the generated images in a structured directory format, with separate folders for each font type, containing the respective images.

### Splitting Data (`split_data.py`)

Once a substantial dataset is generated, it is crucial to split this data into training, validation, and test sets to ensure robust model training and evaluation.

#### Process:
- **Directory Setup**: Ensures that directories for training, validation, and test sets are created for each class (font type).
- **Shuffling and Splitting**: Shuffles the images within each class and splits them according to predefined ratios: 70% for training, 15% for validation, and 15% for testing.
- **File Management**: Moves files into the respective directories and cleans up the original dataset directory to maintain a tidy environment.

#### Result:
This script organizes the data into a structure that is conducive to the training process, facilitating straightforward access and management during model training and evaluation.

By combining these scripts, the project efficiently manages a large volume of synthetic data, simulating a variety of fonts and text scenarios, thus enhancing the classifier’s ability to recognize and differentiate between font styles effectively.


## Model Architecture and Training

### Initial CNN Model

#### Model Description
The initial model, `FontClassifierCNN`, was a custom convolutional neural network designed specifically for font classification. It featured several convolutional and pooling layers, followed by fully connected layers aimed at reducing the input dimensions to classify 10 different font types.

### Loss Function

The choice of CrossEntropyLoss as the loss function is pivotal for the following reasons:

- **Suitability for Classification**: CrossEntropyLoss is specifically designed for classification tasks where each instance is expected to belong to a single class out of multiple classes. This makes it ideal for the font classification task, which involves distinctly categorizing each image into one of several font types.
- **Probabilistic Interpretation**: This loss function measures the performance of a classification model whose output is a probability value between 0 and 1. CrossEntropyLoss penalizes the probability based on the distance from the actual label, promoting more confident and accurate predictions.

#### Training and Performance
The CNN was trained using a similar data augmentation strategy to introduce robustness against overfitting. However, despite optimizations, it achieved a maximum accuracy of 48% on the validation set. This performance indicated potential underfitting and limitations in the model’s capacity to capture the complex features necessary for accurate font classification.

#### Challenges Encountered
- **Complexity Limitations**: The model might not have had sufficient depth or complexity to learn the nuanced differences between various fonts.
- **Overfitting to Noise**: Despite augmentation, the model might have overfitted to noise within the training data rather than generalizing from it.

#### Lessons Learned
The experience with the CNN model highlighted the need for a more complex and capable architecture, leading to the adoption of ResNet-50, which substantially improved the classification accuracy.

### ResNet-50 Model Setup


#### Pre-trained Model
Leveraging a pre-trained ResNet-50 provided a robust starting point due to its prior training on a large dataset.


#### Modifications
The final fully connected layer was replaced to adapt to my 10-class font classification problem.

#### Optimizer
The SGD optimizer was used with a learning rate of 0.001 and momentum of 0.9.

#### Loss Function
CrossEntropyLoss was used for its effectiveness in multi-class classification.


### Training Process
Training was conducted over 15 epochs with real-time augmentation applied to the input data to enhance generalization:

1. **Epoch Execution**: Each epoch involved a forward pass, loss computation, backpropagation, and parameter updates.
2. **Validation Accuracy**: Post each epoch, validation accuracy was computed to monitor performance and adjust training strategies if necessary.


The model achieved an accuracy of approximately 82% on the validation set by the end of the training process, demonstrating significant improvements over the initial model.

## Conclusion
The implementation of the ResNet-50 model has markedly enhanced my project's capability to accurately classify fonts from images. The architectural advantages of ResNet, combined with strategic training approaches, have culminated in a robust model that significantly outperforms the initial setup. The result has achieved 96% accuracy on the test data provided.
