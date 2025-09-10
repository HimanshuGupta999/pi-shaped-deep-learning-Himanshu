# Recurrent Neural Networks (CNN) 

## Project Overview

This repository contains a Jupyter Notebook (CNN.ipynb) that builds and trains a Convolutional Neural Network (CNN) to classify images from the Fashion-MNIST dataset into 10 clothing categories (e.g., t-shirt, trouser, sneaker, bag). The project demonstrates preprocessing, model construction, training, and evaluation of a CNN for image classification tasks.

### Methodology

The project follows these key steps:
1.  **Data Loading**: The Fashion-MNIST dataset is loaded directly from TensorFlow/Keras, containing 60,000 training and 10,000 test grayscale images of size 28×28 pixels across 10 categories.
2.  **Preprocessing**: 
- Pixel values are normalized to the range [0,1] to stabilize training.
- Images are reshaped to (28, 28, 1) to include the channel dimension for CNN input.
- Labels are kept as integers (0–9) for simplicity and used with sparse categorical crossentropy.
3.  **Model Building**: 
- A CNN is constructed using TensorFlow/Keras Sequential API.
- Layers include convolutional layers for feature extraction, max pooling for downsampling, dense layers for classification, and a softmax output for 10-class prediction.
- Dropout regularization is applied to reduce overfitting.
4.  **Training**: 
- The model is trained for 50 epochs with a batch size of 128 using the Adam optimizer.
- Training progress is monitored with validation accuracy and loss on the test set.
5.  **Evaluation**: 
- Model performance is assessed on the test dataset using overall accuracy.
- A **confusion matrix and classification report** provide detailed per-class performance metrics.
- Training history plots (accuracy & loss curves) help visualize learning behavior and detect overfitting.

### How to Run This Project

1.  **Prerequisites**: Make sure you have Python installed, along with the following libraries:
    *   numpy
    *   pandas
    *   scikit-learn
    *   tensorflow
    *   matplotlib
    *   jupyter
    *   seaborn

    You can install them using pip:
    ```bash
    pip install -r requirements.txt    
    ```

2.  **Execution**: Open and run the `CNN.ipynb` file in a Jupyter environment. The notebook is self-contained and includes explanations for each step.

## Core Concepts Questions

### 1. What advantages do CNNs have over traditional fully connected neural networks for image data?
CNNs use convolutional filters to capture spatial patterns, requiring fewer parameters than fully connected networks. This makes them more efficient, less prone to overfitting, and better at detecting local features like edges and textures.

### 2. What is the role of convolutional filters/kernels in a CNN?
Filters (kernels) scan across the image to detect patterns such as edges, shapes, or textures. Each filter learns a specific feature, and stacking filters builds hierarchical feature representations.

### 3. Why do we use pooling layers, and what is the difference between MaxPooling and AveragePooling?
Pooling reduces the size of feature maps, lowering computation cost and improving generalization.
- **MaxPooling:** selects the strongest (maximum) feature in a region.
- **AveragePooling:** computes the average of features in a region.

### 4. Why is normalization of image pixels important before training?  
Normalization scales pixel values (0–255 → 0–1), ensuring inputs are on the same scale. This speeds up convergence, stabilizes training, and prevents exploding/vanishing gradients. 

### 5. How does the softmax activation function work in multi-class classification?
Softmax converts raw scores (logits) into probabilities that sum to 1. Each output neuron represents the probability of a class, and the highest probability determines the predicted label.

### 6. What strategies can help prevent overfitting in CNNs? (e.g., dropout, data augmentation)
- **Dropout:** randomly disables neurons during training.
- **Data Augmentation:** creates new samples by flipping, rotating, or cropping images.
- **Regularization (L2 weight decay).**
- **Early Stopping:** halts training when validation performance stops improving.
- **Batch Normalization:** stabilizes training and reduces overfitting.  

### 7. What does the confusion matrix tell you about model performance?
It shows how many samples were correctly or incorrectly classified per class. It helps identify which classes are confused with each other (e.g., shirts vs t-shirts).

### 8. If you wanted to improve the CNN, what architectural or data changes would you try?   
- Add more convolutional + pooling layers.
- Use Batch Normalization.
- Apply stronger data augmentation.
- Experiment with different optimizers or learning rates.
- Try advanced architectures like ResNet, VGG, or MobileNet.