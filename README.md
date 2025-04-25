# Image Similarity Checker

A deep learning-based web application that classifies images into predefined categories and finds similar images from a dataset using PyTorch and Flask.

## Project Overview

This application uses a convolutional neural network (CNN) trained on the CIFAR-10 dataset to classify uploaded images into one of ten categories (airplane, automobile, bird, cat, dog, frog, horse, ship, truck). When a user uploads an image, the model predicts its class and returns a similar image from the training dataset.

## Features

- Image classification using a custom CNN architecture
- Web interface for easy image uploading
- Display of matched similar images
- Support for common image formats
- Responsive design for desktop and mobile devices

## Repository Structure

```
├── .gitattributes              # Git LFS configuration
├── .gitignore                  # Files to be ignored by Git
├── GetImageLabel.py            # Script to generate labels for images
├── GetTrainingPack.py          # Script to extract and save CIFAR-10 images
├── load_model.py               # Flask server for the web application
├── model_configuration.py      # Model architecture and dataset classes
├── train_model.py              # Script for training the CNN model
├── templates/                  # HTML templates
│   └── index.html              # Main web interface
├── static/                     # Static files (images, CSS, etc.)
│   └── matched/                # Directory containing matched images (gitignored)
├── best_model.pth              # Trained model weights (generated after training)
└── labels.pkl                  # Pickle file containing image labels
```

## Prerequisites

- Python 3.6+
- PyTorch
- Flask
- Torchvision
- PIL (Pillow)
- tqdm

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Image-Similarity-Checker.git
   cd Image-Similarity-Checker
   ```

2. Install dependencies:
   ```
   pip install torch torchvision Flask Pillow tqdm
   ```

3. Prepare the dataset:
   ```
   python GetTrainingPack.py
   ```
   This will download the CIFAR-10 dataset and save images to the `cifar_images` directory.

4. Generate labels for the images:
   ```
   python GetImageLabel.py
   ```
   This will create a `labels.pkl` file with mappings between images and their class labels.

5. Train the model:
   ```
   python train_model.py
   ```
   This will train the CNN model and save the best-performing version as `best_model.pth`.

## Running the Application

1. Start the Flask server:
   ```
   python load_model.py
   ```

2. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

3. Upload an image through the web interface to see the classification result and a matched similar image.

## Model Architecture

The project includes two model architectures:

1. **Basic Model** (`MyModel`): A simple CNN with one convolutional layer.
2. **Improved Model** (`ImprovedModel`): A deeper CNN with three convolutional layers, batch normalization, and dropout for better performance.

The improved model is used by default and achieves higher accuracy on the test set.

## Customization

- To use your own dataset, modify the `GetTrainingPack.py` script to load your images
- To adjust model parameters, edit the architecture in `model_configuration.py`
- To change the web interface appearance, modify `templates/index.html`

## License

[MIT License](LICENSE)

## Contact

Ryan Zhao - ryannayr.zhao@mail.utoronto.ca
