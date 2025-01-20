# MNIST Digit Classification Project 🖊️

This comprehensive project analyzes the MNIST dataset, featuring 70,000 handwritten digits, using various machine learning and deep learning models to predict digit classes with high accuracy. The models explored include K-Nearest Neighbors (KNN), Support Vector Machines (SVM), Random Forests (RF), and Convolutional Neural Networks (CNN).

## Features

- **Data Preprocessing:** Utilized `NumPy` and `Keras` for loading and preprocessing data, transforming images into normalized one-dimensional arrays suitable for model input.
- **Model Implementations:**
  - **KNN:** Achieved 97.05% accuracy; optimal for its simplicity and effectiveness on clear, distinct images.
  - **SVM:** Reached up to 98.26% accuracy with RBF kernel, highlighting its strength in high-dimensional spaces.
  - **Random Forest:** Consistently strong performer with a best accuracy of 97.12%, demonstrating robustness with an ensemble method.
  - **CNN:** Best model with a 99.55% accuracy, showcasing deep learning's capability in image recognition tasks.
  
## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/MNISTClassification.git
   ```

2. Navigate to the project directory:
   ```bash
   cd MNISTClassification
   ```

3. Install required libraries:
   ```bash
   pip install numpy matplotlib keras scikit-learn
   ```

## Usage

1. Load the Jupyter Notebook:
   ```bash
   jupyter notebook ProjectFinal.ipynb
   ```
2. Run the cells sequentially to train and evaluate models.

## Dataset

- **Training Data:** 60,000 grayscale images (28x28 pixels) of handwritten digits.
- **Testing Data:** 10,000 images used for model evaluation.

## Results

- **Highest Accuracy:** CNN model achieved a peak accuracy of 99.55% on the test set, validating its effectiveness in digit recognition.
- **Model Comparison:** Deep learning models, particularly CNN, outperformed traditional machine learning algorithms on image classification tasks.

## Skills Demonstrated

- Machine Learning and Deep Learning
- Image Preprocessing
- Model Optimization and Evaluation
- Performance Metrics Analysis

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
