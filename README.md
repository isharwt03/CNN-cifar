# CNN-cifar
# CNN CIFAR-10 Image Classification Project

## Project Overview
Convolutional Neural Network (CNN) for classifying images from the CIFAR-10 dataset, comprising 10 object classes.

## Dataset
- CIFAR-10: 60,000 32x32 color images
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 50,000 training images, 10,000 test images

## Model Architecture
- Convolutional Layers: 3-4 layers
- Activation: ReLU
- Pooling: Max pooling
- Dropout: Regularization
- Final Layer: Softmax classification

## Requirements
- Python 3.8+
- TensorFlow/Keras
- NumPy
- Matplotlib
- scikit-learn

## Performance Metrics
- Accuracy: ~85-90%
- Validation Split: 20%
- Training Epochs: 50-100

## Key Files
- `model.py`: Neural network architecture
- `train.py`: Training script
- `evaluate.py`: Model evaluation
- `predict.py`: Inference on new images

## Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train.py

# Evaluate model
python evaluate.py

# Predict on new image
python predict.py --image path/to/image.jpg
```

## Model Training
1. Data preprocessing
2. Data augmentation
3. Model compilation
4. Training with early stopping
5. Performance visualization

## Results Visualization
- Confusion matrix
- Training/validation loss curves
- Classification report
