# README.md

# Feedforward Neural Network MNIST Project

This repository contains code for training and evaluating feedforward neural networks on the MNIST dataset using Python. It includes feature extraction, data preprocessing, visualization, and model saving/loading utilities.

## Project Structure

The project is divided into two main parts:
1. **Simple Neural Network**: Implementation of a basic neural network from scratch with a single hidden layer, including its own test script.
2. **Feedforward Neural Network**: Implementation of a feedforward neural network from scratch with multiple hidden layers, along with its test script.

## Directory Structure

```
your_project/
│
├── docs/
│   └── Documentation.pdf
│
├── examples/
│   ├── model_nn/
│   │   ├── mnist_nn_model.pkl
│   │   └── pca_transformer.pkl
│   │
│   ├── models_FNN/
│   │   ├── mnist_nnm_model.pkl
│   │   └── pca_transformer.pkl
│   │
│   ├── test_FNN.ipynb
│   └── test_nn.ipynb
│
├── NeuralNetwork/
│   ├── __init__.py
│   ├── Feedforward_Neural_Network.py
│   └── NeuralNetwork.py
│
├── LICENSE
├──
```

## Getting Started

### Prerequisites

Install dependencies using:

```sh
pip install -r requirements.txt
```

### Files

- [`src/Feedforward Neural Network.py`](src/Feedforward%20Neural%20Network.py): Main feedforward neural network implementation and XOR example.
- [`src/NeuralNetwork.py`](src/NeuralNetwork.py): Simple neural network class (single hidden layer) and utilities.
- [`examples/test FNN.ipynb`](examples/test%20FNN.ipynb): Jupyter notebook for MNIST training, feature extraction (PCA), visualization, and evaluation of the feedforward neural network.
- [`examples/test nn.ipynb`](examples/test%20nn.ipynb): Jupyter notebook for testing the simple neural network.
- [`examples/models_FNN/mnist_nnm_model.pkl`](examples/models_FNN/mnist_nnm_model.pkl): Saved trained model fnn.
-  [`examples/models_FNN/pca_transformer.pkl`](examples/models_FNN/pca_transformer.pkl): Saved PCA transformer from fnn.
-  [`examples/model_nn/mnist_nn_model.pkl`](examples/model_nn/mnist_nn_model.pkl): Saved trained model nn.
- [`examples/model_nn/pca_transformer.pkl`](examples/model_nn/pca_transformer.pkl): Saved PCA transformer from nn.

## Usage

1. Run the notebooks in [`examples/`](examples/) for step-by-step training and evaluation.
2. Use the scripts in [`src/`](src/) for custom experiments or integration.

## Features

- Data loading and preprocessing (normalization, flattening, scaling)
- Feature extraction using PCA
- One-hot encoding for categorical targets
- Model training and evaluation (accuracy, MSE, confusion matrix, classification report)
- Visualization of sample images and data distribution
- Model saving and loading with robust error handling

## Documentation

See [`docs/Documentation.pdf`](docs/Documentation.pdf) for detailed documentation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file
