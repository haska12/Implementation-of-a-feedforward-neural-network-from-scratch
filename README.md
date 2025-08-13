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
│   └── Simple_Neural_Network.py
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

<<<<<<< HEAD
- [`NeuralNetwork/Simple_Neural_Network.py`](NeuralNetwork/Simple_Neural_Network.py): Simple neural network class (single hidden layer) and utilities.
- [`NeuralNetwork/Feedforward_Neural_Network.py`](NeuralNetwork/Feedforward_Neural_Network.py): Main feedforward neural network implementation and XOR example.
- [`examples/test_FNN.ipynb`](examples/test_FNN.ipynb): Jupyter notebook for MNIST training, feature extraction (PCA), visualization, and evaluation of the feedforward neural network.
- [`examples/test_nn.ipynb`](examples/test_nn.ipynb): Jupyter notebook for testing the simple neural network.
- [`examples/models_FNN/mnist_nnm_model.pkl`](examples/models_FNN/mnist_nnm_model.pkl): Saved trained feedforward neural network model.
- [`examples/models_FNN/pca_transformer.pkl`](examples/models_FNN/pca_transformer.pkl): Saved PCA transformer for feedforward neural network.
- [`examples/model_nn/mnist_nn_model.pkl`](examples/model_nn/mnist_nn_model.pkl): Saved trained simple neural network model.
- [`examples/model_nn/pca_transformer.pkl`](examples/model_nn/pca_transformer.pkl): Saved PCA transformer for simple neural network.
=======
- [`src/Feedforward Neural Network.py`](src/Feedforward%20Neural%20Network.py): Main feedforward neural network implementation and XOR example.
- [`src/NeuralNetwork.py`](src/NeuralNetwork.py): Simple neural network class (single hidden layer) and utilities.
- [`examples/test FNN.ipynb`](examples/test%20FNN.ipynb): Jupyter notebook for MNIST training, feature extraction (PCA), visualization, and evaluation of the feedforward neural network.
- [`examples/test nn.ipynb`](examples/test%20nn.ipynb): Jupyter notebook for testing the simple neural network.
- [`examples/models_FNN/mnist_nnm_model.pkl`](examples/models_FNN/mnist_nnm_model.pkl): Saved trained model fnn.
-  [`examples/models_FNN/pca_transformer.pkl`](examples/models_FNN/pca_transformer.pkl): Saved PCA transformer from fnn.
-  [`examples/model_nn/mnist_nn_model.pkl`](examples/model_nn/mnist_nn_model.pkl): Saved trained model nn.
- [`examples/model_nn/pca_transformer.pkl`](examples/model_nn/pca_transformer.pkl): Saved PCA transformer from nn.
>>>>>>> 43be36ce25cb14fabdf7f11a343bc3344b0767f4

## Usage

1. Run the notebooks in [`examples/`](examples/) for step-by-step training and evaluation.
2. Use the scripts in [`src/`](src/) for custom experiments or integration.

## Features

### SimpleNeuralNetwork

- **Architecture:** Single hidden layer.
- **Activation Functions:** Supports sigmoid and ReLU for the hidden layer.
- **Optimization:** Basic gradient descent with fixed learning rate.
- **Loss Function:** Mean Squared Error (MSE) for training progress.
- **Training:** Early stopping based on a predefined error limit (`limite_error`).

### FeedforwardNeuralNetwork

- **Architecture:** Multiple hidden layers, configurable via a list of sizes.
- **Activation Functions:** Supports relu and sigmoid for hidden layers, softmax for output layer.
- **Optimization:** Advanced Adam optimizer.
- **Loss Function:** Categorical cross-entropy for multi-class classification.
- **Training:** Early stopping based on a predefined error limit (`limite_error`) and provides a history

## Documentation

See [`docs/Documentation.pdf`](docs/Documentation.pdf) for detailed documentation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file
