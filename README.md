# Neural Network for Cancer Diagnosis

This project implements a neural network from scratch in C++ for predicting cancer diagnosis. The network is built with object-oriented programming (OOP) principles to ensure modularity and reusability of code. The neural network uses machine learning techniques such as backpropagation and gradient descent for training and optimization. The model achieved an accuracy score of 91% in predicting cancer diagnoses.

## Features

- **Object-Oriented Design:** The neural network is structured using OOP principles, enabling clean, modular, and reusable code.
  - **Network Class:** Handles the structure of the neural network, including the layers, activation functions, and weight initialization.
  - **Node Class:** Represents a node in the network, managing inputs, weights, and activation functions. This class encapsulates the behavior and attributes of individual neurons, allowing for easy modifications and extensions.
  - **Connection Class:** Represents the connections between nodes (edges), including weight management and gradient calculations. This class facilitates the interaction between nodes and supports the backpropagation algorithm.
- **Training and Optimization:** Implements the backpropagation and gradient descent algorithms to train the network.
- **Data Handling:** Includes functionality to load and preprocess cancer diagnosis data for training and evaluation.