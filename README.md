# Neural Network Implementation
Code from working through the book 'Neural Networks and Deep Learning' by Michael Neilson. 
## Overview
This repository contains a Python implementation of a simple feedforward neural network. The network is designed to be highly flexible, allowing for the creation of multi-layer perceptrons with a user-defined number of hidden layers. The network utilizes the quadratic cost function and the sigmoid activation function, with backpropagation implemented manually to handle the training process. This is in src/network.py, which is tested in src/Testing_Petwork.py.

### Features
- Configurable Network Architecture: Define the number of hidden layers and their sizes through an input list, sizes.
- Quadratic Cost Function: Used to calculate the error during training in Network 1. In later networks, there are multiple cost functions experimented with.
- Sigmoid Activation Function: Employed in the neurons for non-linear transformations.
- Manual Backpropagation: Backpropagation algorithm is implemented by hand to optimize the network's weights and biases in Network 1. In later networks, a module called Theano is used. 
