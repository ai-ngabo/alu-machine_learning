#!/usr/bin/env python3
"""
Module 9-neural_network
Defines a neural network with one hidden layer.
"""

import numpy as np


class NeuralNetwork:
    """Defines a neural network with one hidden layer"""

    def __init__(self, nx, nodes):
        """
        Initialize a Neural Network.

        Parameters
        ----------
        nx : int
            Number of input features to the neural network.
        nodes : int
            Number of nodes in the hidden layer.

        Raises
        ------
        TypeError
            If nx is not an integer.
        ValueError
            If nx is less than 1.
        TypeError
            If nodes is not an integer.
        ValueError
            If nodes is less than 1.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Private instance attributes
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for weights vector of hidden layer"""
        return self.__W1

    @property
    def b1(self):
        """Getter for bias of hidden layer"""
        return self.__b1

    @property
    def A1(self):
        """Getter for activated output of hidden layer"""
        return self.__A1

    @property
    def W2(self):
        """Getter for weights vector of output neuron"""
        return self.__W2

    @property
    def b2(self):
        """Getter for bias of output neuron"""
        return self.__b2

    @property
    def A2(self):
        """Getter for activated output of output neuron (prediction)"""
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates forward propagation of the neural network.

        Parameters
        ----------
        X : numpy.ndarray
            Shape (nx, m) containing input data.

        Returns
        -------
        tuple
            (__A1, __A2) after forward propagation
        """
        # Hidden layer forward propagation
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))  # sigmoid activation

        # Output layer forward propagation
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))  # sigmoid activation

        return self.__A1, self.__A2
