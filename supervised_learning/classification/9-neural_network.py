#!/usr/bin/env python3
"""
Module 8-neural_network
Defines a neural network with one hidden layer performing binary classification.
"""

import numpy as np


class NeuralNetwork:
    """Defines a neural network with one hidden layer performing binary classification"""

    def __init__(self, nx, nodes):
        """
        Initialize the NeuralNetwork.

        Parameters
        ----------
        nx : int
            Number of input features.
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
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Hidden layer parameters
        self.__W1 = np.random.randn(nodes, nx)  # weights for hidden layer
        self.__b1 = np.zeros((nodes, 1))        # bias for hidden layer
        self.__A1 = 0                           # activated output for hidden layer

        # Output layer parameters
        self.__W2 = np.random.randn(1, nodes)   # weights for output neuron
        self.__b2 = 0                           # bias for output neuron
        self.__A2 = 0                           # activated output for output neuron

    # Getter methods
    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2
