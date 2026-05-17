#!/usr/bin/env python3
"""
Module 8-neural_network
Defines a neural network with one hidden layer
"""

import numpy as np


class NeuralNetwork:
    """Defines a neural network with one hidden layer """

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

        # Public instance attributes
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
