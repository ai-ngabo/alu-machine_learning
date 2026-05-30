#!/usr/bin/env python3
"""
Defines a deep neural network performing binary classification.
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Represents a deep neural network.
    """

    def __init__(self, nx, layers):
        """
        Initializes the deep neural network.

        Parameters:
            nx (int): The number of input features.
            layers (list): The number of nodes in each layer of the network.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            # First layer uses nx, subsequent layers use the size of the previous layer
            n_in = nx if i == 0 else layers[i - 1]
            n_out = layers[i]

            # He et al. initialization
            self.__weights[f"W{i + 1}"] = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
            # Bias initialization
            self.__weights[f"b{i + 1}"] = np.zeros((n_out, 1))

    @property
    def L(self):
        """Getter for the number of layers."""
        return self.__L

    @property
    def cache(self):
        """Getter for the intermediary values cache."""
        return self.__cache

    @property
    def weights(self):
        """Getter for the weights and biases dictionary."""
        return self.__weights
