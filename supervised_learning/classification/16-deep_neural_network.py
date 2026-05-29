#!/usr/bin/env python3
"""
Defines a deep neural network performing binary classification
using dynamic layers and He et al. initialization.
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
        - nx: number of input features
        - layers: list representing the number of nodes in each layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        # Set up dynamic attributes and confirm valid layer elements
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l in range(self.L):
            if not isinstance(layers[l], int) or layers[l] <= 0:
                raise TypeError("layers must be a list of positive integers")

            # Determine input size for weight matrix dimensions
            # Layer 1 connects to features (nx), subsequent layers connect to l-1
            n_prev = nx if l == 0 else layers[l - 1]

            # He et al. Initialization: variance = 2 / n_prev
            he_variance = np.sqrt(2.0 / n_prev)
            self.weights[f"W{l + 1}"] = (
                np.random.randn(layers[l], n_prev) * he_variance
            )
            self.weights[f"b{l + 1}"] = np.zeros((layers[l], 1))
