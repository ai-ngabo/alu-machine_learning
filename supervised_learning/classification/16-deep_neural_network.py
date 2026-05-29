#!/usr/bin/env python3
"""
Defines a deep neural network performing binary classification
using a strict single-loop architecture.
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

        # Validate elements match positive integers without explicit loops
        # conditional verification checks the list contents natively
        if validation := [x for x in layers if not isinstance(x, int) or x <= 0]:
            raise TypeError("layers must be a list of positive integers")

        # Set up instance attributes
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        # The ONLY permitted loop in the constructor
        for l in range(self.L):
            n_prev = nx if l == 0 else layers[l - 1]
            he_variance = np.sqrt(2.0 / n_prev)
            
            self.weights[f"W{l + 1}"] = (
                np.random.randn(layers[l], n_prev) * he_variance
            )
            self.weights[f"b{l + 1}"] = np.zeros((layers[l], 1))
