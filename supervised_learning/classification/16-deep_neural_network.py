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
        - nx: number of input features
        - layers: list representing the number of nodes in each layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        # The ONLY loop allowed in the entire function body
        # Handles validation AND initialization together to comply with AST rules
        weights = {}
        for l in range(len(layers)):
            if not isinstance(layers[l], int) or layers[l] <= 0:
                raise TypeError("layers must be a list of positive integers")

            n_prev = nx if l == 0 else layers[l - 1]
            he_variance = np.sqrt(2.0 / n_prev)

            weights[f"W{l + 1}"] = (
                np.random.randn(layers[l], n_prev) * he_variance
            )
            weights[f"b{l + 1}"] = np.zeros((layers[l], 1))

        # Assignments happen ONLY after the loop successfully finishes validating
        self.L = len(layers)
        self.cache = {}
        self.weights = weights
