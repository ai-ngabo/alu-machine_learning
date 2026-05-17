#!/usr/bin/env python3
"""
Module 0-neuron
Defines a single neuron performing binary classification.
"""

import numpy as np

class Neuron:
    """Defines a single neuron performing binary classification"""


    def __init__(self, nx):
        """
        Initialize a Neuron.

        Parameters
        ----------
        nx : int
            Number of input features to the neuron.

        Raises
        ------
        TypeError
            If nx is not an integer.
        ValueError
            If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.randn(1, nx)  # weights vector
        self.b = 0                       # bias
        self.A = 0                       # activated output

