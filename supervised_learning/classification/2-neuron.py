#!/usr/bin/env python3
"""
Module 1-neuron
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
            raise ValueError("nx must be positive")

        # Private attributes
        self.__W = np.random.randn(1, nx)  # weights vector
        self.__b = 0                       # bias
        self.__A = 0                       # activated output

    @property
    def W(self):
        """Getter for weights vector"""
        return self.__W

    @property
    def b(self):
        """Getter for bias"""
        return self.__b

    @property
    def A(self):
        """Getter for activated output"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates forward propagation of the neuron.

        Parameters
        ----------
        X : numpy.ndarray
            Shape (nx, m) containing input data.
            nx = number of input features
            m = number of examples

        Returns
        -------
        numpy.ndarray
            Activated output of the neuron (__A).
        """
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))  # sigmoid activation
        return self.__A
